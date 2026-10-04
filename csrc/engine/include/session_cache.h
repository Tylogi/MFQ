#pragma once

#include "mfq_paged_prefix_cache.h"
#include "paged_session_bindings.h"
#include "session_snapshot_cache.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mfq::engine {

// Matching, eviction and restore/store ordering are shared. Backend supplies
// native snapshot/predictor types and the paged payload codec.
template <class Backend> class SessionCache {
    using Snapshot = typename Backend::Snapshot;
    using Restore = typename Backend::Restore;
    using Predictor = typename Backend::Predictor;
    using StateError = typename Backend::StateError;

  public:
    template <class SessionConfig, class PrefixConfig>
    explicit SessionCache(const SessionConfig &session_config, const PrefixConfig &prefix_config,
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {}, bool supported = true,
        int disabled_reason = 0)
        : snapshots_(session_config.snapshots), paged_cache_(std::move(paged_cache)),
          paged_bindings_(paged_cache_, snapshots_.max_sessions()),
          paged_disk_budget_(paged_cache_ ? prefix_config.disk_bytes : 0),
          paged_hot_budget_(paged_cache_ ? prefix_config.hot_bytes : 0),
          trace_(session_config.trace), supported_(supported), disabled_reason_(disabled_reason) {}

    void limit_snapshot_bytes(std::size_t bytes) { snapshots_.limit_bytes(bytes); }

    bool persistent_prefix_enabled() const noexcept { return static_cast<bool>(paged_cache_); }

    template <typename Model>
    Restore restore_best(Model &model, Predictor *mtp, const std::string &requested_session,
        const std::vector<int64_t> &prompt, size_t maximum_prefix_tokens,
        const std::string &input_key = {}) {
        if (!supported_)
            return {};
        if (paged_cache_ && mtp == nullptr && input_key.empty()) {
            return {restore_paged(model, requested_session, prompt, maximum_prefix_tokens), {}};
        }
        if (!model.supports_text_session_state())
            return {};
        auto match = snapshots_.find_best(
            requested_session, prompt, maximum_prefix_tokens, [&](const Snapshot &state) {
            return state.input_key == input_key &&
                   (mtp == nullptr || (mtp->supports_session_state() && state.mtp.has_value()));
        });
        if (!match)
            return {};
        try {
            model.restore_text_session_state(*match->state);
            Restore restored{match->tokens(), {}};
            if (mtp != nullptr) {
                mtp->restore_session_state(*match->state->mtp);
                restored.mtp_last_target_hidden = match->state->mtp->last_target_hidden;
            }
            snapshots_.record_hit(*match);
            if (trace_) {
                std::cerr << "runtime_session_cache action=hit session=" << requested_session
                          << " source=" << match->session_id << " reused_tokens=" << match->tokens()
                          << " prefill_tokens=" << prompt.size() - match->tokens() << std::endl;
            }
            return restored;
        } catch (const std::exception &error) {
            const auto selected_session = match->session_id;
            snapshots_.erase(*match);
            model.reset(1);
            if (mtp != nullptr)
                mtp->reset(1);
            std::cerr << "runtime_session_cache action=invalidate session=" << selected_session
                      << " error=" << error.what() << std::endl;
            return {};
        }
    }

    void store(const std::string &session_id, Snapshot state) {
        if (!supported_)
            return;
        if (paged_cache_ && state.input_key.empty() && !state.mtp.has_value()) {
            store_paged(session_id, state);
            return;
        }
        if (session_id.empty() || !snapshots_.enabled())
            return;
        if (state.bytes > snapshots_.max_bytes()) {
            if (trace_) {
                std::cerr << "runtime_session_cache action=skip session=" << session_id
                          << " bytes=" << state.bytes << " budget=" << snapshots_.max_bytes()
                          << std::endl;
            }
            return;
        }
        const auto stored = snapshots_.store(
            session_id, std::move(state), [](const Snapshot &saved, const Snapshot &candidate) {
            return saved.tokens == candidate.tokens && saved.input_key == candidate.input_key;
        });
        if (trace_ && stored) {
            std::cerr << "runtime_session_cache action=store session=" << session_id
                      << " tokens=" << stored.state->tokens.size()
                      << " bytes=" << stored.state->bytes
                      << " snapshots=" << stored.session_snapshots
                      << " total_bytes=" << stored.total_bytes << std::endl;
        }
    }

    size_t fork_session(const std::string &source_session, const std::string &target_session) {
        const auto copied_paged = paged_bindings_.fork(source_session, target_session);
        const auto copied_snapshots = snapshots_.fork(source_session, target_session);
        if (trace_ && copied_snapshots > 0) {
            std::cerr << "runtime_session_cache action=fork source=" << source_session
                      << " target=" << target_session << " snapshots=" << copied_snapshots
                      << " total_bytes=" << snapshots_.metrics().bytes << std::endl;
        }
        return copied_snapshots + copied_paged;
    }

    size_t close_session(const std::string &session_id) {
        const auto released_paged = paged_bindings_.close(session_id);
        const auto released = snapshots_.close(session_id);
        if (trace_ && released.snapshots > 0) {
            std::cerr << "runtime_session_cache action=close session=" << session_id
                      << " snapshots=" << released.snapshots << " bytes=" << released.bytes
                      << " total_bytes=" << snapshots_.metrics().bytes << std::endl;
        }
        return released.snapshots + released_paged;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        if (paged_cache_) {
            const auto value = paged_cache_->metrics();
            return {
                {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
                {"prefix_cache_disabled_reason", static_cast<double>(disabled_reason_)},
                {"prefix_cache_queries", static_cast<double>(value.queries)},
                {"prefix_cache_hits", static_cast<double>(value.hits)},
                {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
                {"prefix_cache_sessions", static_cast<double>(paged_bindings_.sessions())},
                {"prefix_cache_snapshots", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_tokens", static_cast<double>(paged_bindings_.tokens())},
                {"prefix_cache_bytes", static_cast<double>(value.hot_bytes)},
                {"prefix_cache_max_sessions", static_cast<double>(snapshots_.max_sessions())},
                {"prefix_cache_max_snapshots_per_session", 1.0},
                {"prefix_cache_max_bytes", static_cast<double>(paged_hot_budget_)},
                {"prefix_cache_disk_blocks", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_disk_bytes", static_cast<double>(value.disk_bytes)},
                {"prefix_cache_disk_max_bytes", static_cast<double>(paged_disk_budget_)},
                {"prefix_cache_hot_blocks", static_cast<double>(value.hot_blocks)},
                {"prefix_cache_hot_bytes", static_cast<double>(value.hot_bytes)},
                {"prefix_cache_pending_writes", static_cast<double>(value.pending_writes)},
                {"prefix_cache_pending_bytes", static_cast<double>(value.pending_bytes)},
                {"prefix_cache_pending_max_bytes", static_cast<double>(value.pending_max_bytes)},
                {"prefix_cache_writes", static_cast<double>(value.writes)},
                {"prefix_cache_deduplicated_writes",
                    static_cast<double>(value.deduplicated_writes)},
                {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                {"prefix_cache_hot_hits", static_cast<double>(value.hot_hits)},
                {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
            };
        }
        const auto value = snapshots_.metrics();
        return {
            {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
            {"prefix_cache_disabled_reason", static_cast<double>(disabled_reason_)},
            {"prefix_cache_queries", static_cast<double>(value.queries)},
            {"prefix_cache_hits", static_cast<double>(value.hits)},
            {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
            {"prefix_cache_sessions", static_cast<double>(value.sessions)},
            {"prefix_cache_snapshots", static_cast<double>(value.snapshots)},
            {"prefix_cache_tokens", static_cast<double>(value.tokens)},
            {"prefix_cache_bytes", static_cast<double>(value.bytes)},
            {"prefix_cache_max_sessions", static_cast<double>(snapshots_.max_sessions())},
            {"prefix_cache_max_snapshots_per_session",
                static_cast<double>(snapshots_.max_snapshots_per_session())},
            {"prefix_cache_max_bytes", static_cast<double>(snapshots_.max_bytes())},
        };
    }

    size_t clear_live_sessions() noexcept { return snapshots_.clear() + paged_bindings_.clear(); }

    size_t clear() {
        const auto snapshots = snapshots_.clear();
        paged_bindings_.clear();
        return snapshots + (paged_cache_ ? paged_cache_->clear() : 0);
    }

    uint64_t trim_hot(uint64_t target_bytes) {
        if (!paged_cache_)
            return 0;
        const auto released = paged_cache_->trim_hot(target_bytes);
        if (released > 0)
            Backend::release_host_cache();
        return released;
    }

  private:
    template <typename Model>
    size_t restore_paged(Model &model, const std::string &requested_session,
        const std::vector<int64_t> &prompt, size_t maximum_prefix_tokens) {
        if (snapshots_.max_sessions() == 0 || !model.supports_paged_text_session_state() ||
            prompt.size() < 2) {
            return 0;
        }
        const auto limit = std::min(maximum_prefix_tokens, prompt.size() - 1);
        std::vector<int64_t> candidate(
            prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(limit));
        auto match = paged_cache_->match(candidate, {}, false);
        if (match.matched_tokens == 0) {
            paged_cache_->record_match(0);
            return 0;
        }

        auto payloads = paged_cache_->load_prefix(match.blocks);
        if (payloads.empty()) {
            paged_cache_->record_match(0);
            return 0;
        }
        if (payloads.size() != match.blocks.size()) {
            match.blocks.resize(payloads.size());
            match.matched_tokens = payloads.size() * paged_cache_->block_size_tokens();
        }
        std::vector<int64_t> matched_tokens(
            prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(match.matched_tokens));
        const auto fail = [&](const char *action, const std::exception &error, bool invalidate) {
            if (invalidate && !match.blocks.empty()) {
                paged_cache_->invalidate(match.blocks.back());
            }
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr << "runtime_session_cache action=" << action
                      << " session=" << requested_session << " error=" << error.what() << std::endl;
            return size_t{0};
        };
        const char *invalid_action = "paged_codec_invalidate";
        const char *failure_action = "paged_codec_failed";
        try {
            auto state =
                Backend::decode(payloads, matched_tokens, paged_cache_->block_size_tokens());
            invalid_action = "paged_restore_invalidate";
            failure_action = "paged_restore_failed";
            model.restore_text_session_state(state);
            if (!requested_session.empty()) {
                paged_bindings_.bind(requested_session, match.blocks, match.matched_tokens);
            }
            paged_cache_->record_match(match.matched_tokens);
            if (trace_) {
                std::cerr << "runtime_session_cache action=paged_hit "
                          << "session=" << requested_session
                          << " reused_tokens=" << match.matched_tokens
                          << " prefill_tokens=" << prompt.size() - match.matched_tokens
                          << std::endl;
            }
            return match.matched_tokens;
        } catch (const StateError &error) {
            return fail(invalid_action, error, true);
        } catch (const std::exception &error) {
            return fail(failure_action, error, false);
        }
    }

    void store_paged(const std::string &session_id, const Snapshot &state) {
        if (snapshots_.max_sessions() == 0 || state.tokens.empty()) {
            return;
        }
        const auto block_size = paged_cache_->block_size_tokens();
        const auto full_blocks = state.tokens.size() / block_size;
        if (full_blocks == 0)
            return;

        auto existing = paged_cache_->match(state.tokens, {}, false);
        if (existing.blocks.size() > full_blocks) {
            throw std::runtime_error("paged prefix match exceeds the session state");
        }
        auto blocks = std::move(existing.blocks);
        mfq::cache::BlockHash parent{};
        if (!blocks.empty())
            parent = blocks.back();
        for (size_t index = blocks.size(); index < full_blocks; ++index) {
            const auto token_offset = index * block_size;
            auto payload = Backend::encode(state, block_size, index);
            parent = paged_cache_->store(
                parent, state.tokens.data() + token_offset, block_size, std::move(payload));
            blocks.push_back(parent);
        }
        if (!session_id.empty()) {
            paged_bindings_.bind(session_id, std::move(blocks), full_blocks * block_size);
        }
        if (trace_) {
            std::cerr << "runtime_session_cache action=paged_store "
                      << "session=" << session_id << " tokens=" << full_blocks * block_size
                      << " blocks=" << full_blocks << std::endl;
        }
    }

    mfq::engine::SessionSnapshotCache<Snapshot> snapshots_;
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache_;
    mfq::engine::PagedSessionBindings paged_bindings_;
    uint64_t paged_disk_budget_ = 0;
    uint64_t paged_hot_budget_ = 0;
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
};

} // namespace mfq::engine
