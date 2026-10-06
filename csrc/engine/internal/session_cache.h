#pragma once

#include "mfq_paged_prefix_cache.h"
#include "step_sequence.h"
#include <future>
#include <thread>
#include <variant>
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

    ~SessionCache() {
        // Disk tasks own only the cache. Model/state teardown never waits for persistence.
        if (paged_cache_) {
            try { std::thread([cache = paged_cache_] { cache->flush(); }).detach(); }
            catch (...) {} // Thread creation failure falls back to ordinary cache destruction.
        }
    }

    void limit_snapshot_bytes(std::size_t bytes) { snapshots_.limit_bytes(bytes); }

    bool persistent_prefix_enabled() const noexcept { return static_cast<bool>(paged_cache_); }

    template <typename Model>
    Restore restore_best(Model &model, Predictor *mtp, const std::string &requested_session,
        const std::vector<int64_t> &prompt, size_t maximum_prefix_tokens,
        const std::string &input_key = {}) {
        return mfq::finish_steps(restore_steps(model, mtp, requested_session, prompt,
            maximum_prefix_tokens, input_key));
    }

    template <typename Model>
    mfq::StepSequence<Restore> restore_steps(Model &model, Predictor *mtp,
        std::string requested_session, const std::vector<int64_t> &prompt,
        size_t maximum_prefix_tokens, std::string input_key = {}) {
        if (!supported_ || clearing()) { co_yield Restore{}; co_return; }
        if (paged_cache_ && mtp == nullptr && input_key.empty()) {
            auto sequence = restore_paged(model, requested_session, prompt, maximum_prefix_tokens);
            while (auto step = sequence.next()) co_yield std::move(step);
            co_return;
        }
        if (!model.supports_text_session_state()) { co_yield Restore{}; co_return; }
        auto match = snapshots_.find_best(requested_session, prompt, maximum_prefix_tokens,
            [&](const Snapshot &state) {
                return state.input_key == input_key &&
                    (mtp == nullptr || (mtp->supports_session_state() && state.mtp.has_value()));
            });
        if (!match) { co_yield Restore{}; co_return; }
        // Controls may evict the cache between steps. Retain native buffers by value.
        auto state = *match->state;
        snapshots_.record_hit(*match);
        auto sequence = restore_model(model, state);
        bool failed = false;
        while (true) {
            mfq::StepResult<std::monostate> step;
            try { step = sequence.next(); }
            catch (const std::exception &error) {
                snapshots_.close(match->session_id);
                model.reset(1);
                if (mtp) mtp->reset(1);
                failed = true;
                std::cerr << "runtime_session_cache action=invalidate error=" << error.what() << '\n';
            }
            if (failed || !step) break;
            co_yield step.state;
        }
        if (failed) { co_yield Restore{}; co_return; }
        Restore restored{state.tokens.size(), {}};
        try {
            if (mtp) {
                mtp->restore_session_state(*state.mtp);
                restored.mtp_last_target_hidden = state.mtp->last_target_hidden;
            }
        } catch (const std::exception &error) {
            snapshots_.close(match->session_id);
            model.reset(1);
            mtp->reset(1);
            restored = {};
            std::cerr << "runtime_session_cache action=invalidate error=" << error.what() << '\n';
        }
        co_yield std::move(restored);
    }

    mfq::StepSequence<std::monostate> store_steps(std::string session_id, Snapshot state) {
        if (!supported_ || clearing()) { co_yield std::monostate{}; co_return; }
        if (paged_cache_ && state.input_key.empty() && !state.mtp.has_value()) {
            auto sequence = store_paged(session_id, state);
            while (auto step = sequence.next()) co_yield std::move(step);
        } else store_memory(session_id, std::move(state));
        co_yield std::monostate{};
    }

    void store(const std::string &session_id, Snapshot state) {
        (void)mfq::finish_steps(store_steps(session_id, std::move(state)));
    }

    void store_memory(const std::string &session_id, Snapshot state) {
        if (!supported_)
            return;
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
            const auto value = clearing() ? mfq::cache::PagedPrefixCacheMetrics{} : paged_cache_->metrics();
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
                {"prefix_cache_skipped_writes", static_cast<double>(value.skipped_writes)},
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

    size_t clear_live_sessions() noexcept { ++epoch_; return snapshots_.clear() + paged_bindings_.clear(); }

    size_t clear() {
        ++epoch_;
        const auto snapshots = snapshots_.clear();
        paged_bindings_.clear();
        if (!paged_cache_ || clearing()) return snapshots;
        const auto blocks = paged_cache_->metrics().disk_blocks;
        std::packaged_task<size_t()> task([cache = paged_cache_] { return cache->clear(); });
        clearing_ = task.get_future();
        std::thread(std::move(task)).detach();
        return snapshots + blocks;
    }

    uint64_t trim_hot(uint64_t target_bytes) {
        if (!paged_cache_ || clearing())
            return 0;
        const auto released = paged_cache_->trim_hot(target_bytes);
        if (released > 0)
            Backend::release_host_cache();
        return released;
    }

  private:
    bool clearing() const {
        return clearing_.valid() && clearing_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    }
    template <class Model>
    static mfq::StepSequence<std::monostate> restore_model(Model &model, const Snapshot &state) {
        if constexpr (requires { model.restore_text_session_steps(state); }) {
            auto sequence = model.restore_text_session_steps(state);
            while (auto step = sequence.next()) co_yield std::move(step);
        } else model.restore_text_session_state(state);
        co_yield std::monostate{};
    }

    template <typename Model>
    mfq::StepSequence<Restore> restore_paged(Model &model, const std::string &requested_session,
        const std::vector<int64_t> &prompt, size_t maximum_prefix_tokens) {
        if (snapshots_.max_sessions() == 0 || !model.supports_paged_text_session_state() || prompt.size() < 2) {
            co_yield Restore{}; co_return;
        }
        const auto epoch = epoch_;
        const auto limit = std::min(maximum_prefix_tokens, prompt.size() - 1);
        std::vector<int64_t> candidate(prompt.begin(), prompt.begin() + limit);
        auto match = paged_cache_->match(candidate, {}, false);
        std::vector<mfq::cache::PagedPrefixPayload> payloads;
        for (const auto &block : match.blocks) {
            // One owned disk task per cache. Destroying a cancelled coroutine never joins it.
            while (read_.valid() && read_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                co_yield mfq::StepState::waiting;
            std::packaged_task<std::vector<mfq::cache::PagedPrefixPayload>()> task(
                [cache = paged_cache_, block] { return cache->load_prefix({block}); });
            read_ = task.get_future();
            std::thread(std::move(task)).detach();
            while (read_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                co_yield mfq::StepState::waiting;
            auto part = read_.get();
            if (part.empty() || epoch != epoch_) break;
            payloads.push_back(std::move(part.front()));
            co_yield mfq::StepState::advanced;
        }
        if (payloads.empty() || epoch != epoch_) {
            paged_cache_->record_match(0); co_yield Restore{}; co_return;
        }
        match.blocks.resize(payloads.size());
        match.matched_tokens = payloads.size() * paged_cache_->block_size_tokens();
        candidate.resize(match.matched_tokens);
        auto decode = decode_steps(payloads, candidate, paged_cache_->block_size_tokens());
        std::optional<Snapshot> state;
        bool failed = false, invalid = false;
        while (true) {
            mfq::StepResult<Snapshot> step;
            try { step = decode.next(); }
            catch (const StateError &error) { failed = invalid = true; }
            catch (const std::exception &error) {
                failed = true;
                std::cerr << "runtime_session_cache action=paged_codec_failed error=" << error.what() << '\n';
            }
            if (failed || !step) break;
            if (step.value) state = std::move(*step.value);
            co_yield step.state;
        }
        if (state && !failed) {
            auto restore = restore_model(model, *state);
            while (true) {
                mfq::StepResult<std::monostate> step;
                try { step = restore.next(); }
                catch (const StateError &error) { failed = invalid = true; }
                catch (const std::exception &error) {
                    failed = true;
                    model.reset(1);
                    std::cerr << "runtime_session_cache action=paged_restore_failed error=" << error.what() << '\n';
                }
                if (failed || !step) break;
                co_yield step.state;
            }
        }
        if (invalid && epoch == epoch_) {
            std::packaged_task<std::vector<mfq::cache::PagedPrefixPayload>()> task(
                [cache = paged_cache_, block = match.blocks.back()] {
                    cache->invalidate(block);
                    return std::vector<mfq::cache::PagedPrefixPayload>{};
                });
            read_ = task.get_future();
            std::thread(std::move(task)).detach();
        }
        if (failed || !state || epoch != epoch_) {
            model.reset(1); paged_cache_->record_match(0); co_yield Restore{}; co_return;
        }
        if (!requested_session.empty()) paged_bindings_.bind(requested_session, match.blocks, match.matched_tokens);
        paged_cache_->record_match(match.matched_tokens);
        if (trace_) std::cerr << "runtime_session_cache action=paged_hit session=" << requested_session
            << " reused_tokens=" << match.matched_tokens << '\n';
        co_yield Restore{match.matched_tokens, {}};
    }

    static mfq::StepSequence<Snapshot> decode_steps(
        const std::vector<mfq::cache::PagedPrefixPayload> &payloads,
        const std::vector<int64_t> &tokens, size_t block_size) {
        if constexpr (requires { Backend::decode_steps(payloads, tokens, block_size); }) {
            auto sequence = Backend::decode_steps(payloads, tokens, block_size);
            while (auto step = sequence.next()) co_yield std::move(step);
        } else co_yield Backend::decode(payloads, tokens, block_size);
    }

    static mfq::StepSequence<mfq::cache::PagedPrefixPayload> encode_steps(
        const Snapshot &state, size_t block_size, size_t index) {
        if constexpr (requires { Backend::encode_steps(state, block_size, index); }) {
            auto sequence = Backend::encode_steps(state, block_size, index);
            while (auto step = sequence.next()) co_yield std::move(step);
        } else co_yield Backend::encode(state, block_size, index);
    }

    mfq::StepSequence<std::monostate> store_paged(const std::string &session_id, const Snapshot &state) {
        if (snapshots_.max_sessions() == 0 || state.tokens.empty()) {
            co_return;
        }
        const auto epoch = epoch_;
        const auto block_size = paged_cache_->block_size_tokens();
        const auto full_blocks = state.tokens.size() / block_size;
        if (full_blocks == 0)
            co_return;

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
            auto encode = encode_steps(state, block_size, index);
            mfq::cache::PagedPrefixPayload payload;
            while (auto step = encode.next()) {
                if (step.value) payload = std::move(*step.value);
                co_yield step.state;
            }
            if (epoch != epoch_) co_return;
            if (!payload) throw std::runtime_error("paged encoder returned no payload");
            parent = paged_cache_->store(
                parent, state.tokens.data() + token_offset, block_size, std::move(payload));
            blocks.push_back(parent);
            co_yield mfq::StepState::advanced;
        }
        if (epoch != epoch_) co_return;
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
    std::future<std::vector<mfq::cache::PagedPrefixPayload>> read_;
    std::future<size_t> clearing_;
    std::uint64_t epoch_ = 0;
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
};

} // namespace mfq::engine
