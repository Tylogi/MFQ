#pragma once

#include "prefix_cache.h"
#include "step_sequence.h"
#include <future>
#include <thread>
#include <variant>
#include "paged.h"
#include "snapshot.h"

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
    template <class Visitor> void visit_snapshots(Visitor&& visitor) const {
        snapshots_.visit_states(std::forward<Visitor>(visitor));
    }
    template <class SessionConfig, class PrefixConfig>
    explicit SessionCache(const SessionConfig &session_config, const PrefixConfig &prefix_config,
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {}, bool supported = true,
        int disabled_reason = 0, bool native_hot_snapshots = false)
        : snapshots_([&] {
              auto configuration = session_config.snapshots;
              if (native_hot_snapshots) configuration.max_bytes = prefix_config.enabled ? prefix_config.hot_bytes : 0;
              return configuration;
          }()), paged_cache_(std::move(paged_cache)),
          paged_bindings_(paged_cache_, snapshots_.max_sessions()),
          paged_disk_budget_(paged_cache_ ? prefix_config.disk_bytes : 0),
          paged_hot_budget_(paged_cache_ ? prefix_config.hot_bytes : 0),
          paged_ram_budget_(paged_cache_ && native_hot_snapshots ? prefix_config.ram_bytes : 0),
          trace_(session_config.trace), supported_(supported), disabled_reason_(disabled_reason),
          native_hot_snapshots_(native_hot_snapshots) {}

    ~SessionCache() {
        // Disk tasks own only the cache. Model/state teardown never waits for persistence.
        if (paged_cache_) {
            try { std::thread([cache = paged_cache_] { cache->blocks.flush(); }).detach(); }
            catch (...) {} // Thread creation failure falls back to ordinary cache destruction.
        }
    }

    void limit_snapshot_bytes(std::size_t bytes) { snapshots_.limit_bytes(bytes); }

    bool persistent_prefix_enabled() const noexcept {
        return static_cast<bool>(paged_cache_) || (native_hot_snapshots_ && supported_ && snapshots_.enabled());
    }

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
        if (paged_cache_ && !native_hot_snapshots_ && input_key.empty()) {
            auto sequence = restore_paged(model, requested_session, prompt, maximum_prefix_tokens, mtp);
            while (auto step = sequence.next()) co_yield std::move(step);
            co_return;
        }
        if (!model.supports_text_session_state()) { co_yield Restore{}; co_return; }
        auto match = snapshots_.find_best(
            native_hot_snapshots_ && requested_session.empty() ? "\x01prefix-global" : requested_session,
            prompt, maximum_prefix_tokens, [&](const Snapshot &state) {
                return state.input_key == input_key &&
                    (mtp == nullptr || (mtp->supports_session_state() && state.mtp.has_value()));
            });
        if (!match) {
            if (paged_cache_ && native_hot_snapshots_ && input_key.empty()) {
                auto sequence = restore_paged(model, requested_session, prompt, maximum_prefix_tokens, mtp);
                while (auto step = sequence.next()) co_yield std::move(step);
            } else co_yield Restore{};
            co_return;
        }
        // Controls may evict the cache between steps. Retain native buffers by value.
        auto state = *match->state;
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
            snapshots_.record_hit(*match);
            if (paged_cache_ && native_hot_snapshots_) paged_cache_->record_match(state.tokens.size());
            if (trace_) {
                std::cerr << "runtime_session_cache action=hit session=" << requested_session
                          << " source=" << match->session_id << " reused_tokens=" << state.tokens.size()
                          << " prefill_tokens=" << prompt.size() - state.tokens.size() << std::endl;
            }
        } catch (const std::exception &error) {
            snapshots_.close(match->session_id);
            model.reset(1);
            if (mtp) mtp->reset(1);
            restored = {};
            std::cerr << "runtime_session_cache action=invalidate error=" << error.what() << '\n';
        }
        co_yield std::move(restored);
    }

    mfq::StepSequence<std::monostate> store_steps(std::string session_id, Snapshot state) {
        if (!supported_ || clearing()) { co_yield std::monostate{}; co_return; }
        bool paged_mtp=false;
        if constexpr(requires { Backend::paged_mtp(state); })paged_mtp=Backend::paged_mtp(state);
        if (paged_cache_ && state.input_key.empty() && (!state.mtp.has_value() || paged_mtp)) {
            auto sequence = store_paged(session_id, state);
            while (auto step = sequence.next()) co_yield std::move(step);
            if (native_hot_snapshots_) store_memory(session_id, std::move(state));
        } else store_memory(session_id, std::move(state));
        co_yield std::monostate{};
    }

    void store(const std::string &session_id, Snapshot state) {
        (void)mfq::finish_steps(store_steps(session_id, std::move(state)));
    }

    void store_memory(const std::string &session_id, Snapshot state) {
        if (!supported_)
            return;
        if ((!native_hot_snapshots_ && session_id.empty()) || !snapshots_.enabled())
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
            native_hot_snapshots_ && session_id.empty() ? "\x01prefix-" + std::to_string(++anonymous_snapshots_) : session_id,
            std::move(state), [](const Snapshot &saved, const Snapshot &candidate) {
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
            const auto native = snapshots_.metrics();
            std::vector<std::pair<std::string, double>> result{
                {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
                {"prefix_cache_disabled_reason", static_cast<double>(disabled_reason_)},
                {"prefix_cache_queries", static_cast<double>(value.queries)},
                {"prefix_cache_hits", static_cast<double>(value.hits)},
                {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
                {"prefix_cache_sessions", static_cast<double>(paged_bindings_.sessions())},
                {"prefix_cache_snapshots", static_cast<double>(value.blocks ? value.blocks : native.snapshots)},
                {"prefix_cache_tokens", static_cast<double>(paged_bindings_.tokens())},
                {"prefix_cache_bytes", static_cast<double>(native_hot_snapshots_ ? native.bytes : value.hot_bytes)},
                {"prefix_cache_max_sessions", static_cast<double>(snapshots_.max_sessions())},
                {"prefix_cache_max_snapshots_per_session", 1.0},
                {"prefix_cache_max_bytes", static_cast<double>(paged_hot_budget_)},
                {"prefix_cache_disk_blocks", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_disk_bytes", static_cast<double>(value.disk_bytes)},
                {"prefix_cache_disk_max_bytes", static_cast<double>(paged_disk_budget_)},
                {"prefix_cache_hot_blocks", static_cast<double>(native_hot_snapshots_ ? native.snapshots : value.hot_blocks)},
                {"prefix_cache_hot_bytes", static_cast<double>(native_hot_snapshots_ ? native.bytes : value.hot_bytes)},
                {"prefix_cache_native_hot", native_hot_snapshots_ ? 1.0 : 0.0},
                {"prefix_cache_ram_bytes", static_cast<double>(native_hot_snapshots_ ? value.resident_bytes : value.hot_bytes)},
                {"prefix_cache_ram_max_bytes", static_cast<double>(paged_ram_budget_)},
                {"prefix_cache_ram_blocks", static_cast<double>(native_hot_snapshots_ ? value.hot_blocks : 0)},
                {"prefix_cache_ram_hits", static_cast<double>(native_hot_snapshots_ ? value.hot_hits : 0)},
                {"prefix_cache_hot_pressure_bytes", static_cast<double>(native_hot_snapshots_ ? 0 : value.hot_pressure_bytes)},
                {"prefix_cache_ram_pressure_bytes", static_cast<double>(native_hot_snapshots_ ? value.hot_pressure_bytes : 0)},
                {"prefix_cache_dynamic_budget", 1.0},
                {"prefix_cache_pending_writes", static_cast<double>(value.pending_writes)},
                {"prefix_cache_pending_bytes", static_cast<double>(value.pending_bytes)},
                {"prefix_cache_pending_max_bytes", static_cast<double>(value.pending_max_bytes)},
                {"prefix_cache_writes", static_cast<double>(value.writes)},
                {"prefix_cache_skipped_writes", static_cast<double>(value.skipped_writes)},
                {"prefix_cache_deduplicated_writes",
                    static_cast<double>(value.deduplicated_writes)},
                {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                {"prefix_cache_hot_hits", static_cast<double>(native_hot_snapshots_ ? native.hits : value.hot_hits)},
                {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
                {"prefix_cache_low_disk_space_skips", static_cast<double>(value.low_disk_space_skips)},
                {"prefix_cache_failed_writes", static_cast<double>(value.failed_writes)},
            };
            if (native_hot_snapshots_)
                result.emplace_back("prefix_cache_resident_sessions", static_cast<double>(native.sessions));
            return result;
        }
        const auto value = snapshots_.metrics();
        std::vector<std::pair<std::string, double>> result{
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
        if (native_hot_snapshots_) {
            result.insert(result.end(), {{"prefix_cache_native_hot", 1.0}, {"prefix_cache_dynamic_budget", 1.0},
                {"prefix_cache_hot_bytes", static_cast<double>(value.bytes)},
                {"prefix_cache_hot_blocks", static_cast<double>(value.snapshots)},
                {"prefix_cache_resident_sessions", static_cast<double>(value.sessions)},
                {"prefix_cache_ram_bytes", 0.0}});
        }
        return result;
    }

    size_t clear_live_sessions() noexcept { ++epoch_; return snapshots_.clear() + paged_bindings_.clear(); }

    size_t clear() {
        ++epoch_;
        const auto snapshots = snapshots_.clear();
        paged_bindings_.clear();
        if (!paged_cache_ || clearing()) return snapshots;
        const auto blocks = paged_cache_->metrics().disk_blocks;
        std::packaged_task<size_t()> task([cache = paged_cache_] { return cache->blocks.clear(); });
        clearing_ = task.get_future();
        std::thread(std::move(task)).detach();
        return snapshots + blocks;
    }

    uint64_t trim_hot(uint64_t target_bytes) {
        if (native_hot_snapshots_) return snapshots_.trim(target_bytes);
        if (!paged_cache_ || clearing()) return 0;
        const auto released = paged_cache_->blocks.trim_hot(target_bytes);
        if (released > 0)
            Backend::release_host_cache();
        return released;
    }

    uint64_t set_hot_limit(uint64_t max_bytes) {
        if (native_hot_snapshots_) {
            paged_hot_budget_ = max_bytes;
            return snapshots_.set_limit(max_bytes);
        }
        if (!paged_cache_) return 0;
        paged_hot_budget_ = max_bytes;
        const auto released = paged_cache_->blocks.set_hot_limit(max_bytes);
        if (released > 0) Backend::release_host_cache();
        return released;
    }

    uint64_t set_disk_limit(uint64_t max_bytes) {
        if (!paged_cache_) return 0;
        paged_disk_budget_ = max_bytes;
        return paged_cache_->blocks.set_disk_limit(max_bytes);
    }

    uint64_t set_ram_limit(uint64_t max_bytes) {
        if (!paged_cache_ || !native_hot_snapshots_) return 0;
        paged_ram_budget_ = max_bytes;
        const auto released = paged_cache_->blocks.set_hot_limit(max_bytes);
        if (released) Backend::release_host_cache();
        return released;
    }

    void reserve_ram(uint64_t bytes) {
        if (paged_cache_ && native_hot_snapshots_ && paged_cache_->blocks.reserve_ram(bytes))
            Backend::release_host_cache();
    }

    uint64_t refresh_disk_index() {
        if (!paged_cache_) return 0;
        const auto removed = paged_cache_->blocks.refresh_disk_index();
        if (removed > 0) Backend::release_host_cache();
        return removed;
    }

  private:
    bool clearing() const {
        return clearing_.valid() &&
            clearing_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
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
        const std::vector<int64_t> &prompt, size_t maximum_prefix_tokens, Predictor* mtp) {
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
                [cache = paged_cache_, block] { return cache->blocks.load_prefix({block}); });
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
        bool invalid_metadata=false;
        try {
            if constexpr(requires { Backend::decodable_blocks(payloads); })
                payloads.resize(Backend::decodable_blocks(payloads));
            if(!payloads.empty()) {
                match.blocks.resize(payloads.size());
                match.matched_tokens=payloads.size()*paged_cache_->block_size_tokens();
                if constexpr(requires { Backend::token_count(payloads.front()); }) {
                    match.matched_tokens=0;
                    for(const auto& payload:payloads) {
                        const auto count=Backend::token_count(payload);
                        if(count==0 || count>paged_cache_->block_size_tokens() ||
                            match.matched_tokens>limit || count>limit-match.matched_tokens)
                            throw StateError("invalid paged prefix token count");
                        match.matched_tokens+=count;
                    }
                }
                if(match.matched_tokens>limit)throw StateError("paged prefix exceeds requested limit");
            }
        } catch(const StateError&) { invalid_metadata=true; }
        if(payloads.empty() || invalid_metadata) {
            paged_cache_->record_match(0); co_yield Restore{}; co_return;
        }
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
        if (state && !failed && mtp && (!mtp->supports_session_state() || !state->mtp)) {
            paged_cache_->record_match(0); co_yield Restore{}; co_return;
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
        Restore restored{match.matched_tokens, {}};
        if (state && !failed && mtp) {
            try {
                mtp->restore_session_state(*state->mtp);
                restored.mtp_last_target_hidden=state->mtp->last_target_hidden;
            } catch(const StateError&) { failed=invalid=true; }
            catch(const std::exception& error) {
                failed=true;
                std::cerr<<"runtime_session_cache action=paged_mtp_restore_failed error="<<error.what()<<'\n';
            }
        }
        if (invalid && epoch == epoch_) {
            std::packaged_task<std::vector<mfq::cache::PagedPrefixPayload>()> task(
                [cache = paged_cache_, block = match.blocks.back()] {
                    cache->blocks.invalidate(block);
                    return std::vector<mfq::cache::PagedPrefixPayload>{};
                });
            read_ = task.get_future();
            std::thread(std::move(task)).detach();
        }
        if (failed || !state || epoch != epoch_) {
            model.reset(1);
            if(mtp)mtp->reset(1);
            paged_cache_->record_match(0); co_yield Restore{}; co_return;
        }
        if (native_hot_snapshots_)
            snapshots_.store(requested_session.empty()
                    ? "\x01prefix-" + std::to_string(++anonymous_snapshots_)
                    : requested_session,
                std::move(*state), [](const Snapshot& saved, const Snapshot& candidate) {
                    return saved.tokens == candidate.tokens && saved.input_key == candidate.input_key;
                });
        if (!requested_session.empty()) paged_bindings_.bind(requested_session, match.blocks, match.matched_tokens);
        paged_cache_->record_match(match.matched_tokens);
        if (trace_) std::cerr << "runtime_session_cache action=paged_hit session=" << requested_session
            << " reused_tokens=" << match.matched_tokens << '\n';
        co_yield std::move(restored);
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
        bool partial=false;
        if constexpr(requires { Backend::partial_blocks(state); })partial=Backend::partial_blocks(state);
        const auto full_blocks = state.tokens.size() / block_size + (partial && state.tokens.size()%block_size ? 1 : 0);
        if (full_blocks == 0)
            co_return;

        auto existing = paged_cache_->match(state.tokens, {}, false);
        // A shorter tail cannot become the parent of a subsequent full block.
        existing.blocks.resize(std::min(existing.blocks.size(),existing.matched_tokens/block_size));
        if (existing.blocks.size() > full_blocks) {
            throw std::runtime_error("paged prefix match exceeds the session state");
        }
        auto blocks = std::move(existing.blocks);
        mfq::cache::BlockHash parent{};
        if (!blocks.empty())
            parent = blocks.back();
        for (size_t index = blocks.size(); index < full_blocks; ++index) {
            const auto token_offset = index * block_size;
            const auto count=std::min(block_size,state.tokens.size()-token_offset);
            auto encode = encode_steps(state, block_size, index);
            mfq::cache::PagedPrefixPayload payload;
            while (auto step = encode.next()) {
                if (step.value) payload = std::move(*step.value);
                co_yield step.state;
            }
            if (epoch != epoch_) co_return;
            if (!payload) throw std::runtime_error("paged encoder returned no payload");
            parent = paged_cache_->store(
                parent, state.tokens.data() + token_offset, count, std::move(payload));
            blocks.push_back(parent);
            co_yield mfq::StepState::advanced;
        }
        if (epoch != epoch_) co_return;
        if (!session_id.empty()) {
            paged_bindings_.bind(session_id, std::move(blocks), partial?state.tokens.size():full_blocks * block_size);
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
    uint64_t paged_ram_budget_ = 0;
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
    bool native_hot_snapshots_ = false;
    std::uint64_t anonymous_snapshots_ = 0;
};

} // namespace mfq::engine
