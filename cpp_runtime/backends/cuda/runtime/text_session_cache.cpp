#include "text_session_cache.h"

#include "causal_lm.h"
#include "mtp.h"
#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mfq::cuda::internal {

static uint64_t cuda_cache_environment_bytes(
        const char * name, uint64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char * end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

static std::filesystem::path default_cuda_prefix_cache_directory() {
    if (const char * configured =
            std::getenv("MFQ_RUNTIME_PREFIX_CACHE_DIR")) {
        if (configured[0] != '\0') return configured;
    }
#ifdef _WIN32
    if (const char * local = std::getenv("LOCALAPPDATA")) {
        if (local[0] != '\0') {
            return std::filesystem::path(local) /
                "TyloQuant" / "MFQ" / "prefix-cache";
        }
    }
#else
    if (const char * xdg = std::getenv("XDG_CACHE_HOME")) {
        if (xdg[0] != '\0') {
            return std::filesystem::path(xdg) /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
    if (const char * home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" /
                "tyloquant" / "mfq" / "prefix-cache";
        }
    }
#endif
    return std::filesystem::temp_directory_path() /
        "tyloquant-mfq-prefix-cache";
}

static std::string cuda_prefix_cache_compatibility_key(
        const mfq::ModelSource& source,
        int64_t max_position_embeddings) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=cuda-full-attention-kv-v1\n"
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto& path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " +
                error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source timestamp: " +
                error.message());
        }
        const auto modified_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(modified.time_since_epoch()).count();
        key << "source=" << index << ':' << path.filename().string() << ':'
            << size << ':' << modified_ns << '\n';
    }
    for (const auto& tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':'
            << tensor.nbytes << '\n';
    }
    return key.str();
}

std::shared_ptr<mfq::cache::PagedPrefixCache>
make_cuda_paged_prefix_cache(
        const mfq::ModelSource& source,
        int64_t max_position_embeddings,
        bool supports_paged_text_session_state) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if ((format != source.metadata().end() &&
         format->second == "hf-safetensors") ||
        !supports_paged_text_session_state) {
        return {};
    }
    if (const char * disabled =
            std::getenv("MFQ_RUNTIME_DISABLE_PREFIX_CACHE")) {
        if (disabled[0] == '1') return {};
    }
    const auto block_size = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS", 256);
    if (block_size == 0 || block_size > 65536) {
        throw std::runtime_error(
            "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS must be in [1, 65536]");
    }
    mfq::cache::PagedPrefixCacheConfig config;
    config.cache_dir = default_cuda_prefix_cache_directory();
    config.compatibility_key =
        cuda_prefix_cache_compatibility_key(
            source, max_position_embeddings);
    config.block_size_tokens = static_cast<size_t>(block_size);
    config.max_disk_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES",
        100ULL * 1024ULL * 1024ULL * 1024ULL);
    config.max_hot_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES",
        2ULL * 1024ULL * 1024ULL * 1024ULL);
    config.max_pending_writes = static_cast<size_t>(
        cuda_cache_environment_bytes(
            "MFQ_RUNTIME_PREFIX_CACHE_PENDING_WRITES", 64));
    config.max_pending_bytes = cuda_cache_environment_bytes(
        "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES",
        512ULL * 1024ULL * 1024ULL);
    return std::make_shared<mfq::cache::PagedPrefixCache>(
        std::move(config));
}

struct TextSessionCache::Impl {
private:
    struct PagedBinding {
        std::vector<mfq::cache::BlockHash> blocks;
        size_t tokens = 0;
        uint64_t last_used = 0;
    };

public:
    explicit Impl(
            std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {},
            bool supported = true,
            int disabled_reason = 0)
        : paged_cache_(std::move(paged_cache)),
          supported_(supported),
          disabled_reason_(disabled_reason) {
        const char * entries =
            std::getenv("MFQ_RUNTIME_MAX_KV_SESSIONS");
        if (entries != nullptr) {
            max_sessions_ = static_cast<size_t>(std::strtoull(
                entries, nullptr, 10));
        }
        const char * snapshots =
            std::getenv("MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION");
        if (snapshots != nullptr) {
            max_snapshots_per_session_ = static_cast<size_t>(std::strtoull(
                snapshots, nullptr, 10));
        }
        const char * bytes =
            std::getenv("MFQ_RUNTIME_KV_SESSION_BYTES");
        if (bytes != nullptr) {
            max_bytes_ = static_cast<size_t>(std::strtoull(
                bytes, nullptr, 10));
        }
        const char * trace =
            std::getenv("MFQ_RUNTIME_TRACE_SESSION_CACHE");
        trace_ = trace != nullptr && trace[0] == '1';
        if (paged_cache_) {
            paged_disk_budget_ = cuda_cache_environment_bytes(
                "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES",
                100ULL * 1024ULL * 1024ULL * 1024ULL);
            paged_hot_budget_ = cuda_cache_environment_bytes(
                "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES",
                2ULL * 1024ULL * 1024ULL * 1024ULL);
        }
    }

    bool persistent_prefix_enabled() const noexcept {
        return static_cast<bool>(paged_cache_);
    }

    template <typename Model>
    TextSessionRestore restore_best(
            Model& model,
            MtpModule* mtp,
            const std::string & requested_session,
            const std::vector<int64_t> & prompt,
            size_t maximum_prefix_tokens,
            const std::string& input_key = {}) {
        if (!supported_) return {};
        if (paged_cache_ && mtp == nullptr && input_key.empty()) {
            return {restore_paged(
                model,
                requested_session,
                prompt,
                maximum_prefix_tokens), {}};
        }
        if (requested_session.empty() || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 ||
                max_bytes_ == 0 || !model.supports_text_session_state()) {
            return {};
        }
        ++queries_;
        std::string selected_session;
        size_t selected_snapshot = 0;
        size_t selected_tokens = 0;
        for (const auto & [session_id, history] : states_) {
            for (size_t index = 0; index < history.size(); ++index) {
                const auto & state = history[index];
                const auto & tokens = state.tokens;
                if (tokens.empty() || tokens.size() >= prompt.size() ||
                        tokens.size() > maximum_prefix_tokens ||
                        tokens.size() < selected_tokens ||
                        state.input_key != input_key ||
                        (mtp != nullptr &&
                         (!mtp->supports_session_state() ||
                          !state.mtp.has_value())) ||
                        !std::equal(
                            tokens.begin(), tokens.end(), prompt.begin())) {
                    continue;
                }
                const bool requested_tie =
                    tokens.size() == selected_tokens &&
                    session_id == requested_session &&
                    selected_session != requested_session;
                if (tokens.size() > selected_tokens || requested_tie) {
                    selected_session = session_id;
                    selected_snapshot = index;
                    selected_tokens = tokens.size();
                }
            }
        }
        if (selected_session.empty()) return {};
        auto & selected = states_.at(selected_session)[selected_snapshot];
        try {
            model.restore_text_session_state(selected);
            TextSessionRestore restored{selected_tokens, {}};
            if (mtp != nullptr) {
                mtp->restore_session_state(*selected.mtp);
                restored.mtp_last_target_hidden =
                    selected.mtp->last_target_hidden;
            }
            selected.last_used = ++clock_;
            ++hits_;
            hit_tokens_ += selected_tokens;
            if (trace_) {
                std::cerr << "runtime_session_cache action=hit session="
                          << requested_session
                          << " source=" << selected_session
                          << " reused_tokens=" << selected_tokens
                          << " prefill_tokens="
                          << prompt.size() - selected_tokens << std::endl;
            }
            return restored;
        } catch (const std::exception & error) {
            erase_snapshot(selected_session, selected_snapshot, "invalidate");
            model.reset(1);
            if (mtp != nullptr) mtp->reset(1);
            std::cerr << "runtime_session_cache action=invalidate session="
                      << selected_session << " error=" << error.what()
                      << std::endl;
            return {};
        }
    }

    void store(
            const std::string & session_id,
            TextSessionState state) {
        if (!supported_) return;
        if (paged_cache_ && state.input_key.empty() &&
                !state.mtp.has_value()) {
            store_paged(session_id, state);
            return;
        }
        if (session_id.empty() || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return;
        }
        if (state.bytes > max_bytes_) {
            if (trace_) {
                std::cerr << "runtime_session_cache action=skip session="
                          << session_id << " bytes=" << state.bytes
                          << " budget=" << max_bytes_ << std::endl;
            }
            return;
        }
        state.last_used = ++clock_;
        const uint64_t protected_clock = state.last_used;
        auto & history = states_[session_id];
        auto previous = std::find_if(
            history.begin(), history.end(),
            [&](const TextSessionState & saved) {
                return saved.tokens == state.tokens &&
                    saved.input_key == state.input_key;
            });
        if (previous != history.end()) {
            bytes_ -= previous->bytes;
            *previous = std::move(state);
        } else {
            history.push_back(std::move(state));
        }
        const auto stored = std::find_if(
            history.begin(), history.end(),
            [&](const TextSessionState & saved) {
                return saved.last_used == protected_clock;
            });
        if (stored == history.end()) {
            throw std::runtime_error("stored session snapshot is unavailable");
        }
        bytes_ += stored->bytes;
        evict_history_to_limit(session_id, protected_clock);
        evict_to_budget(session_id, protected_clock);
        sync_telemetry();
        if (trace_) {
            const auto & saved_history = states_.at(session_id);
            const auto saved = std::find_if(
                saved_history.begin(), saved_history.end(),
                [&](const TextSessionState & candidate) {
                    return candidate.last_used == protected_clock;
                });
            if (saved == saved_history.end()) {
                throw std::runtime_error(
                    "protected session snapshot was evicted");
            }
            std::cerr << "runtime_session_cache action=store session="
                      << session_id << " tokens=" << saved->tokens.size()
                      << " bytes=" << saved->bytes
                      << " snapshots=" << saved_history.size()
                      << " total_bytes=" << bytes_ << std::endl;
        }
    }

    size_t fork_session(
            const std::string & source_session,
            const std::string & target_session) {
        if (paged_cache_) {
            const auto source = paged_bindings_.find(source_session);
            if (source == paged_bindings_.end() ||
                    source_session.empty() || target_session.empty() ||
                    source_session == target_session) {
                return 0;
            }
            bind_paged_session(
                target_session,
                source->second.blocks,
                source->second.tokens);
            return source->second.blocks.size();
        }
        if (source_session.empty() || target_session.empty() ||
                source_session == target_session || max_sessions_ == 0 ||
                max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return 0;
        }
        const auto source = states_.find(source_session);
        if (source == states_.end()) return 0;
        std::vector<TextSessionState> copied = source->second;
        close_session(target_session);
        auto & target = states_[target_session];
        uint64_t protected_clock = 0;
        for (auto & snapshot : copied) {
            snapshot.last_used = ++clock_;
            protected_clock = snapshot.last_used;
            bytes_ += snapshot.bytes;
            target.push_back(std::move(snapshot));
        }
        evict_history_to_limit(target_session, protected_clock);
        evict_to_budget(target_session, protected_clock);
        const auto remaining = states_.find(target_session);
        const size_t copied_snapshots = remaining == states_.end()
            ? 0 : remaining->second.size();
        sync_telemetry();
        if (trace_) {
            std::cerr << "runtime_session_cache action=fork source="
                      << source_session << " target=" << target_session
                      << " snapshots=" << copied_snapshots
                      << " total_bytes=" << bytes_ << std::endl;
        }
        return copied_snapshots;
    }

    size_t close_session(const std::string & session_id) {
        if (paged_cache_) return close_paged_session(session_id);
        auto found = states_.find(session_id);
        if (found == states_.end()) return 0;
        const size_t released = found->second.size();
        size_t released_bytes = 0;
        for (const auto & snapshot : found->second) {
            released_bytes += snapshot.bytes;
        }
        bytes_ -= released_bytes;
        states_.erase(found);
        sync_telemetry();
        if (trace_) {
            std::cerr << "runtime_session_cache action=close session="
                      << session_id << " snapshots=" << released
                      << " bytes=" << released_bytes
                      << " total_bytes=" << bytes_ << std::endl;
        }
        return released;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        if (paged_cache_) {
            const auto value = paged_cache_->metrics();
            return {
                {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
                {"prefix_cache_disabled_reason",
                    static_cast<double>(disabled_reason_)},
                {"prefix_cache_queries", static_cast<double>(value.queries)},
                {"prefix_cache_hits", static_cast<double>(value.hits)},
                {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
                {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
                {"prefix_cache_snapshots", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
                {"prefix_cache_bytes", static_cast<double>(value.hot_bytes)},
                {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
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
                {"prefix_cache_deduplicated_writes", static_cast<double>(value.deduplicated_writes)},
                {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                {"prefix_cache_hot_hits", static_cast<double>(value.hot_hits)},
                {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
            };
        }
        return {
            {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
            {"prefix_cache_disabled_reason",
                static_cast<double>(disabled_reason_)},
            {"prefix_cache_queries", static_cast<double>(queries_.load())},
            {"prefix_cache_hits", static_cast<double>(hits_.load())},
            {"prefix_cache_hit_tokens", static_cast<double>(hit_tokens_.load())},
            {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
            {"prefix_cache_snapshots", static_cast<double>(metric_snapshots_.load())},
            {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
            {"prefix_cache_bytes", static_cast<double>(metric_bytes_.load())},
            {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
            {"prefix_cache_max_snapshots_per_session",
                static_cast<double>(max_snapshots_per_session_)},
            {"prefix_cache_max_bytes", static_cast<double>(max_bytes_)},
        };
    }

    size_t clear_live_sessions() noexcept {
        if (paged_cache_) {
            const auto sessions = paged_bindings_.size();
            for (const auto & [session, binding] : paged_bindings_) {
                (void)session;
                paged_cache_->unpin(binding.blocks);
            }
            paged_bindings_.clear();
            sync_paged_telemetry();
            return sessions;
        }
        size_t snapshots = 0;
        for (const auto & [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
        }
        states_.clear();
        bytes_ = 0;
        sync_telemetry();
        return snapshots;
    }

    size_t clear() {
        if (paged_cache_) {
            clear_live_sessions();
            return paged_cache_->clear();
        }
        return clear_live_sessions();
    }

    uint64_t trim_hot(uint64_t target_bytes) {
        if (!paged_cache_) return 0;
        const auto released = paged_cache_->trim_hot(target_bytes);
        if (released > 0) mfq_release_host_allocator_cache();
        return released;
    }

private:
    template <typename Model>
    size_t restore_paged(
            Model& model,
            const std::string & requested_session,
            const std::vector<int64_t> & prompt,
            size_t maximum_prefix_tokens) {
        if (max_sessions_ == 0 ||
                !model.supports_paged_text_session_state() ||
                prompt.size() < 2) {
            return 0;
        }
        const auto limit = std::min(
            maximum_prefix_tokens, prompt.size() - 1);
        std::vector<int64_t> candidate(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(limit));
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
            match.matched_tokens =
                payloads.size() * paged_cache_->block_size_tokens();
        }
        std::vector<int64_t> matched_tokens(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(
                match.matched_tokens));
        std::optional<TextSessionState> state;
        try {
            state.emplace(decode_cuda_paged_session(
                payloads,
                matched_tokens,
                paged_cache_->block_size_tokens()));
        } catch (const std::exception & error) {
            if (!match.blocks.empty()) {
                paged_cache_->invalidate(match.blocks.back());
            }
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr
                << "runtime_session_cache backend=cuda "
                << "action=paged_codec_invalidate "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
        try {
            model.restore_text_session_state(*state);
            if (!requested_session.empty()) {
                bind_paged_session(
                    requested_session, match.blocks, match.matched_tokens);
            }
            paged_cache_->record_match(match.matched_tokens);
            if (trace_) {
                std::cerr
                    << "runtime_session_cache backend=cuda action=paged_hit "
                    << "session=" << requested_session
                    << " reused_tokens=" << match.matched_tokens
                    << " prefill_tokens="
                    << prompt.size() - match.matched_tokens << std::endl;
            }
            return match.matched_tokens;
        } catch (const std::exception & error) {
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr
                << "runtime_session_cache backend=cuda "
                << "action=paged_restore_failed "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
    }

    void store_paged(
            const std::string & session_id,
            const TextSessionState & state) {
        if (max_sessions_ == 0 || state.tokens.empty()) {
            return;
        }
        const auto block_size = paged_cache_->block_size_tokens();
        const auto full_blocks = state.tokens.size() / block_size;
        if (full_blocks == 0) return;

        auto existing = paged_cache_->match(state.tokens, {}, false);
        if (existing.blocks.size() > full_blocks) {
            throw std::runtime_error(
                "paged prefix match exceeds the CUDA session state");
        }
        auto blocks = std::move(existing.blocks);
        mfq::cache::BlockHash parent{};
        if (!blocks.empty()) parent = blocks.back();
        for (size_t index = blocks.size(); index < full_blocks; ++index) {
            const auto token_offset = index * block_size;
            auto payload = encode_cuda_paged_block(
                state, block_size, index);
            parent = paged_cache_->store(
                parent,
                state.tokens.data() + token_offset,
                block_size,
                std::move(payload));
            blocks.push_back(parent);
        }
        if (!session_id.empty()) {
            bind_paged_session(
                session_id,
                std::move(blocks),
                full_blocks * block_size);
        }
        if (trace_) {
            std::cerr
                << "runtime_session_cache backend=cuda action=paged_store "
                << "session=" << session_id
                << " tokens=" << full_blocks * block_size
                << " blocks=" << full_blocks << std::endl;
        }
    }

    void bind_paged_session(
            const std::string & session_id,
            std::vector<mfq::cache::BlockHash> blocks,
            size_t tokens) {
        close_paged_session(session_id);
        paged_cache_->pin(blocks);
        paged_bindings_[session_id] = PagedBinding{
            std::move(blocks), tokens, ++clock_};
        while (paged_bindings_.size() > max_sessions_) {
            auto victim = paged_bindings_.end();
            for (auto iterator = paged_bindings_.begin();
                    iterator != paged_bindings_.end(); ++iterator) {
                if (iterator->first == session_id) continue;
                if (victim == paged_bindings_.end() ||
                        iterator->second.last_used <
                            victim->second.last_used) {
                    victim = iterator;
                }
            }
            if (victim == paged_bindings_.end()) break;
            close_paged_session(victim->first);
        }
        sync_paged_telemetry();
    }

    size_t close_paged_session(const std::string & session_id) {
        auto found = paged_bindings_.find(session_id);
        if (found == paged_bindings_.end()) return 0;
        const auto blocks = found->second.blocks.size();
        paged_cache_->unpin(found->second.blocks);
        paged_bindings_.erase(found);
        sync_paged_telemetry();
        return blocks;
    }

    void sync_paged_telemetry() noexcept {
        size_t tokens = 0;
        for (const auto & [session, binding] : paged_bindings_) {
            (void)session;
            tokens += binding.tokens;
        }
        metric_sessions_.store(paged_bindings_.size());
        metric_tokens_.store(tokens);
    }

    void sync_telemetry() noexcept {
        size_t snapshots = 0;
        size_t tokens = 0;
        for (const auto & [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
            for (const auto & snapshot : history) {
                tokens += snapshot.tokens.size();
            }
        }
        metric_sessions_.store(states_.size());
        metric_snapshots_.store(snapshots);
        metric_tokens_.store(tokens);
        metric_bytes_.store(bytes_);
    }

    void evict_history_to_limit(
            const std::string & session_id,
            uint64_t protected_clock) {
        auto found = states_.find(session_id);
        while (found != states_.end() &&
                found->second.size() > max_snapshots_per_session_) {
            size_t victim = found->second.size();
            for (size_t index = 0; index < found->second.size(); ++index) {
                const auto & snapshot = found->second[index];
                if (snapshot.last_used == protected_clock) continue;
                if (victim == found->second.size() ||
                        snapshot.last_used <
                            found->second[victim].last_used) {
                    victim = index;
                }
            }
            if (victim == found->second.size()) break;
            erase_snapshot(session_id, victim, "history_evict");
            found = states_.find(session_id);
        }
    }

    void evict_to_budget(
            const std::string & protected_session,
            uint64_t protected_clock) {
        while (states_.size() > max_sessions_) {
            auto victim = states_.end();
            uint64_t victim_last_used = 0;
            for (auto it = states_.begin(); it != states_.end(); ++it) {
                if (it->first == protected_session) continue;
                uint64_t session_last_used = 0;
                for (const auto & snapshot : it->second) {
                    session_last_used = std::max(
                        session_last_used, snapshot.last_used);
                }
                if (victim == states_.end() ||
                        session_last_used < victim_last_used) {
                    victim = it;
                    victim_last_used = session_last_used;
                }
            }
            if (victim == states_.end()) break;
            close_session(victim->first);
        }
        while (bytes_ > max_bytes_) {
            std::string victim_session;
            size_t victim_snapshot = 0;
            uint64_t victim_last_used = 0;
            bool found_victim = false;
            for (const auto & [session_id, history] : states_) {
                for (size_t index = 0; index < history.size(); ++index) {
                    const auto & snapshot = history[index];
                    if (session_id == protected_session &&
                            snapshot.last_used == protected_clock) {
                        continue;
                    }
                    if (!found_victim ||
                            snapshot.last_used < victim_last_used) {
                        victim_session = session_id;
                        victim_snapshot = index;
                        victim_last_used = snapshot.last_used;
                        found_victim = true;
                    }
                }
            }
            if (!found_victim) break;
            erase_snapshot(victim_session, victim_snapshot, "budget_evict");
        }
    }

    void erase_snapshot(
            const std::string & session_id,
            size_t index,
            const char * action) {
        auto found = states_.find(session_id);
        if (found == states_.end() || index >= found->second.size()) return;
        const size_t removed_bytes = found->second[index].bytes;
        if (trace_) {
            std::cerr << "runtime_session_cache action=" << action
                      << " session=" << session_id
                      << " tokens=" << found->second[index].tokens.size()
                      << " bytes=" << removed_bytes << std::endl;
        }
        bytes_ -= removed_bytes;
        found->second.erase(found->second.begin() +
            static_cast<std::ptrdiff_t>(index));
        if (found->second.empty()) states_.erase(found);
        sync_telemetry();
    }

    std::unordered_map<
        std::string, std::vector<TextSessionState>> states_;
    std::unordered_map<std::string, PagedBinding> paged_bindings_;
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache_;
    uint64_t paged_disk_budget_ = 0;
    uint64_t paged_hot_budget_ = 0;
    size_t max_sessions_ = 4;
    size_t max_snapshots_per_session_ = 4;
    size_t max_bytes_ = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    size_t bytes_ = 0;
    uint64_t clock_ = 0;
    std::atomic<uint64_t> queries_{0};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> hit_tokens_{0};
    std::atomic<size_t> metric_sessions_{0};
    std::atomic<size_t> metric_snapshots_{0};
    std::atomic<size_t> metric_tokens_{0};
    std::atomic<size_t> metric_bytes_{0};
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
};





TextSessionCache::TextSessionCache(
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache,
        bool supported,
        int disabled_reason)
    : impl_(std::make_unique<Impl>(
          std::move(paged_cache), supported, disabled_reason)) {}

TextSessionCache::~TextSessionCache() = default;

bool TextSessionCache::persistent_prefix_enabled() const noexcept {
    return impl_->persistent_prefix_enabled();
}

template <typename Model>
TextSessionRestore TextSessionCache::restore_best(
        Model& model,
        MtpModule* mtp,
        const std::string& requested_session,
        const std::vector<int64_t>& prompt,
        size_t maximum_prefix_tokens,
        const std::string& input_key) {
    return impl_->restore_best(
        model, mtp, requested_session, prompt,
        maximum_prefix_tokens, input_key);
}

void TextSessionCache::store(
        const std::string& session_id,
        TextSessionState state) {
    impl_->store(session_id, std::move(state));
}

size_t TextSessionCache::fork_session(
        const std::string& source_session,
        const std::string& target_session) {
    return impl_->fork_session(source_session, target_session);
}

size_t TextSessionCache::close_session(const std::string& session_id) {
    return impl_->close_session(session_id);
}

std::vector<std::pair<std::string, double>>
TextSessionCache::metrics() const {
    return impl_->metrics();
}

size_t TextSessionCache::clear_live_sessions() noexcept {
    return impl_->clear_live_sessions();
}

size_t TextSessionCache::clear() {
    return impl_->clear();
}

uint64_t TextSessionCache::trim_hot(uint64_t target_bytes) {
    return impl_->trim_hot(target_bytes);
}

#define MFQ_INSTANTIATE_SESSION_CACHE(BACKBONE)                           \
    template TextSessionRestore TextSessionCache::restore_best(          \
        mfq::cuda::CausalLmFor<BACKBONE>&, MtpModule*,                   \
        const std::string&, const std::vector<int64_t>&,                 \
        size_t, const std::string&);

MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::generic_qwen)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::minicpmo45)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::minicpmo_tts)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::gemma4)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::glm_dsa)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::glm5_next)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::qwen4_exp)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::deepseek_v4)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::CudaBackbone::deepseek_v41)

#undef MFQ_INSTANTIATE_SESSION_CACHE

} // namespace mfq::cuda::internal
