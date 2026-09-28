#include "text_session_cache.h"

#include "runtime_config.h"
#include "models/causal_lm.h"
#include "mtp.h"
#include "mfq_paged_prefix_cache.h"
#include "paged_session_bindings.h"
#include "session_snapshot_cache.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mfq::cuda::internal {

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
        bool supports_paged_text_session_state,
        const CudaPrefixCacheConfig& config) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if (!config.enabled ||
            (format != source.metadata().end() &&
             format->second == "hf-safetensors") ||
            !supports_paged_text_session_state) {
        return {};
    }
    mfq::cache::PagedPrefixCacheConfig cache;
    cache.cache_dir = config.directory;
    cache.compatibility_key = cuda_prefix_cache_compatibility_key(
        source, max_position_embeddings);
    cache.block_size_tokens = static_cast<size_t>(
        config.block_tokens);
    cache.max_disk_bytes = config.disk_bytes;
    cache.max_hot_bytes = config.hot_bytes;
    cache.max_pending_writes = config.pending_writes;
    cache.max_pending_bytes = config.pending_bytes;
    return std::make_shared<mfq::cache::PagedPrefixCache>(
        std::move(cache));
}

struct TextSessionCache::Impl {
public:
    explicit Impl(
            const CudaSessionCacheConfig& session_config,
            const CudaPrefixCacheConfig& prefix_config,
            std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {},
            bool supported = true,
            int disabled_reason = 0)
        : snapshots_(session_config.snapshots),
          paged_cache_(std::move(paged_cache)),
          paged_bindings_(paged_cache_, snapshots_.max_sessions()),
          paged_disk_budget_(paged_cache_ ? prefix_config.disk_bytes : 0),
          paged_hot_budget_(paged_cache_ ? prefix_config.hot_bytes : 0),
          trace_(session_config.trace),
          supported_(supported),
          disabled_reason_(disabled_reason) {}

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
        if (!model.supports_text_session_state()) return {};
        auto match = snapshots_.find_best(
            requested_session, prompt, maximum_prefix_tokens,
            [&](const TextSessionState& state) {
                return state.input_key == input_key &&
                    (mtp == nullptr ||
                     (mtp->supports_session_state() && state.mtp.has_value()));
            });
        if (!match) return {};
        try {
            model.restore_text_session_state(*match->state);
            TextSessionRestore restored{match->tokens(), {}};
            if (mtp != nullptr) {
                mtp->restore_session_state(*match->state->mtp);
                restored.mtp_last_target_hidden =
                    match->state->mtp->last_target_hidden;
            }
            snapshots_.record_hit(*match);
            if (trace_) {
                std::cerr << "runtime_session_cache action=hit session="
                          << requested_session
                          << " source=" << match->session_id
                          << " reused_tokens=" << match->tokens()
                          << " prefill_tokens="
                          << prompt.size() - match->tokens() << std::endl;
            }
            return restored;
        } catch (const std::exception & error) {
            const auto selected_session = match->session_id;
            snapshots_.erase(*match);
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
        if (session_id.empty() || !snapshots_.enabled()) return;
        if (state.bytes > snapshots_.max_bytes()) {
            if (trace_) {
                std::cerr << "runtime_session_cache action=skip session="
                          << session_id << " bytes=" << state.bytes
                          << " budget=" << snapshots_.max_bytes() << std::endl;
            }
            return;
        }
        const auto stored = snapshots_.store(
            session_id, std::move(state),
            [](const TextSessionState& saved,
               const TextSessionState& candidate) {
                return saved.tokens == candidate.tokens &&
                    saved.input_key == candidate.input_key;
            });
        if (trace_ && stored) {
            std::cerr << "runtime_session_cache action=store session="
                      << session_id << " tokens=" << stored.state->tokens.size()
                      << " bytes=" << stored.state->bytes
                      << " snapshots=" << stored.session_snapshots
                      << " total_bytes=" << stored.total_bytes << std::endl;
        }
    }

    size_t fork_session(
            const std::string & source_session,
            const std::string & target_session) {
        if (paged_cache_) {
            return paged_bindings_.fork(source_session, target_session);
        }
        const auto copied_snapshots =
            snapshots_.fork(source_session, target_session);
        if (trace_ && copied_snapshots > 0) {
            std::cerr << "runtime_session_cache action=fork source="
                      << source_session << " target=" << target_session
                      << " snapshots=" << copied_snapshots
                      << " total_bytes=" << snapshots_.metrics().bytes
                      << std::endl;
        }
        return copied_snapshots;
    }

    size_t close_session(const std::string & session_id) {
        if (paged_cache_) return paged_bindings_.close(session_id);
        const auto released = snapshots_.close(session_id);
        if (trace_ && released.snapshots > 0) {
            std::cerr << "runtime_session_cache action=close session="
                      << session_id << " snapshots=" << released.snapshots
                      << " bytes=" << released.bytes
                      << " total_bytes=" << snapshots_.metrics().bytes
                      << std::endl;
        }
        return released.snapshots;
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
                {"prefix_cache_deduplicated_writes", static_cast<double>(value.deduplicated_writes)},
                {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                {"prefix_cache_hot_hits", static_cast<double>(value.hot_hits)},
                {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
            };
        }
        const auto value = snapshots_.metrics();
        return {
            {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
            {"prefix_cache_disabled_reason",
                static_cast<double>(disabled_reason_)},
            {"prefix_cache_queries", static_cast<double>(value.queries)},
            {"prefix_cache_hits", static_cast<double>(value.hits)},
            {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
            {"prefix_cache_sessions", static_cast<double>(value.sessions)},
            {"prefix_cache_snapshots", static_cast<double>(value.snapshots)},
            {"prefix_cache_tokens", static_cast<double>(value.tokens)},
            {"prefix_cache_bytes", static_cast<double>(value.bytes)},
            {"prefix_cache_max_sessions",
                static_cast<double>(snapshots_.max_sessions())},
            {"prefix_cache_max_snapshots_per_session",
                static_cast<double>(snapshots_.max_snapshots_per_session())},
            {"prefix_cache_max_bytes",
                static_cast<double>(snapshots_.max_bytes())},
        };
    }

    size_t clear_live_sessions() noexcept {
        if (paged_cache_) return paged_bindings_.clear();
        return snapshots_.clear();
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
        if (snapshots_.max_sessions() == 0 ||
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
                paged_bindings_.bind(
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
        if (snapshots_.max_sessions() == 0 || state.tokens.empty()) {
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
            paged_bindings_.bind(
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

    mfq::engine::SessionSnapshotCache<TextSessionState> snapshots_;
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache_;
    mfq::engine::PagedSessionBindings paged_bindings_;
    uint64_t paged_disk_budget_ = 0;
    uint64_t paged_hot_budget_ = 0;
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
};





TextSessionCache::TextSessionCache(
        const CudaSessionCacheConfig& session_config,
        const CudaPrefixCacheConfig& prefix_config,
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache,
        bool supported,
        int disabled_reason)
    : impl_(std::make_unique<Impl>(
          session_config, prefix_config, std::move(paged_cache),
          supported, disabled_reason)) {}

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
