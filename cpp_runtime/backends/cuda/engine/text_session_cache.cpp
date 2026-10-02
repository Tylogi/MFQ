#include "text_session_cache.h"

#include "cuda_runtime_config.h"
#include "models/causal_models.h"
#include "models/session_state.h"
#include "mtp.h"
#include "mfq_paged_prefix_cache.h"
#include "paged_session_bindings.h"
#include "session_cache.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mfq::cuda::internal {

static std::string cuda_prefix_cache_compatibility_key(
    const mfq::ModelSource &source, int64_t max_position_embeddings) {
    std::ostringstream key;
    key << "mfq-cuda-prefix-v1\n"
        << "codec=cuda-full-attention-kv-v1\n"
        << "context=" << max_position_embeddings << '\n'
        << "architecture=" << source.architecture() << '\n';
    for (std::size_t index = 0; index < source.source_paths().size(); ++index) {
        const auto &path = source.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify model source for prefix cache: " + error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error("cannot identify model source timestamp: " + error.message());
        }
        const auto modified_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(modified.time_since_epoch())
                .count();
        key << "source=" << index << ':' << path.filename().string() << ':' << size << ':'
            << modified_ns << '\n';
    }
    for (const auto &tensor : source.tensors()) {
        key << "tensor=" << tensor.name << ':' << tensor.dtype << ':' << tensor.nbytes << '\n';
    }
    return key.str();
}

std::shared_ptr<mfq::cache::PagedPrefixCache> make_cuda_paged_prefix_cache(
    const mfq::ModelSource &source, int64_t max_position_embeddings,
    bool supports_paged_text_session_state, const CudaPrefixCacheConfig &config) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if (!config.enabled ||
        (format != source.metadata().end() && format->second == "hf-safetensors") ||
        !supports_paged_text_session_state) {
        return {};
    }
    mfq::cache::PagedPrefixCacheConfig cache;
    cache.cache_dir = config.directory;
    cache.compatibility_key = cuda_prefix_cache_compatibility_key(source, max_position_embeddings);
    cache.block_size_tokens = static_cast<size_t>(config.block_tokens);
    cache.max_disk_bytes = config.disk_bytes;
    cache.max_hot_bytes = config.hot_bytes;
    cache.max_pending_writes = config.pending_writes;
    cache.max_pending_bytes = config.pending_bytes;
    return std::make_shared<mfq::cache::PagedPrefixCache>(std::move(cache));
}

struct CudaSessionOps {
    using Snapshot = TextSessionState;
    using Restore = TextSessionRestore;
    using Predictor = MtpModule;
    using StateError = CudaSessionStateError;
    template <class Payloads>
    static auto decode(
        const Payloads &payloads, const std::vector<int64_t> &tokens, size_t block_size) {
        return decode_cuda_paged_session(payloads, tokens, block_size);
    }
    static auto encode(const Snapshot &state, size_t block_size, size_t index) {
        return encode_cuda_paged_block(state, block_size, index);
    }
    static void release_host_cache() { mfq_release_host_allocator_cache(); }
};

struct TextSessionCache::Impl : mfq::engine::SessionCache<CudaSessionOps> {
    using SessionCache::SessionCache;
};

TextSessionCache::TextSessionCache(const CudaSessionCacheConfig &session_config,
    const CudaPrefixCacheConfig &prefix_config,
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache, bool supported, int disabled_reason)
    : impl_(std::make_unique<Impl>(
          session_config, prefix_config, std::move(paged_cache), supported, disabled_reason)) {}

TextSessionCache::~TextSessionCache() = default;

bool TextSessionCache::persistent_prefix_enabled() const noexcept {
    return impl_->persistent_prefix_enabled();
}

template <typename Model>
TextSessionRestore TextSessionCache::restore_best(Model &model, MtpModule *mtp,
    const std::string &requested_session, const std::vector<int64_t> &prompt,
    size_t maximum_prefix_tokens, const std::string &input_key) {
    return impl_->restore_best(
        model, mtp, requested_session, prompt, maximum_prefix_tokens, input_key);
}

void TextSessionCache::store(const std::string &session_id, TextSessionState state) {
    impl_->store(session_id, std::move(state));
}

size_t TextSessionCache::fork_session(
    const std::string &source_session, const std::string &target_session) {
    return impl_->fork_session(source_session, target_session);
}

size_t TextSessionCache::close_session(const std::string &session_id) {
    return impl_->close_session(session_id);
}

std::vector<std::pair<std::string, double>> TextSessionCache::metrics() const {
    return impl_->metrics();
}

size_t TextSessionCache::clear_live_sessions() noexcept { return impl_->clear_live_sessions(); }

size_t TextSessionCache::clear() { return impl_->clear(); }

uint64_t TextSessionCache::trim_hot(uint64_t target_bytes) { return impl_->trim_hot(target_bytes); }

#define MFQ_INSTANTIATE_SESSION_CACHE(MODEL)                                                       \
    template TextSessionRestore TextSessionCache::restore_best(MODEL &,                            \
        MtpModule *,                                                                               \
        const std::string &,                                                                       \
        const std::vector<int64_t> &,                                                              \
        size_t,                                                                                    \
        const std::string &);

MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen35CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMO45CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::MiniCPMOTtsCausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Gemma4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::GlmDsaCausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Glm5CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::Qwen4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV4CausalLm)
MFQ_INSTANTIATE_SESSION_CACHE(mfq::cuda::DeepseekV41CausalLm)

#undef MFQ_INSTANTIATE_SESSION_CACHE

} // namespace mfq::cuda::internal
