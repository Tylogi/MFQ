#pragma once

#include "text_cache.h"
#include "state.h"
#include "session_cache.h"

namespace mfq::cuda::internal {
struct CudaSessionOps {
    using Snapshot = TextSessionState;
    using Restore = TextSessionRestore;
    using Predictor = MtpModule;
    using StateError = CudaSessionStateError;
    static TextSessionState decode(const std::vector<CudaPagedPayload>& payloads, const std::vector<int64_t>& tokens, size_t block_size);
    static CudaPagedPayload encode(const Snapshot& state, size_t block_size, size_t index);
    static mfq::StepSequence<TextSessionState> decode_steps(const std::vector<CudaPagedPayload>& payloads, const std::vector<int64_t>& tokens, size_t block_size);
    static mfq::StepSequence<CudaPagedPayload> encode_steps(const Snapshot& state, size_t block_size, size_t index);
    static void release_host_cache();
};

struct TextSessionCache::Impl : mfq::engine::SessionCache<CudaSessionOps> {
    using SessionCache::SessionCache;
};

template <typename Model>
TextSessionRestore TextSessionCache::restore_best(Model& model, MtpModule* mtp,
        const std::string& session, const std::vector<int64_t>& prompt, size_t limit, const std::string& key) {
    return impl_->restore_best(model, mtp, session, prompt, limit, key);
}
template <typename Model>
mfq::StepSequence<TextSessionRestore> TextSessionCache::restore_steps(Model& model, MtpModule* mtp,
        const std::string& session, const std::vector<int64_t>& prompt, size_t limit, const std::string& key) {
    return impl_->restore_steps(model, mtp, session, prompt, limit, key);
}
} // namespace mfq::cuda::internal
