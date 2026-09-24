#pragma once

#include "minicpmo45_runtime.h"
#include "mfq/runtime.h"

#include <mutex>
#include <optional>
#include <vector>

namespace mfq::cuda::minicpmo45 {

int32_t generate_multimodal_tokens(
    MiniCPMO45Runtime& runtime,
    std::mutex& model_mutex,
    const std::vector<int64_t>& prompt,
    const MfqVisionInput& vision,
    const MfqSamplingParams& sampling,
    const MfqTokenCallback& on_token,
    const MfqPrefillCallback& on_prefill,
    const MfqTokenConstraintPtr& token_constraint);

MfqDuplexBackend make_cuda_minicpmo45_duplex_backend(
    MiniCPMO45Runtime& runtime,
    std::mutex& model_mutex,
    std::optional<MiniCPMO45DuplexSession>& session);

} // namespace mfq::cuda::minicpmo45
