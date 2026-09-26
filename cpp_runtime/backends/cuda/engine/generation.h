#pragma once

#include "cuda_execution.h"
#include "decode_graph.h"
#include "mfq/runtime.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

struct MtpModule;

namespace mfq::cuda::internal {

class TextSessionCache;

template <typename Model>
using PreparedPromptFactory =
    std::function<std::optional<CudaPreparedPrompt>(Model&)>;

template <typename Model>
std::int32_t generate(
    Model& model,
    std::mutex& model_mutex,
    DecodeGraphCache& graph_cache,
    TextSessionCache& session_cache,
    const std::vector<std::int64_t>& prompt,
    const MfqSamplingParams& sampling,
    const MfqTokenCallback& on_token,
    const MfqPrefillCallback& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    MtpModule* mtp = nullptr,
    std::int64_t prefill_chunk_size = 2048,
    PreparedPromptFactory<Model> prepare_prompt = {},
    MfqCancellationCheck cancelled = {});

} // namespace mfq::cuda::internal
