#pragma once

#include "generation_step.h"
#include "cuda_execution.h"
#include "decode_graph.h"
#include "cuda_runtime_config.h"

#include <optional>
#include <utility>

struct MtpModule;

namespace mfq::cuda::internal {
class TextSessionCache;

template <typename Model>
mfq::engine::Generation generate(
    Model& model, DecodeGraphCache& graph, TextSessionCache& cache,
    const CudaRuntimeConfig& config, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, MtpModule* mtp = nullptr,
    std::optional<CudaPreparedPrompt> prepared = {});
} // namespace mfq::cuda::internal
