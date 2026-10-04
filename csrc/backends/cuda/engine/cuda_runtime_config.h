#pragma once

#include "runtime_config.h"
#include "mfq/cuda/engine.h"

#include <cstdint>

namespace mfq::cuda {

struct CudaDecodeGraphConfig {
    bool enabled = true;
    bool trace = false;
    std::int32_t minimum_generation_tokens = 16;
};

struct CudaContinuousBatchConfig : mfq::engine::ContinuousBatchConfig {
    bool greedy = true;
    bool cuda_graph = true;
    bool paged_kv = true;
    std::int32_t cuda_graph_minimum_tokens = 16;
};

struct CudaRuntimeConfig : mfq::engine::RuntimeConfig {
    CudaDecodeGraphConfig decode_graph;
    CudaContinuousBatchConfig continuous_batch;
};

} // namespace mfq::cuda

namespace mfq::cuda::internal {

CudaRuntimeConfig resolve_cuda_runtime_config(const CudaEngineOptions& options);

} // namespace mfq::cuda::internal
