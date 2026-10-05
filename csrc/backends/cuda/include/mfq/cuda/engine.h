#pragma once

#include "mfq/engine.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mfq::cuda {

struct CudaParallelOptions {
    std::vector<int> devices;
    std::vector<double> split;
};

struct CudaLoadOptions {
    std::string model_path, config_path, tokenizer_model;
    std::string moe_cache_profile_path;
    CudaParallelOptions tensor_parallel;
    CudaParallelOptions expert_parallel;
    CudaParallelOptions layer_parallel;
    std::vector<int> cpu_offload_layers;
    std::optional<int> cpu_threads;
    std::optional<int> n_gpu_layers;
    double moe_gpu_cache_gb = 0.0;
    int64_t context_size = 0;
    bool parallel_test_duplicates = false;
};

struct CudaEngineOptions : CudaLoadOptions {
    int continuous_batching = 0;
    int64_t prefill_chunk_size = 2048;
    std::size_t memory_budget_bytes = 0; // Per-device execution budget; zero uses available memory.
};

std::unique_ptr<mfq::engine::Engine> load_cuda_engine(CudaEngineOptions options);

} // namespace mfq::cuda
