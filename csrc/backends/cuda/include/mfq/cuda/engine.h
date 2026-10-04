#pragma once

#include <engine.h>

#include <cstdint>
#include <memory>
#include <string>

namespace mfq::cuda {

struct CudaLoadOptions {
    std::string model_path, config_path, tokenizer_model;
    std::string cpu_offload_layers_arg, moe_cache_profile_path;
    std::string tensor_parallel_arg, tensor_split_arg;
    std::string expert_parallel_arg, expert_split_arg;
    std::string layer_parallel_arg, layer_split_arg;
    double moe_gpu_cache_gb = 0.0;
    int cpu_threads = 0;
    int64_t context_size = 0;
    bool parallel_test_duplicates = false;
    bool n_gpu_layers_set = false;
    bool cpu_threads_set = false;
    int n_gpu_layers = -1;
};

struct CudaEngineOptions : CudaLoadOptions {
    int continuous_batching = 0;
    int64_t prefill_chunk_size = 2048;
    std::size_t memory_budget_bytes = 0; // Per-device execution budget; zero uses available memory.
};

std::unique_ptr<mfq::engine::Engine> load_cuda_engine(CudaEngineOptions options);

} // namespace mfq::cuda
