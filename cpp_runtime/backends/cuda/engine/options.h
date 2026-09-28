#pragma once

#include <cstdint>
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
};

struct CudaRuntimeConfig;

} // namespace mfq::cuda

namespace mfq::cuda::internal {

CudaRuntimeConfig resolve_cuda_runtime_config(const CudaEngineOptions& options);
void setup_cuda_load(const CudaLoadOptions& options);
void reset_cuda_load() noexcept;

} // namespace mfq::cuda::internal
