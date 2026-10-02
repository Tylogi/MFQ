#pragma once

#include "generation_policy.h"
#include "session_snapshot_cache.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace mfq::cuda {

struct CudaDecodeGraphConfig {
    bool enabled = true;
    bool trace = false;
    std::int32_t minimum_generation_tokens = 16;
};

struct CudaContinuousBatchConfig {
    std::size_t max_sequences = 0;
    bool greedy = true;
    bool cuda_graph = true;
    bool paged_kv = true;
    std::int32_t cuda_graph_minimum_tokens = 16;
    std::int64_t prefill_token_budget = 2048;
};

struct CudaSessionCacheConfig {
    mfq::engine::SessionSnapshotCacheConfig snapshots;
    bool trace = false;
};

struct CudaPrefixCacheConfig {
    std::filesystem::path directory;
    std::uint64_t block_tokens = 256;
    std::uint64_t disk_bytes = 100ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t hot_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    std::size_t pending_writes = 64;
    std::uint64_t pending_bytes = 512ULL * 1024ULL * 1024ULL;
    bool enabled = true;
};

struct CudaRuntimeConfig {
    mfq::engine::GenerationConfig generation;
    CudaDecodeGraphConfig decode_graph;
    CudaContinuousBatchConfig continuous_batch;
    CudaSessionCacheConfig session_cache;
    CudaPrefixCacheConfig prefix_cache;
};

} // namespace mfq::cuda

struct CudaExecutionContext;

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
void setup_cuda_load(
    const CudaLoadOptions& options,
    CudaExecutionContext& execution);
int with_cuda_load(
    const CudaLoadOptions& options,
    const std::function<int(CudaExecutionContext&)>& run);

} // namespace mfq::cuda::internal
