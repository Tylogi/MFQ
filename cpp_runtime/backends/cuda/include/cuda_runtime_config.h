#pragma once

#include "generation_policy.h"
#include "session_snapshot_cache.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

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
