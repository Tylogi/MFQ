#pragma once

#include "generation_policy.h"
#include "session_snapshot_cache.h"

#include <cstdint>
#include <filesystem>

namespace mfq::engine {

struct SessionCacheConfig {
    SessionSnapshotCacheConfig snapshots;
    bool trace = false;
};

struct PrefixCacheConfig {
    std::filesystem::path directory;
    std::uint64_t block_tokens = 256;
    std::uint64_t disk_bytes = 100ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t hot_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    std::size_t pending_writes = 64;
    std::uint64_t pending_bytes = 512ULL * 1024ULL * 1024ULL;
    bool enabled = true;
};

struct ContinuousBatchConfig {
    std::size_t max_sequences = 0;
    std::int64_t prefill_token_budget = 2048;
};

struct RuntimeConfig {
    GenerationConfig generation;
    SessionCacheConfig session_cache;
    PrefixCacheConfig prefix_cache;
};

std::uint64_t environment_uint64(const char* name, std::uint64_t fallback);
bool environment_enabled(const char* name, bool fallback = true);
RuntimeConfig resolve_runtime_config(std::int64_t prefill_chunk_size);
ContinuousBatchConfig resolve_batch_config(int capacity, std::int64_t prefill_chunk_size);

} // namespace mfq::engine
