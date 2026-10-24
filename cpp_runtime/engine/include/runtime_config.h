#pragma once

#include "generation_policy.h"
#include "kv_admission.h"
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
    std::uint64_t ram_bytes = 2ULL << 30;
    std::size_t pending_writes = 64;
    std::uint64_t pending_bytes = 512ULL * 1024ULL * 1024ULL;
    bool enabled = true;
};

struct ContinuousBatchConfig {
    // Raising this ceiling requires validating every native batch path.
    static constexpr std::size_t validated_max_sequences = 32;
    std::size_t max_sequences = 0;
    std::int64_t prefill_token_budget = 2048;
    std::size_t sequence_limit = validated_max_sequences;
    KvAdmissionConfig admission;
    std::size_t kv_budget_bytes = 0;
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
