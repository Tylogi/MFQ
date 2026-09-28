#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace mfq::engine {

struct GenerationConfig {
    std::int64_t prefill_chunk_size = 2048;
};

struct GenerationPlan {
    std::size_t prompt_tokens = 0;
    std::size_t stable_prefix_tokens = 0;
    std::int32_t generation_tokens = 0;
};

GenerationPlan plan_generation(
    std::span<const std::int64_t> prompt,
    std::int64_t vocab_size,
    std::int64_t context_capacity,
    std::int32_t requested_generation_tokens,
    std::size_t stable_prefix_tokens = 0,
    std::int64_t occupied_context = -1);

struct PrefillChunk {
    std::int64_t offset = 0;
    std::int64_t count = 0;
    bool last = false;
};

PrefillChunk next_prefill_chunk(
    std::int64_t total_tokens,
    std::int64_t offset,
    std::int64_t chunk_size);

} // namespace mfq::engine
