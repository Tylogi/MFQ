#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

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

struct TokenGenerationResult {
    std::int64_t tokens = 0;
    bool hit_eos = false;
};

// Synchronous evaluation/TTS shares token acceptance and termination ordering.
// Native bindings provide tensor sampling and one decode operation at a time.
template <class Sample, class Accept, class Stop, class Advance>
TokenGenerationResult generate_tokens(std::int64_t limit, Sample sample, Accept accept, Stop stop,
                                      Advance advance) {
    if (limit < 0)
        throw std::invalid_argument("negative generation limit");
    for (std::int64_t step = 0; step < limit; ++step) {
        auto token = sample(step);
        accept(token, step);
        if (stop(token))
            return {step + 1, true};
        if (step + 1 < limit)
            advance(token);
    }
    return {limit, false};
}

} // namespace mfq::engine
