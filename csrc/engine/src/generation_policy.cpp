#include "generation_policy.h"

#include <algorithm>
#include <stdexcept>

namespace mfq::engine {

GenerationPlan plan_generation(
        std::span<const std::int64_t> prompt,
        std::int64_t vocab_size,
        std::int64_t context_capacity,
        std::int32_t requested_generation_tokens,
        std::size_t stable_prefix_tokens,
        std::int64_t occupied_context) {
    if (prompt.empty()) {
        throw std::invalid_argument("generation prompt must not be empty");
    }
    if (vocab_size <= 0 || context_capacity <= 0 ||
            prompt.size() > static_cast<std::size_t>(context_capacity)) {
        throw std::invalid_argument("generation prompt exceeds model capacity");
    }
    if (requested_generation_tokens < 0) {
        throw std::invalid_argument("generation token count must be non-negative");
    }
    for (const auto token : prompt) {
        if (token < 0 || token >= vocab_size) {
            throw std::invalid_argument(
                "generation prompt token is outside the vocabulary");
        }
    }
    if (occupied_context < 0) {
        occupied_context = static_cast<std::int64_t>(prompt.size());
    }
    if (occupied_context < static_cast<std::int64_t>(prompt.size()) ||
            occupied_context > context_capacity) {
        throw std::invalid_argument(
            "generation occupied context is outside model capacity");
    }
    const auto available = context_capacity - occupied_context;
    const auto limit = std::min<std::int64_t>(
        requested_generation_tokens, available);
    return {
        prompt.size(),
        std::min(stable_prefix_tokens, prompt.size()),
        static_cast<std::int32_t>(limit),
    };
}

PrefillChunk next_prefill_chunk(
        std::int64_t total_tokens,
        std::int64_t offset,
        std::int64_t chunk_size) {
    if (total_tokens <= 0 || offset < 0 || offset >= total_tokens ||
            chunk_size <= 0) {
        throw std::invalid_argument("invalid prefill chunk geometry");
    }
    const auto count = std::min(chunk_size, total_tokens - offset);
    return {offset, count, offset + count == total_tokens};
}

} // namespace mfq::engine
