#pragma once

#include <cstdint>

namespace mfq::cuda {

enum class SamplingKind : std::int32_t { inactive, greedy, softmax, radix, fallback };

// One pinned packet per row. Uniforms come from the request's existing host RNG;
// counts and masks are stable device addresses owned by the request/batch.
struct SamplingRow {
    const std::int32_t* counts = nullptr;
    const float* mask = nullptr;
    float temperature = 1.0f;
    float top_p = 1.0f;
    float random = 0.0f;
    float presence = 0.0f;
    float frequency = 0.0f;
    float repetition = 1.0f;
    std::int32_t top_k = 0;
    std::int32_t source = 0;
    std::int32_t retry = -1;
    SamplingKind kind = SamplingKind::inactive;
};

// Host-only dispatch facts; no device readback is needed to select kernels.
// A captured plan fixes row/candidate counts, available kinds and radix capacity;
// replay updates row values/pointers and may disable work within that capacity.
struct SamplingPlan {
    int rows = 0;
    int candidates = 0;
    int radix_k = 0;
    bool greedy = false;
    bool softmax = false;
    bool fallback = false;
    bool transform = false;
    bool constrained = false;
};

} // namespace mfq::cuda
