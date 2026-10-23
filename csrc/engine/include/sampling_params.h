#pragma once

#include <cstdint>

namespace mfq {

struct SamplingParams {
    int32_t max_tokens = 4096;
    double temperature = 1.0;
    int32_t top_k = 100;
    double top_p = 0.95;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    bool enable_thinking = true;
    bool enable_vision = true;
    bool enable_mtp = true;
    int32_t mtp_max_draft_tokens = 3;
    uint64_t seed = 0;
};

} // namespace mfq
