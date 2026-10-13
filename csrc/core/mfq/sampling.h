#pragma once

#include "mfq/token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

// Describes the portion of a rendered prompt whose KV state is stable across
// requests. A backend may retain this exact token prefix after generation and
// reuse it only when the next request starts with the same token sequence.
struct PromptCachePlan {
    std::string session_id;
    size_t stable_prefix_tokens = 0;
    bool cache_output_tokens = false;
};

struct PrefillTiming {
    size_t prompt_tokens = 0;
    // Language-model prompt evaluation only.
    double llm_ms = 0.0;
    // Vision/audio encoder and multimodal projector work preceding the LLM.
    double multimodal_ms = 0.0;
    // Complete model-side prefill wall time.
    double model_ms = 0.0;
};

} // namespace mfq
