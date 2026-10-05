#pragma once

#include "chat.h"
#include "mfq/runtime.h"
#include "mtp_policy.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::engine {

enum class InferenceInputErrorCode {
    Invalid,
    Unsupported,
    ContextLength,
};

class InferenceInputError final : public std::runtime_error {
public:
    InferenceInputError(
        InferenceInputErrorCode code,
        std::string message,
        std::string field = {});

    InferenceInputErrorCode code;
    std::string field;
};

struct ChatInput {
    common_chat_templates_inputs template_inputs;
    std::optional<std::string> preformatted_prompt;
    std::optional<bool> parallel_tool_calls;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_AUTO;
};

struct InferenceInput {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    std::string prompt;
    std::optional<ChatInput> chat_input;
    std::vector<std::string> stops;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    std::optional<MfqMultimodalInput> media;
};

struct ChatTemplateCapabilities {
    bool thinking = false;
    std::vector<std::string> reasoning_effort_values;
};

struct InferenceMetrics {
    using Clock = std::chrono::steady_clock;

    Clock::time_point started = Clock::now();
    Clock::time_point first_token;
    Clock::time_point last_token;
    std::size_t prefill_tokens = 0;
    double prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    bool saw_token = false;
    bool saw_prefill = false;
    mtp::GenerationStats mtp;

    void mark_prefill(const MfqPrefillTiming& timing);
    void mark_token();
};

struct OutputDelta {
    std::vector<std::int64_t> token_ids;
    std::vector<common_chat_msg_diff> diffs;
};

} // namespace mfq::engine
