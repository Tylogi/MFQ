#pragma once

#include "inference.h"
#include "chat.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

// Engine-owned model text semantics: template application, tokenization,
// grammar constraints, output parsing, stop handling, and media placeholders.
class TextProcessor {
public:
    TextProcessor(
        const std::string& tokenizer_path,
        std::int32_t expected_vocabulary,
        std::string model_type);
    TextProcessor(
        const std::vector<std::uint8_t>& tokenizer_gguf,
        std::int32_t expected_vocabulary,
        std::string model_type);
    ~TextProcessor();

    TextProcessor(const TextProcessor&) = delete;
    TextProcessor& operator=(const TextProcessor&) = delete;

    std::int32_t vocab_size() const;
    ChatTemplateCapabilities chat_template_capabilities() const;
    InferenceRequest prepare(
        InferenceInput input,
        std::int64_t max_context) const;
    const MfqTokenizer& tokenizer() const;

    void prepare_duplex_session(
        const std::string& system_prompt,
        MfqDuplexSessionParams& parameters) const;
    void prepare_duplex_step(
        const std::string& text,
        MfqDuplexStepInput& step) const;
    std::string decode_tokens(
        const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::engine
