#pragma once

#include "inference.h"
#include "engine.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace mfq { class ModelSource; }

namespace mfq::engine {

// Engine-owned model text semantics: template application, tokenization,
// grammar constraints, output parsing, stop handling, and media placeholders.
class TextProcessor {
public:
    static std::unique_ptr<TextProcessor> load(const ModelSource& source,
        const std::string& tokenizer_path, std::int32_t vocabulary, const std::string& model_type);
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
