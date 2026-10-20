#pragma once

#include "engine.h"
#include "chat.h"
#include "tokenizer.h"
#include "emitter.h"

#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace mfq::engine {

std::unordered_set<std::int64_t> model_stop_token_ids(std::string_view config, std::int64_t vocabulary);

struct InferenceRequest {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    common_chat_parser_params chat_parser;
    std::unordered_set<std::int64_t> preserved_tokens;
    std::vector<std::int64_t> prompt;
    std::vector<std::string> stops;
    std::unordered_set<std::int64_t> stop_token_ids;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    MfqTokenConstraintPtr token_constraint;
    std::optional<MfqMultimodalInput> vision;
};

struct InferenceResult {
    std::string text;
    std::string reasoning_text;
    std::vector<common_chat_tool_call> tool_calls;
    std::string finish_reason = "length";
    std::int32_t completion_tokens = 0;
    bool client_connected = true;
    bool cancelled = false;
};

using TokenOutput = OutputDelta;

// Engine-owned parser state. append/finish only return values.
class InferenceOutput {
public:
    InferenceOutput(const InferenceRequest& request,
                    const MfqTokenizer* tokenizer, std::string request_id);
    void prepare(const MfqTokenizer* tokenizer);
    TokenOutput append(const std::vector<std::int64_t>& tokens);
    TokenOutput finish();
    bool stopped() const noexcept;
    InferenceResult result;
    InferenceMetrics metrics;
    // Internal teardown result, checked before the terminal event is built.
    std::exception_ptr cleanup_failure;
private:
    void parse(std::string piece, bool partial, TokenOutput& output);
    const InferenceRequest& request_;
    const MfqTokenizer* tokenizer_;
    std::string id_, generated_;
    TextEmitter emitter_;
    common_chat_msg message_;
    std::vector<std::string> tool_ids_;
    bool stopped_ = false, finished_ = false;
};

} // namespace mfq::engine
