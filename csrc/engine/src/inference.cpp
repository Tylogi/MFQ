#include "inference.h"

#include "text_emitter.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace mfq::engine {

void InferenceMetrics::mark_prefill(const MfqPrefillTiming& timing) {
    prefill_tokens = timing.prompt_tokens;
    prefill_ms = timing.llm_ms;
    multimodal_ms = timing.multimodal_ms;
    model_prefill_ms = timing.model_ms;
    saw_prefill = timing.llm_ms > 0.0 || timing.model_ms > 0.0;
}

void InferenceMetrics::mark_token() {
    if (saw_token) return;
    first_token = Clock::now();
    saw_token = true;
}

InferenceOutput::InferenceOutput(const InferenceRequest& request,
        const MfqTokenizer* tokenizer, std::string request_id)
    : request_(request), tokenizer_(tokenizer), id_(std::move(request_id)),
      emitter_(request.stops) {
    prepare(tokenizer);
}

void InferenceOutput::prepare(const MfqTokenizer* tokenizer) {
    tokenizer_ = tokenizer;
    emitter_ = TextEmitter(request_.stops);
    if (tokenizer_ && request_.chat && request_.chat_parser.is_continuation &&
            !request_.chat_parser.echo)
        message_ = common_chat_parse("", true, request_.chat_parser);
}

bool InferenceOutput::stopped() const noexcept {
    return stopped_ || result.cancelled ||
        result.completion_tokens >= request_.sampling.max_tokens;
}

void InferenceOutput::parse(std::string piece, bool partial, TokenOutput& output) {
    if (!tokenizer_) return;
    if (!request_.chat) {
        result.text += piece;
        if (!piece.empty()) {
            common_chat_msg_diff diff;
            diff.content_delta = std::move(piece);
            output.diffs.push_back(std::move(diff));
        }
        return;
    }
    generated_ += piece;
    auto parsed = common_chat_parse(generated_, partial, request_.chat_parser);
    if (parsed.empty()) return;
    if (partial) parsed.tool_calls.clear();
    parsed.set_tool_call_ids(tool_ids_, [&] {
        return "call_" + id_ + "_" + std::to_string(tool_ids_.size());
    });
    auto diffs = common_chat_msg_diff::compute_diffs(message_, parsed);
    output.diffs.insert(output.diffs.end(),
        std::make_move_iterator(diffs.begin()), std::make_move_iterator(diffs.end()));
    message_ = std::move(parsed);
    result.text = message_.content;
    result.reasoning_text = message_.reasoning_content;
    result.tool_calls = message_.tool_calls;
}

TokenOutput InferenceOutput::append(const std::vector<std::int64_t>& tokens) {
    TokenOutput output;
    for (auto token : tokens) {
        if (stopped()) break;
        metrics.mark_token();
        ++result.completion_tokens;
        output.token_ids.push_back(token);
        if (!tokenizer_) continue;
        if (tokenizer_->is_eog(token)) {
            stopped_ = true;
            result.finish_reason = "stop";
            break;
        }
        emitter_.append(tokenizer_->piece(token, request_.preserved_tokens.count(token)));
        parse(emitter_.take(), true, output);
        if (emitter_.stopped()) {
            stopped_ = true;
            result.finish_reason = "stop";
        }
    }
    return output;
}

TokenOutput InferenceOutput::finish() {
    TokenOutput output;
    if (finished_) return output;
    finished_ = true;
    if (!emitter_.stopped()) emitter_.flush();
    parse(emitter_.take(), false, output);
    if (result.cancelled) result.finish_reason = "cancelled";
    else if (!result.tool_calls.empty()) result.finish_reason = "tool_calls";
    return output;
}

} // namespace mfq::engine
