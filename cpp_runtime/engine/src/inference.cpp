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

namespace {

class ChatOutputParser {
public:
    ChatOutputParser(
            const common_chat_parser_params& params,
            InferenceEmit emit,
            std::function<std::string()> make_tool_call_id)
        : params_(params),
          emit_(std::move(emit)),
          make_tool_call_id_(std::move(make_tool_call_id)) {
        if (params_.is_continuation && !params_.echo) {
            message_ = common_chat_parse("", true, params_);
        }
    }

    bool append(const std::string& piece) {
        generated_ += piece;
        return update(true);
    }

    bool flush() { return update(false); }
    const common_chat_msg& message() const { return message_; }

private:
    bool update(bool partial) {
        common_chat_msg parsed =
            common_chat_parse(generated_, partial, params_);
        if (parsed.empty()) return true;
        if (partial) parsed.tool_calls.clear();
        parsed.set_tool_call_ids(tool_call_ids_, make_tool_call_id_);
        const auto diffs =
            common_chat_msg_diff::compute_diffs(message_, parsed);
        message_ = std::move(parsed);
        for (const auto& diff : diffs) {
            if (!emit_(diff)) return false;
        }
        return true;
    }

    common_chat_parser_params params_;
    InferenceEmit emit_;
    std::function<std::string()> make_tool_call_id_;
    std::string generated_;
    common_chat_msg message_;
    std::vector<std::string> tool_call_ids_;
};

} // namespace

InferenceResult run_inference(
        const InferenceRequest& request,
        const MfqTokenizer& tokenizer,
        const InferenceExecute& execute,
        const std::function<bool()>& cancelled,
        const InferenceEmit& emit,
        InferenceMetrics* metrics,
        bool defer_token_parsing,
        const std::function<std::string()>& make_tool_call_id) {
    InferenceResult result;
    const auto emit_parsed = [&](const common_chat_msg_diff& diff) {
        result.client_connected = emit(diff);
        return result.client_connected;
    };
    std::unique_ptr<ChatOutputParser> chat_parser;
    if (request.chat) {
        chat_parser = std::make_unique<ChatOutputParser>(
            request.chat_parser, emit_parsed, make_tool_call_id);
    }
    TextEmitter emitter(request.stops, [&](const std::string& text) {
        if (chat_parser) return chat_parser->append(text);
        result.text += text;
        common_chat_msg_diff diff;
        diff.content_delta = text;
        return emit_parsed(diff);
    });

    const bool defer_tokens =
        defer_token_parsing && request.stops.empty();
    std::vector<std::int64_t> deferred_tokens;
    if (defer_tokens) {
        deferred_tokens.reserve(static_cast<std::size_t>(
            std::max(request.sampling.max_tokens, 0)));
    }
    const auto on_token = [&](std::int64_t token) {
        if (metrics) metrics->mark_token();
        if (tokenizer.is_eog(token)) {
            result.finish_reason = "stop";
            return false;
        }
        if (defer_tokens) {
            deferred_tokens.push_back(token);
            return true;
        }
        const bool preserve =
            request.preserved_tokens.find(token) !=
            request.preserved_tokens.end();
        if (!emitter.append(tokenizer.piece(token, preserve))) {
            if (emitter.stopped()) result.finish_reason = "stop";
            return false;
        }
        return true;
    };
    const auto on_prefill = [&](const MfqPrefillTiming& timing) {
        if (metrics) metrics->mark_prefill(timing);
    };
    result.completion_tokens = execute(on_token, on_prefill);
    if (cancelled && cancelled()) {
        result.cancelled = true;
        result.finish_reason = "cancelled";
    }
    if (defer_tokens) {
        std::string text;
        text.reserve(deferred_tokens.size() * 8);
        for (const auto token : deferred_tokens) {
            const bool preserve =
                request.preserved_tokens.find(token) !=
                request.preserved_tokens.end();
            text += tokenizer.piece(token, preserve);
        }
        if (!text.empty() && !emitter.append(text) && emitter.stopped()) {
            result.finish_reason = "stop";
        }
    }
    if (result.client_connected && !emitter.stopped()) emitter.flush();
    if (result.client_connected && chat_parser) {
        chat_parser->flush();
        const auto& message = chat_parser->message();
        result.text = message.content;
        result.reasoning_text = message.reasoning_content;
        result.tool_calls = message.tool_calls;
    }
    if (emitter.stopped() && !result.cancelled) {
        result.finish_reason = "stop";
    }
    if (!result.cancelled && !result.tool_calls.empty()) {
        result.finish_reason = "tool_calls";
    }
    return result;
}

} // namespace mfq::engine
