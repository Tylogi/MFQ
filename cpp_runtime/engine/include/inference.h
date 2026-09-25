#pragma once

#include "chat.h"
#include "mfq/runtime.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace mfq::engine {

struct InferenceRequest {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    common_chat_parser_params chat_parser;
    std::unordered_set<std::int64_t> preserved_tokens;
    std::vector<std::int64_t> prompt;
    std::vector<std::string> stops;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    MfqTokenConstraintPtr token_constraint;
    std::optional<MfqMultimodalInput> vision;
};

struct InferenceMetrics {
    using Clock = std::chrono::steady_clock;

    Clock::time_point started = Clock::now();
    Clock::time_point first_token;
    std::size_t prefill_tokens = 0;
    double prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    bool saw_token = false;
    bool saw_prefill = false;

    void mark_prefill(const MfqPrefillTiming& timing);
    void mark_token();
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

using InferenceExecute = std::function<std::int32_t(
    const MfqTokenCallback&, const MfqPrefillCallback&)>;
using InferenceEmit = std::function<bool(const common_chat_msg_diff&)>;

InferenceResult run_inference(
    const InferenceRequest& request,
    const MfqTokenizer& tokenizer,
    const InferenceExecute& execute,
    const std::function<bool()>& cancelled,
    const InferenceEmit& emit,
    InferenceMetrics* metrics,
    bool defer_token_parsing,
    const std::function<std::string()>& make_tool_call_id);

struct PrefillResult {
    std::int64_t token;
    MfqPrefillTiming timing;
};

// A whole-model request: adapters own device tensors, sampling and physical
// cache state. cache_position counts evaluated tokens, not emitted tokens.
namespace detail {
template <class Model, class Generate>
std::int32_t run_request(
        Model& model,
        const std::vector<std::int64_t>& prompt,
        const MfqTokenCallback& on_token,
        const MfqPromptCachePlan& cache_plan,
        Generate&& generate) {
    const auto stable = std::min(cache_plan.stable_prefix_tokens, prompt.size());
    const bool cache_enabled = stable > 0 && model.supports_cache() &&
        (!cache_plan.session_id.empty() || model.persistent_prefix_enabled());
    std::vector<std::int64_t> history = prompt;
    std::size_t last_snapshot = 0;
    const auto snapshot = [&](std::size_t length) {
        if (cache_enabled && length > 0 && length != last_snapshot &&
                length <= history.size() &&
                model.cache_position() == static_cast<std::int64_t>(length)) {
            model.snapshot(std::vector<std::int64_t>(
                history.begin(), history.begin() + length));
            last_snapshot = length;
        }
    };
    try {
        const auto reused = cache_enabled ? model.restore(stable) : std::size_t{0};
        if (reused == 0) model.reset();
        const auto emit = [&](std::int64_t token) {
            const bool keep_going = !on_token || on_token(token);
            if (keep_going) history.push_back(token);
            return keep_going;
        };
        const auto generated = generate(
            reused, cache_enabled ? stable : 0, snapshot, emit);
        // The last emitted token may not yet have been evaluated. Never
        // persist it as though it were part of the model's KV state.
        const auto position = model.cache_position();
        if (position > 0 && static_cast<std::size_t>(position) <= history.size()) {
            snapshot(static_cast<std::size_t>(position));
        }
        return generated;
    } catch (...) {
        model.reset();
        throw;
    }
}
} // namespace detail

// Shared target-token lifecycle. Backends provide prefill, advance and accept;
// optimized execution stays behind those operations.
template <class Model>
std::int32_t generate_target(
        Model& model,
        std::size_t reused,
        std::size_t stable,
        const std::function<void(std::size_t)>& checkpoint,
        const MfqTokenCallback& emit,
        const MfqPrefillCallback& on_prefill,
        std::int32_t max_tokens) {
    if (max_tokens <= 0) return 0;
    const auto first = model.prefill(reused, stable, checkpoint);
    if (stable == model.prompt_size()) checkpoint(stable);
    if (on_prefill) on_prefill(first.timing);
    std::int32_t generated = 0;
    std::int64_t token = first.token;
    while (generated < max_tokens) {
        ++generated;
        if (!emit(token)) break;
        model.accept(token);
        if (generated == max_tokens) break;
        token = model.advance();
    }
    return generated;
}

// One whole-model lifecycle. Adapters choose target or MTP generation;
// neither device loops nor predictor transactions belong in the Engine.
template <class Model>
std::int32_t generate(
        Model& model,
        const std::vector<std::int64_t>& prompt,
        const MfqSamplingParams& sampling,
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill,
        const MfqPromptCachePlan& cache_plan) {
    return detail::run_request(
        model, prompt, on_token, cache_plan,
        [&](std::size_t reused, std::size_t stable,
            const auto& snapshot, const auto& emit) {
            if (sampling.max_tokens <= 0) return std::int32_t{0};
            return model.generate(reused, stable, snapshot, emit,
                                  on_prefill, sampling.max_tokens);
        });
}

} // namespace mfq::engine
