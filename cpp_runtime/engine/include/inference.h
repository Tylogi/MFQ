#pragma once

#include "mfq/runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mfq::engine {

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
