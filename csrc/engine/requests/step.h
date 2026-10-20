#pragma once

#include "engine.h"
#include "policy.h"
#include "inference.h"
#include "step_sequence.h"
#include "activity.h"

#include <exception>
#include <functional>
#include <stdexcept>
#include <utility>

namespace mfq::engine {

using mfq::StepState;
using Generation = mfq::StepSequence<EventData>;

inline void require_execution(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

// Prompt progress published while prefill runs: llm_ms is the language-model
// part, multimodal_ms the encoder work that preceded it, model_ms both.
inline PrefillProgress prefill_progress(
    std::size_t prompt_tokens, double llm_ms, double multimodal_ms = 0.0) {
    return PrefillProgress{{prompt_tokens, llm_ms, multimodal_ms, llm_ms + multimodal_ms}};
}

// Resource owners release exactly once, including coroutine destruction. The
// executor observes teardown failures before publishing a terminal event.
template <class Release> struct ExecutionCleanup {
    std::exception_ptr& failure;
    Release release;
    bool pending = true;
    void finish() {
        if (!std::exchange(pending, false)) return;
        try { release(); }
        catch (...) { failure = std::current_exception(); throw; }
    }
    ~ExecutionCleanup() { try { finish(); } catch (...) {} }
};
template <class Release>
ExecutionCleanup(std::exception_ptr&, Release) -> ExecutionCleanup<Release>;

// Ops performs one device prefill/decode operation. Chunking, cancellation,
// token acceptance and publication order are independent of tensor storage.
template <class Operations>
Generation generate_sequence(Operations operations, InferenceOutput &output, std::int64_t prompt_tokens,
    std::int64_t reused_tokens, std::int64_t stable_prefix_tokens, std::int64_t chunk_size) {
    std::unwrap_reference_t<Operations>& ops = operations;
    double elapsed = 0.0;
    auto offset = reused_tokens;
    while (offset < prompt_tokens && !output.stopped()) {
        const auto end = offset < stable_prefix_tokens
                             ? std::min(prompt_tokens, stable_prefix_tokens)
                             : prompt_tokens;
        auto chunk = next_prefill_chunk(end, offset, chunk_size);
        if constexpr (requires { ops.prefill_chunk(chunk); }) chunk = ops.prefill_chunk(chunk);
        if constexpr (requires { ops.schedule_prefill(chunk); }) {
            ops.schedule_prefill(chunk);
            co_yield StepState::advanced;
            while (!ops.ready() && !output.stopped()) co_yield StepState::waiting;
            if (output.stopped()) co_return;
        }
        {
            PrefillActivity activity;
            elapsed += ops.prefill(chunk);
        }
        offset += chunk.count;
        co_yield prefill_progress(static_cast<std::size_t>(offset - reused_tokens), elapsed);
    }
    if (output.stopped())
        co_return;
    if constexpr (requires { ops.generate_decode(output); }) {
        auto generation = ops.generate_decode(output);
        if (generation) {
            while (auto step = generation->next()) co_yield std::move(step);
            co_return;
        }
    }
    auto token = ops.first_token();
    while (!output.stopped()) {
        auto delta = output.append(std::vector<std::int64_t>{token});
        ops.accept(token);
        if constexpr (requires { ops.schedule_decode(); }) {
            if (!output.stopped()) ops.schedule_decode();
        }
        co_yield std::move(delta);
        if (!output.stopped()) {
            if constexpr (requires { ops.schedule_decode(); }) {
                while (!ops.ready() && !output.stopped()) co_yield StepState::waiting;
                if (output.stopped()) co_return;
            }
            token = ops.advance();
        }
    }
}

} // namespace mfq::engine
