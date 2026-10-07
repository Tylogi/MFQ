#pragma once

#include "engine.h"
#include "generation_policy.h"
#include "step_sequence.h"
#include "prefill_activity.h"

#include <exception>
#include <functional>
#include <utility>

namespace mfq::engine {

using mfq::StepState;
using Generation = mfq::StepSequence<EventData>;

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
        const auto chunk = next_prefill_chunk(end, offset, chunk_size);
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
        co_yield PrefillProgress{
            {static_cast<std::size_t>(offset - reused_tokens), elapsed, 0.0, elapsed}};
    }
    if (output.stopped())
        co_return;
    auto token = ops.first_token();
    while (!output.stopped()) {
        auto delta = output.append(std::vector<std::int64_t>{token});
        ops.accept(token);
        co_yield std::move(delta);
        if (!output.stopped()) {
            if constexpr (requires { ops.schedule_decode(); }) {
                ops.schedule_decode();
                co_yield StepState::advanced;
                while (!ops.ready() && !output.stopped()) co_yield StepState::waiting;
                if (output.stopped()) co_return;
            }
            token = ops.advance();
        }
    }
}

} // namespace mfq::engine
