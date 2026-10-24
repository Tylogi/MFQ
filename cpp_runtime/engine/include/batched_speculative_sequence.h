#pragma once

#include "generation_step.h"
#include "execution_check.h"

namespace mfq::engine {

// Numerical verification may finish before a paused consumer can publish it.
// Resolve only the prefix actually emitted by InferenceOutput (EOS/stop/limit).
template<class Request, class Schedule, class Ready, class Check, class Publish>
Generation batched_speculative_sequence(Request& request, InferenceOutput& output,
        int64_t first, Schedule schedule, Ready ready, Check check, Publish publish,
        std::function<void(const TokenOutput&)> accept_initial = {}) {
    auto initial = output.append(std::vector<int64_t>{first});
    if (accept_initial) accept_initial(initial);
    if constexpr (requires { request.accepted_tokens; })
        request.accepted_tokens.insert(request.accepted_tokens.end(), initial.token_ids.begin(), initial.token_ids.end());
    request.produced += initial.token_ids.size();
    request.pending_token = first;
    co_yield std::move(initial);
    while (!output.stopped()) {
        schedule();
        co_yield StepState::advanced;
        while (!ready() && !output.stopped()) co_yield StepState::waiting;
        if (output.stopped()) co_return;
        check();
        auto delta = publish(output);
        require_execution(!delta.token_ids.empty(), "batched MTP published an empty live window");
        if constexpr (requires { request.accepted_tokens; })
            request.accepted_tokens.insert(request.accepted_tokens.end(), delta.token_ids.begin(), delta.token_ids.end());
        request.produced += delta.token_ids.size();
        request.pending_token = delta.token_ids.back();
        co_yield std::move(delta);
    }
}

} // namespace mfq::engine
