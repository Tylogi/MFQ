#pragma once
#include "causal_forward.h"
#include <utility>

namespace mfq::models {

// Verification keeps the confirmed prefix and checkpoints only the draft suffix.
// Slice/concat and checkpoint storage are supplied by the tensor backend.
template <class Forward, class Checkpoint, class Rollback, class Concat>
auto recurrent_window(int64_t tokens, bool cache, int64_t confirmed, bool pending, Forward forward,
                      Checkpoint checkpoint, Rollback rollback, Concat concat) {
    require_model(tokens > 0 && confirmed >= 0 && confirmed <= tokens && (!confirmed || cache),
                  "invalid recurrent verification window");
    if (confirmed == 0 || confirmed == tokens)
        return forward(0, tokens);
    require_model(!pending, "resolve previous recurrent checkpoint");
    auto prefix = forward(0, confirmed);
    checkpoint();
    try {
        auto suffix = forward(confirmed, tokens - confirmed);
        return concat(std::move(prefix), std::move(suffix));
    } catch (...) {
        rollback();
        throw;
    }
}

// GDN and KDA have the same graph. The decay operator defines their different
// scalar/per-channel equations; native recurrence kernels return fresh state.
template <class Project, class Convolve, class Gates, class Recur, class NormGate, class Output,
          class Commit>
auto gated_delta_attention(bool cache, Project project, Convolve convolve, Gates gates, Recur recur,
                           NormGate norm_gate, Output output, Commit commit) {
    auto projected = project();
    auto convolution = convolve(std::move(projected));
    auto decay = gates();
    auto recurrent = recur(convolution, std::move(decay));
    auto result = output(norm_gate(recurrent[0]));
    if (cache)
        commit(convolution, recurrent[1]);
    return result;
}

} // namespace mfq::models
