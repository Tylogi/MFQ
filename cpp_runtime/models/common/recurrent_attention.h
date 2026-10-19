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

// Mamba-2 has the same state lifetime as GDN but different equations: the
// depthwise convolution state and the selective-scan state are separate, and
// the gated norm runs after the scan (norm_before_gate=false).
// project() returns {gate(z), convolution input(xBC), raw per-head dt};
// convolve() turns that input into {x, B, C} and advances the convolution
// state; scan() returns {y, final recurrent state} and updates the recurrent
// state in place when it is a single decode step. commit() receives the final
// state produced by a prefill scan (undefined for decode).
template <class Tensor, class Project, class Convolve, class Scan, class GatedNorm, class Output,
          class Commit>
Tensor selective_state_attention(bool cache, Project project, Convolve convolve, Scan scan,
                                GatedNorm norm_gate, Output output, Commit commit) {
    auto projected = project();
    if (projected.size() != 3)
        throw std::runtime_error("Mamba-2 projection requires z, xBC and dt");
    auto signals = convolve(projected);
    auto scanned = scan(signals, projected[2]);
    auto result = output(norm_gate(scanned[0], projected[0]));
    if (cache)
        commit(scanned[1]);
    return result;
}

} // namespace mfq::models
