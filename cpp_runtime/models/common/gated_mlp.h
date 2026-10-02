#pragma once
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::models {

enum class GatedActivation { silu, gelu };

// Canonical gate/up -> activation product -> down graph. Native operations may
// replace a contiguous part of this graph with an equivalent fused operation.
template <class Tensor, class Fused, class FusedProjection, class Project, class Activate,
          class Down, class FusedDown>
Tensor gated_mlp(Tensor input, bool geglu, double swiglu_limit, Fused fused,
                 FusedProjection fused_projection, Project project, Activate activate, Down down,
                 FusedDown fused_down) {
    const auto activation = geglu ? GatedActivation::gelu : GatedActivation::silu;
    const double limit = geglu ? 0.0 : swiglu_limit;
    if (auto output = fused(input, activation, limit))
        return std::move(*output);
    if (auto hidden = fused_projection(input, activation, limit))
        return down(std::move(*hidden));
    auto parts = project(std::move(input));
    if (parts.size() != 2)
        throw std::runtime_error("gated MLP requires gate and up projections");
    if (auto output = fused_down(parts[0], parts[1], activation, limit))
        return std::move(*output);
    return down(activate(std::move(parts[0]), std::move(parts[1]), activation, limit));
}

// Independent low/high neuron branches may run concurrently. Scheduling is
// native; selecting both results and merging them is part of the model graph.
template <class Branch, class Parallel, class Add>
auto additive_branches(Branch branch, Parallel parallel, Add add) {
    using Tensor = decltype(branch(0));
    std::vector<Tensor> outputs;
    if (!parallel(branch, outputs)) {
        outputs.push_back(branch(0));
        outputs.push_back(branch(1));
    }
    if (outputs.size() != 2)
        throw std::runtime_error("additive FFN requires two branch outputs");
    return add(std::move(outputs[0]), std::move(outputs[1]));
}

template <class Tensor, class Up, class Activate, class Down>
auto mlp(Tensor hidden, Up up, Activate activate, Down down) {
    return down(activate(up(std::move(hidden))));
}

} // namespace mfq::models
