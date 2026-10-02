#pragma once
#include <stdexcept>
#include <utility>

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

} // namespace mfq::models
