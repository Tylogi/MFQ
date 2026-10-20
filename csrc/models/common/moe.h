#pragma once
#include "gated_mlp.h"
#include <utility>

namespace mfq::models {

enum class RouterActivation { softmax, sigmoid, sqrt_softplus };
struct MoeRouting {
    RouterActivation activation = RouterActivation::softmax;
    bool normalize = false;
    bool delayed_softmax = false;
    double scale = 1.0;
};

// Both branches consume the same input. Native routing may prepare an expert
// cache before shared computation, but never changes the selected experts.
template <class Router, class Select, class Shared, class Experts, class Combine>
auto mixture_of_experts(Router router, Select select, Shared shared, Experts experts,
                        Combine combine) {
    auto selected = select(router());
    auto shared_output = shared(selected);
    auto routed = experts(selected);
    return combine(std::move(routed), std::move(shared_output), selected);
}

template <class Project, class Activate, class Down, class Reduce>
auto routed_experts(Project project, Activate activate, Down down, Reduce reduce) {
    auto projections = project();
    auto hidden = activate(std::move(projections));
    auto pairs = down(std::move(hidden));
    return reduce(std::move(pairs));
}

// Fused reduction may include shared gating and addition. Otherwise the same
// graph is evaluated explicitly, including the ungated shared-expert variant.
template <class Tensor, class Fused, class Reduce, class Gate, class Add, class GatedAdd>
auto combine_experts(Tensor pairs, Tensor shared, bool reduced, bool ungated, Fused fused,
                     Reduce reduce, Gate gate, Add add, GatedAdd gated_add) {
    if (!reduced)
        if (auto result = fused(pairs, shared))
            return std::move(*result);
    auto routed = reduced ? std::move(pairs) : reduce(std::move(pairs));
    if (ungated)
        return add(std::move(routed), std::move(shared));
    auto logits = gate();
    return gated_add(std::move(routed), std::move(shared), std::move(logits));
}

} // namespace mfq::models
