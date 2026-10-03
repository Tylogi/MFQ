#pragma once
#include <optional>
#include <utility>

namespace mfq::models {

template <class Tensor> struct AttentionHeads {
    Tensor query, key, value;
    std::optional<Tensor> gate;
};

// The model orders these operations. Fused prepare operations return true only
// after replacing the complete norm/rotary/cache span (or rotary/cache span).
template <class Tensor, class Project, class FusedPrepare, class Normalize, class FusedRotaryCache,
          class Rotary, class Cache, class Attend, class Output>
Tensor full_attention(Tensor normalized, Project project, FusedPrepare fused_prepare,
                      Normalize normalize, FusedRotaryCache fused_rotary_cache, Rotary rotary,
                      Cache cache, Attend attend, Output output) {
    auto heads = project(std::move(normalized));
    if (!fused_prepare(heads)) {
        normalize(heads);
        if (!fused_rotary_cache(heads)) {
            rotary(heads);
            cache(heads);
        }
    }
    auto attended = attend(heads);
    return output(std::move(attended), std::move(heads.gate));
}

} // namespace mfq::models
