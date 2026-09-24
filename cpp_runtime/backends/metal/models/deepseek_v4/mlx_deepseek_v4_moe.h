#pragma once

#include "deepseek_v4_model.h"
#include "mlx_grouped_linear.h"
#include "mlx_moe.h"
#include "mlx_ssd_expert_cache.h"
#include "mlx_tensor.h"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV4MoeResult {
    mlx::core::array output;
    mlx::core::array expert_ids;
    mlx::core::array expert_weights;
};

struct MlxDeepseekV4MoeBranches {
    mlx::core::array routed;
    mlx::core::array shared;
    mlx::core::array expert_ids;
    mlx::core::array expert_weights;
};

// Complete DeepSeek-V4 shared plus routed expert subgraph. Ordinary router,
// shared gate and shared up projections use one grouped packed dispatch when
// their formats support it. Routed gate/up and down remain MFE-native.
class MlxDeepseekV4Moe {
public:
    static MlxDeepseekV4Moe load(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        std::size_t layer,
        const std::optional<mlx::core::array>& available =
            std::nullopt,
        std::shared_ptr<MlxMfeOffloadCache> offload =
            nullptr,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache =
            nullptr);

    // Load a complete eager MoE stored below an arbitrary namespace such as
    // ``mtp.0``.  This is the shared small-M operator used by DSpark stages;
    // routed experts retain their packed MFQ representation.
    static MlxDeepseekV4Moe load_named(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        const std::string& prefix,
        const std::optional<mlx::core::array>& available =
            std::nullopt,
        std::shared_ptr<MlxMfeOffloadCache> offload =
            nullptr,
        std::size_t expert_cache_layer = 0,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache =
            nullptr);

    MlxDeepseekV4Moe(
        DeepseekV4Config config,
        MlxLinear router,
        MlxLinear shared_gate,
        MlxLinear shared_up,
        MlxLinear shared_down,
        MlxRoutedLinear routed_gate_up,
        MlxRoutedLinear routed_down,
        std::optional<mlx::core::array> router_bias =
            std::nullopt,
        std::optional<mlx::core::array> token_experts =
            std::nullopt,
        std::optional<mlx::core::array> available =
            std::nullopt,
        std::optional<mlx::core::array> visual_router_bias =
            std::nullopt);

    MlxDeepseekV4MoeResult forward_with_routing(
        const mlx::core::array& input,
        const mlx::core::array& token_ids) const;

    // Preserve routed and shared branches so the following HC expansion can
    // consume their sum directly without materializing an intermediate.
    MlxDeepseekV4MoeBranches forward_branches(
        const mlx::core::array& input,
        const mlx::core::array& token_ids) const;

    // Begin the full-layer native-expert read before the layer's attention
    // work. The caller keeps the handle alive and passes it back to the
    // three-argument forward_branches overload after submitting attention.
    std::optional<MlxSsdPrefetchedExpertLayer> prefetch_routed(
        std::size_t rows) const;
    MlxDeepseekV4MoeBranches forward_branches(
        const mlx::core::array& input,
        const mlx::core::array& token_ids,
        MlxSsdPrefetchedExpertLayer* prefetched) const;
    MlxDeepseekV4MoeBranches forward_branches(
        const mlx::core::array& input,
        const mlx::core::array& token_ids,
        MlxSsdPrefetchedExpertLayer* prefetched,
        bool may_contain_visual_tokens) const;

    mlx::core::array forward(
        const mlx::core::array& input,
        const mlx::core::array& token_ids) const;

    mlx::core::array operator()(
        const mlx::core::array& input,
        const mlx::core::array& token_ids) const {
        return forward(input, token_ids);
    }

    bool uses_grouped_shared_projection() const noexcept {
        return (projection_batch_.has_value() &&
                projection_batch_->grouped_projection_count() > 0) ||
            (shared_gate_up_batch_.has_value() &&
             shared_gate_up_batch_->grouped_projection_count() > 0);
    }
    bool uses_streamed_experts() const noexcept {
        return static_cast<bool>(expert_offload_)
            || static_cast<bool>(ssd_expert_cache_);
    }
    int recommended_prefill_chunk_size() const noexcept;

private:
    MlxDeepseekV4Moe(
        DeepseekV4Config config,
        MlxLinear router,
        MlxLinear shared_gate,
        MlxLinear shared_up,
        MlxLinear shared_down,
        std::optional<MlxRoutedLinear> routed_gate_up,
        std::optional<MlxRoutedLinear> routed_gate,
        std::optional<MlxRoutedLinear> routed_up,
        std::optional<MlxRoutedLinear> routed_down,
        std::shared_ptr<MlxMfeOffloadCache>
            expert_offload,
        std::shared_ptr<MlxMoeSsdExpertCache>
            ssd_expert_cache,
        std::size_t layer,
        std::string streamed_gate_up_name,
        std::string streamed_gate_name,
        std::string streamed_up_name,
        std::string streamed_down_name,
        std::optional<mlx::core::array> router_bias,
        std::optional<mlx::core::array> token_experts,
        std::optional<mlx::core::array> available,
        std::optional<mlx::core::array> visual_router_bias);

    std::vector<mlx::core::array> project_shared(
        const mlx::core::array& input,
        float swiglu_limit,
        bool project_router) const;

    DeepseekV4Config config_;
    MlxLinear router_;
    MlxLinear shared_gate_;
    MlxLinear shared_up_;
    MlxLinear shared_down_;
    std::optional<MlxRoutedLinear> routed_gate_up_;
    std::optional<MlxRoutedLinear> routed_gate_;
    std::optional<MlxRoutedLinear> routed_up_;
    std::optional<MlxRoutedLinear> routed_down_;
    std::shared_ptr<MlxMfeOffloadCache>
        expert_offload_;
    std::shared_ptr<MlxMoeSsdExpertCache>
        ssd_expert_cache_;
    std::size_t layer_ = 0;
    std::string streamed_gate_up_name_;
    std::string streamed_gate_name_;
    std::string streamed_up_name_;
    std::string streamed_down_name_;
    std::optional<MlxProjectionBatch> projection_batch_;
    std::optional<MlxProjectionBatch> shared_gate_up_batch_;
    bool fused_shared_swiglu_ = true;
    bool fused_dense_router_ = true;
    std::optional<mlx::core::array> router_bias_;
    std::optional<mlx::core::array> visual_router_bias_;
    std::optional<mlx::core::array> token_experts_;
    mlx::core::array available_;
    int available_count_ = 0;
};

} // namespace mfq::metal
