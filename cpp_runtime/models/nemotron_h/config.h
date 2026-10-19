#pragma once

#include "mfq_model_graph.h"
#include "models/common/model_config.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models::nemotron_h {

// Every Nemotron-H block is exactly one pre-normed mixer, so the block
// schedule alone decides the graph topology of a layer.
enum class MixerKind { linear_attention, full_attention, moe, mlp };

// Nemotron-H text geometry: Mamba-2 scan parameters and latent relu-squared
// MoE. The architecture adapter only translates this contract into the shared
// mamba/MoE/attention graphs; it owns no reusable behavior.
struct Config : ModelConfig {
    std::vector<MixerKind> mixers;
    // Mamba-2 mixer.
    std::int64_t mamba_heads = 0;
    std::int64_t mamba_head_dim = 0;
    std::int64_t ssm_state_size = 0;
    std::int64_t mamba_groups = 0;
    std::int64_t conv_kernel = 0;
    bool conv_bias = false;
    double time_step_limit_lo = 0.0;
    double time_step_limit_hi = 0.0;
    // Latent MoE. Mixed and shared experts are plain relu-squared MLPs.
    std::int64_t num_routed_experts = 0;
    std::int64_t num_experts_per_tok = 0;
    std::int64_t moe_intermediate_size = 0;
    std::int64_t moe_latent_size = 0;
    std::int64_t moe_shared_intermediate_size = 0;
    bool norm_topk_prob = false;
    double routed_scaling_factor = 1.0;

    std::int64_t mamba_inner_size() const noexcept { return mamba_heads * mamba_head_dim; }

    // Width of the depthwise convolution input: x followed by grouped B and C.
    std::int64_t mamba_conv_width() const noexcept {
        return mamba_inner_size() + 2 * mamba_groups * ssm_state_size;
    }

    // Rows of the fused in_proj: z, xBC and the per-head time step.
    std::int64_t mamba_projection_width() const noexcept {
        return 2 * mamba_inner_size() + 2 * mamba_groups * ssm_state_size + mamba_heads;
    }

    MixerKind mixer(int64_t layer) const { return mixers.at(static_cast<std::size_t>(layer)); }

    static MixerKind mixer_kind(std::string_view type);
    static Config from_json(std::string_view payload, const MfqModelGraph& graph);
};

} // namespace mfq::models::nemotron_h
