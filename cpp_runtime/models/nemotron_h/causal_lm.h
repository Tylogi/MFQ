#pragma once
#include "config.h"
#include "models/common/weight_loading.h"
#include "models/common/causal_model.h"
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::models::nemotron_h {

// One pre-normed Mamba-2 mixer. The fused projection keeps z, xBC and dt in
// one matrix; the graph slices them instead of splitting the checkpoint
// tensor, which would requantize the narrow time-step tail.
template <class Block, class Loader>
void load_linear_attention(Block& block, Loader& ops, const Config& config, int layer,
                           std::string_view tensor_root = "model") {
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    const auto linear = layer_prefix + "linear_attention.";
    block.layer = layer;
    block.rms_norm_eps = config.rms_norm_eps;
    block.mamba_heads = config.mamba_heads;
    block.mamba_head_dim = config.mamba_head_dim;
    block.ssm_state_size = config.ssm_state_size;
    block.mamba_groups = config.mamba_groups;
    block.conv_kernel = config.conv_kernel;
    block.time_step_limit_lo = config.time_step_limit_lo;
    block.time_step_limit_hi = config.time_step_limit_hi;
    block.mixer_norm = ops.dense(layer_prefix + "attention.norm.weight");
    block.in_proj = ops.linear(linear + "in_projection.weight");
    block.conv_weight = ops.dense(linear + "conv.weight");
    if (ops.has(linear + "conv.bias"))
        block.conv_bias = ops.dense(linear + "conv.bias");
    block.a_log = ops.dense(linear + "a");
    block.d = ops.dense(linear + "d");
    block.dt_bias = ops.dense(linear + "dt_bias");
    block.scan_norm = ops.dense(linear + "norm.weight");
    block.out_proj = ops.linear(linear + "output.weight");
    const auto shape = [&](const auto& value) { return ops.shape(value); };
    const std::vector<int64_t> expected_convolution{config.mamba_conv_width(), 1,
                                                    config.conv_kernel};
    if (block.in_proj.out() != config.mamba_projection_width() ||
        block.in_proj.neuron_len() != config.hidden_size ||
        shape(block.conv_weight) != expected_convolution ||
        ops.elements(block.scan_norm) != config.mamba_inner_size() ||
        block.out_proj.out() != config.hidden_size ||
        block.out_proj.neuron_len() != config.mamba_inner_size() ||
        ops.elements(block.mixer_norm) != config.hidden_size ||
        ops.elements(block.a_log) != config.mamba_heads ||
        ops.elements(block.d) != config.mamba_heads ||
        ops.elements(block.dt_bias) != config.mamba_heads ||
        (config.conv_bias && ops.elements(block.conv_bias) != config.mamba_conv_width())) {
        throw std::runtime_error("Nemotron-H mamba tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
}

// One pre-normed NoPE attention mixer. The layer keeps the shared full
// attention block; only the block's FFN tail is absent.
template <class Block, class Loader>
void load_attention(Block& block, Loader& ops, const Config& config, int layer,
                    std::string_view tensor_root = "model") {
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    models::load_attention_mixer(block, ops, config, layer, layer_prefix, 0.0, false);
}

// One pre-normed latent MoE (or dense relu-squared MLP) mixer. The router and
// the shared expert read the residual stream; only the routed experts run in
// the projected latent space.
template <class Block, class Loader>
void load_moe(Block& block, Loader& ops, const Config& config, int layer,
              std::string_view tensor_root = "model") {
    if (config.num_routed_experts <= 0 || config.moe_latent_size <= 0)
        throw std::runtime_error("Nemotron-H MoE layer has no routed-expert geometry");
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    const auto mlp = layer_prefix + "mlp.";
    block.layer = layer;
    block.rms_norm_eps = config.rms_norm_eps;
    block.ffn_norm = ops.dense(layer_prefix + "attention.norm.weight");
    auto& ffn = block.ffn;
    ffn = models::load_moe_weights<typename Loader::Ffn>(ops, mlp, layer, true, true, true);
    ffn.moe_top_k = static_cast<int>(config.num_experts_per_tok);
    ffn.moe_use_sigmoid = true;
    ffn.moe_use_sqrt_softplus = false;
    ffn.moe_normalize = config.norm_topk_prob;
    ffn.moe_delayed_softmax = false;
    ffn.moe_router_scale = config.routed_scaling_factor;
    ffn.moe_shared_ungated = true;
    const bool dense_bank = ffn.moe_dense_up.defined() || ffn.moe_dense_down.defined();
    const auto rows = static_cast<int64_t>(config.num_routed_experts);
    const bool expert_shapes =
        dense_bank ? ffn.moe_dense_up.dim() == 3 &&
                         ffn.moe_dense_up.size(0) == rows &&
                         ffn.moe_dense_up.size(1) == config.moe_intermediate_size &&
                         ffn.moe_dense_up.size(2) == config.moe_latent_size &&
                         ffn.moe_dense_down.dim() == 3 &&
                         ffn.moe_dense_down.size(0) == rows &&
                         ffn.moe_dense_down.size(1) == config.moe_latent_size &&
                         ffn.moe_dense_down.size(2) == config.moe_intermediate_size
                   : ffn.moe_up.n_experts == config.num_routed_experts &&
                         ffn.moe_up.neuron_len == config.moe_latent_size &&
                         ffn.moe_up.out_per_expert == config.moe_intermediate_size &&
                         ffn.moe_down.n_experts == config.num_routed_experts &&
                         ffn.moe_down.neuron_len == config.moe_intermediate_size &&
                         ffn.moe_down.out_per_expert == config.moe_latent_size;
    if (!expert_shapes || ffn.moe_router.dim() != 2 ||
        ffn.moe_router.size(0) != config.num_routed_experts ||
        ffn.moe_router.size(1) != config.hidden_size ||
        ops.elements(ffn.moe_router_bias) != config.num_routed_experts ||
        ffn.moe_latent_in.out() != config.moe_latent_size ||
        ffn.moe_latent_in.neuron_len() != config.hidden_size ||
        ffn.moe_latent_out.out() != config.hidden_size ||
        ffn.moe_latent_out.neuron_len() != config.moe_latent_size ||
        ffn.shared->gate_up.outs.size() != 1 ||
        ffn.shared->gate_up.outs[0] != config.moe_shared_intermediate_size ||
        ffn.shared->down.out() != config.hidden_size ||
        ffn.shared->down.neuron_len() != config.moe_shared_intermediate_size ||
        ops.elements(block.ffn_norm) != config.hidden_size) {
        throw std::runtime_error("Nemotron-H MoE tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
}

// The dense relu-squared MLP block of the same family shares the canonical
// non-gated MLP graph, so it needs no separate kernel.
template <class Block, class Loader>
void load_dense_mlp(Block& block, Loader& ops, const Config& config, int layer,
                    std::string_view tensor_root = "model") {
    const auto layer_prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    const auto mlp = layer_prefix + "mlp.";
    block.layer = layer;
    block.rms_norm_eps = config.rms_norm_eps;
    block.ffn_norm = ops.dense(layer_prefix + "attention.norm.weight");
    auto& ffn = block.ffn;
    ffn.down = ops.linear(mlp + "down.weight");
    ffn.gate_up = ops.gate_up({mlp + "up.weight"}, ffn.down);
    ffn.activation = models::MlpActivation::relu_squared;
    ops.workspace(ffn);
    if (ffn.gate_up.outs.size() != 1 || ffn.gate_up.outs[0] != config.intermediate_size ||
        ffn.down.out() != config.hidden_size ||
        ffn.down.neuron_len() != config.intermediate_size) {
        throw std::runtime_error("Nemotron-H dense MLP shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "nemotron_h"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph& graph, const Source& source) {
        auto& config = this->config;
        auto& metadata = this->metadata;

        config = Config::from_json(payload, graph);
        metadata.vocab_size = config.vocab_size;
        metadata.hidden_size = config.hidden_size;
        metadata.num_hidden_layers = config.num_hidden_layers;
        metadata.num_attention_heads = config.num_attention_heads;
        metadata.num_key_value_heads = config.num_key_value_heads;
        metadata.head_dim = config.head_dim;
        metadata.max_position_embeddings = config.max_position_embeddings;
        metadata.rotary_dim = 0;
        metadata.num_experts = config.num_routed_experts;
        metadata.rope_base = config.rope_base;
        metadata.rms_norm_eps = config.rms_norm_eps;
        metadata.norm_weight_offset = 0.0;
        metadata.tie_word_embeddings = config.tie_word_embeddings;
        metadata.no_rope = true;
        metadata.model_type = config.model_type;
        metadata.layer_types = config.layer_types;
        this->adapter_validate_components(graph);
    }
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
    bool adapter_supports_speculation() const noexcept { return true; }
};

} // namespace mfq::models::nemotron_h
