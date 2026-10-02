#pragma once

#include "model_config.h"
#include <memory>
#include <string>
#include <vector>

namespace mfq::models {

enum class ExpertProjection { gate, up, gate_up, down };

// Canonical parameter relationships are shared; the loader materializes native
// weights and chooses packing, projection fusion and device placement.
template <class Ffn, class Loader>
Ffn load_dense_ffn(Loader& ops, const ModelConfig& config, const std::string& prefix) {
    Ffn result;
    const auto down = prefix + "down.weight", gate = prefix + "gate.weight",
               up = prefix + "up.weight";
    result.down = ops.linear(down);
    result.gate_up = ops.gate_up({gate, up}, result.down);
    ops.important_neurons(config.hidden_size, config.intermediate_size, result, down, gate, up);
    ops.workspace(result);
    return result;
}

template <class Ffn, class Loader>
Ffn load_moe_weights(Loader& ops, const std::string& prefix, int layer,
                    bool router_bias_required = false) {
    const auto gate_up = prefix + "experts.gate_up.weight", gate = prefix + "experts.gate.weight",
               up = prefix + "experts.up.weight", down = prefix + "experts.down.weight";
    const bool fused = ops.has(gate_up), split = ops.has(gate);
    if (split != ops.has(up) || fused == split || !ops.has(down))
        throw std::runtime_error("routed MoE requires down and exactly one fused or split "
                                 "Gate/Up representation at layer " + std::to_string(layer));
    Ffn result;
    result.is_moe = true;
    result.moe_split_gate_up = split;
    result.moe_layer = layer;
    if (split) {
        ops.experts(result, ExpertProjection::gate, gate, layer);
        ops.experts(result, ExpertProjection::up, up, layer);
    } else {
        ops.experts(result, ExpertProjection::gate_up, gate_up, layer);
    }
    ops.experts(result, ExpertProjection::down, down, layer);
    result.moe_router = ops.router_parameter(prefix + "router.weight");
    if (router_bias_required || ops.has(prefix + "router.bias"))
        result.moe_router_bias = ops.router_parameter(prefix + "router.bias");
    result.shared = std::make_unique<Ffn>();
    result.shared->down = ops.linear(prefix + "shared_expert.down.weight");
    result.shared->gate_up = ops.gate_up(
        {prefix + "shared_expert.gate.weight", prefix + "shared_expert.up.weight"}, result.shared->down);
    ops.workspace(*result.shared);
    return result;
}

template <class Block, class Loader>
void load_full_attention(Block& block, Loader& ops, const ModelConfig& config, int layer,
                         const std::string& prefix, double norm_offset, bool output_gate = false) {
    block.layer = layer;
    block.attention_heads = config.num_attention_heads;
    block.kv_heads = config.num_key_value_heads;
    block.attention_head_dim = config.head_dim;
    block.max_position_embeddings = config.max_position_embeddings;
    block.rms_norm_eps = config.rms_norm_eps;
    block.norm_weight_offset = norm_offset;
    block.attention_output_gate = output_gate;
    block.attn_norm = ops.dense(prefix + "attention.norm.weight");
    block.ffn_norm = ops.dense(prefix + "mlp.norm.weight");
    const auto attention = prefix + "attention.";
    ops.qkv(block, {attention + "query.weight", attention + "key.weight", attention + "value.weight"});
    block.o = ops.linear(attention + "output.weight");
    if (ops.has(attention + "query_norm.weight"))
        block.q_norm = ops.dense(attention + "query_norm.weight");
    if (ops.has(attention + "key_norm.weight"))
        block.k_norm = ops.dense(attention + "key_norm.weight");
}

} // namespace mfq::models
