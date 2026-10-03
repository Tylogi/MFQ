#pragma once
#include "config.h"
#include "models/common/causal_model.h"
#include <array>
#include <bit>
#include <cmath>
#include <utility>

namespace mfq::models::gemma4 {

template <class Tensor> struct FfnInputs {
    Tensor residual, dense, router, experts;
};

template <class Tensor, class Fused, class PostNorm, class Add, class DenseNorm, class RouterNorm,
          class ExpertNorm>
auto prepare_ffn(Tensor residual, Tensor attention, bool moe, Fused fused, PostNorm post_norm,
                 Add add, DenseNorm dense_norm, RouterNorm router_norm, ExpertNorm expert_norm) {
    if (auto prepared = fused(residual, attention))
        return std::move(*prepared);
    auto hidden = add(std::move(residual), post_norm(std::move(attention)));
    FfnInputs<Tensor> result{hidden, dense_norm(hidden), {}, {}};
    if (moe) {
        result.router = router_norm(hidden);
        result.experts = expert_norm(hidden);
    }
    return result;
}

enum class FfnNorm { dense, experts, output };

// Fusion replaces the same normalization/merge span. The ordinary graph stays
// here so a backend cannot silently choose different residual or scale order.
template <class Tensor, class Dense, class Experts, class Normalize, class Add, class Residual,
          class Scale, class Fused>
auto feed_forward(bool moe, bool has_scale, bool fused_merge, Tensor residual, Dense dense,
                  Experts experts, Normalize normalize, Add add, Residual add_residual, Scale scale,
                  Fused fused) {
    auto output = dense();
    if (moe) {
        if (!fused_merge)
            output = normalize(std::move(output), FfnNorm::dense);
        auto routed = experts();
        if (fused_merge)
            return fused(std::move(output), std::move(routed), std::move(residual));
        output = add(std::move(output), normalize(std::move(routed), FfnNorm::experts));
    }
    output = add_residual(std::move(residual), normalize(std::move(output), FfnNorm::output));
    return has_scale ? scale(std::move(output)) : output;
}

template <class Router, class Select, class Scale, class Project, class Reduce>
auto experts(Router router, Select select, Scale scale, Project project, Reduce reduce) {
    auto selected = select(router());
    scale(selected);
    auto pairs = project(selected);
    return reduce(std::move(pairs), selected);
}

struct LayerSpec {
    bool moe, sliding, value_equals_key;
    int64_t heads, kv_heads, head_dim, window, rotary_pairs;
    double rope_base;
};

inline LayerSpec layer_spec(const Config &c, std::string_view type) {
    require_model(type == "full_attention" || type == "sliding_attention",
                  "unsupported Gemma4 layer type");
    const bool sliding = type == "sliding_attention";
    const auto width = sliding ? c.head_dim : c.global_head_dim;
    return {c.num_experts > 0,
            sliding,
            !sliding && c.attention_key_equals_value,
            c.num_attention_heads,
            sliding ? c.num_key_value_heads : c.num_global_key_value_heads,
            width,
            sliding ? c.sliding_window : 0,
            sliding ? -1 : int64_t(std::llround(c.full_rotary_factor * double(width) / 2.0)),
            sliding ? c.sliding_rope_base : c.rope_base};
}

template <class Block, class Loader>
void load_block(Block &b, Loader &ops, const Config &c, int layer, std::string_view type) {
    const auto spec = layer_spec(c, type);
    b.layer = layer;
    b.gemma4_moe = spec.moe;
    b.max_position_embeddings = c.max_position_embeddings;
    b.rms_norm_eps = c.rms_norm_eps;
    b.norm_weight_offset = 0.0;
    b.sliding = spec.sliding;
    b.value_equals_key = spec.value_equals_key;
    b.attention_heads = spec.heads;
    b.kv_heads = spec.kv_heads;
    b.attention_head_dim = b.attention_rotary_dim = spec.head_dim;
    b.attention_window = spec.window;
    b.attention_scale = 1.0;
    b.attention_rope = ops.rope(c.max_position_embeddings, spec);
    const auto root = "model.block." + std::to_string(layer) + ".";
    const auto attention = root + "attention.";
    b.attn_norm = ops.dense(attention + "norm.weight");
    b.attn_post_norm = ops.dense(attention + "output_norm.weight");
    b.q_norm = ops.dense(attention + "query_norm.weight");
    b.k_norm = ops.dense(attention + "key_norm.weight");
    b.v_norm = ops.ones(spec.head_dim);
    std::vector<std::string> projections{attention + "query.weight", attention + "key.weight"};
    if (!spec.value_equals_key)
        projections.push_back(attention + "value.weight");
    b.qkv = ops.projections(projections);
    b.o = ops.linear(attention + "output.weight");
    const auto mlp = root + "mlp.";
    b.ffn_norm = ops.dense(mlp + "dense.input_norm.weight");
    b.ffn_post_norm = ops.dense(mlp + "output_norm.weight");
    b.layer_scale = ops.layer_scale(root + "output_scale");
    if (spec.moe) {
        b.ffn_post_norm_1 = ops.dense(mlp + "dense.output_norm.weight");
        b.ffn_pre_norm_2 = ops.dense(mlp + "experts.input_norm.weight");
        b.ffn_post_norm_2 = ops.dense(mlp + "experts.output_norm.weight");
    }
    b.ffn.geglu = true;
    b.ffn.down = ops.linear(mlp + "down.weight");
    b.ffn.gate_up = ops.gate_up({mlp + "gate.weight", mlp + "up.weight"}, b.ffn.down);
    ops.workspace(b.ffn);
    if (!spec.moe)
        return;
    b.gemma_moe_gate_up = ops.experts(mlp + "experts.gate_up.weight", layer, "gate_up");
    b.gemma_moe_down = ops.experts(mlp + "experts.down.weight", layer, "down");
    b.gemma_router = ops.router_parameter(mlp + "router.weight");
    b.gemma_router_norm_scale = ops.normalized_router_parameter(mlp + "router.norm.weight",
                                                                std::sqrt(double(c.hidden_size)));
    b.gemma_expert_scale = ops.router_parameter(mlp + "router.expert_scale");
    b.gemma_top_k = static_cast<int>(c.num_experts_per_tok);
    require_model(
        ops.expert_shape(b.gemma_moe_gate_up) ==
                std::array<int64_t, 3>{c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size} &&
            ops.expert_shape(b.gemma_moe_down) ==
                std::array<int64_t, 3>{c.num_experts, c.hidden_size, c.moe_intermediate_size} &&
            ops.shape(b.gemma_router) == std::vector<int64_t>{c.num_experts, c.hidden_size} &&
            ops.elements(b.gemma_router_norm_scale) == c.hidden_size &&
            ops.elements(b.gemma_expert_scale) == c.num_experts,
        "Gemma4 MoE tensor shapes disagree with config");
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "gemma4"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;
        auto &embed_scale = this->embed_scale;

        config = Config::from_json(payload);
        auto bits = std::bit_cast<std::uint32_t>(
            static_cast<float>(std::sqrt(static_cast<double>(config.hidden_size))));
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        embed_scale = std::bit_cast<float>(bits & 0xffff0000U);
        metadata.vocab_size = config.vocab_size;
        metadata.hidden_size = config.hidden_size;
        metadata.num_hidden_layers = config.num_hidden_layers;
        metadata.num_attention_heads = config.num_attention_heads;
        metadata.num_key_value_heads = config.num_key_value_heads;
        metadata.head_dim = config.head_dim;
        metadata.max_position_embeddings = config.max_position_embeddings;
        metadata.rotary_dim = config.rotary_dim;
        metadata.num_experts = config.num_experts;
        metadata.rope_base = config.rope_base;
        metadata.rms_norm_eps = config.rms_norm_eps;
        metadata.final_logit_softcapping = config.final_logit_softcapping;
        metadata.embedding_scale = embed_scale;
        metadata.tie_word_embeddings = config.tie_word_embeddings;
        metadata.gemma4 = true;
        metadata.decode_graph_double_warmup = true;
        metadata.model_type = config.model_type;
        metadata.layer_types = config.layer_types;
    }
    Tensor adapter_prepare_hidden(Tensor hidden, int64_t batch, int64_t tokens) const {
        return this->scale_hidden(std::move(hidden), this->embed_scale);
    }
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
};

} // namespace mfq::models::gemma4
