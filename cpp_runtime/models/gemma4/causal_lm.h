#pragma once
#include "config.h"
#include "models/common/causal_model.h"
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

// Gemma combines a dense branch with routed experts before its final residual.
// Dense-post and merge may fuse the family's normalization arithmetic.
template <class Dense, class DensePost, class Experts, class Merge, class Finish>
auto feed_forward(bool moe, Dense dense, DensePost dense_post, Experts experts, Merge merge,
                  Finish finish_dense) {
    auto output = dense();
    if (!moe)
        return finish_dense(std::move(output));
    output = dense_post(std::move(output));
    auto routed = experts();
    return merge(std::move(output), std::move(routed));
}

template <class Router, class Select, class Scale, class Project, class Reduce>
auto experts(Router router, Select select, Scale scale, Project project, Reduce reduce) {
    auto selected = select(router());
    scale(selected);
    auto pairs = project(selected);
    return reduce(std::move(pairs), selected);
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
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
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
};

} // namespace mfq::models::gemma4
