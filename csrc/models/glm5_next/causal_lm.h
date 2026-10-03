#pragma once
#include "config.h"
#include "models/common/causal_model.h"
#include "models/common/predictor.h"
#include "models/common/recurrent_attention.h"
#include "models/common/transformer_layer.h"

namespace mfq::models::glm5_next {

template <class Loader>
auto load_dense_ffn(Loader &ops, const std::string &p, double limit) {
    auto gate = ops.linear(p + ".gate.weight"), up = ops.linear(p + ".up.weight"),
         down = ops.linear(p + ".down.weight");
    return ops.gated_mlp(std::move(gate), std::move(up), std::move(down), limit);
}

template <class Loader>
auto load_ffn(Loader &ops, const Config &c, int layer, const std::string &p, bool predictor) {
    if (c.dense_layer(layer, predictor))
        return load_dense_ffn(ops, p, c.swiglu_limit);
    auto gate_up = ops.routed_gate_up(p, layer, c.experts, c.moe_intermediate, c.hidden);
    auto down = ops.routed(p + ".experts.down.weight", layer, c.experts, c.hidden, c.moe_intermediate);
    auto router = ops.linear(p + ".router.weight");
    auto shared = load_dense_ffn(ops, p + ".shared_expert", c.swiglu_limit);
    auto bias = ops.fp32(ops.dense(p + ".router.bias"));
    return ops.moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared),
                   std::move(bias), c);
}

template <class Loader>
auto load_mla(Loader &ops, const Config &c, int layer, const std::string &a) {
    typename Loader::MlaWeights w{
        ops.linear(a + ".query_a.weight"), ops.linear(a + ".key_value_a.weight"),
        ops.linear(a + ".query_b.weight"), ops.linear(a + ".output.weight"),
        ops.linear(a + ".indexer.query.weight"), ops.linear(a + ".indexer.key.weight"),
        ops.linear(a + ".indexer.score.weight"),
        ops.headwise(ops.routed(a + ".latent.query_embedding.weight", layer, c.heads, c.latent, c.nope),
                     c.heads, c.latent),
        ops.headwise(ops.routed(a + ".latent.output_unembedding.weight", layer, c.heads, c.value_width, c.latent),
                     c.heads, c.value_width),
        ops.dense(a + ".query_a_norm.weight"), ops.dense(a + ".key_value_a_norm.weight"),
        ops.dense(a + ".indexer.key_norm.weight"), ops.dense(a + ".indexer.key_norm.bias"),
        ops.dense(a + ".indexer.pool.gate"), ops.dense(a + ".indexer.pool.position")};
    return ops.mla(std::move(w), c);
}

template <class Loader>
auto load_mhc(Loader &ops, const std::string &p) {
    auto function = ops.dense(p + ".function"), base = ops.dense(p + ".base"),
         scale = ops.dense(p + ".scale");
    return ops.mhc(std::move(function), std::move(base), std::move(scale));
}

template <class Block, class Loader>
void load_block(Block &b, Loader &ops, const Config &c, int layer) {
    const auto p = "model.block." + std::to_string(layer);
    b.config = c;
    b.attention_hc = load_mhc(ops, p + ".attention.mhc.pre");
    b.ffn_hc = load_mhc(ops, p + ".mlp.mhc.pre");
    b.attention_norm = ops.dense(p + ".attention.norm.weight");
    b.ffn_norm = ops.dense(p + ".mlp.norm.weight");
    b.ffn = load_ffn(ops, c, layer, p + ".mlp", false);
    if (c.linear_layer(layer)) {
        const auto a = p + ".linear_attention";
        typename Loader::KdaWeights w{
            ops.linear(a + ".query.weight"), ops.linear(a + ".key.weight"),
            ops.linear(a + ".value.weight"), ops.linear(a + ".beta.weight"),
            ops.linear(a + ".gate_a.weight"), ops.linear(a + ".gate_b.weight"),
            ops.linear(a + ".output.weight"),
            ops.concat({ops.dense(a + ".query_conv.weight"), ops.dense(a + ".key_conv.weight"),
                        ops.dense(a + ".value_conv.weight")}),
            ops.dense(a + ".forget_a.weight"), ops.dense(a + ".forget_b.weight"),
            ops.dense(a + ".dt_bias"), ops.dense(a + ".a"), ops.dense(a + ".output_norm.weight")};
        b.kda = ops.kda(std::move(w), c);
    } else {
        b.mla = load_mla(ops, c, layer, p + ".attention");
    }
}

template <class Predictor, class Loader>
bool load_predictor(Predictor &result, Loader &ops, const Config &c) {
    const auto count = c.predictor_layers;
    if (!ops.has("predictor.embedding_norm.weight") || count <= 0) {
        require_model(!ops.has_prefix("predictor"),
                      "GLM5-Next model source contains an incomplete or undeclared MTP head");
        return false;
    }
    result.config = c;
    result.embedding_norm = ops.fp32(ops.dense("predictor.embedding_norm.weight"));
    result.hidden_norm = ops.fp32(ops.dense("predictor.hidden_norm.weight"));
    result.output_norm = ops.dense("predictor.output_norm.weight");
    result.fusion = ops.linear("predictor.fusion.weight");
    for (int64_t i = 0; i < count; ++i) {
        const auto p = "predictor.block." + std::to_string(i);
        typename Predictor::Layer layer;
        layer.attention_norm = ops.dense(p + ".attention.norm.weight");
        layer.ffn_norm = ops.dense(p + ".mlp.norm.weight");
        layer.ffn = load_ffn(ops, c, int(i), p + ".mlp", true);
        layer.attention = load_mla(ops, c, int(i), p + ".attention");
        result.layers.push_back(std::move(layer));
    }
    require_model(ops.shape(result.embedding_norm) == std::vector<int64_t>{c.hidden} &&
                      ops.shape(result.hidden_norm) == std::vector<int64_t>{c.hidden} &&
                      ops.elements(result.output_norm) == c.hidden,
                  "GLM5-Next MTP normalization width disagrees with backbone");
    result.lengths.resize(count, 0);
    return true;
}

template <class Project, class Cache, class Absorb, class Dense, class Pool, class IndexQuery,
          class Score, class Select, class Sparse, class Unembed, class Output, class Rollback>
auto sparse_mla(int64_t length, int64_t budget, bool cache, Project project, Cache append,
                Absorb absorb, Dense dense, Pool pool, IndexQuery index_query, Score score,
                Select select, Sparse sparse, Unembed unembed, Output output, Rollback rollback) {
    auto projected = project();
    try {
        if (cache)
            append(projected);
        auto query = absorb(projected);
        auto attended = [&] {
            if (length <= budget)
                return dense(query, projected);
            auto pooled = pool(projected);
            auto iq = index_query(projected);
            auto scores = score(std::move(iq), std::move(pooled));
            auto selected = select(std::move(scores));
            return sparse(query, projected, std::move(selected));
        }();
        return output(unembed(std::move(attended)));
    } catch (...) {
        if (cache)
            rollback();
        throw;
    }
}

template <class Tensor, class AttentionPre, class Normalize, class LinearAttention,
          class SparseAttention, class Post, class FfnPre, class Ffn>
auto decoder_layer(Tensor hidden, bool linear, AttentionPre attention_pre, Normalize normalize,
                   LinearAttention linear_attention, SparseAttention sparse_attention, Post post,
                   FfnPre ffn_pre, Ffn ffn) {
    return hyperconnection_layer(
        std::move(hidden), attention_pre,
        [&](const auto &mix) {
            auto branch = normalize(mix[2], 0);
            return linear ? linear_attention(std::move(branch))
                          : sparse_attention(std::move(branch));
        },
        post, [&](const auto &value, const auto &) { return ffn_pre(value); },
        [&](const auto &mix) { return ffn(normalize(mix[2], 1)); }, post, [](const auto &) {});
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "glm5_next"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = Config::from_json(payload);
        metadata.vocab_size = config.vocab;
        metadata.hidden_size = config.hidden;
        metadata.num_hidden_layers = config.layers;
        metadata.num_attention_heads = config.heads;
        metadata.num_key_value_heads = 1;
        metadata.head_dim = config.nope;
        metadata.max_position_embeddings = config.maximum;
        metadata.num_experts = config.experts;
        metadata.hc_mult = config.streams;
        metadata.rope_base = 1.0;
        metadata.rms_norm_eps = config.eps;
        metadata.hc_eps = config.hc_eps;
        metadata.tie_word_embeddings = config.tied_embeddings;
        metadata.flash_next = true;
        metadata.model_type = "glm5_next";
        metadata.layer_types = config.layer_types;
    }
    Tensor adapter_finalize_hidden(Tensor hidden, const Tensor &output_norm, int64_t batch,
                                   int64_t tokens) const {
        auto collapsed = this->collapse_hidden(std::move(hidden), batch, tokens);
        return this->normalize_hidden(std::move(collapsed), output_norm, batch, tokens);
    }
    Tensor adapter_prepare_hidden(Tensor hidden, int64_t batch, int64_t tokens) const {
        return this->expand_hidden(std::move(hidden), batch, tokens, this->metadata.hc_mult);
    }
    void adapter_set_max_position_embeddings(int64_t value) { this->config.maximum = value; }
    void adapter_validate_forward(int64_t, int64_t tokens, int64_t cache_position,
                                  bool has_position_override, bool has_cache_position_override,
                                  bool has_attention_mask) const {
        require_model(
            tokens > 0 && cache_position + tokens <= this->metadata.max_position_embeddings &&
                !has_position_override && !has_cache_position_override && !has_attention_mask,
            "GLM Flash-Next currently requires contiguous causal cache positions "
            "without an external mask");
    }
    bool adapter_supports_speculation() const noexcept { return true; }
};

} // namespace mfq::models::glm5_next
