#pragma once
#include "config.h"
#include "models/common/causal_model.h"
#include "models/common/predictor.h"
#include "models/common/recurrent_attention.h"
#include "models/common/transformer_layer.h"

namespace mfq::models::glm5_next {

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
