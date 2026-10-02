#pragma once
#include "config.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include <utility>

namespace mfq::models::glm_dsa {

template <class Input, class QueryNorm, class Query, class Prepare, class Cache, class Index,
          class Absorb, class Attend, class Unembed, class Output>
auto attention(bool full_indexer, Input input, QueryNorm query_norm, Query query, Prepare prepare,
               Cache cache, Index index, Absorb absorb, Attend attend, Unembed unembed,
               Output output) {
    auto first = input();
    require_model(first.size() == (full_indexer ? 4u : 2u),
                  "GLM DSA input projection count mismatch");
    auto second = query(query_norm(first[0]));
    require_model(second.size() == (full_indexer ? 2u : 1u), "GLM DSA q projection count mismatch");
    auto qkv = prepare(first[1], second[0]);
    cache(qkv);
    if (full_indexer)
        index(first[2], first[3], second[1]);
    return output(unembed(attend(absorb(qkv))));
}

template <class Dense, class DensePrefix, class Sparse, class Concat>
auto indexed_attention(bool indexed, int64_t tokens, int64_t prefix, int64_t sparse_rows,
                       Dense dense, DensePrefix dense_prefix, Sparse sparse, Concat concat) {
    if (!indexed)
        return dense();
    require_model(prefix >= 0 && sparse_rows > 0 && prefix + sparse_rows == tokens,
                  "GLM DSA shared index state has the wrong row count");
    if (prefix == 0)
        return sparse();
    auto first = dense_prefix();
    auto rest = sparse();
    return concat(std::move(first), std::move(rest));
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "glm_dsa"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = Config::from_json(payload);
        config.layer_types.assign(static_cast<std::size_t>(config.num_hidden_layers), "glm_dsa");
        config.rotary_dim = config.qk_rope_head_dim;
        this->adapter_validate_model_geometry();
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
        metadata.norm_weight_offset = 1.0;
        metadata.tie_word_embeddings = config.tie_word_embeddings;
        metadata.decode_graph_double_warmup = true;
        metadata.model_type = config.model_type;
        metadata.layer_types = config.layer_types;
    }
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
};

} // namespace mfq::models::glm_dsa
