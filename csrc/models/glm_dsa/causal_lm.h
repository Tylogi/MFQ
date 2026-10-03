#pragma once
#include "config.h"
#include "models/common/weight_loading.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include <utility>

namespace mfq::models::glm_dsa {

template <class Ffn, class Loader>
void load_ffn(Loader& ops, const Config& c, int i, Ffn& f) {
    const auto &config = c;
    const std::string p = "model.block." + std::to_string(i) + ".mlp.";
    if (config.mlp_layer_types.at(static_cast<size_t>(i)) == "sparse") {
        f = models::load_moe_weights<Ffn>(ops, p, i, true);
        f.moe_top_k = static_cast<int>(c.num_experts_per_tok);
        f.moe_use_sigmoid = true;
        f.moe_use_sqrt_softplus = false;
        f.moe_normalize = c.norm_topk_prob;
        f.moe_delayed_softmax = false;
        f.moe_shared_ungated = true;
        f.moe_router_scale = c.routed_scaling_factor;
        if (f.moe_gate_up.n_experts != c.num_experts || f.moe_down.n_experts != c.num_experts ||
            f.moe_gate_up.neuron_len != c.hidden_size ||
            f.moe_gate_up.out_per_expert != 2 * c.moe_intermediate_size ||
            f.moe_down.neuron_len != c.moe_intermediate_size ||
            f.moe_down.out_per_expert != c.hidden_size || ops.shape(f.moe_router).size() != 2 ||
            ops.shape(f.moe_router)[0] != c.num_experts || ops.shape(f.moe_router)[1] != c.hidden_size ||
            ops.elements(f.moe_router_bias) != c.num_experts) {
            throw std::runtime_error("GLM DSA MoE tensor shapes disagree with config at layer " +
                                     std::to_string(i));
        }
        return;
    }
    f = models::load_dense_ffn<Ffn>(ops, c, p);
}

template <class Block, class Loader>
void load_block(Block& b, Loader& ops, const Config& c, int i, const std::string& type) {
    const auto &config = c;
    if (type != "glm_dsa") {
        throw std::runtime_error("invalid GLM DSA block loader state");
    }
    const std::string lp = "model.block." + std::to_string(i) + ".";
    const std::string ap = lp + "attention.";
    b.config = c;
    b.layer = i;
    b.full_indexer = config.indexer_types.at(static_cast<size_t>(i)) == "full";
    b.attn_norm = ops.dense(ap + "norm.weight");
    b.ffn_norm = ops.dense(lp + "mlp.norm.weight");
    b.q_a_norm = ops.dense(ap + "query_a_norm.weight");
    b.kv_a_norm = ops.dense(ap + "key_value_a_norm.weight");
    std::vector<std::string> first_names = {
        ap + "query_a.weight",
        ap + "key_value_a.weight",
    };
    std::vector<std::string> second_names = {
        ap + "query_b.weight",
    };
    if (b.full_indexer) {
        first_names.push_back(ap + "indexer.key.weight");
        first_names.push_back(ap + "indexer.score.weight");
        second_names.push_back(ap + "indexer.query.weight");
        b.index_k_norm = ops.dense(ap + "indexer.key_norm.weight");
        b.index_k_bias = ops.dense(ap + "indexer.key_norm.bias");
    }
    b.input_proj = ops.projections(first_names);
    b.q_proj = ops.projections(second_names);
    b.embed_q = ops.headwise(ap + "latent.query_embedding.weight");
    b.unembed_out = ops.headwise(ap + "latent.output_unembedding.weight");
    b.o_proj = ops.linear(ap + "output.weight");
    load_ffn(ops, c, i, b.ffn);

    const bool input_shape_ok = b.input_proj.outs.size() == (b.full_indexer ? 4u : 2u) &&
                                b.input_proj.outs[0] == c.q_lora_rank &&
                                b.input_proj.outs[1] == c.kv_lora_rank + c.qk_rope_head_dim &&
                                (!b.full_indexer || (b.input_proj.outs[2] == c.index_head_dim &&
                                                      b.input_proj.outs[3] == c.index_n_heads));
    const bool q_shape_ok =
        b.q_proj.outs.size() == (b.full_indexer ? 2u : 1u) &&
        b.q_proj.outs[0] == c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) &&
        (!b.full_indexer || b.q_proj.outs[1] == c.index_n_heads * c.index_head_dim);
    const bool head_shape_ok = b.embed_q.n_experts == c.num_attention_heads &&
                               b.embed_q.neuron_len == c.qk_nope_head_dim &&
                               b.embed_q.out_per_expert == c.kv_lora_rank &&
                               b.unembed_out.n_experts == c.num_attention_heads &&
                               b.unembed_out.neuron_len == c.kv_lora_rank &&
                               b.unembed_out.out_per_expert == c.v_head_dim &&
                               b.o_proj.neuron_len() == c.num_attention_heads * c.v_head_dim &&
                               b.o_proj.out() == c.hidden_size;
    if (!input_shape_ok || !q_shape_ok || !head_shape_ok || ops.elements(b.attn_norm) != c.hidden_size ||
        ops.elements(b.ffn_norm) != c.hidden_size || ops.elements(b.q_a_norm) != c.q_lora_rank ||
        ops.elements(b.kv_a_norm) != c.kv_lora_rank ||
        (b.full_indexer && (ops.elements(b.index_k_norm) != c.index_head_dim ||
                             ops.elements(b.index_k_bias) != c.index_head_dim))) {
        throw std::runtime_error("GLM DSA tensor shapes disagree with config at layer " +
                                 std::to_string(i));
    }
}


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
