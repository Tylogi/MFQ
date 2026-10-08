#pragma once
#include "config.h"
#include "models/common/causal_model.h"
#include "models/common/predictor.h"
#include "models/common/recurrent_attention.h"
#include "models/common/transformer_layer.h"

namespace mfq::models::qwen4_exp {

template <class Loader>
auto load_residual(Loader &ops, const Config &c, const std::string &p, bool combine) {
    auto norm = ops.fp32(ops.dense(p + ".norm.weight"));
    auto down = ops.residual_linear(p + ".down.weight"), up = ops.residual_linear(p + ".up.weight");
    decltype(down) injection{};
    if (combine)
        injection = ops.residual_linear(p.substr(0, p.size() - 4) + ".post.inject.weight");
    return ops.residual(std::move(norm), std::move(down), std::move(up), std::move(injection), c);
}

template <class Loader>
auto load_ffn(Loader &ops, const Config &c, int layer, const std::string &p) {
    auto gate_up = ops.routed_gate_up(p, layer, c.experts, c.moe_width, c.hidden);
    auto down = ops.routed(p + ".experts.down.weight", layer, c.experts, c.hidden, c.moe_width);
    auto router = ops.linear(p + ".router.weight");
    auto shared_gate = ops.linear(p + ".shared_expert.router.weight");
    auto sg = ops.linear(p + ".shared_expert.gate.weight"),
         su = ops.linear(p + ".shared_expert.up.weight"),
         sd = ops.linear(p + ".shared_expert.down.weight");
    return ops.moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared_gate),
                   std::move(sg), std::move(su), std::move(sd), c);
}

template <class Loader>
auto load_ple(Loader &ops, const Config &c, const std::string &p) {
    std::vector<typename Loader::Embedding> shards;
    int64_t rows = 0, width = c.hidden / ((c.ngram - 1) * c.ngram_heads);
    for (int64_t i = 0; i < c.shards; ++i) {
        auto [embedding, shape] = ops.embedding(p + ".ngram.shard." + std::to_string(i) + ".weight");
        require_model(shape[1] == width && shape[0] > 0 && (!rows || shape[0] == rows),
                      "Qwen4 PLE embedding shard dimensions disagree");
        rows = shape[0];
        shards.push_back(std::move(embedding));
    }
    auto multipliers = ops.integers(p + ".ngram.layer_multipliers"),
         offsets = ops.integers(p + ".ngram.head_offsets"),
         sizes = ops.integers(p + ".ngram.head_vocab_sizes");
    typename Loader::PleWeights w{
        ops.linear(p + ".key.weight"), ops.linear(p + ".value.weight"),
        ops.dense(p + ".key_norm.weight"), ops.dense(p + ".query_norm.weight"),
        ops.dense(p + ".conv_norm.weight"), ops.dense(p + ".conv.weight")};
    return ops.ple(std::move(shards), rows, width, c, std::move(multipliers), std::move(offsets),
                   std::move(sizes), std::move(w));
}

// The same layer definition loads the backbone and its QSA-only predictor.
template <class Block, class Loader>
void load_block(Block &b, Loader &ops, const Config &c, int layer, bool predictor = false) {
    const auto p = std::string(predictor ? "predictor" : "model") + ".block." + std::to_string(layer);
    b.config = c;
    b.attention_gr = load_residual(ops, c, p + ".attention.mhc.pre", true);
    b.ffn_gr = load_residual(ops, c, p + ".mlp.mhc.pre", true);
    b.ffn = load_ffn(ops, c, layer, p + ".mlp");
    if (c.linear_layer(layer, predictor)) {
        const auto a = p + ".linear_attention";
        typename Loader::GdnWeights w{
            ops.linear(a + ".qkv.weight"), ops.linear(a + ".gate.weight"),
            ops.linear(a + ".alpha.weight"), ops.linear(a + ".beta.weight"),
            ops.linear(a + ".output.weight"), ops.dense(a + ".conv.weight"),
            ops.dense(a + ".dt_bias"), ops.dense(a + ".a"), ops.dense(a + ".norm.weight")};
        b.gdn = ops.gdn(std::move(w), c);
    } else {
        const auto a = p + ".attention";
        typename Loader::QsaWeights w{
            ops.linear(a + ".query.weight"), ops.linear(a + ".key.weight"),
            ops.linear(a + ".value.weight"), ops.linear(a + ".output.weight"),
            ops.linear(a + ".indexer.query_key.weight"), ops.dense(a + ".query_norm.weight"),
            ops.dense(a + ".key_norm.weight"), ops.dense(a + ".indexer.query_norm.weight"),
            ops.dense(a + ".indexer.key_norm.weight")};
        b.qsa = ops.qsa(std::move(w), c);
    }
    if (c.position_embedding_layer(layer, predictor))
        b.ple = load_ple(ops, c, p + ".position_embedding");
}

template <class Predictor, class Loader>
bool load_predictor(Predictor &result, Loader &ops, const Config &c) {
    const auto count = c.predictor_layers;
    if (!ops.has("predictor.embedding_norm.weight") || count <= 0) {
        require_model(!ops.has_prefix("predictor"),
                      "Qwen4-Exp model source contains an incomplete or undeclared MTP head");
        return false;
    }
    result.config = c;
    result.embedding_norm = ops.fp32(ops.dense("predictor.embedding_norm.weight"));
    result.hidden_norm = ops.fp32(ops.dense("predictor.hidden_norm.weight"));
    result.embedding_fusion = ops.linear("predictor.fusion.embedding.weight");
    result.hidden_fusion = ops.linear("predictor.fusion.hidden.weight");
    result.final_mixer = ops.final_mixer(load_residual(ops, c, "predictor.mhc.pre", false));
    for (int64_t i = 0; i < count; ++i)
        result.layers.push_back(ops.block(c, int(i), true));
    require_model(ops.shape(result.embedding_norm) == std::vector<int64_t>{c.hidden} &&
                      ops.shape(result.hidden_norm) == std::vector<int64_t>{c.hidden * c.streams},
                  "Qwen4-Exp MTP normalization width disagrees with backbone");
    result.positions.resize(count);
    result.lengths.resize(count, 0);
    return true;
}

// The sparse indexer consumes pooled keys only after the dense budget is exceeded.
template <class Project, class Cache, class Dense, class Pool, class Score, class Select,
          class Sparse, class Output, class Rollback>
auto sparse_attention(int64_t length, int64_t budget, bool cache, Project project, Cache append,
                      Dense dense, Pool pool, Score score, Select select, Sparse sparse,
                      Output output, Rollback rollback) {
    auto qkv = project();
    try {
        if (cache)
            append(qkv);
        auto attended = [&] {
            if (length <= budget)
                return dense(qkv);
            auto pooled = pool(qkv);
            auto scores = score(qkv, pooled);
            auto selected = select(scores, qkv, pooled);
            return sparse(qkv, std::move(selected));
        }();
        return output(std::move(attended), qkv);
    } catch (...) {
        if (cache)
            rollback();
        throw;
    }
}

template <class Embed, class Key, class Query, class Gate, class Normalize, class Convolve,
          class Add, class Commit, class Rollback>
auto position_embedding(bool cache, Embed embed, Key key, Query query, Gate gate,
                        Normalize normalize, Convolve convolve, Add add, Commit commit,
                        Rollback rollback) {
    try {
        auto embeddings = embed();
        auto keys = key(embeddings);
        auto queries = query();
        auto gated = gate(keys, queries, embeddings);
        auto convolution = convolve(normalize(gated));
        auto output = add(std::move(gated), convolution[0]);
        if (cache)
            commit(convolution[1]);
        return output;
    } catch (...) {
        rollback();
        throw;
    }
}

template <class Tensor, class PositionEmbedding, class Add, class AttentionPre,
          class LinearAttention, class SparseAttention, class AttentionPost, class FfnPre,
          class Ffn, class FfnPost>
auto decoder_layer(Tensor hidden, bool has_ple, bool linear, PositionEmbedding position_embedding,
                   Add add, AttentionPre attention_pre, LinearAttention linear_attention,
                   SparseAttention sparse_attention, AttentionPost attention_post, FfnPre ffn_pre,
                   Ffn ffn, FfnPost ffn_post) {
    if (has_ple) {
        auto positional = position_embedding(hidden);
        hidden = add(std::move(hidden), std::move(positional));
    }
    return hyperconnection_layer(
        std::move(hidden), attention_pre,
        [&](const auto &mix) {
            return linear ? linear_attention(mix[0]) : sparse_attention(mix[0]);
        },
        [&](auto branch, const auto &, const auto &mix) {
            return attention_post(std::move(branch), mix);
        },
        [&](const auto &value, const auto &) { return ffn_pre(value); },
        [&](const auto &mix) { return ffn(mix[0]); },
        [&](auto branch, const auto &, const auto &mix) {
            return ffn_post(std::move(branch), mix);
        },
        [](const auto &) {});
}

template <class Tensor, class PositionEmbedding, class Add, class AttentionPre,
          class LinearAttention, class SparseAttention, class FfnPreAfter, class Ffn, class FfnPost>
auto decoder_layer_chained(Tensor hidden, bool has_ple, bool linear,
    PositionEmbedding position_embedding, Add add, AttentionPre attention_pre,
    LinearAttention linear_attention, SparseAttention sparse_attention,
    FfnPreAfter ffn_pre_after, Ffn ffn, FfnPost ffn_post) {
    if (has_ple) {
        auto positional = position_embedding(hidden);
        hidden = add(std::move(hidden), std::move(positional));
    }
    auto attention_mix = attention_pre(hidden);
    auto attention_branch = linear ? linear_attention(attention_mix[0]) : sparse_attention(attention_mix[0]);
    auto ffn_mix = ffn_pre_after(std::move(attention_branch), attention_mix);
    auto ffn_branch = ffn(ffn_mix[0]);
    return ffn_post(std::move(ffn_branch), ffn_mix);
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "qwen4_exp"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = Config::from_json(payload);
        metadata.vocab_size = config.vocab;
        metadata.hidden_size = config.hidden;
        metadata.num_hidden_layers = config.layers;
        metadata.num_attention_heads = config.heads;
        metadata.num_key_value_heads = config.kv_heads;
        metadata.head_dim = config.width;
        metadata.max_position_embeddings = config.maximum;
        metadata.rotary_dim = config.rotary;
        metadata.num_experts = config.experts;
        metadata.hc_mult = config.streams;
        metadata.rope_base = config.rope_base;
        metadata.rms_norm_eps = config.eps;
        metadata.tie_word_embeddings = config.tied_embeddings;
        metadata.multi_axis_positions = true;
        metadata.flash_next = true;
        metadata.model_type = "qwen4_exp";
        metadata.layer_types = config.layer_types;
    }
    Tensor adapter_prepare_hidden(Tensor hidden, int64_t batch, int64_t tokens) const {
        return this->repeat_hidden(std::move(hidden), this->metadata.hc_mult);
    }
    CausalPositions<Tensor> adapter_prepare_positions(Tensor current, int64_t, int64_t) {
        const auto rank = Backend::rank(current);
        if ((rank == 2 || rank == 3) && Backend::size(current, 0) == 4)
            current = this->position_axes(std::move(current), 1, 3);
        auto full = this->defined(this->positions)
                        ? this->concat_positions(this->positions, current)
                        : current;
        return {std::move(current), std::move(full)};
    }
    void adapter_validate_positions(const Tensor &positions, int64_t batch, int64_t tokens,
                                    bool has_mrope) const {
        const auto rank = Backend::rank(positions);
        require_model((rank == 1 || (rank == 2 && Backend::size(positions, 0) == 3) ||
                       (rank == 3 && Backend::size(positions, 0) == 3 &&
                        Backend::size(positions, 1) == batch)) &&
                          Backend::size(positions, -1) == tokens,
                      "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    }
    void adapter_set_max_position_embeddings(int64_t value) { this->config.maximum = value; }
    void adapter_validate_forward(int64_t new_batch, int64_t tokens, int64_t cache_position, bool,
                                  bool has_cache_position_override, bool has_attention_mask) const {
        require_model(tokens > 0 &&
                          cache_position + tokens <= this->metadata.max_position_embeddings &&
                          !has_cache_position_override && !has_attention_mask,
                      "Qwen4 requires unpadded causal cache positions");
        (void)new_batch;
    }
    bool adapter_requires_batch_reset(int64_t new_batch) const noexcept {
        return this->batch != 0 && this->batch != new_batch;
    }
    bool adapter_allows_speculative_position_override() const noexcept { return true; }
    bool adapter_force_cache_advance() const noexcept { return true; }
    bool adapter_supports_speculation() const noexcept { return true; }
};

} // namespace mfq::models::qwen4_exp
