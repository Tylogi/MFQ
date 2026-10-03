#pragma once
#include "config.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include <vector>

namespace mfq::models::deepseek_v4 {

template <class Tensor> struct Projections {
    Tensor rank, queries, values;
    std::vector<Tensor> compressor, indexer;
};

template <class Query, class Values, class Compressor, class Indexer, class Prefill, class Decode,
          class Concat>
auto attention(int64_t tokens, int64_t position, bool lengths, int64_t ratio, Query query,
               Values values, Compressor compressor, Indexer indexer, Prefill prefill,
               Decode decode, Concat concat) {
    auto q = query();
    using Tensor = decltype(values());
    Projections<Tensor> projected{q[0], q[1], values(), {}, {}};
    if (ratio > 0)
        projected.compressor = compressor();
    if (ratio == 4)
        projected.indexer = indexer();
    if (tokens > 1 && position == 0 && !lengths)
        return prefill(projected);
    std::vector<Tensor> output;
    output.reserve(tokens);
    for (int64_t token = 0; token < tokens; ++token)
        output.push_back(decode(projected, token, position + token + 1));
    return concat(output);
}

template <class Index, class All, class Empty>
auto select_compressed(int64_t ratio, int64_t visible, Index index, All all, Empty empty) {
    if (ratio == 4 && visible > 512)
        return index();
    if (visible > 0)
        return all();
    return empty();
}

template <class Write, class Compress, class Index, class Select, class Plan, class Cache,
          class Attend, class Inverse, class Output>
auto compressed_step(int64_t ratio, bool prefill, Write write, Compress compress, Index index,
                     Select select, Plan plan, Cache cache, Attend attend, Inverse inverse,
                     Output output) {
    write();
    const auto visible = ratio > 0 ? compress() : 0;
    if (ratio == 4) {
        const auto indexed = index();
        require_model(!prefill || indexed == visible,
                      "DeepSeek V4 compressor/indexer prefill length mismatch");
    }
    auto selected = select(visible);
    auto indices = plan(std::move(selected), visible);
    auto values = cache(visible);
    return output(inverse(attend(std::move(values), indices)));
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "deepseek_v4"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = Config::from_json(payload);
        config.layer_types.assign(static_cast<std::size_t>(config.num_hidden_layers),
                                  "deepseek_v4");
        metadata.vocab_size = config.vocab_size;
        metadata.hidden_size = config.hidden_size;
        metadata.num_hidden_layers = config.num_hidden_layers;
        metadata.num_attention_heads = config.num_attention_heads;
        metadata.num_key_value_heads = config.num_key_value_heads;
        metadata.head_dim = config.head_dim;
        metadata.max_position_embeddings = config.max_position_embeddings;
        metadata.rotary_dim = config.rotary_dim;
        metadata.num_experts = config.num_experts;
        metadata.hc_mult = config.hc_mult;
        metadata.rope_base = config.rope_base;
        metadata.rms_norm_eps = config.rms_norm_eps;
        metadata.hc_eps = config.hc_eps;
        metadata.tie_word_embeddings = config.tie_word_embeddings;
        metadata.model_type = config.model_type;
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
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
};

} // namespace mfq::models::deepseek_v4
