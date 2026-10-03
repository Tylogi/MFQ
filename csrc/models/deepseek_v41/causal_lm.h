#pragma once
#include "config.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include "models/common/transformer_layer.h"
#include <optional>
#include <vector>

namespace mfq::models::deepseek_v41 {

// Both the target and DSpark pass the new pre-weights from attention into
// the FFN collapse, and commit them only after the layer completes.
template <class Tensor, class Collapse, class Attention, class Expand, class Ffn>
auto mega_layer(Tensor hidden, Tensor &previous_pre, Collapse collapse, Attention attention,
                Expand expand, Ffn ffn) {
    return hyperconnection_layer(
        std::move(hidden), [&](auto value) { return collapse(std::move(value), previous_pre, 0); },
        [&](const auto &mix) { return attention(mix.branch); },
        [&](auto branch, auto residual, const auto &mix) {
            return expand(std::move(branch), std::move(residual), mix, 0);
        },
        [&](auto value, const auto &attention_mix) {
            return collapse(std::move(value), attention_mix.next_pre, 1);
        },
        [&](const auto &mix) { return ffn(mix.branch); },
        [&](auto branch, auto residual, const auto &mix) {
            return expand(std::move(branch), std::move(residual), mix, 1);
        },
        [&](auto &mix) { previous_pre = std::move(mix.next_pre); });
}

template <class Tensor, class Engram, class Capture, class Collapse, class Attention, class Expand,
          class Ffn>
auto decoder_layer(Tensor hidden, bool has_engram, Tensor &previous_pre, Engram engram,
                   Capture capture, Collapse collapse, Attention attention, Expand expand,
                   Ffn ffn) {
    if (has_engram)
        hidden = engram(std::move(hidden));
    capture(hidden);
    return mega_layer(std::move(hidden), previous_pre, collapse, attention, expand, ffn);
}

template <class State, class Direct, class Project, class Pool, class Save, class Write,
          class PartialPool, class Concat>
auto compress(State &state, int64_t tokens, int64_t position, int64_t ratio, Direct direct,
              Project project, Pool pool, Save save, Write write, PartialPool partial_pool,
              Concat concat) {
    using Tensor = decltype(direct());
    require_model(ratio > 0, "invalid compression ratio");
    if (ratio == 1)
        return std::optional<Tensor>(direct());
    project();
    std::vector<Tensor> emitted;
    if (state.partial_length == 0 && position % ratio == 0) {
        const auto complete = tokens / ratio, cutoff = complete * ratio;
        if (complete > 0)
            emitted.push_back(pool(complete, cutoff));
        const auto remainder = tokens - cutoff;
        if (remainder > 0)
            save(cutoff, remainder);
        state.partial_length = remainder;
    } else {
        for (int64_t token = 0; token < tokens; ++token) {
            const auto slot = (position + token) % ratio;
            write(token, slot);
            if (slot + 1 == ratio)
                emitted.push_back(partial_pool());
        }
        state.partial_length = (position + tokens) % ratio;
    }
    if (emitted.empty())
        return std::optional<Tensor>{};
    return std::optional<Tensor>(emitted.size() == 1 ? std::move(emitted.front())
                                                     : concat(emitted));
}

template <class Empty, class Reuse, class Score, class Candidates, class SelectCandidates,
          class Select, class Publish>
auto hierarchical_index(int64_t ratio, bool source, int64_t layer, int64_t candidate_source,
                        int64_t pool_length, Empty empty, Reuse reuse, Score score,
                        Candidates candidates, SelectCandidates select_candidates, Select select,
                        Publish publish) {
    if (ratio == 0)
        return empty();
    if (!source)
        return reuse();
    auto selected = empty();
    if (pool_length > 0) {
        auto scores = score();
        if (layer == candidate_source)
            candidates(scores);
        selected = layer > candidate_source ? select_candidates(scores) : select(scores);
    }
    publish(selected);
    return selected;
}

template <class Rank, class Query, class Local, class Publish, class Validate, class Index,
          class Cache, class Plan, class Attend, class Inverse, class Commit, class Output>
auto attention(bool source, int64_t ratio, Rank rank, Query query, Local local, Publish publish,
               Validate validate, Index index, Cache cache, Plan plan, Attend attend,
               Inverse inverse, Commit commit, Output output) {
    auto q_rank = rank();
    auto q = query(q_rank);
    auto kv = local();
    if (source)
        publish();
    else if (ratio > 0)
        validate();
    auto topk = index(q_rank);
    auto values = cache(std::move(kv));
    auto indices = plan(std::move(topk));
    auto hidden = inverse(attend(std::move(q), std::move(values), indices));
    commit();
    return output(std::move(hidden));
}

template <class Gather, class Project, class Gate, class Add>
auto engram(Gather gather, Project project, Gate gate, Add add) {
    auto key_value = project(gather());
    auto gated = gate(key_value);
    return add(std::move(gated), key_value);
}

template <class Embed, class Expand, class InitialPre, class Layer, class Collapse, class Logits,
          class Slice>
auto dspark_draft(int64_t width, int64_t requested, size_t stages, Embed embed, Expand expand,
                  InitialPre initial_pre, Layer layer, Collapse collapse, Logits logits,
                  Slice slice) {
    require_model(requested > 0 && requested <= width && stages > 0, "invalid DSpark draft width");
    auto embedded = embed(width);
    auto hidden = expand(embedded);
    auto previous_pre = initial_pre(embedded);
    for (size_t index = 0; index < stages; ++index)
        hidden = layer(index, std::move(hidden), previous_pre);
    return slice(logits(collapse(std::move(hidden), previous_pre)), requested);
}

template <class Project, class Keys, class Store>
void dspark_context(int64_t &position, int64_t tokens, size_t stages, Project project, Keys keys,
                    Store store) {
    auto main = project();
    for (size_t index = 0; index < stages; ++index)
        store(index, keys(index, main));
    position += tokens;
}

template <class Query, class Key, class Cache, class Plan, class Attend, class Inverse,
          class Output>
auto dspark_attention(int64_t context, Query query, Key key, Cache cache, Plan plan, Attend attend,
                      Inverse inverse, Output output) {
    require_model(context > 0, "DeepSeek-V4.1 DSpark draft requires committed context");
    auto q = query();
    auto keys = cache(key());
    auto indices = plan(keys);
    return output(inverse(attend(std::move(q), std::move(keys), indices)));
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "deepseek_v41"; }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = Config::from_json(payload);
        metadata.vocab_size = config.vocab;
        metadata.hidden_size = config.hidden;
        metadata.num_hidden_layers = config.n_layers;
        metadata.num_attention_heads = config.n_heads;
        metadata.num_key_value_heads = config.n_kv_heads;
        metadata.head_dim = config.head_dim;
        metadata.max_position_embeddings = config.max_position_embeddings;
        metadata.rotary_dim = config.rope_head_dim;
        metadata.num_experts = config.n_experts;
        metadata.hc_mult = config.hc_mult;
        metadata.rope_base = config.rope_theta;
        metadata.rms_norm_eps = config.rms_eps;
        metadata.hc_eps = config.hc_eps;
        metadata.model_type = config.text_model_type;
        metadata.layer_types.assign(static_cast<std::size_t>(config.n_layers), "deepseek_v41");
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
    void adapter_validate_forward(int64_t, int64_t tokens, int64_t cache_position,
                                  bool has_position_override, bool has_cache_position_override,
                                  bool has_attention_mask) const {
        require_model(
            tokens > 0 && cache_position + tokens <= this->metadata.max_position_embeddings &&
                !has_position_override && !has_cache_position_override && !has_attention_mask,
            "DeepSeek-V4.1 currently requires contiguous causal cache positions "
            "without an external mask");
    }
    bool adapter_supports_suffix_speculation() const noexcept { return true; }
};

} // namespace mfq::models::deepseek_v41
