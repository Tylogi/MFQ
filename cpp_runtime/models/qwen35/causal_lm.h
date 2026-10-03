#pragma once
#include "config.h"
#include "models/common/weight_loading.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include <optional>
#include <utility>

namespace mfq::models::qwen35 {

struct LinearWeightNames {
    std::string qkv, qk, value, gate, alpha, beta;
};

template <class Ffn, class Loader>
Ffn load_ffn(Loader& ops, const Config& config, int layer, std::string_view tensor_root) {
    const std::string prefix =
        std::string(tensor_root) + ".block." + std::to_string(layer) + ".mlp.";
    if (!ops.has(prefix + "experts.gate_up.weight") &&
        !ops.has(prefix + "experts.gate.weight") &&
        !ops.has(prefix + "experts.up.weight") &&
        !ops.has(prefix + "experts.down.weight")) {
        return models::load_dense_ffn<Ffn>(ops, config, prefix);
    }
    if (config.num_experts <= 0 || config.num_experts_per_tok <= 0 ||
        config.moe_intermediate_size <= 0 || config.shared_expert_intermediate_size <= 0) {
        throw std::runtime_error("Qwen MoE config fields are missing");
    }

    auto ffn = models::load_moe_weights<Ffn>(ops, prefix, layer);
    ffn.moe_shared_gate = ops.router_parameter(prefix + "shared_expert.router.weight");
    ffn.moe_top_k = static_cast<int>(config.num_experts_per_tok);
    const auto routing = config.routing();
    ffn.moe_use_sqrt_softplus = routing.activation == mfq::models::RouterActivation::sqrt_softplus;
    ffn.moe_normalize = routing.normalize;
    ffn.moe_delayed_softmax = routing.delayed_softmax;
    ffn.moe_router_scale = routing.scale;
    const bool routed_gate_shapes =
        ffn.moe_split_gate_up
            ? ffn.moe_gate.n_experts == config.num_experts &&
                  ffn.moe_up.n_experts == config.num_experts &&
                  ffn.moe_gate.neuron_len == config.hidden_size &&
                  ffn.moe_up.neuron_len == config.hidden_size &&
                  ffn.moe_gate.out_per_expert == config.moe_intermediate_size &&
                  ffn.moe_up.out_per_expert == config.moe_intermediate_size
            : ffn.moe_gate_up.n_experts == config.num_experts &&
                  ffn.moe_gate_up.neuron_len == config.hidden_size &&
                  ffn.moe_gate_up.out_per_expert == 2 * config.moe_intermediate_size;
    if (!routed_gate_shapes || ffn.moe_down.n_experts != config.num_experts ||
        ffn.moe_down.neuron_len != config.moe_intermediate_size ||
        ffn.moe_down.out_per_expert != config.hidden_size || ops.shape(ffn.moe_router).size() != 2 ||
        ops.shape(ffn.moe_router)[0] != config.num_experts ||
        ops.shape(ffn.moe_router)[1] != config.hidden_size || ops.shape(ffn.moe_shared_gate).size() != 2 ||
        ops.shape(ffn.moe_shared_gate)[0] != 1 || ops.shape(ffn.moe_shared_gate)[1] != config.hidden_size) {
        throw std::runtime_error("Qwen MoE tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
    return ffn;
}


template <class Loader>
auto load_block(Loader& ops, const Config& config, int layer, const std::string& type,
                std::string_view tensor_root = "model") {
    const auto prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    if (Config::attention_kind(type) == Config::AttentionKind::full) {
        auto block = ops.full_block();
        models::load_full_attention(*block, ops, config, layer, prefix,
                                    config.legacy_tensor_layout.norm_weight_offset,
                                    config.attention_output_gate);
        block->ffn = load_ffn<typename Loader::Ffn>(ops, config, layer, tensor_root);
        return ops.finish(std::move(block));
    }
    auto block = ops.linear_block();
    block->qwen_config = config;
    block->tiled_v_heads = config.legacy_tensor_layout.qwen_gdn_gguf_layout;
    block->attn_norm = ops.dense(prefix + "attention.norm.weight");
    block->ffn_norm = ops.dense(prefix + "mlp.norm.weight");
    const auto linear = prefix + "linear_attention.";
    LinearWeightNames names{linear + "qkv.weight", linear + "qk.weight", linear + "value.weight",
                            linear + "gate.weight", linear + "alpha.weight", linear + "beta.weight"};
    ops.linear_projections(*block, names, ops.has(names.qk) && ops.has(names.value));
    block->conv_weight = ops.dense(linear + "conv.weight");
    if (ops.has(linear + "conv.bias")) block->conv_bias = ops.dense(linear + "conv.bias");
    block->dt_bias = ops.dense(linear + "dt_bias");
    auto a = ops.dense(linear + "a");
    block->a_log = config.legacy_tensor_layout.linear_attention_a_is_log ? a : ops.log_negative(a);
    block->linear_norm = ops.dense(linear + "norm.weight");
    ops.linear_output(*block, linear + "output.weight");
    block->ffn = load_ffn<typename Loader::Ffn>(ops, config, layer, tensor_root);
    return ops.finish(std::move(block));
}

template <class Tensor> struct LinearProjections {
    Tensor qkv, qk, value, output_gate, alpha, beta;
};

// Convolution/QK normalization can be one fused native operator. GDN updates
// its native recurrent state; the model owns its placement in the graph.
template <class Tensor, class Project, class Gates, class Convolve, class Recur, class Normalize,
          class Output>
Tensor linear_attention(Tensor normalized, Project project, Gates gates, Convolve convolve,
                        Recur recur, Normalize normalize, Output output) {
    auto projected = project(std::move(normalized));
    auto recurrence_gates = gates(projected);
    auto qkv = convolve(projected);
    auto hidden = recur(std::move(qkv), std::move(recurrence_gates));
    auto normed = normalize(std::move(hidden));
    return output(std::move(normed), std::move(projected.output_gate));
}

enum class PredictorNorm { embedding, hidden, output };

template <class Ops>
auto mtp_predictor(Ops &ops, typename Ops::Tensor hidden, typename Ops::Tensor ids,
                   typename Ops::Tensor positions) {
    auto &model = ops.model;
    const auto &config = model.config;
    require_model(
        ops.rank(hidden) == 3 && ops.rank(ids) == 2 && ops.size(hidden, 0) == ops.size(ids, 0) &&
            ops.size(hidden, 1) == ops.size(ids, 1) && ops.size(hidden, 2) == config.hidden_size &&
            ops.size(hidden, 1) > 0,
        "Qwen MTP inputs must be matching [B,T,H] hidden states and [B,T] next-token IDs");
    const auto batch = ops.size(hidden, 0), tokens = ops.size(hidden, 1);
    require_model(model.cache_pos + tokens <= config.max_position_embeddings,
                  "Qwen MTP history exceeds context capacity");
    auto embedded = ops.embed(std::move(ids), hidden);
    auto e = ops.normalize(std::move(embedded), PredictorNorm::embedding);
    auto h = ops.normalize(std::move(hidden), PredictorNorm::hidden);
    ops.trace("mtp.embedding_norm", e);
    ops.trace("mtp.hidden_norm", h);
    auto current = ops.fuse(std::move(e), std::move(h));
    ops.trace("mtp.fusion", current);
    const bool explicit_positions = ops.defined(positions);
    auto pos = ops.positions(std::move(positions), model.cache_pos, tokens);
    require_model((ops.rank(pos) == 1 && ops.elements(pos) == tokens) ||
                      (ops.rank(pos) == 2 && ops.size(pos, 1) == tokens &&
                       (ops.size(pos, 0) == batch ||
                        (ops.size(pos, 0) == 3 && !config.mrope_sections.empty()))),
                  "Qwen MTP positions must have shape [T], [B,T], or configured grid-MRoPE [3,T]");
    std::optional<typename Ops::Tensor> lengths, cache_positions;
    if (tokens == 1 && model.cache_pos > 0)
        lengths = ops.lengths(batch, model.cache_pos + 1, pos);
    if (explicit_positions)
        cache_positions = ops.cache_positions(pos, model.cache_pos, tokens);
    for (auto &block : model.blocks)
        current = ops.layer(block, std::move(current), pos, lengths, cache_positions);
    model.cache_pos += tokens;
    auto output = ops.normalize(std::move(current), PredictorNorm::output);
    ops.trace("mtp.output_norm", output);
    return output;
}

template <class Backend> struct CausalLm : models::CausalModelBase<Backend, CausalLm<Backend>> {
    using Tensor = typename Backend::Tensor;
    static bool accepts_backbone(std::string_view backbone) {
        return backbone == "qwen3_5" || backbone == "generic_qwen";
    }
    template <class Graph, class Source>
    void adapter_load_config(std::string_view payload, const Graph &graph, const Source &source) {
        auto &config = this->config;
        auto &metadata = this->metadata;

        config = Config::from_json(payload, graph);
        config.legacy_tensor_layout = source.legacy_tensor_compatibility().layout;
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
        metadata.norm_weight_offset = config.legacy_tensor_layout.norm_weight_offset;
        metadata.tie_word_embeddings = config.tie_word_embeddings;
        metadata.model_type = config.model_type;
        metadata.layer_types = config.layer_types;
        this->adapter_validate_components(graph);
    }
    void adapter_validate_positions(const Tensor &positions, int64_t batch, int64_t tokens,
                                    bool has_mrope) const {
        const auto rank = Backend::rank(positions);
        require_model(this->config.valid_positions(rank, rank > 1 ? Backend::size(positions, 0) : 0,
                                                   Backend::size(positions, -1), batch, tokens,
                                                   has_mrope),
                      "position_ids must have shape [tokens], [batch,tokens], or configured "
                      "grid-MRoPE [3,tokens]");
    }
    void adapter_set_max_position_embeddings(int64_t value) {
        this->config.max_position_embeddings = value;
    }
    bool adapter_supports_speculation() const noexcept { return true; }
};

} // namespace mfq::models::qwen35
