#pragma once
#include "models/common/causal_forward.h"
#include <optional>
#include <utility>

namespace mfq::models::qwen35 {

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

} // namespace mfq::models::qwen35
