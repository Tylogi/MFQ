#pragma once
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace mfq::models {

inline void require_model(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Model, class Guard> void reset_model(Model &model, int64_t batch, Guard guard) {
    model.cache_pos = model.decode_position_delta = 0;
    model.adapter_reset(batch);
    model.speculative_start = -1;
    model.speculative_confirmed = 0;
    model.speculative_suffix_forward = false;
    for (auto &block : model.blocks) {
        auto scope = guard(block);
        block->reset(batch);
    }
}

template <class Model, class Guard>
void begin_speculative_suffix(Model &model, int64_t tokens, Guard guard) {
    require_model(model.supports_suffix_speculation() && model.speculative_start < 0 &&
                      tokens > 0 && model.cache_pos + tokens <= model.max_position_embeddings(),
                  "invalid speculative suffix");
    model.speculative_start = model.cache_pos;
    model.speculative_confirmed = 0;
    model.adapter_begin_speculative();
    size_t begun = 0;
    try {
        for (auto &block : model.blocks) {
            auto scope = guard(block);
            block->begin_speculative(tokens);
            ++begun;
        }
    } catch (...) {
        for (size_t i = 0; i < begun; ++i) {
            try {
                auto scope = guard(model.blocks[i]);
                model.blocks[i]->commit_speculative();
            } catch (...) {
            }
        }
        try {
            model.adapter_rollback_speculative(model.cache_pos);
        } catch (...) {
        }
        model.speculative_start = -1;
        model.speculative_confirmed = 0;
        throw;
    }
}

template <class Model, class Guard>
void finish_speculative(Model &model, bool commit, int64_t accepted, Guard guard) {
    require_model(model.speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    const auto keep = model.speculative_start + model.speculative_confirmed + accepted;
    for (auto &block : model.blocks) {
        auto scope = guard(block);
        if (commit)
            block->commit_speculative();
        else
            block->rollback_speculative(keep);
    }
    if (commit)
        model.adapter_commit_speculative();
    else {
        model.adapter_rollback_speculative(keep);
        model.cache_pos = keep;
    }
    model.speculative_start = -1;
    model.speculative_confirmed = 0;
}

// The model's logical cache and layer traversal are independent of storage.
// Ops places tensors on devices and supplies each family's native operators.
template <class Ops> auto causal_forward(Ops &ops) {
    auto &model = ops.model;
    ops.prepare_inputs();
    const auto B = ops.size(ops.ids, 0), T = ops.size(ops.ids, 1);
    const auto &embeddings = ops.input_embeddings;
    require_model(ops.rank(embeddings) == 3 && ops.size(embeddings, 0) == B &&
                      ops.size(embeddings, 1) == T &&
                      ops.size(embeddings, 2) == model.hidden_size(),
                  "inputs_embeds shape must match [batch,tokens,hidden_size]");
    if (model.adapter_requires_batch_reset(B))
        model.reset(B);
    model.adapter_validate_forward(B, T, model.cache_pos, ops.pos_override.has_value(),
                                   ops.cache_positions_override.has_value(),
                                   ops.attention_mask.has_value());
    require_model(model.speculative_start < 0 || model.speculative_suffix_forward,
                  "commit or roll back the pending speculative pass before forwarding");
    const auto confirmed = ops.confirmed_prefix;
    require_model(
        confirmed >= 0 &&
            (confirmed == 0 || (confirmed < T && B == 1 && model.cache_pos > 0 &&
                                (!ops.pos_override.has_value() ||
                                 model.adapter_allows_speculative_position_override()) &&
                                !ops.cache_positions_override.has_value() &&
                                !ops.attention_mask.has_value() && model.supports_speculation())),
        "unsupported speculative backbone geometry");
    if (confirmed > 0) {
        require_model(model.cache_pos + T <= model.max_position_embeddings(),
                      "speculative pass exceeds context capacity");
        model.speculative_start = model.cache_pos;
        model.speculative_confirmed = confirmed;
    }
    if (model.cache_pos == 0)
        model.reset(B);
    ops.cache_positions = ops.cache_positions_override
                              ? ops.device_ids(*ops.cache_positions_override)
                              : ops.position_range(model.cache_pos, T);
    ops.pos = ops.pos_override
                  ? ops.device_ids(*ops.pos_override)
                  : (model.decode_position_delta == 0
                         ? ops.cache_positions
                         : ops.offset_positions(ops.cache_positions, model.decode_position_delta));
    auto prepared = model.adapter_prepare_positions(std::move(ops.pos), B, T);
    ops.pos = std::move(prepared.positions);
    ops.full_positions = std::move(prepared.full_positions);
    const auto &positions = ops.cache_positions;
    require_model((ops.rank(positions) == 1 && ops.elements(positions) == T) ||
                      (ops.rank(positions) == 2 && ops.size(positions, 0) == B &&
                       ops.size(positions, 1) == T),
                  "cache_positions must have shape [tokens] or [batch,tokens]");
    model.adapter_validate_positions(ops.pos, B, T, ops.has_mrope());
    if (ops.attention_mask.has_value()) {
        const auto &mask = *ops.attention_mask;
        require_model(ops.rank(mask) == 2 && ops.size(mask, 0) == B &&
                          ops.size(mask, 1) >= model.cache_pos + T,
                      "attention_mask must cover [batch,cached_plus_current_tokens]");
    }
    ops.effective_attention_mask =
        model.adapter_attention_mask(ops.attention_mask, T, model.cache_pos);
    auto hidden = ops.prepare_hidden(B, T);
    model.adapter_begin_forward(ops.raw_hidden != nullptr);
    ops.trace(hidden);
    for (auto &block : model.blocks) {
        hidden = ops.layer(block, std::move(hidden));
        ops.trace(hidden);
    }
    model.adapter_finish_forward(ops.full_positions, B, T);
    if (model.adapter_force_cache_advance() || !ops.pos_override.has_value() ||
        ops.advance_cache_with_position_ids)
        model.cache_pos += T;
    return ops.finish(std::move(hidden), B, T);
}
} // namespace mfq::models
