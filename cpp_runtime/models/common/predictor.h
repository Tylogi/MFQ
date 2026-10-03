#pragma once
#include "causal_forward.h"

namespace mfq::models {
struct PredictorCursor {
    int64_t layer, start;
};

template <class State, class Reset>
PredictorCursor begin_predictor(State &state, int64_t batch, int64_t tokens, int64_t depth,
                                int64_t capacity, bool cache, Reset reset) {
    const auto count = int64_t(state.lengths.size());
    require_model(count > 0 && batch > 0 && tokens > 0, "invalid predictor geometry");
    const auto layer = (depth % count + count) % count;
    if (cache && state.batch && state.batch != batch)
        reset(batch);
    const auto start = cache ? state.lengths[layer] : 0;
    require_model(start >= 0 && tokens <= capacity - start,
                  "predictor history exceeds context capacity");
    return {layer, start};
}

template <class State>
void commit_predictor(State &state, PredictorCursor cursor, int64_t batch, int64_t tokens) {
    state.batch = batch;
    state.lengths[cursor.layer] = cursor.start + tokens;
}
template <class State, class Reset, class Embed, class Positions, class Embedding, class Hidden,
          class Fuse, class Layer, class Output, class SavePositions>
auto projected_predictor(State &state, int64_t batch, int64_t tokens, int64_t depth, bool cache,
                         Reset reset, Embed embed, Positions positions, Embedding embedding,
                         Hidden hidden, Fuse fuse, Layer layer, Output output,
                         SavePositions save_positions) {
    const auto cursor =
        begin_predictor(state, batch, tokens, depth, state.config.maximum, cache, reset);
    auto embeds = embed();
    auto pos = positions(cursor);
    auto e = embedding(std::move(embeds), pos);
    auto h = hidden();
    auto multi = layer(fuse(std::move(e), std::move(h)), cursor.layer, pos);
    auto result = output(multi);
    if (cache) {
        save_positions(cursor.layer, pos);
        commit_predictor(state, cursor, batch, tokens);
    }
    return std::pair{std::move(result), std::move(multi)};
}

} // namespace mfq::models
