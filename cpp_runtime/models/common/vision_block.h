#pragma once
#include <utility>

#include "models/common/gated_mlp.h"
#include "models/common/transformer_layer.h"

namespace mfq::models {

// Non-causal self-attention of an image encoder: project q/k/v, attend over the
// whole patch sequence, merge the heads and project out.  Unlike the decoder's
// attention there is no causal mask and no positional rotation inside the
// block, so the family only supplies the projections and the attention kernel.
template <class Tensor, class Query, class Key, class Value, class Attend, class Output>
Tensor vision_attention(Tensor hidden, Query query, Key key, Value value, Attend attend,
                        Output output) {
    auto q = query(hidden);
    auto k = key(hidden);
    auto v = value(hidden);
    return output(attend(std::move(q), std::move(k), std::move(v)));
}

// Up projection -> activation -> down projection, the MLP half of every vision
// transformer block (the activation variant belongs to the family).
template <class Tensor, class Up, class Activate, class Down>
Tensor vision_mlp(Tensor hidden, Up up, Activate activate, Down down) {
    return mlp(std::move(hidden), std::move(up), std::move(activate), std::move(down));
}

// One pre-LayerNorm vision transformer block:
//   h = h + attention(norm1(h));  h = h + mlp(norm2(h))
template <class Tensor, class Norm, class Attention, class FeedForward, class Add>
Tensor vision_block(Tensor hidden, Norm norm, Attention attention, FeedForward feed_forward,
                    Add add) {
    return pre_norm_layer(std::move(hidden), std::move(norm), std::move(attention),
                          std::move(feed_forward), std::move(add));
}

} // namespace mfq::models
