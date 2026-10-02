#pragma once
#include <utility>

namespace mfq::models {

// Operators may fuse adjacent arithmetic, but the model owns their order and
// the residual branch. All values keep their backend's native tensor type.
template <class Tensor, class Norm, class Attention, class Tail>
auto attention_layer(Tensor hidden, Norm normalize, Attention attention, Tail tail) {
    auto normalized = normalize(hidden);
    auto projected = attention(std::move(normalized));
    return tail(std::move(hidden), std::move(projected));
}

template <class Tensor, class AddNorm, class FeedForward>
auto residual_ffn(Tensor residual, Tensor attention, AddNorm add_norm, FeedForward feed_forward) {
    auto prepared = add_norm(std::move(residual), std::move(attention));
    return feed_forward(std::move(prepared[0]), std::move(prepared[1]));
}

template <class Tensor, class Fused, class FeedForward, class Add>
auto feed_forward(Tensor residual, Tensor normalized, Fused fused, FeedForward project, Add add) {
    if (auto output = fused(normalized, residual))
        return std::move(*output);
    auto output = project(std::move(normalized));
    return add(std::move(residual), std::move(output));
}

// DeepSeek HC/MHC uses a separate residual stream and branch preparation for
// attention and FFN. MHC's next_pre travels from attention to the FFN branch.
template <class Tensor, class AttnPre, class Attention, class AttnPost, class FfnPre,
          class FeedForward, class FfnPost, class Commit>
auto hyperconnection_layer(Tensor hidden, AttnPre attention_pre, Attention attention,
                           AttnPost attention_post, FfnPre ffn_pre, FeedForward feed_forward,
                           FfnPost ffn_post, Commit commit) {
    auto attention_mix = attention_pre(hidden);
    auto attended = attention(attention_mix);
    hidden = attention_post(std::move(attended), hidden, attention_mix);
    auto ffn_mix = ffn_pre(hidden, attention_mix);
    auto output = feed_forward(ffn_mix);
    hidden = ffn_post(std::move(output), hidden, ffn_mix);
    commit(ffn_mix);
    return hidden;
}

template <class Tensor, class Layers, class Apply>
Tensor layer_stack(Tensor hidden, Layers &layers, Apply apply) {
    for (auto &layer : layers)
        hidden = apply(layer, std::move(hidden));
    return hidden;
}

template <class Tensor, class Norm, class Attention, class FeedForward, class Add>
Tensor pre_norm_layer(Tensor hidden, Norm norm, Attention attention, FeedForward ffn, Add add) {
    auto branch = attention(norm(hidden, 0));
    hidden = add(std::move(hidden), std::move(branch));
    branch = ffn(norm(hidden, 1));
    return add(std::move(hidden), std::move(branch));
}

} // namespace mfq::models
