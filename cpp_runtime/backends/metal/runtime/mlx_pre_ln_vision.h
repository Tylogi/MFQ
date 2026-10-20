#pragma once

#include "mfq_container.h"
#include "mlx_tensor.h"

#include <optional>
#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

// Activation used by a pre-LayerNorm vision MLP. RADIO's "gelu" is the exact
// erf form; SigLIP-style towers use the tanh approximation.
enum class MlxVisionActivation { gelu_erf, gelu_tanh };

mlx::core::array mlx_vision_gelu(
    const mlx::core::array& input,
    MlxVisionActivation activation);

struct MlxBiasedLinear {
    MlxLinear weight;
    std::optional<mlx::core::array> bias;

    mlx::core::array operator()(const mlx::core::array& input) const;
};

class MlxVisionLayerNorm {
public:
    MlxVisionLayerNorm(
        mlx::core::array weight,
        mlx::core::array bias,
        float eps);

    mlx::core::array operator()(const mlx::core::array& input) const;

    int width() const noexcept {
        return width_;
    }

private:
    mlx::core::array weight_;
    mlx::core::array bias_;
    float eps_;
    int width_ = 0;
};

// LayerNorm(x + Attention(LN(x))) followed by LayerNorm(x + MLP(LN(x))).
// The attention runs over the full token axis (no causal mask) and uses a
// biased, optionally fused q/k/v projection.
class MlxPreLnVisionBlock {
public:
    // Loads canonical `norm1`/`norm2`, `attention.{qkv|query/key/value}` and
    // `attention.output`, and `mlp.up`/`mlp.down` tensors under `prefix`.
    static MlxPreLnVisionBlock load(
        const MfqContainer& model,
        const std::string& prefix,
        int heads,
        float layer_norm_eps,
        MlxVisionActivation activation);

    MlxPreLnVisionBlock(
        MlxVisionLayerNorm norm1,
        MlxBiasedLinear qkv,
        MlxBiasedLinear output,
        MlxVisionLayerNorm norm2,
        MlxBiasedLinear up,
        MlxBiasedLinear down,
        int heads,
        MlxVisionActivation activation);

    mlx::core::array forward(
        const mlx::core::array& input,
        const std::optional<mlx::core::array>& mask = std::nullopt) const;

    int heads() const noexcept {
        return heads_;
    }

private:
    void validate_components() const;

    MlxVisionLayerNorm norm1_;
    MlxBiasedLinear qkv_;
    MlxBiasedLinear output_;
    MlxVisionLayerNorm norm2_;
    MlxBiasedLinear up_;
    MlxBiasedLinear down_;
    int heads_ = 0;
    MlxVisionActivation activation_;
};

} // namespace mfq::metal
