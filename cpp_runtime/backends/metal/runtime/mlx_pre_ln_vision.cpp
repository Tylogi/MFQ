#include "mlx_pre_ln_vision.h"

#include "mlx_legacy_tensor_compat.h"
#include "mlx_transformer.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

array dense(const MfqContainer& model, const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" && record.dtype != "F32") {
        throw std::runtime_error(
            "vision tensor must be dense BF16/F16/F32: " + name);
    }
    return mlx::core::contiguous(load_dense_array(record.dtype, model.read(name)));
}

array float32_contiguous(array value) {
    if (value.dtype() != mlx::core::float32) {
        value = mlx::core::astype(value, mlx::core::float32);
    }
    return mlx::core::contiguous(std::move(value));
}

} // namespace

array mlx_vision_gelu(const array& input, MlxVisionActivation activation) {
    if (activation == MlxVisionActivation::gelu_erf) {
        constexpr float kInvSqrtTwo = 0.7071067811865475f;
        return 0.5f * input *
            (1.0f + mlx::core::erf(input * kInvSqrtTwo));
    }
    constexpr float kScale = 0.7978845608028654f;
    return 0.5f * input *
        (1.0f + mlx::core::tanh(
                    kScale * (input + 0.044715f * input * input * input)));
}

array MlxBiasedLinear::operator()(const array& input) const {
    auto output = weight(input);
    if (bias) {
        output = output + mlx::core::astype(*bias, output.dtype());
    }
    return output;
}

MlxVisionLayerNorm::MlxVisionLayerNorm(array weight, array bias, float eps)
    : weight_(float32_contiguous(std::move(weight))),
      bias_(float32_contiguous(std::move(bias))),
      eps_(eps) {
    if (weight_.ndim() != 1 || weight_.shape() != bias_.shape() ||
        !std::isfinite(eps_) || eps_ <= 0.0f) {
        throw std::runtime_error("vision LayerNorm parameters are invalid");
    }
    width_ = weight_.shape(0);
}

array MlxVisionLayerNorm::operator()(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != width_) {
        throw std::runtime_error("vision LayerNorm input width mismatch");
    }
    const auto dtype = input.dtype();
    auto source = mlx::core::astype(input, mlx::core::float32);
    const auto mean = mlx::core::mean(source, -1, true);
    const auto centered = source - mean;
    const auto variance = mlx::core::mean(centered * centered, -1, true);
    const auto normalized =
        centered / mlx::core::sqrt(variance + array(eps_, variance.dtype()));
    return mlx::core::astype(
        normalized * weight_ + bias_, dtype);
}

MlxPreLnVisionBlock MlxPreLnVisionBlock::load(
    const MfqContainer& model,
    const std::string& prefix,
    int heads,
    float layer_norm_eps,
    MlxVisionActivation activation) {
    const auto attention = prefix + ".attention.";
    const auto load_qkv = [&]() -> MlxBiasedLinear {
        if (!model.contains(attention + "qkv.weight")) {
            auto weight = mlx::core::concatenate(
                {dense(model, attention + "query.weight"),
                 dense(model, attention + "key.weight"),
                 dense(model, attention + "value.weight")},
                0);
            std::optional<array> bias;
            if (model.contains(attention + "query.bias")) {
                bias = mlx::core::concatenate(
                    {dense(model, attention + "query.bias"),
                     dense(model, attention + "key.bias"),
                     dense(model, attention + "value.bias")},
                    0);
            }
            return MlxBiasedLinear{
                MlxLinear(std::move(weight)), std::move(bias)};
        }
        std::optional<array> bias;
        if (model.contains(attention + "qkv.bias")) {
            bias = dense(model, attention + "qkv.bias");
        }
        return MlxBiasedLinear{
            MlxLinear::load(model, attention + "qkv.weight"), std::move(bias)};
    };
    const auto load_biased = [&](const std::string& name) {
        std::optional<array> bias;
        if (model.contains(name + ".bias")) {
            bias = dense(model, name + ".bias");
        }
        return MlxBiasedLinear{
            MlxLinear::load(model, name + ".weight"), std::move(bias)};
    };
    auto qkv = load_qkv();
    auto output = load_biased(attention + "output");
    auto up = load_biased(prefix + ".mlp.up");
    auto down = load_biased(prefix + ".mlp.down");
    const auto hidden = qkv.weight.input_size();
    const auto qkv_width = qkv.weight.output_size();
    if (qkv_width != 3 * hidden) {
        throw std::runtime_error(
            "vision pre-LN block qkv width does not match hidden_size");
    }
    return MlxPreLnVisionBlock(
        MlxVisionLayerNorm(
            dense(model, prefix + ".norm1.weight"),
            dense(model, prefix + ".norm1.bias"),
            layer_norm_eps),
        std::move(qkv),
        std::move(output),
        MlxVisionLayerNorm(
            dense(model, prefix + ".norm2.weight"),
            dense(model, prefix + ".norm2.bias"),
            layer_norm_eps),
        std::move(up),
        std::move(down),
        heads,
        activation);
}

MlxPreLnVisionBlock::MlxPreLnVisionBlock(
    MlxVisionLayerNorm norm1,
    MlxBiasedLinear qkv,
    MlxBiasedLinear output,
    MlxVisionLayerNorm norm2,
    MlxBiasedLinear up,
    MlxBiasedLinear down,
    int heads,
    MlxVisionActivation activation)
    : norm1_(std::move(norm1)),
      qkv_(std::move(qkv)),
      output_(std::move(output)),
      norm2_(std::move(norm2)),
      up_(std::move(up)),
      down_(std::move(down)),
      heads_(heads),
      activation_(activation) {
    validate_components();
}

void MlxPreLnVisionBlock::validate_components() const {
    const int hidden = norm1_.width();
    const int intermediate = up_.weight.output_size();
    if (heads_ <= 0 || hidden % heads_ != 0 ||
        norm2_.width() != hidden ||
        qkv_.weight.input_size() != hidden ||
        qkv_.weight.output_size() != 3 * hidden ||
        output_.weight.input_size() != hidden ||
        output_.weight.output_size() != hidden ||
        up_.weight.input_size() != hidden ||
        up_.weight.output_size() != intermediate ||
        down_.weight.input_size() != intermediate ||
        down_.weight.output_size() != hidden) {
        throw std::runtime_error(
            "vision pre-LN block component dimensions are inconsistent");
    }
}

array MlxPreLnVisionBlock::forward(
    const array& input,
    const std::optional<array>& mask) const {
    if (input.ndim() != 3 || input.shape(-1) != norm1_.width()) {
        throw std::runtime_error(
            "vision pre-LN block input must be [batch,tokens,hidden]");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    const int hidden = norm1_.width();
    const int head_dim = hidden / heads_;
    auto qkv = qkv_(norm1_(input));
    qkv = mlx::core::reshape(
        qkv, Shape{batch, tokens, 3, heads_, head_dim});
    const auto parts = mlx::core::split(qkv, 3, 2);
    const int head_count = heads_;
    const auto project = [batch, tokens, head_count, head_dim](const array& part) {
        return mlx::core::transpose(
            mlx::core::reshape(
                part, Shape{batch, tokens, head_count, head_dim}),
            {0, 2, 1, 3});
    };
    const auto query = project(parts.at(0));
    const auto key = project(parts.at(1));
    const auto value = project(parts.at(2));
    auto attended = scaled_dot_product_attention(
        query, key, value, false, 0.0f, mask);
    attended = mlx::core::reshape(
        mlx::core::transpose(attended, {0, 2, 1, 3}),
        Shape{batch, tokens, hidden});
    auto hidden_state = input + output_(attended);
    auto branch = up_(norm2_(hidden_state));
    branch = mlx_vision_gelu(branch, activation_);
    return hidden_state + down_(branch);
}

} // namespace mfq::metal
