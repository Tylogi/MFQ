#pragma once

#include "causal_lm.h"
#include "grid_vision.h"
#include "mfq_cuda_ops.h"
#include "mfq/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::grid_vision_runtime {

using Tensor = mfq_tensor_backend::Tensor;

namespace detail {

inline Tensor required_vector(
        const mfq::ModelSource& model,
        GridVisionTensorRole role,
        int64_t width,
        size_t block = 0) {
    const auto name = grid_vision_canonical_name(role, block);
    auto value = load_dense_gpu(model, name);
    if (value.dim() != 1 || value.size(0) != width) {
        throw std::runtime_error("grid-ViT vector shape mismatch: " + name);
    }
    return value;
}

struct Affine {
    QuantLinear linear;
    Tensor bias;

    static Affine load(
            const mfq::ModelSource& model,
            GridVisionTensorRole weight,
            GridVisionTensorRole bias_role,
            size_t block = 0) {
        const auto weight_name = grid_vision_canonical_name(weight, block);
        const auto bias_name = grid_vision_canonical_name(bias_role, block);
        Affine result{load_quant_linear(model, weight_name), {}};
        if (has_tensor(model, bias_name)) {
            result.bias = load_dense_gpu(model, bias_name);
            if (result.bias.dim() != 1 ||
                    result.bias.size(0) != result.linear.out()) {
                throw std::runtime_error(
                    "grid-ViT affine bias shape mismatch: " + bias_name);
            }
        }
        return result;
    }

    Tensor operator()(Tensor input) const {
        auto output = linear.forward(std::move(input));
        return bias.defined()
            ? output + bias.to(output.scalar_type())
            : output;
    }
};

struct LayerNorm {
    Tensor weight;
    Tensor bias;
    double epsilon = 1e-6;

    static LayerNorm load(
            const mfq::ModelSource& model,
            GridVisionTensorRole weight,
            GridVisionTensorRole bias,
            int64_t width,
            double epsilon,
            size_t block = 0) {
        return {
            required_vector(model, weight, width, block),
            required_vector(model, bias, width, block),
            epsilon,
        };
    }

    Tensor operator()(const Tensor& input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error(
                "grid-ViT LayerNorm input width mismatch");
        }
        return mfq_tensor_backend::layer_norm(
            input, {weight.size(0)}, weight, bias, epsilon);
    }
};

inline Tensor apply_axial_rope(
        const Tensor& input,
        const std::vector<int32_t>& positions,
        double theta) {
    if (input.dim() != 3 || input.size(0) <= 0 ||
            input.size(2) <= 0 || input.size(2) % 4 != 0 ||
            positions.size() != static_cast<size_t>(input.size(0) * 2)) {
        throw std::runtime_error("grid-ViT axial RoPE geometry mismatch");
    }
    const int64_t tokens = input.size(0);
    const int64_t head_dim = input.size(2);
    const int64_t half = head_dim / 2;
    const int64_t frequencies = half / 2;
    std::vector<float> cosine(static_cast<size_t>(tokens * head_dim));
    std::vector<float> sine(cosine.size());
    for (int64_t token = 0; token < tokens; ++token) {
        for (int axis = 0; axis < 2; ++axis) {
            const float position = static_cast<float>(
                positions[static_cast<size_t>(token * 2 + axis)]);
            for (int64_t index = 0; index < frequencies; ++index) {
                const float inverse = std::pow(
                    static_cast<float>(theta),
                    -static_cast<float>(2 * index) / static_cast<float>(half));
                const float angle = position * inverse;
                const int64_t column = axis * frequencies + index;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto offset = static_cast<size_t>(
                        token * head_dim + repeat * half + column);
                    cosine[offset] = std::cos(angle);
                    sine[offset] = std::sin(angle);
                }
            }
        }
    }
    const auto cpu = mfq_tensor_backend::TensorOptions()
        .dtype(mfq_tensor_backend::kFloat32)
        .device(mfq_tensor_backend::kCPU);
    auto cos = mfq_tensor_backend::tensor(cosine, cpu)
        .reshape({tokens, 1, head_dim}).to(input.device());
    auto sin = mfq_tensor_backend::tensor(sine, cpu)
        .reshape({tokens, 1, head_dim}).to(input.device());
    auto source = input.to(mfq_tensor_backend::kFloat32);
    auto first = source.narrow(-1, 0, half);
    auto second = source.narrow(-1, half, half);
    auto rotated = mfq_tensor_backend::cat({-second, first}, -1);
    return (source * cos + rotated * sin).to(input.scalar_type()).contiguous();
}

inline Tensor segmented_attention(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const std::vector<int32_t>& lengths) {
    std::vector<Tensor> outputs;
    outputs.reserve(lengths.size());
    int64_t offset = 0;
    for (const int32_t length : lengths) {
        if (length <= 0 || offset + length > query.size(0)) {
            throw std::runtime_error(
                "grid-ViT attention segment is invalid");
        }
        const auto segment = [offset, length](const Tensor& value) {
            return value.narrow(0, offset, length)
                .transpose(0, 1).unsqueeze(0).contiguous();
        };
        auto attended = attention_cuda(
            segment(query).to(mfq_tensor_backend::kFloat16),
            segment(key).to(mfq_tensor_backend::kFloat16),
            segment(value).to(mfq_tensor_backend::kFloat16),
            1.0 / std::sqrt(static_cast<double>(query.size(2))), false);
        outputs.push_back(attended.index({0}).transpose(0, 1).contiguous());
        offset += length;
    }
    if (outputs.empty() || offset != query.size(0)) {
        throw std::runtime_error(
            "grid-ViT attention segments do not cover every patch");
    }
    return outputs.size() == 1 ? outputs.front() :
        mfq_tensor_backend::cat(outputs, 0);
}

struct VisionAttention {
    Affine qkv;
    Affine output;
    int64_t hidden = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    double theta = 10'000.0;

    Tensor operator()(const Tensor& input, const GridVisionLayout& layout) const {
        const int64_t tokens = input.size(0);
        auto projected = qkv(input).reshape({tokens, 3, heads, head_dim});
        auto query = projected.select(1, 0);
        auto key = projected.select(1, 1);
        auto value = projected.select(1, 2);
        query = apply_axial_rope(query, layout.positions, theta);
        key = apply_axial_rope(key, layout.positions, theta);
        auto attended = segmented_attention(
            query, key, value, layout.segment_lengths);
        return output(attended.reshape({tokens, hidden}));
    }
};

inline Tensor gelu_tanh(const Tensor& input) {
    constexpr double sqrt_two_over_pi = 0.7978845608028654;
    auto source = input.to(mfq_tensor_backend::kFloat32);
    auto output = 0.5 * source *
        (1.0 + mfq_tensor_backend::tanh(
            sqrt_two_over_pi *
            (source + 0.044715 * source * source * source)));
    return output.to(input.scalar_type());
}

struct VisionBlock {
    LayerNorm norm1;
    VisionAttention attention;
    LayerNorm norm2;
    Affine mlp_up;
    Affine mlp_down;

    Tensor operator()(const Tensor& input, const GridVisionLayout& layout) const {
        auto attention_output = attention(norm1(input), layout);
        auto hidden = input + attention_output.to(input.scalar_type());
        auto mlp_output = mlp_down(gelu_tanh(mlp_up(norm2(hidden))));
        return hidden + mlp_output.to(hidden.scalar_type());
    }
};

} // namespace detail

class CudaGridVisionEncoder {
public:
    static CudaGridVisionEncoder load(
            const mfq::ModelSource& model,
            const GridVisionConfig& config) {
        config.validate();
        const auto hidden = config.hidden_size;
        auto patch_weight = load_dense_gpu(
            model, grid_vision_canonical_name(
                GridVisionTensorRole::patch_weight));
        const std::vector<int64_t> expected_patch{
            hidden, config.in_channels, config.temporal_patch_size,
            config.patch_size, config.patch_size};
        if (patch_weight.sizes().vec() != expected_patch) {
            throw std::runtime_error("grid-ViT patch weight shape mismatch");
        }
        auto patch_bias = detail::required_vector(
            model, GridVisionTensorRole::patch_bias, hidden);
        auto position_weight = load_dense_gpu(
            model, grid_vision_canonical_name(
                GridVisionTensorRole::position_weight));
        if (position_weight.dim() != 2 ||
                position_weight.size(0) != config.num_position_embeddings ||
                position_weight.size(1) != hidden) {
            throw std::runtime_error(
                "grid-ViT learned position table shape mismatch");
        }
        std::vector<detail::VisionBlock> blocks;
        blocks.reserve(static_cast<size_t>(config.depth));
        for (size_t index = 0; index < static_cast<size_t>(config.depth); ++index) {
            detail::VisionBlock block{
                detail::LayerNorm::load(
                    model, GridVisionTensorRole::block_norm1_weight,
                    GridVisionTensorRole::block_norm1_bias, hidden,
                    config.layer_norm_eps, index),
                {
                    detail::Affine::load(
                        model, GridVisionTensorRole::block_attention_qkv_weight,
                        GridVisionTensorRole::block_attention_qkv_bias, index),
                    detail::Affine::load(
                        model, GridVisionTensorRole::block_attention_output_weight,
                        GridVisionTensorRole::block_attention_output_bias, index),
                    hidden, config.num_heads, config.head_dim(), config.rope_theta,
                },
                detail::LayerNorm::load(
                    model, GridVisionTensorRole::block_norm2_weight,
                    GridVisionTensorRole::block_norm2_bias, hidden,
                    config.layer_norm_eps, index),
                detail::Affine::load(
                    model, GridVisionTensorRole::block_mlp_up_weight,
                    GridVisionTensorRole::block_mlp_up_bias, index),
                detail::Affine::load(
                    model, GridVisionTensorRole::block_mlp_down_weight,
                    GridVisionTensorRole::block_mlp_down_bias, index),
            };
            if (block.attention.qkv.linear.neuron_len() != hidden ||
                    block.attention.qkv.linear.out() != 3 * hidden ||
                    block.attention.output.linear.neuron_len() != hidden ||
                    block.attention.output.linear.out() != hidden ||
                    block.mlp_up.linear.neuron_len() != hidden ||
                    block.mlp_up.linear.out() != config.intermediate_size ||
                    block.mlp_down.linear.neuron_len() != config.intermediate_size ||
                    block.mlp_down.linear.out() != hidden) {
                throw std::runtime_error(
                    "grid-ViT transformer block geometry mismatch");
            }
            blocks.push_back(std::move(block));
        }
        return CudaGridVisionEncoder(
            config, std::move(patch_weight), std::move(patch_bias),
            std::move(position_weight), std::move(blocks));
    }

    Tensor encode(Tensor pixels, const std::vector<GridShape>& grids) const {
        const auto layout = make_grid_vision_layout(
            grids, static_cast<int32_t>(config_.spatial_merge_size));
        if (pixels.dim() == 5) {
            pixels = pixels.reshape({pixels.size(0), -1});
        }
        if (pixels.dim() != 2 || pixels.size(0) != layout.patch_count ||
                pixels.size(1) != config_.patch_width()) {
            throw std::runtime_error(
                "grid-ViT pixel values must be flattened CxTxHxW patches");
        }
        auto flat_weight = patch_weight_.reshape(
            {config_.hidden_size, config_.patch_width()});
        auto hidden = mfq_tensor_backend::matmul(
            pixels.to(flat_weight.scalar_type()), flat_weight.transpose(0, 1));
        hidden = hidden + patch_bias_.to(hidden.scalar_type());

        const int side = static_cast<int>(std::sqrt(
            static_cast<double>(config_.num_position_embeddings)));
        const auto interpolation = make_learned_position_interpolation(
            grids, static_cast<int32_t>(config_.spatial_merge_size), side);
        auto index = mfq_tensor_backend::tensor(
            interpolation.indices,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt32)
                .device(mfq_tensor_backend::kCPU))
            .to(hidden.device()).to(mfq_tensor_backend::kInt64);
        auto weights = mfq_tensor_backend::tensor(
            interpolation.weights,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kFloat32)
                .device(mfq_tensor_backend::kCPU))
            .reshape({layout.patch_count, 4, 1}).to(hidden.device());
        auto learned = mfq_tensor_backend::sum(
            position_weight_.index_select(0, index.reshape({-1}))
                .reshape({layout.patch_count, 4, config_.hidden_size}) * weights,
            1);
        hidden = hidden + learned.to(hidden.scalar_type());
        for (const auto& block : blocks_) hidden = block(hidden, layout);
        return hidden;
    }

    const GridVisionConfig& config() const noexcept { return config_; }

private:
    CudaGridVisionEncoder(
            GridVisionConfig config,
            Tensor patch_weight,
            Tensor patch_bias,
            Tensor position_weight,
            std::vector<detail::VisionBlock> blocks)
        : config_(std::move(config)),
          patch_weight_(std::move(patch_weight)),
          patch_bias_(std::move(patch_bias)),
          position_weight_(std::move(position_weight)),
          blocks_(std::move(blocks)) {}

    GridVisionConfig config_;
    Tensor patch_weight_;
    Tensor patch_bias_;
    Tensor position_weight_;
    std::vector<detail::VisionBlock> blocks_;
};

class CudaGridVisionPromptComponent {
public:
    static CudaGridVisionPromptComponent load(
            const mfq::ModelSource& model,
            const GridVisionConfig& vision,
            int64_t image_token_id,
            int64_t video_token_id,
            std::string input_contract,
            std::string position_policy) {
        if (input_contract != kMfqGridVisionInputContract) {
            throw std::invalid_argument(
                "unsupported grid-Vision input contract: " + input_contract);
        }
        if (position_policy != kMfqGridMropePositionPolicy) {
            throw std::invalid_argument(
                "unsupported grid-Vision position policy: " + position_policy);
        }
        if (image_token_id < 0 || video_token_id < 0 ||
                image_token_id == video_token_id) {
            throw std::invalid_argument(
                "invalid grid-Vision placeholder token IDs");
        }
        vision.validate();
        const int64_t unit = vision.spatial_merge_size *
            vision.spatial_merge_size;
        auto merger_norm = detail::LayerNorm::load(
            model, GridVisionTensorRole::merger_norm_weight,
            GridVisionTensorRole::merger_norm_bias, vision.hidden_size,
            vision.layer_norm_eps);
        auto merger_up = detail::Affine::load(
            model, GridVisionTensorRole::merger_mlp_up_weight,
            GridVisionTensorRole::merger_mlp_up_bias);
        auto merger_down = detail::Affine::load(
            model, GridVisionTensorRole::merger_mlp_down_weight,
            GridVisionTensorRole::merger_mlp_down_bias);
        if (merger_up.linear.neuron_len() != unit * vision.hidden_size ||
                merger_down.linear.neuron_len() != merger_up.linear.out() ||
                merger_down.linear.out() != vision.out_hidden_size) {
            throw std::runtime_error("grid-Vision merger geometry mismatch");
        }
        return CudaGridVisionPromptComponent(
            CudaGridVisionEncoder::load(model, vision),
            std::move(merger_norm), std::move(merger_up),
            std::move(merger_down), image_token_id, video_token_id,
            std::move(input_contract), std::move(position_policy));
    }

    template <typename Model>
    CudaPreparedPrompt prepare(
            Model& language,
            const std::vector<int64_t>& token_ids,
            const MfqMultimodalInput& media) const {
        if (media.processor != MfqMultimodalProcessor::grid_vision ||
                media.processor_name != input_contract_ ||
                media.vision_grid_shape != std::vector<int64_t>{1, 3} ||
                media.vision_grid.size() != 3 ||
                media.vision_types != std::vector<int32_t>{1} ||
                !media.video_grid.empty()) {
            throw std::invalid_argument(
                "CUDA grid-Vision milestone accepts exactly one image and no video");
        }
        const GridShape grid{
            media.vision_grid[0], media.vision_grid[1], media.vision_grid[2]};
        const auto expected_pixel_shape = std::vector<int64_t>{
            static_cast<int64_t>(grid.temporal) * grid.height * grid.width,
            encoder_.config().patch_width()};
        if (grid.temporal != 1 ||
                media.image_grid != media.vision_grid ||
                media.image_grid_shape != std::vector<int64_t>{1, 3} ||
                media.pixel_shape != expected_pixel_shape ||
                media.pixel_values.size() != static_cast<size_t>(
                    expected_pixel_shape[0] * expected_pixel_shape[1])) {
            throw std::invalid_argument(
                "CUDA grid-Vision image geometry is invalid");
        }
        auto pixels = mfq_tensor_backend::from_blob(
            const_cast<float*>(media.pixel_values.data()), media.pixel_shape,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kFloat32)
                .device(mfq_tensor_backend::kCPU)).clone()
            .to(mfq_tensor_backend::kCUDA);
        auto patches = encoder_.encode(std::move(pixels), {grid});
        const int64_t unit = encoder_.config().spatial_merge_size *
            encoder_.config().spatial_merge_size;
        if (patches.size(0) % unit != 0) {
            throw std::runtime_error(
                "grid-Vision patch count is not divisible by merge unit");
        }
        auto merged = merger_down_(mfq_tensor_backend::gelu(
            merger_up_(merger_norm_(patches).reshape(
                {patches.size(0) / unit,
                 unit * encoder_.config().hidden_size})), "none"));

        auto ids = mfq_tensor_backend::tensor(
            token_ids,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, -1});
        auto embeddings = language.embed_forward(ids).contiguous();
        std::vector<int64_t> placeholders;
        for (size_t index = 0; index < token_ids.size(); ++index) {
            if (token_ids[index] == image_token_id_) {
                placeholders.push_back(static_cast<int64_t>(index));
            } else if (token_ids[index] == video_token_id_) {
                throw std::invalid_argument(
                    "CUDA grid-Vision does not support video placeholders");
            }
        }
        if (static_cast<int64_t>(placeholders.size()) != merged.size(0)) {
            throw std::invalid_argument(
                "image placeholders disagree with merged vision patches");
        }
        auto placeholder_index = mfq_tensor_backend::tensor(
            placeholders,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA));
        embeddings.index_copy_(
            1, placeholder_index,
            merged.to(embeddings.scalar_type()).unsqueeze(0));

        const auto positions = ::mfq::build_grid_mrope_positions(
            token_ids, image_token_id_, video_token_id_,
            static_cast<int32_t>(encoder_.config().spatial_merge_size),
            {grid}, {});
        auto position_tensor = mfq_tensor_backend::tensor(
            positions.values,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt32)
                .device(mfq_tensor_backend::kCPU))
            .reshape({3, positions.token_count})
            .to(mfq_tensor_backend::kCUDA)
            .to(mfq_tensor_backend::kInt64);
        return {
            token_ids, std::move(embeddings), std::move(position_tensor),
            positions.decode_delta};
    }

    const std::string& input_contract() const noexcept {
        return input_contract_;
    }
    const std::string& position_policy() const noexcept {
        return position_policy_;
    }

private:
    CudaGridVisionPromptComponent(
            CudaGridVisionEncoder encoder,
            detail::LayerNorm merger_norm,
            detail::Affine merger_up,
            detail::Affine merger_down,
            int64_t image_token_id,
            int64_t video_token_id,
            std::string input_contract,
            std::string position_policy)
        : encoder_(std::move(encoder)),
          merger_norm_(std::move(merger_norm)),
          merger_up_(std::move(merger_up)),
          merger_down_(std::move(merger_down)),
          image_token_id_(image_token_id),
          video_token_id_(video_token_id),
          input_contract_(std::move(input_contract)),
          position_policy_(std::move(position_policy)) {}

    CudaGridVisionEncoder encoder_;
    detail::LayerNorm merger_norm_;
    detail::Affine merger_up_;
    detail::Affine merger_down_;
    int64_t image_token_id_ = -1;
    int64_t video_token_id_ = -1;
    std::string input_contract_;
    std::string position_policy_;
};

} // namespace mfq::cuda::grid_vision_runtime
