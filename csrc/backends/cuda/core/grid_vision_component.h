#pragma once

#include "storage/weight_loader.h"
#include "models/common/grid_vision_model.h"
#include <array>

#include "core/causal_model.h"
#include "grid_vision.h"
#include "mfq/runtime.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_paged_prefix_cache.h"

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

inline Tensor required_vector(CudaExecutionContext &execution, const mfq::ModelSource &model,
                              GridVisionTensorRole role, int64_t width, size_t block = 0) {
    const auto name = grid_vision_canonical_name(role, block);
    auto value = load_dense_gpu(execution, model, name);
    if (value.dim() != 1 || value.size(0) != width) {
        throw std::runtime_error("grid-ViT vector shape mismatch: " + name);
    }
    return value;
}

struct Affine {
    QuantLinear linear;
    Tensor bias;

    static Affine load(CudaExecutionContext &execution, const mfq::ModelSource &model,
                       GridVisionTensorRole weight, GridVisionTensorRole bias_role,
                       size_t block = 0) {
        const auto weight_name = grid_vision_canonical_name(weight, block);
        const auto bias_name = grid_vision_canonical_name(bias_role, block);
        Affine result{load_quant_linear(execution, model, weight_name), {}};
        if (has_tensor(model, bias_name)) {
            result.bias = load_dense_gpu(execution, model, bias_name);
            if (result.bias.dim() != 1 || result.bias.size(0) != result.linear.out()) {
                throw std::runtime_error("grid-ViT affine bias shape mismatch: " + bias_name);
            }
        }
        return result;
    }

    Tensor operator()(CudaExecutionContext &execution, Tensor input) const {
        auto output = linear.forward(execution, std::move(input));
        return bias.defined() ? output + bias.to(output.scalar_type()) : output;
    }
};

struct LayerNorm {
    Tensor weight;
    Tensor bias;
    double epsilon = 1e-6;

    static LayerNorm load(CudaExecutionContext &execution, const mfq::ModelSource &model,
                          GridVisionTensorRole weight, GridVisionTensorRole bias, int64_t width,
                          double epsilon, size_t block = 0) {
        return {
            required_vector(execution, model, weight, width, block),
            required_vector(execution, model, bias, width, block),
            epsilon,
        };
    }

    Tensor operator()(const Tensor &input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error("grid-ViT LayerNorm input width mismatch");
        }
        return mfq_tensor_backend::layer_norm(input, {weight.size(0)}, weight, bias, epsilon);
    }
};

inline Tensor apply_axial_rope(const Tensor &input, const std::vector<int32_t> &positions,
                               double theta) {
    if (input.dim() != 3 || input.size(0) <= 0 || input.size(2) <= 0 || input.size(2) % 4 != 0 ||
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
            const float position =
                static_cast<float>(positions[static_cast<size_t>(token * 2 + axis)]);
            for (int64_t index = 0; index < frequencies; ++index) {
                const float inverse =
                    std::pow(static_cast<float>(theta),
                             -static_cast<float>(2 * index) / static_cast<float>(half));
                const float angle = position * inverse;
                const int64_t column = axis * frequencies + index;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto offset =
                        static_cast<size_t>(token * head_dim + repeat * half + column);
                    cosine[offset] = std::cos(angle);
                    sine[offset] = std::sin(angle);
                }
            }
        }
    }
    const auto cpu = mfq_tensor_backend::TensorOptions()
                         .dtype(mfq_tensor_backend::kFloat32)
                         .device(mfq_tensor_backend::kCPU);
    auto cos =
        mfq_tensor_backend::tensor(cosine, cpu).reshape({tokens, 1, head_dim}).to(input.device());
    auto sin =
        mfq_tensor_backend::tensor(sine, cpu).reshape({tokens, 1, head_dim}).to(input.device());
    auto source = input.to(mfq_tensor_backend::kFloat32);
    auto first = source.narrow(-1, 0, half);
    auto second = source.narrow(-1, half, half);
    auto rotated = mfq_tensor_backend::cat({-second, first}, -1);
    return (source * cos + rotated * sin).to(input.scalar_type()).contiguous();
}

inline Tensor segmented_attention(const Tensor &query, const Tensor &key, const Tensor &value,
                                  const std::vector<int32_t> &lengths) {
    std::vector<Tensor> outputs;
    outputs.reserve(lengths.size());
    int64_t offset = 0;
    for (const int32_t length : lengths) {
        if (length <= 0 || offset + length > query.size(0)) {
            throw std::runtime_error("grid-ViT attention segment is invalid");
        }
        const auto segment = [offset, length](const Tensor &value) {
            return value.narrow(0, offset, length).transpose(0, 1).unsqueeze(0).contiguous();
        };
        auto attended = attention_cuda(segment(query).to(mfq_tensor_backend::kFloat16),
                                       segment(key).to(mfq_tensor_backend::kFloat16),
                                       segment(value).to(mfq_tensor_backend::kFloat16),
                                       1.0 / std::sqrt(static_cast<double>(query.size(2))), false);
        outputs.push_back(attended.index({0}).transpose(0, 1).contiguous());
        offset += length;
    }
    if (outputs.empty() || offset != query.size(0)) {
        throw std::runtime_error("grid-ViT attention segments do not cover every patch");
    }
    return outputs.size() == 1 ? outputs.front() : mfq_tensor_backend::cat(outputs, 0);
}

struct VisionAttention {
    Affine qkv;
    Affine output;
    int64_t hidden = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    double theta = 10'000.0;

    Tensor operator()(CudaExecutionContext &execution, const Tensor &input,
                      const GridVisionLayout &layout) const {
        const int64_t tokens = input.size(0);
        return mfq::models::grid_vision::attention(
            input, [&](Tensor value) { return qkv(execution, std::move(value)); },
            [&](Tensor value) {
                auto projected = value.reshape({tokens, 3, heads, head_dim});
                return std::array<Tensor, 3>{projected.select(1, 0), projected.select(1, 1),
                                             projected.select(1, 2)};
            },
            [&](Tensor value) { return apply_axial_rope(value, layout.positions, theta); },
            [&](Tensor q, Tensor k, Tensor v) {
                return segmented_attention(q, k, v, layout.segment_lengths);
            },
            [&](Tensor value) { return output(execution, value.reshape({tokens, hidden})); });
    }
};

inline Tensor gelu_tanh(const Tensor &input) {
    constexpr double sqrt_two_over_pi = 0.7978845608028654;
    auto source = input.to(mfq_tensor_backend::kFloat32);
    auto output = 0.5 * source *
                  (1.0 + mfq_tensor_backend::tanh(sqrt_two_over_pi *
                                                  (source + 0.044715 * source * source * source)));
    return output.to(input.scalar_type());
}

struct VisionBlock {
    LayerNorm norm1;
    VisionAttention attention;
    LayerNorm norm2;
    Affine mlp_up;
    Affine mlp_down;

    Tensor operator()(CudaExecutionContext &execution, const Tensor &input,
                      const GridVisionLayout &layout) const {
        return mfq::models::pre_norm_layer(
            input, [&](const Tensor &x, int stage) { return stage == 0 ? norm1(x) : norm2(x); },
            [&](Tensor x) { return attention(execution, x, layout); },
            [&](Tensor x) {
                return mfq::models::mlp(
                    std::move(x), [&](Tensor value) { return mlp_up(execution, std::move(value)); },
                    gelu_tanh, [&](Tensor value) { return mlp_down(execution, std::move(value)); });
            },
            [](Tensor residual, Tensor branch) {
                return residual + branch.to(residual.scalar_type());
            });
    }
};

} // namespace detail

class CudaGridVisionEncoder {
  public:
    static CudaGridVisionEncoder load(CudaExecutionContext &execution,
                                      const mfq::ModelSource &model,
                                      const GridVisionConfig &config) {
        config.validate();
        const auto hidden = config.hidden_size;
        auto patch_weight = load_dense_gpu(
            execution, model, grid_vision_canonical_name(GridVisionTensorRole::patch_weight));
        const std::vector<int64_t> expected_patch{hidden, config.in_channels,
                                                  config.temporal_patch_size, config.patch_size,
                                                  config.patch_size};
        if (patch_weight.sizes().vec() != expected_patch) {
            throw std::runtime_error("grid-ViT patch weight shape mismatch");
        }
        auto patch_bias =
            detail::required_vector(execution, model, GridVisionTensorRole::patch_bias, hidden);
        auto position_weight = load_dense_gpu(
            execution, model, grid_vision_canonical_name(GridVisionTensorRole::position_weight));
        if (position_weight.dim() != 2 ||
            position_weight.size(0) != config.num_position_embeddings ||
            position_weight.size(1) != hidden) {
            throw std::runtime_error("grid-ViT learned position table shape mismatch");
        }
        std::vector<detail::VisionBlock> blocks;
        blocks.reserve(static_cast<size_t>(config.depth));
        for (size_t index = 0; index < static_cast<size_t>(config.depth); ++index) {
            detail::VisionBlock block{
                detail::LayerNorm::load(execution, model, GridVisionTensorRole::block_norm1_weight,
                                        GridVisionTensorRole::block_norm1_bias, hidden,
                                        config.layer_norm_eps, index),
                {
                    detail::Affine::load(execution, model,
                                         GridVisionTensorRole::block_attention_qkv_weight,
                                         GridVisionTensorRole::block_attention_qkv_bias, index),
                    detail::Affine::load(execution, model,
                                         GridVisionTensorRole::block_attention_output_weight,
                                         GridVisionTensorRole::block_attention_output_bias, index),
                    hidden,
                    config.num_heads,
                    config.head_dim(),
                    config.rope_theta,
                },
                detail::LayerNorm::load(execution, model, GridVisionTensorRole::block_norm2_weight,
                                        GridVisionTensorRole::block_norm2_bias, hidden,
                                        config.layer_norm_eps, index),
                detail::Affine::load(execution, model, GridVisionTensorRole::block_mlp_up_weight,
                                     GridVisionTensorRole::block_mlp_up_bias, index),
                detail::Affine::load(execution, model, GridVisionTensorRole::block_mlp_down_weight,
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
                throw std::runtime_error("grid-ViT transformer block geometry mismatch");
            }
            blocks.push_back(std::move(block));
        }
        return CudaGridVisionEncoder(config, std::move(patch_weight), std::move(patch_bias),
                                     std::move(position_weight), std::move(blocks));
    }

    mfq::StepSequence<Tensor> encode(CudaExecutionContext &execution, Tensor pixels,
                  const std::vector<GridShape> &grids) const {
        return mfq::models::grid_vision::encode(
            std::move(pixels), config_, grids, blocks_,
            [&](Tensor input, int64_t count) {
                if (input.dim() == 5)
                    input = input.reshape({input.size(0), -1});
                if (input.dim() != 2 || input.size(0) != count ||
                    input.size(1) != config_.patch_width())
                    throw std::runtime_error(
                        "grid-ViT pixel values must be flattened CxTxHxW patches");
                auto weight = patch_weight_.reshape({config_.hidden_size, config_.patch_width()});
                auto output = mfq_tensor_backend::matmul(input.to(weight.scalar_type()),
                                                         weight.transpose(0, 1));
                return output + patch_bias_.to(output.scalar_type());
            },
            [&](const LearnedPositionInterpolation &interpolation) {
                auto index = mfq_tensor_backend::tensor(interpolation.indices,
                                                        mfq_tensor_backend::TensorOptions()
                                                            .dtype(mfq_tensor_backend::kInt32)
                                                            .device(mfq_tensor_backend::kCPU))
                                 .to(position_weight_.device())
                                 .to(mfq_tensor_backend::kInt64);
                auto weights = mfq_tensor_backend::tensor(interpolation.weights,
                                                          mfq_tensor_backend::TensorOptions()
                                                              .dtype(mfq_tensor_backend::kFloat32)
                                                              .device(mfq_tensor_backend::kCPU))
                                   .reshape({interpolation.patch_count, 4, 1})
                                   .to(position_weight_.device());
                return mfq_tensor_backend::sum(
                    position_weight_.index_select(0, index.reshape({-1}))
                            .reshape({interpolation.patch_count, 4, config_.hidden_size}) *
                        weights,
                    1);
            },
            [](Tensor hidden, Tensor learned) { return hidden + learned.to(hidden.scalar_type()); },
            [&](const auto &block, Tensor value, const GridVisionLayout &layout) {
                return block(execution, std::move(value), layout);
            });
    }

    const GridVisionConfig &config() const noexcept { return config_; }

  private:
    CudaGridVisionEncoder(GridVisionConfig config, Tensor patch_weight, Tensor patch_bias,
                          Tensor position_weight, std::vector<detail::VisionBlock> blocks)
        : config_(std::move(config)), patch_weight_(std::move(patch_weight)),
          patch_bias_(std::move(patch_bias)), position_weight_(std::move(position_weight)),
          blocks_(std::move(blocks)) {}

    GridVisionConfig config_;
    Tensor patch_weight_;
    Tensor patch_bias_;
    Tensor position_weight_;
    std::vector<detail::VisionBlock> blocks_;
};

class CudaGridVisionPromptComponent {
  public:
    static CudaGridVisionPromptComponent
    load(CudaExecutionContext &execution, const mfq::ModelSource &model,
         const GridVisionConfig &vision, int64_t image_token_id, int64_t video_token_id,
         std::string input_contract, std::string position_policy) {
        if (input_contract != kMfqGridVisionInputContract) {
            throw std::invalid_argument("unsupported grid-Vision input contract: " +
                                        input_contract);
        }
        if (position_policy != kMfqGridMropePositionPolicy) {
            throw std::invalid_argument("unsupported grid-Vision position policy: " +
                                        position_policy);
        }
        if (image_token_id < 0 || video_token_id < 0 || image_token_id == video_token_id) {
            throw std::invalid_argument("invalid grid-Vision placeholder token IDs");
        }
        vision.validate();
        const int64_t unit = vision.spatial_merge_size * vision.spatial_merge_size;
        auto merger_norm = detail::LayerNorm::load(
            execution, model, GridVisionTensorRole::merger_norm_weight,
            GridVisionTensorRole::merger_norm_bias, vision.hidden_size, vision.layer_norm_eps);
        auto merger_up =
            detail::Affine::load(execution, model, GridVisionTensorRole::merger_mlp_up_weight,
                                 GridVisionTensorRole::merger_mlp_up_bias);
        auto merger_down =
            detail::Affine::load(execution, model, GridVisionTensorRole::merger_mlp_down_weight,
                                 GridVisionTensorRole::merger_mlp_down_bias);
        if (merger_up.linear.neuron_len() != unit * vision.hidden_size ||
            merger_down.linear.neuron_len() != merger_up.linear.out() ||
            merger_down.linear.out() != vision.out_hidden_size) {
            throw std::runtime_error("grid-Vision merger geometry mismatch");
        }
        return CudaGridVisionPromptComponent(CudaGridVisionEncoder::load(execution, model, vision),
                                             std::move(merger_norm), std::move(merger_up),
                                             std::move(merger_down), image_token_id, video_token_id,
                                             std::move(input_contract), std::move(position_policy));
    }

    template <typename Model>
    mfq::StepSequence<CudaPreparedPrompt> prepare(Model &language, const std::vector<int64_t> &token_ids,
                               const MfqMultimodalInput &media) const {
        if (media.processor != MfqMultimodalProcessor::grid_vision ||
            media.processor_name != input_contract_ ||
            media.vision_grid_shape != std::vector<int64_t>{1, 3} ||
            media.vision_grid.size() != 3 || media.vision_types != std::vector<int32_t>{1} ||
            !media.video_grid.empty()) {
            throw std::invalid_argument(
                "CUDA grid-Vision milestone accepts exactly one image and no video");
        }
        const GridShape grid{media.vision_grid[0], media.vision_grid[1], media.vision_grid[2]};
        const auto expected_pixel_shape =
            std::vector<int64_t>{static_cast<int64_t>(grid.temporal) * grid.height * grid.width,
                                 encoder_.config().patch_width()};
        if (grid.temporal != 1 || media.image_grid != media.vision_grid ||
            media.image_grid_shape != std::vector<int64_t>{1, 3} ||
            media.pixel_shape != expected_pixel_shape ||
            media.pixel_values.size() !=
                static_cast<size_t>(expected_pixel_shape[0] * expected_pixel_shape[1])) {
            throw std::invalid_argument("CUDA grid-Vision image geometry is invalid");
        }
        std::string cache_key =
            input_contract_ + '|' + position_policy_ + '|' +
            mfq::cache::block_hash_hex(mfq::cache::sha256(
                media.pixel_values.data(), media.pixel_values.size() * sizeof(float)));
        for (const auto value : media.pixel_shape) {
            cache_key.push_back('|');
            cache_key += std::to_string(value);
        }
        for (const auto value : media.vision_grid) {
            cache_key.push_back('|');
            cache_key += std::to_string(value);
        }

        Tensor merged;
        if (cached_vision_key_ == cache_key && cached_vision_.defined()) {
            merged = cached_vision_;
        } else {
            auto pixels = mfq_tensor_backend::from_blob(
                              const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                              mfq_tensor_backend::TensorOptions()
                                  .dtype(mfq_tensor_backend::kFloat32)
                                  .device(mfq_tensor_backend::kCPU))
                              .clone()
                              .to(mfq_tensor_backend::kCUDA);
            co_yield mfq::StepState::advanced;
            auto encoding = encoder_.encode(*language.execution, std::move(pixels), {grid});
            Tensor patches;
            while (auto step = encoding.next()) {
                if (step.value) patches = std::move(*step.value);
                else co_yield step.state;
            }
            merged = mfq::models::grid_vision::merge(
                patches, patches.size(0), encoder_.config(),
                [&](Tensor value) { return merger_norm_(value); },
                [](Tensor value, int64_t rows, int64_t width) {
                    return value.reshape({rows, width});
                },
                [&](Tensor value) { return merger_up_(*language.execution, std::move(value)); },
                [](Tensor value) { return mfq_tensor_backend::gelu(value, "none"); },
                [&](Tensor value) { return merger_down_(*language.execution, std::move(value)); });
            // ponytail: one image is the current native contract; use an LRU
            // only when alternating concurrent images becomes measurable.
            cached_vision_key_ = cache_key;
            cached_vision_ = merged;
            co_yield mfq::StepState::advanced;
        }

        auto prepared = mfq::models::grid_vision::prepare(
            token_ids, merged, merged.size(0), image_token_id_, video_token_id_, encoder_.config(),
            {grid},
            [&](const auto &tokens) {
                auto ids =
                    mfq_tensor_backend::tensor(tokens, mfq_tensor_backend::TensorOptions()
                                                           .dtype(mfq_tensor_backend::kInt64)
                                                           .device(mfq_tensor_backend::kCUDA))
                        .reshape({1, -1});
                return language.embed_forward(ids).contiguous();
            },
            [](Tensor &embeddings, const Tensor &vision, const std::vector<int64_t> &indices) {
                auto index =
                    mfq_tensor_backend::tensor(indices, mfq_tensor_backend::TensorOptions()
                                                            .dtype(mfq_tensor_backend::kInt64)
                                                            .device(mfq_tensor_backend::kCUDA));
                embeddings.index_copy_(1, index, vision.to(embeddings.scalar_type()).unsqueeze(0));
            },
            [](const GridMropePositions &positions) {
                return mfq_tensor_backend::tensor(positions.values,
                                                  mfq_tensor_backend::TensorOptions()
                                                      .dtype(mfq_tensor_backend::kInt32)
                                                      .device(mfq_tensor_backend::kCPU))
                    .reshape({3, positions.token_count})
                    .to(mfq_tensor_backend::kCUDA)
                    .to(mfq_tensor_backend::kInt64);
            });
        co_yield CudaPreparedPrompt{token_ids, std::move(prepared.embeddings), std::move(prepared.positions),
                prepared.decode_delta, std::move(cache_key)};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const std::string &position_policy() const noexcept { return position_policy_; }

  private:
    CudaGridVisionPromptComponent(CudaGridVisionEncoder encoder, detail::LayerNorm merger_norm,
                                  detail::Affine merger_up, detail::Affine merger_down,
                                  int64_t image_token_id, int64_t video_token_id,
                                  std::string input_contract, std::string position_policy)
        : encoder_(std::move(encoder)), merger_norm_(std::move(merger_norm)),
          merger_up_(std::move(merger_up)), merger_down_(std::move(merger_down)),
          image_token_id_(image_token_id), video_token_id_(video_token_id),
          input_contract_(std::move(input_contract)), position_policy_(std::move(position_policy)) {
    }

    CudaGridVisionEncoder encoder_;
    detail::LayerNorm merger_norm_;
    detail::Affine merger_up_;
    detail::Affine merger_down_;
    int64_t image_token_id_ = -1;
    int64_t video_token_id_ = -1;
    std::string input_contract_;
    std::string position_policy_;
    mutable std::string cached_vision_key_;
    mutable Tensor cached_vision_;
};

} // namespace mfq::cuda::grid_vision_runtime
