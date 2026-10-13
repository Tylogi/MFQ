#pragma once

#include "storage/weight_loader.h"

#include "minicpmo45/causal_lm.h"
#include "common/transformer_layer.h"
#include "native/tensor_backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
inline constexpr const char *MINICPMO45_RESAMPLER_POS_EMBED_ASSET =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";

struct MiniCPMO45Linear {
    QuantLinear weight;
    mfq_tensor_backend::Tensor bias;

    static MiniCPMO45Linear load(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                 const std::string &prefix, bool with_bias = true) {
        MiniCPMO45Linear result;
        result.weight = load_quant_linear(execution, mfq, prefix + ".weight");
        const std::string bias_name = prefix + ".bias";
        if (with_bias && has_tensor(mfq, bias_name)) {
            result.bias = load_dense_gpu(execution, mfq, bias_name);
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input) const {
        const auto output_dtype = input.scalar_type();
        if (weight.is_dense()) {
            const auto dense_dtype = weight.dense.scalar_type();
            std::optional<mfq_tensor_backend::Tensor> dense_bias = std::nullopt;
            if (bias.defined()) {
                dense_bias = bias.to(dense_dtype);
            }
            return mfq_linear(input.to(dense_dtype), weight.dense, dense_bias)
                .to(output_dtype)
                .contiguous();
        }
        auto output = weight.forward(execution, input);
        if (bias.defined()) {
            output = output + bias.to(output.scalar_type());
        }
        return output.to(output_dtype).contiguous();
    }
};

inline mfq_tensor_backend::Tensor
minicpmo45_embedding(const QuantLinear &embedding, mfq_tensor_backend::Tensor ids,
                     mfq_tensor_backend::ScalarType output_dtype = mfq_tensor_backend::kBFloat16) {
    ids = ids.contiguous().to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64);
    auto output = quant_embedding_lookup(embedding, ids);
    return output.to(output_dtype).contiguous();
}

inline mfq_tensor_backend::Tensor minicpmo45_layer_norm(mfq_tensor_backend::Tensor input,
                                                        const mfq_tensor_backend::Tensor &weight,
                                                        const mfq_tensor_backend::Tensor &bias,
                                                        double eps) {
    const auto dtype = input.scalar_type();
    auto output = mfq_tensor_backend::layer_norm(input, {input.size(-1)}, weight.to(dtype),
                                                 bias.to(dtype), eps);
    return output.to(dtype).contiguous();
}

inline mfq_tensor_backend::Tensor
minicpmo45_attention(mfq_tensor_backend::Tensor query, mfq_tensor_backend::Tensor key,
                     mfq_tensor_backend::Tensor value,
                     MfqOptional<mfq_tensor_backend::Tensor> additive_mask, double scale) {
    auto scores = mfq_tensor_backend::matmul(query, key.transpose(-2, -1)) * scale;
    if (additive_mask.has_value()) {
        scores = scores + additive_mask.value().to(scores.scalar_type());
    }
    auto probabilities = mfq_tensor_backend::softmax(scores, -1, mfq_tensor_backend::kFloat32)
                             .to(value.scalar_type());
    return mfq_tensor_backend::matmul(probabilities, value).contiguous();
}

struct MiniCPMO45VisionAttention {
    MiniCPMO45Linear q;
    MiniCPMO45Linear k;
    MiniCPMO45Linear v;
    MiniCPMO45Linear output;
    int64_t heads = 16;
    int64_t head_dim = 72;

    static MiniCPMO45VisionAttention load(CudaExecutionContext &execution,
                                          const mfq::ModelSource &mfq, const std::string &prefix) {
        MiniCPMO45VisionAttention result;
        result.q = MiniCPMO45Linear::load(execution, mfq, prefix + ".query");
        result.k = MiniCPMO45Linear::load(execution, mfq, prefix + ".key");
        result.v = MiniCPMO45Linear::load(execution, mfq, prefix + ".value");
        result.output = MiniCPMO45Linear::load(execution, mfq, prefix + ".output");
        return result;
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input,
                                       MfqOptional<mfq_tensor_backend::Tensor> mask) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](mfq_tensor_backend::Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        };
        using Tensor = mfq_tensor_backend::Tensor;
        return mfq::models::minicpmo45::encoder_attention(
            input, false, [&](const Tensor &x) { return reshape(q.forward(execution, x)); },
            [&](const Tensor &x) { return reshape(k.forward(execution, x)); },
            [&](const Tensor &x) { return reshape(v.forward(execution, x)); },
            [](Tensor &, Tensor &) {},
            [&](Tensor q, Tensor k, Tensor v) {
                return minicpmo45_attention(q, k, v, mask,
                                            1.0 / std::sqrt(static_cast<double>(head_dim)));
            },
            [&](Tensor attended) {
                return output.forward(
                    execution, attended.transpose(1, 2).reshape({batch, tokens, heads * head_dim}));
            });
    }
};

struct MiniCPMO45VisionLayer {
    MiniCPMO45VisionAttention attention;
    mfq_tensor_backend::Tensor norm1_weight;
    mfq_tensor_backend::Tensor norm1_bias;
    mfq_tensor_backend::Tensor norm2_weight;
    mfq_tensor_backend::Tensor norm2_bias;
    MiniCPMO45Linear fc1;
    MiniCPMO45Linear fc2;

    static MiniCPMO45VisionLayer load(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                      int index) {
        const std::string prefix = "vision.block." + std::to_string(index);
        MiniCPMO45VisionLayer result;
        result.attention = MiniCPMO45VisionAttention::load(execution, mfq, prefix + ".attention");
        result.norm1_weight = load_dense_native_gpu(execution, mfq, prefix + ".norm1.weight");
        result.norm1_bias = load_dense_native_gpu(execution, mfq, prefix + ".norm1.bias");
        result.norm2_weight = load_dense_native_gpu(execution, mfq, prefix + ".norm2.weight");
        result.norm2_bias = load_dense_native_gpu(execution, mfq, prefix + ".norm2.bias");
        result.fc1 = MiniCPMO45Linear::load(execution, mfq, prefix + ".mlp.up");
        result.fc2 = MiniCPMO45Linear::load(execution, mfq, prefix + ".mlp.down");
        return result;
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input,
                                       MfqOptional<mfq_tensor_backend::Tensor> mask) const {
        using Tensor = mfq_tensor_backend::Tensor;
        return mfq::models::pre_norm_layer(
            std::move(input),
            [&](const Tensor &x, int stage) {
                return minicpmo45_layer_norm(x, stage == 0 ? norm1_weight : norm2_weight,
                                             stage == 0 ? norm1_bias : norm2_bias, 1e-6);
            },
            [&](Tensor x) { return attention.forward(execution, x, mask); },
            [&](Tensor x) {
                return mfq::models::mlp(
                    x, [&](Tensor x) { return fc1.forward(execution, x); },
                    [](Tensor x) { return mfq_tensor_backend::gelu(x, "tanh"); },
                    [&](Tensor x) { return fc2.forward(execution, x); });
            },
            [](Tensor residual, Tensor branch) { return (residual + branch).contiguous(); });
    }
};

struct MiniCPMO45VisionEncoder {
    mfq_tensor_backend::Tensor patch_weight;
    mfq_tensor_backend::Tensor patch_bias;
    mfq_tensor_backend::Tensor position_embedding;
    std::vector<MiniCPMO45VisionLayer> layers;
    mfq_tensor_backend::Tensor post_norm_weight;
    mfq_tensor_backend::Tensor post_norm_bias;
    int64_t patch_size = 14;
    int64_t position_side = 70;

    static MiniCPMO45VisionEncoder load(CudaExecutionContext &execution,
                                        const mfq::ModelSource &mfq) {
        MiniCPMO45VisionEncoder result;
        result.patch_weight =
            load_dense_native_gpu(execution, mfq, "vision.patch_embedding.weight");
        result.patch_bias = load_dense_native_gpu(execution, mfq, "vision.patch_embedding.bias");
        result.position_embedding =
            load_dense_native_gpu(execution, mfq, "vision.position_embedding.weight");
        result.post_norm_weight =
            load_dense_native_gpu(execution, mfq, "vision.output_norm.weight");
        result.post_norm_bias = load_dense_native_gpu(execution, mfq, "vision.output_norm.bias");
        result.layers.reserve(27);
        for (int index = 0; index < 27; ++index) {
            result.layers.push_back(MiniCPMO45VisionLayer::load(execution, mfq, index));
        }
        if (result.patch_weight.sizes().vec() != std::vector<int64_t>({1152, 3, 14, 14}) ||
            result.position_embedding.sizes().vec() != std::vector<int64_t>({4900, 1152})) {
            throw std::runtime_error("MiniCPM-o SigLIP tensor shapes disagree with version 4.5");
        }
        return result;
    }

    mfq::StepSequence<mfq_tensor_backend::Tensor> forward_steps(
        CudaExecutionContext &execution,
        mfq_tensor_backend::Tensor pixels,
        mfq_tensor_backend::Tensor patch_mask,
        mfq_tensor_backend::Tensor target_sizes) const {
        if (patch_mask.dim() == 3) {
            patch_mask = patch_mask.flatten(1);
        }
        if (pixels.dim() != 4 || pixels.size(1) != 3 || patch_mask.dim() != 2 ||
            target_sizes.dim() != 2 || target_sizes.size(1) != 2 ||
            pixels.size(0) != patch_mask.size(0) || pixels.size(0) != target_sizes.size(0) ||
            target_sizes.device().is_cuda()) {
            throw std::runtime_error("MiniCPM-o vision input geometry is invalid");
        }
        using Tensor = mfq_tensor_backend::Tensor;
        MfqOptional<Tensor> attention_mask = mfq_nullopt;
        auto encoding = mfq::models::minicpmo45::vision_encoder(
            pixels, layers,
            [&](Tensor pixels) {
                return mfq_tensor_backend::conv2d(
                           pixels.to(patch_weight.scalar_type()), patch_weight, patch_bias,
                           std::vector<int64_t>{patch_size, patch_size}, std::vector<int64_t>{0, 0},
                           std::vector<int64_t>{1, 1}, 1)
                    .flatten(2)
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](Tensor embedded) {
                if (embedded.size(1) != patch_mask.size(1)) {
                    throw std::runtime_error(
                        "MiniCPM-o patch mask length does not match patch convolution");
                }
                auto sizes = target_sizes.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                                 .contiguous();
                auto mask_cpu =
                    patch_mask.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kBool).contiguous();
                const auto *size_data = sizes.data_ptr<int64_t>();
                const auto *mask_data = mask_cpu.data_ptr<bool>();
                auto layout = mfq::models::minicpmo45::patch_positions(
                    embedded.size(0), embedded.size(1), position_side, size_data, mask_data);
                auto &position_ids = layout.ids;
                const bool all_patches_active = layout.all_active;
                auto ids =
                    mfq_tensor_backend::from_blob(
                        position_ids.data(),
                        std::vector<int64_t>{embedded.size(0), embedded.size(1)},
                        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                        .clone()
                        .to(mfq_tensor_backend::kCUDA);
                embedded =
                    embedded +
                    position_embedding.index_select(0, ids.reshape({-1})).reshape(embedded.sizes());

                if (!all_patches_active) {
                    auto invalid =
                        patch_mask.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool)
                            .logical_not()
                            .unsqueeze(1)
                            .unsqueeze(2);
                    attention_mask =
                        mfq_tensor_backend::zeros(
                            {embedded.size(0), 1, embedded.size(1), embedded.size(1)},
                            embedded.options().dtype(mfq_tensor_backend::kFloat32))
                            .masked_fill(invalid, -std::numeric_limits<float>::infinity());
                }
                return embedded;
            },
            [&](const auto &layer, Tensor hidden) {
                return layer.forward(execution, hidden, attention_mask);
            },
            [&](Tensor hidden) {
                return minicpmo45_layer_norm(hidden, post_norm_weight, post_norm_bias, 1e-6);
            });
        while (auto step = encoding.next()) co_yield std::move(step);
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor pixels, mfq_tensor_backend::Tensor mask,
        mfq_tensor_backend::Tensor sizes) const {
        return mfq::finish_steps(
            forward_steps(execution, std::move(pixels), std::move(mask), std::move(sizes)));
    }
};

struct MiniCPMO45Resampler {
    mfq_tensor_backend::Tensor query;
    mfq_tensor_backend::Tensor position_embedding;
    MiniCPMO45Linear kv_projection;
    mfq_tensor_backend::Tensor q_norm_weight;
    mfq_tensor_backend::Tensor q_norm_bias;
    mfq_tensor_backend::Tensor kv_norm_weight;
    mfq_tensor_backend::Tensor kv_norm_bias;
    mfq_tensor_backend::Tensor post_norm_weight;
    mfq_tensor_backend::Tensor post_norm_bias;
    mfq_tensor_backend::Tensor in_projection_weight;
    mfq_tensor_backend::Tensor in_projection_bias;
    mfq_tensor_backend::Tensor out_projection_weight;
    mfq_tensor_backend::Tensor out_projection_bias;
    mfq_tensor_backend::Tensor final_projection;
    int64_t heads = 32;
    int64_t head_dim = 128;

    static MiniCPMO45Resampler load(CudaExecutionContext &execution, const mfq::ModelSource &mfq) {
        MiniCPMO45Resampler result;
        if (!mfq.has_asset(MINICPMO45_RESAMPLER_POS_EMBED_ASSET)) {
            throw std::runtime_error("MiniCPM-o MFQ is missing the exact Resampler position asset; "
                                     "repack it with the current runtime assets tool");
        }
        auto position_blob = read_asset(mfq, MINICPMO45_RESAMPLER_POS_EMBED_ASSET);
        constexpr size_t position_header_bytes = 20;
        constexpr size_t position_value_bytes =
            static_cast<size_t>(70) * 70 * 4096 * sizeof(uint16_t);
        if (position_blob.size() != position_header_bytes + position_value_bytes ||
            std::memcmp(position_blob.data(), "MFQRSPB1", 8) != 0) {
            throw std::runtime_error("MiniCPM-o Resampler position asset has an invalid format");
        }
        size_t position_offset = 8;
        const uint32_t position_height = read_u32_from(position_blob, position_offset);
        const uint32_t position_width = read_u32_from(position_blob, position_offset);
        const uint32_t position_dim = read_u32_from(position_blob, position_offset);
        if (position_height != 70 || position_width != 70 || position_dim != 4096 ||
            position_offset != position_header_bytes) {
            throw std::runtime_error("MiniCPM-o Resampler position asset shape is not 70x70x4096");
        }
        {
            MfqCudaGuard guard(active_weight_load_device(execution));
            result.position_embedding =
                mfq_tensor_backend::from_blob(
                    position_blob.data() + position_offset, std::vector<int64_t>{70, 70, 4096},
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16))
                    .clone()
                    .to(mfq_tensor_backend::kCUDA)
                    .contiguous();
        }
        result.query = load_dense_native_gpu(execution, mfq, "vision.resampler.query");
        result.kv_projection =
            MiniCPMO45Linear::load(execution, mfq, "vision.resampler.key_value", false);
        result.q_norm_weight =
            load_dense_native_gpu(execution, mfq, "vision.resampler.query_norm.weight");
        result.q_norm_bias =
            load_dense_native_gpu(execution, mfq, "vision.resampler.query_norm.bias");
        result.kv_norm_weight =
            load_dense_native_gpu(execution, mfq, "vision.resampler.key_value_norm.weight");
        result.kv_norm_bias =
            load_dense_native_gpu(execution, mfq, "vision.resampler.key_value_norm.bias");
        result.post_norm_weight =
            load_dense_native_gpu(execution, mfq, "vision.resampler.output_norm.weight");
        result.post_norm_bias =
            load_dense_native_gpu(execution, mfq, "vision.resampler.output_norm.bias");
        result.in_projection_weight =
            load_dense_native_gpu(execution, mfq, "vision.resampler.attention.qkv.weight");
        result.in_projection_bias =
            load_dense_native_gpu(execution, mfq, "vision.resampler.attention.qkv.bias");
        result.out_projection_weight =
            load_dense_native_gpu(execution, mfq, "vision.resampler.attention.output.weight");
        result.out_projection_bias =
            load_dense_native_gpu(execution, mfq, "vision.resampler.attention.output.bias");
        result.final_projection =
            load_dense_native_gpu(execution, mfq, "vision.resampler.output.weight");
        if (result.query.sizes().vec() != std::vector<int64_t>({64, 4096}) ||
            result.in_projection_weight.sizes().vec() != std::vector<int64_t>({12288, 4096})) {
            throw std::runtime_error("MiniCPM-o Resampler tensor shapes disagree with version 4.5");
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input,
                                       mfq_tensor_backend::Tensor target_sizes) const {
        if (input.dim() != 3 || input.size(2) != 1152 || target_sizes.dim() != 2 ||
            target_sizes.size(1) != 2 || target_sizes.size(0) != input.size(0) ||
            target_sizes.device().is_cuda() ||
            input.scalar_type() != mfq_tensor_backend::kBFloat16) {
            throw std::runtime_error("MiniCPM-o Resampler input geometry is invalid");
        }
        const int64_t batch = input.size(0);
        const int64_t length = input.size(1);
        auto sizes =
            target_sizes.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        const auto *size_data = sizes.data_ptr<int64_t>();
        std::vector<mfq_tensor_backend::Tensor> positions;
        positions.reserve(static_cast<size_t>(batch));
        std::vector<uint8_t> key_padding(static_cast<size_t>(batch * length), uint8_t{1});
        for (int64_t index = 0; index < batch; ++index) {
            const int64_t height = size_data[2 * index];
            const int64_t width = size_data[2 * index + 1];
            const int64_t patches = height * width;
            if (height <= 0 || width <= 0 || height > position_embedding.size(0) ||
                width > position_embedding.size(1) || patches > length) {
                throw std::runtime_error("MiniCPM-o Resampler target size is invalid");
            }
            auto position = position_embedding.narrow(0, 0, height)
                                .narrow(1, 0, width)
                                .reshape({patches, 4096});
            if (patches < length) {
                position = mfq_tensor_backend::cat(
                    {position,
                     mfq_tensor_backend::zeros({length - patches, 4096}, position.options())},
                    0);
            }
            positions.push_back(position);
            for (int64_t patch = 0; patch < patches; ++patch) {
                key_padding[static_cast<size_t>(index * length + patch)] = 0;
            }
        }
        auto position = mfq_tensor_backend::stack(positions, 0);
        auto key_padding_mask =
            mfq_tensor_backend::from_blob(
                key_padding.data(), std::vector<int64_t>{batch, length},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8))
                .clone()
                .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool);
        const auto dtype = input.scalar_type();
        auto project = [&](mfq_tensor_backend::Tensor value, int64_t offset) {
            auto weight = in_projection_weight.narrow(0, offset, 4096);
            auto bias = in_projection_bias.narrow(0, offset, 4096);
            return mfq_linear(value.to(weight.scalar_type()), weight, bias).to(dtype).contiguous();
        };
        using Tensor = mfq_tensor_backend::Tensor;
        return mfq::models::minicpmo45::resample(
            [&] { return kv_projection.forward(execution, input); },
            [&](Tensor kv) {
                return minicpmo45_layer_norm(kv, kv_norm_weight, kv_norm_bias, 1e-6);
            },
            [&] {
                auto normalized = minicpmo45_layer_norm(query, q_norm_weight, q_norm_bias, 1e-6);
                return normalized.unsqueeze(0)
                    .expand({batch, normalized.size(0), normalized.size(1)})
                    .contiguous();
            },
            [&](Tensor repeated_query) {
                return project(repeated_query, 0)
                    .reshape({batch, 64, heads, head_dim})
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](Tensor kv) {
                return project(kv + position, 4096)
                    .reshape({batch, length, heads, head_dim})
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](Tensor kv) {
                return project(kv, 8192)
                    .reshape({batch, length, heads, head_dim})
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](Tensor q, Tensor k, Tensor v) {
                auto mask = mfq_tensor_backend::zeros({batch, 1, 1, length},
                                                      mfq_tensor_backend::TensorOptions()
                                                          .device(mfq_tensor_backend::kCUDA)
                                                          .dtype(dtype))
                                .masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),
                                             -std::numeric_limits<float>::infinity())
                                .expand({batch, heads, 1, length})
                                .reshape({batch * heads, 1, length});
                auto q_flat = q.reshape({batch * heads, 64, head_dim});
                auto k_flat = k.reshape({batch * heads, length, head_dim});
                auto v_flat = v.reshape({batch * heads, length, head_dim});
                auto attention_weights = mfq_tensor_backend::baddbmm(
                    mask, q_flat * std::sqrt(1.0 / static_cast<double>(head_dim)),
                    k_flat.transpose(1, 2));
                attention_weights = mfq_tensor_backend::softmax(attention_weights, -1);
                return mfq_tensor_backend::bmm(attention_weights, v_flat)
                    .reshape({batch, heads, 64, head_dim})
                    .transpose(1, 2)
                    .reshape({batch, 64, 4096});
            },
            [&](Tensor attended) {
                return mfq_linear(attended.to(out_projection_weight.scalar_type()),
                                  out_projection_weight, out_projection_bias)
                    .to(dtype)
                    .contiguous();
            },
            [&](Tensor attended) {
                return minicpmo45_layer_norm(attended, post_norm_weight, post_norm_bias, 1e-6);
            },
            [&](Tensor attended) {
                return mfq_tensor_backend::matmul(attended.to(final_projection.scalar_type()),
                                                  final_projection)
                    .to(dtype)
                    .contiguous();
            });
    }
};

struct MiniCPMO45WhisperAttention {
    MiniCPMO45Linear q;
    MiniCPMO45Linear k;
    MiniCPMO45Linear v;
    MiniCPMO45Linear output;
    int64_t heads = 16;
    int64_t head_dim = 64;

    static MiniCPMO45WhisperAttention load(CudaExecutionContext &execution,
                                           const mfq::ModelSource &mfq, const std::string &prefix) {
        MiniCPMO45WhisperAttention result;
        result.q = MiniCPMO45Linear::load(execution, mfq, prefix + ".query");
        result.k = MiniCPMO45Linear::load(execution, mfq, prefix + ".key", false);
        result.v = MiniCPMO45Linear::load(execution, mfq, prefix + ".value");
        result.output = MiniCPMO45Linear::load(execution, mfq, prefix + ".output");
        return result;
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input,
                                       MfqOptional<mfq_tensor_backend::Tensor> mask,
                                       mfq_tensor_backend::Tensor *key_cache = nullptr,
                                       mfq_tensor_backend::Tensor *value_cache = nullptr) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](mfq_tensor_backend::Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        };
        using Tensor = mfq_tensor_backend::Tensor;
        return mfq::models::minicpmo45::encoder_attention(
            input, key_cache && value_cache,
            [&](const Tensor &x) { return reshape(q.forward(execution, x)); },
            [&](const Tensor &x) { return reshape(k.forward(execution, x)); },
            [&](const Tensor &x) { return reshape(v.forward(execution, x)); },
            [&](Tensor &key, Tensor &value) {
                if (key_cache->defined()) {
                    key = mfq_tensor_backend::cat({*key_cache, key}, 2).contiguous();
                    value = mfq_tensor_backend::cat({*value_cache, value}, 2).contiguous();
                }
                *key_cache = key;
                *value_cache = value;
            },
            [&](Tensor query, Tensor key, Tensor value) {
                std::optional<mfq_tensor_backend::Tensor> attention_mask = std::nullopt;
                if (mask.has_value()) {
                    attention_mask = mask.value().to(query.scalar_type());
                }
                return mfq_scaled_dot_product_attention(query, key, value, attention_mask, 0.0,
                                                        false, std::nullopt, false)
                    .contiguous();
            },
            [&](Tensor attended) {
                attended = attended.transpose(1, 2).reshape({batch, tokens, heads * head_dim});
                return output.forward(execution, attended);
            });
    }
};

struct MiniCPMO45WhisperLayer {
    MiniCPMO45WhisperAttention attention;
    mfq_tensor_backend::Tensor attention_norm_weight;
    mfq_tensor_backend::Tensor attention_norm_bias;
    mfq_tensor_backend::Tensor final_norm_weight;
    mfq_tensor_backend::Tensor final_norm_bias;
    MiniCPMO45Linear fc1;
    MiniCPMO45Linear fc2;
    mfq_tensor_backend::Tensor key_cache;
    mfq_tensor_backend::Tensor value_cache;

    static MiniCPMO45WhisperLayer load(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                       int index) {
        const std::string prefix = "audio.block." + std::to_string(index);
        MiniCPMO45WhisperLayer result;
        result.attention = MiniCPMO45WhisperAttention::load(execution, mfq, prefix + ".attention");
        result.attention_norm_weight =
            load_dense_native_gpu(execution, mfq, prefix + ".attention.norm.weight");
        result.attention_norm_bias =
            load_dense_native_gpu(execution, mfq, prefix + ".attention.norm.bias");
        result.final_norm_weight =
            load_dense_native_gpu(execution, mfq, prefix + ".mlp.norm.weight");
        result.final_norm_bias = load_dense_native_gpu(execution, mfq, prefix + ".mlp.norm.bias");
        result.fc1 = MiniCPMO45Linear::load(execution, mfq, prefix + ".mlp.up");
        result.fc2 = MiniCPMO45Linear::load(execution, mfq, prefix + ".mlp.down");
        return result;
    }

    void reset() {
        key_cache = mfq_tensor_backend::Tensor();
        value_cache = mfq_tensor_backend::Tensor();
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input,
                                       MfqOptional<mfq_tensor_backend::Tensor> mask,
                                       bool use_cache) {
        using Tensor = mfq_tensor_backend::Tensor;
        return mfq::models::pre_norm_layer(
            input,
            [&](const Tensor &x, int stage) {
                return minicpmo45_layer_norm(
                    x, stage == 0 ? attention_norm_weight : final_norm_weight,
                    stage == 0 ? attention_norm_bias : final_norm_bias, 1e-5);
            },
            [&](Tensor x) {
                return attention.forward(execution, x, mask, use_cache ? &key_cache : nullptr,
                                         use_cache ? &value_cache : nullptr);
            },
            [&](Tensor x) {
                return mfq::models::mlp(
                    x, [&](Tensor x) { return fc1.forward(execution, x); },
                    [](Tensor x) { return mfq_tensor_backend::gelu(x); },
                    [&](Tensor x) { return fc2.forward(execution, x); });
            },
            [](Tensor x, Tensor branch) { return (x + branch).contiguous(); });
    }
};

struct MiniCPMO45AudioEncoder {
    mfq_tensor_backend::Tensor conv1_weight;
    mfq_tensor_backend::Tensor conv1_bias;
    mfq_tensor_backend::Tensor conv2_weight;
    mfq_tensor_backend::Tensor conv2_bias;
    mfq_tensor_backend::Tensor position_embedding;
    std::vector<MiniCPMO45WhisperLayer> layers;
    mfq_tensor_backend::Tensor final_norm_weight;
    mfq_tensor_backend::Tensor final_norm_bias;
    MiniCPMO45Linear projector1;
    MiniCPMO45Linear projector2;

    static MiniCPMO45AudioEncoder load(CudaExecutionContext &execution,
                                       const mfq::ModelSource &mfq) {
        MiniCPMO45AudioEncoder result;
        result.conv1_weight =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv1.weight");
        result.conv1_bias =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv1.bias");
        result.conv2_weight =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv2.weight");
        result.conv2_bias =
            load_dense_native_gpu(execution, mfq, "audio.patch_embedding.conv2.bias");
        result.position_embedding =
            load_dense_native_gpu(execution, mfq, "audio.position_embedding.weight");
        result.final_norm_weight =
            load_dense_native_gpu(execution, mfq, "audio.output_norm.weight");
        result.final_norm_bias = load_dense_native_gpu(execution, mfq, "audio.output_norm.bias");
        result.projector1 = MiniCPMO45Linear::load(execution, mfq, "audio.projector.input");
        result.projector2 = MiniCPMO45Linear::load(execution, mfq, "audio.projector.output");
        result.layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            result.layers.push_back(MiniCPMO45WhisperLayer::load(execution, mfq, index));
        }
        if (result.conv1_weight.sizes().vec() != std::vector<int64_t>({1024, 80, 3}) ||
            result.conv2_weight.sizes().vec() != std::vector<int64_t>({1024, 1024, 3}) ||
            result.position_embedding.sizes().vec() != std::vector<int64_t>({1500, 1024})) {
            throw std::runtime_error("MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
        return result;
    }

    void reset() {
        for (auto &layer : layers)
            layer.reset();
    }

    int64_t cache_length() const {
        if (layers.empty() || !layers.front().key_cache.defined())
            return 0;
        return layers.front().key_cache.size(2);
    }

    mfq::StepSequence<mfq_tensor_backend::Tensor> forward_streaming_steps(CudaExecutionContext &execution,
                                                 mfq_tensor_backend::Tensor features,
                                                 int64_t prefix_extra_frames,
                                                 int64_t suffix_extra_frames) {
        if (features.dim() != 3 || features.size(0) != 1 || features.size(1) != 80 ||
            prefix_extra_frames < 0 || suffix_extra_frames < 0) {
            throw std::runtime_error("MiniCPM-o streaming audio expects [1,80,frames] and "
                                     "non-negative extra-frame counts");
        }
        const int64_t conv_tokens_before_crop = (features.size(2) - 1) / 2 + 1;
        if (cache_length() + conv_tokens_before_crop >= position_embedding.size(0)) {
            reset();
        }

        using Tensor = mfq_tensor_backend::Tensor;
        auto sequence = mfq::models::minicpmo45::audio_encoder(
            features, layers,
            [&](Tensor features) {
                return mfq_tensor_backend::conv1d(
                    features.to(conv1_weight.scalar_type()), conv1_weight, conv1_bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](Tensor x) { return mfq_tensor_backend::gelu(x); },
            [&](Tensor hidden) {
                return mfq_tensor_backend::conv1d(hidden, conv2_weight, conv2_bias,
                                                  std::vector<int64_t>{2}, std::vector<int64_t>{1},
                                                  std::vector<int64_t>{1}, 1);
            },
            [&](Tensor hidden) {
                const int64_t prefix_to_remove =
                    prefix_extra_frames > 0 ? (prefix_extra_frames + 1) / 2 : 0;
                const int64_t suffix_to_remove =
                    suffix_extra_frames > 0 ? (suffix_extra_frames + 1) / 2 : 0;
                if (prefix_to_remove + suffix_to_remove >= hidden.size(2)) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming audio extra context removes every frame");
                }
                if (prefix_to_remove > 0) {
                    hidden = hidden.narrow(2, prefix_to_remove, hidden.size(2) - prefix_to_remove);
                }
                if (suffix_to_remove > 0) {
                    hidden = hidden.narrow(2, 0, hidden.size(2) - suffix_to_remove);
                }
                hidden = hidden.transpose(1, 2).contiguous();

                const int64_t past = cache_length();
                const int64_t tokens = hidden.size(1);
                if (past + tokens > position_embedding.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
                }
                hidden = hidden + position_embedding.narrow(0, past, tokens);
                auto attention_mask =
                    mfq_tensor_backend::zeros({1, 1, tokens, past + tokens}, hidden.options());
                return std::array<Tensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, Tensor hidden, Tensor mask) {
                return layer.forward(execution, hidden, mask, true);
            },
            [&](Tensor hidden) {
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
            },
            [&](Tensor hidden) { return projector1.forward(execution, hidden); },
            [](Tensor hidden) { return mfq_tensor_backend::relu(hidden); },
            [&](Tensor hidden) { return projector2.forward(execution, hidden); },
            [&](Tensor hidden) {
                if (hidden.size(1) < 5) {
                    throw std::runtime_error(
                        "MiniCPM-o streaming audio chunk is too short for stride-5 pooling");
                }
                return mfq_tensor_backend::avg_pool1d(
                           hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
                    .transpose(1, 2)
                    .contiguous();
            });
        while (auto step = sequence.next()) co_yield std::move(step);
    }

    mfq_tensor_backend::Tensor forward_streaming(CudaExecutionContext &execution,
        mfq_tensor_backend::Tensor features, int64_t prefix, int64_t suffix) {
        return mfq::finish_steps(forward_streaming_steps(execution, std::move(features), prefix, suffix));
    }

    static std::vector<int64_t> pooled_lengths(mfq_tensor_backend::Tensor raw_lengths) {
        auto lengths =
            raw_lengths.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        const auto *values = lengths.data_ptr<int64_t>();
        std::vector<int64_t> result(static_cast<size_t>(lengths.numel()));
        for (int64_t index = 0; index < lengths.numel(); ++index) {
            result[static_cast<size_t>(index)] =
                mfq::models::minicpmo45::pooled_audio_length(values[index]);
        }
        return result;
    }

    mfq::StepSequence<mfq_tensor_backend::Tensor> forward_steps(
        CudaExecutionContext &execution,
        mfq_tensor_backend::Tensor features,
        mfq_tensor_backend::Tensor raw_lengths,
        bool use_cache = false) {
        if (features.dim() != 3 || features.size(1) != 80 || raw_lengths.dim() != 1 ||
            raw_lengths.size(0) != features.size(0)) {
            throw std::runtime_error("MiniCPM-o audio input geometry is invalid");
        }
        using Tensor = mfq_tensor_backend::Tensor;
        auto encoding = mfq::models::minicpmo45::audio_encoder(
            features, layers,
            [&](Tensor features) {
                return mfq_tensor_backend::conv1d(
                    features.to(conv1_weight.scalar_type()), conv1_weight, conv1_bias,
                    std::vector<int64_t>{1}, std::vector<int64_t>{1}, std::vector<int64_t>{1}, 1);
            },
            [](Tensor x) { return mfq_tensor_backend::gelu(x); },
            [&](Tensor hidden) {
                return mfq_tensor_backend::conv1d(hidden, conv2_weight, conv2_bias,
                                                  std::vector<int64_t>{2}, std::vector<int64_t>{1},
                                                  std::vector<int64_t>{1}, 1);
            },
            [&](Tensor hidden) {
                hidden = hidden.transpose(1, 2).contiguous();
                const int64_t tokens = hidden.size(1);
                int64_t past = 0;
                if (use_cache && !layers.empty() && layers.front().key_cache.defined()) {
                    past = layers.front().key_cache.size(2);
                }
                if (past + tokens > position_embedding.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o Whisper position cache exceeds 1500 frames");
                }
                hidden = hidden + position_embedding.narrow(0, past, tokens);

                // Preserve the official graph: its padding comparison uses the raw
                // Whisper feature lengths against the post-convolution sequence.
                auto length_tensor =
                    raw_lengths.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64)
                        .contiguous();
                auto key_positions = mfq_tensor_backend::arange(
                    past + tokens, mfq_tensor_backend::TensorOptions()
                                       .device(mfq_tensor_backend::kCUDA)
                                       .dtype(mfq_tensor_backend::kInt64));
                auto valid_keys = key_positions.unsqueeze(0) < (length_tensor + past).unsqueeze(1);
                auto query_positions =
                    mfq_tensor_backend::arange(past, past + tokens,
                                               mfq_tensor_backend::TensorOptions()
                                                   .device(mfq_tensor_backend::kCUDA)
                                                   .dtype(mfq_tensor_backend::kInt64));
                // Official MiniCPM-o 4.5 uses audio_chunk_length=1.0. Whisper emits
                // 50 encoder frames per second, and every query can see its current
                // 50-frame chunk plus all preceding chunks.
                auto chunk_visible =
                    key_positions.unsqueeze(0) <
                    ((query_positions / mfq::models::minicpmo45::audio_chunk_frames + 1) *
                     mfq::models::minicpmo45::audio_chunk_frames)
                        .unsqueeze(1);
                auto visible = valid_keys.unsqueeze(1) & chunk_visible.unsqueeze(0);
                auto attention_mask =
                    mfq_tensor_backend::zeros({features.size(0), 1, tokens, past + tokens},
                                              mfq_tensor_backend::TensorOptions()
                                                  .device(mfq_tensor_backend::kCUDA)
                                                  .dtype(mfq_tensor_backend::kFloat32))
                        .masked_fill(visible.logical_not().unsqueeze(1),
                                     -std::numeric_limits<float>::infinity());
                return std::array<Tensor, 2>{hidden, attention_mask};
            },
            [&](auto &layer, Tensor hidden, Tensor mask) {
                return layer.forward(execution, hidden, mask, use_cache);
            },
            [&](Tensor hidden) {
                return minicpmo45_layer_norm(hidden, final_norm_weight, final_norm_bias, 1e-5);
            },
            [&](Tensor hidden) { return projector1.forward(execution, hidden); },
            [](Tensor hidden) { return mfq_tensor_backend::relu(hidden); },
            [&](Tensor hidden) { return projector2.forward(execution, hidden); },
            [&](Tensor hidden) {
                return mfq_tensor_backend::avg_pool1d(
                           hidden.transpose(1, 2), std::vector<int64_t>{5}, std::vector<int64_t>{5})
                    .transpose(1, 2)
                    .contiguous();
            });
        while (auto step = encoding.next()) co_yield std::move(step);
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor features, mfq_tensor_backend::Tensor lengths, bool use_cache = false) {
        return mfq::finish_steps(
            forward_steps(execution, std::move(features), std::move(lengths), use_cache));
    }
};
#include <random>
#include "generation_policy.h"
#include "sampling.h"
#include "models/common/full_block.h"
#include "ops.h"

struct MiniCPMO45TtsSamplingOps {
    using Tensor = mfq_tensor_backend::Tensor;
    std::mt19937 *evaluator_rng = nullptr;
    static Tensor temperature(Tensor scores, double value) { return scores / value; }
    static Tensor penalties(Tensor scores, std::span<const Tensor> history, double penalty) {
        auto counts = mfq_tensor_backend::zeros_like(scores);
        for (const auto &token : history)
            counts.scatter_add_(1, token.reshape({1, 1}),
                                mfq_tensor_backend::ones({1, 1}, counts.options()));
        auto alpha =
            mfq_tensor_backend::pow(mfq_tensor_backend::full_like(counts, penalty), counts);
        return mfq_tensor_backend::where(scores < 0, scores * alpha, scores / alpha);
    }
    static void mask_eos(Tensor &scores, int64_t token) {
        scores.index_put_({Slice(), token}, -std::numeric_limits<float>::infinity());
    }
    Tensor reference(Tensor scores, int64_t top_k, double top_p, int64_t minimum_keep) const {
        auto host = scores.reshape({-1})
                        .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
                        .contiguous();
        auto token = mfq::engine::sample_top_k_top_p(
            std::span<const float>(host.data_ptr<float>(), host.numel()), top_k, top_p,
            minimum_keep, *evaluator_rng);
        return mfq_tensor_backend::tensor(std::vector<int64_t>{token},
                                          mfq_tensor_backend::TensorOptions()
                                              .device(mfq_tensor_backend::kCUDA)
                                              .dtype(mfq_tensor_backend::kInt64));
    }
    static Tensor top_k(Tensor scores, int64_t count) {
        if (count >= scores.size(1))
            return scores;
        auto values = std::get<0>(mfq_tensor_backend::topk(scores, count, -1, true, true));
        auto threshold = values.select(1, count - 1).unsqueeze(1);
        return scores.masked_fill(scores < threshold, -std::numeric_limits<float>::infinity());
    }
    static Tensor top_p(Tensor scores, double value) {
        auto sorted = mfq_tensor_backend::sort(scores, -1, true);
        auto logits = std::get<0>(sorted);
        auto indices = std::get<1>(sorted);
        auto cumulative = mfq_tensor_backend::cumsum(mfq_tensor_backend::softmax(logits, -1), -1);
        auto keep = cumulative <= value;
        keep.narrow(1, 1, keep.size(1) - 1).copy_(keep.narrow(1, 0, keep.size(1) - 1).clone());
        keep.index_put_({Slice(), 0}, true);
        logits = logits.masked_fill(keep.logical_not(), -std::numeric_limits<float>::infinity());
        scores = mfq_tensor_backend::full_like(scores, -std::numeric_limits<float>::infinity());
        scores.scatter_(1, indices, logits);
        return scores;
    }
    static Tensor min_p(Tensor scores, double value) {
        auto maximum = std::get<0>(mfq_tensor_backend::max(scores, -1, true));
        return scores.masked_fill(scores < maximum + std::log(value),
                                  -std::numeric_limits<float>::infinity());
    }
    static Tensor sample(Tensor scores) {
        return mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(scores, -1), 1)
            .reshape({1})
            .to(mfq_tensor_backend::kInt64);
    }
};

struct MiniCPMO45TtsDecoder {
    CudaExecutionContext* execution = nullptr;
    mfq::models::ModelConfig config;
    RopeCache rope;
    QuantLinear text_embedding;
    QuantLinear code_embedding;
    MiniCPMO45Linear semantic_projector1;
    MiniCPMO45Linear semantic_projector2;
    MiniCPMO45Linear speaker_projector1;
    MiniCPMO45Linear speaker_projector2;
    std::vector<std::unique_ptr<Block>> blocks;
    mfq_tensor_backend::Tensor output_norm;
    mfq_tensor_backend::Tensor code_head;
    int64_t cache_position = 0;


    static MiniCPMO45TtsDecoder load(
            CudaExecutionContext& execution,
            const mfq::ModelSource& mfq) {
        MiniCPMO45TtsDecoder result;
        result.execution = &execution;
        result.config = mfq::models::minicpmo45::tts_decoder_config();
        result.rope = RopeCache(
            result.config.max_position_embeddings,
            result.config.rotary_dim,
            result.config.rope_base);
        result.text_embedding = load_quant_linear(execution,
            mfq, "tts.text_embedding.weight");
        result.code_embedding = load_quant_linear(execution,
            mfq, "tts.code_embedding.0.weight");
        result.semantic_projector1 = MiniCPMO45Linear::load(execution,
            mfq, "tts.semantic_projector.input");
        result.semantic_projector2 = MiniCPMO45Linear::load(execution,
            mfq, "tts.semantic_projector.output");
        result.speaker_projector1 = MiniCPMO45Linear::load(execution,
            mfq, "tts.speaker_projector.input");
        result.speaker_projector2 = MiniCPMO45Linear::load(execution,
            mfq, "tts.speaker_projector.output");
        result.output_norm = load_dense_native_gpu(execution,
            mfq, "tts.output_norm.weight");
        auto head_g = load_dense_native_gpu(execution,
            mfq, "tts.code_output.0.weight_norm.magnitude");
        auto head_v = load_dense_native_gpu(execution,
            mfq, "tts.code_output.0.weight_norm.direction");
        if (head_g.sizes().vec() != std::vector<int64_t>({6562, 1}) ||
                head_v.sizes().vec() != std::vector<int64_t>({6562, 768}) ||
                result.text_embedding.out() != 152064 ||
                result.text_embedding.neuron_len() != 768 ||
                result.code_embedding.out() != 6562 ||
                result.code_embedding.neuron_len() != 768) {
            throw std::runtime_error(
                "MiniCPM-o TTS tensor shapes disagree with version 4.5");
        }
        auto head_v_f32 = head_v.to(mfq_tensor_backend::kFloat32);
        auto row_norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(head_v_f32.square(), -1, true))
            .clamp_min(1e-12);
        result.code_head = (
            head_g.to(mfq_tensor_backend::kFloat32) * head_v_f32 /
            row_norm).to(head_v.scalar_type()).contiguous();
        result.blocks.reserve(20);
        for (int index = 0; index < 20; ++index) {
            auto block = mfq::cuda::minicpmo45::load_language_block(
                execution, mfq, result.config, index,
                "full_attention", mfq::models::minicpmo45::LanguageComponent::tts, "tts");
            block->cuda_device =
                execution.layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch) {
        cache_position = 0;
        for (auto & block : blocks) block->reset(batch);
    }

    mfq_tensor_backend::Tensor semantic_projection(mfq_tensor_backend::Tensor hidden) const {
        auto projected = semantic_projector1.forward(
            *execution, hidden);
        projected = mfq_tensor_backend::relu(projected);
        projected = semantic_projector2.forward(
            *execution, projected);
        auto norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(projected.to(mfq_tensor_backend::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    mfq_tensor_backend::Tensor speaker_projection(mfq_tensor_backend::Tensor hidden) const {
        return speaker_projector2.forward(
            *execution,
            mfq_tensor_backend::relu(
                speaker_projector1.forward(*execution, hidden)));
    }

    mfq_tensor_backend::Tensor condition(
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden) const {
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096) {
            throw std::runtime_error(
                "MiniCPM-o TTS condition expects one aligned text/hidden span");
        }
        auto text = minicpmo45_embedding(text_embedding, text_ids);
        auto semantic = semantic_projection(language_hidden)
            .to(text.scalar_type());
        auto merged = (text + semantic).contiguous();
        auto suffix_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{151692, 151687},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 2});
        auto suffix = minicpmo45_embedding(
            text_embedding, suffix_ids, text.scalar_type());
        return mfq_tensor_backend::cat({merged, suffix}, 1).contiguous();
    }

    mfq_tensor_backend::Tensor duplex_condition(
            mfq_tensor_backend::Tensor text_ids,
            mfq_tensor_backend::Tensor language_hidden,
            int64_t audio_bos_token) const {
        if (text_ids.dim() == 1) text_ids = text_ids.unsqueeze(0);
        if (language_hidden.dim() == 2) {
            language_hidden = language_hidden.unsqueeze(0);
        }
        if (text_ids.dim() != 2 || language_hidden.dim() != 3 ||
                text_ids.size(0) != 1 || language_hidden.size(0) != 1 ||
                text_ids.size(1) != language_hidden.size(1) ||
                language_hidden.size(2) != 4096 ||
                audio_bos_token < 0 || audio_bos_token >= text_embedding.out()) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS condition has invalid geometry or audio BOS");
        }
        mfq_tensor_backend::Tensor condition;
        if (text_ids.size(1) > 0) {
            auto text = minicpmo45_embedding(text_embedding, text_ids);
            auto semantic = semantic_projection(language_hidden)
                .to(text.scalar_type());
            condition = (text + semantic).contiguous();
        }
        auto bos_ids = mfq_tensor_backend::tensor(
            std::vector<int64_t>{audio_bos_token},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)).reshape({1, 1});
        auto bos = minicpmo45_embedding(
            text_embedding, bos_ids, mfq_tensor_backend::kBFloat16);
        return condition.defined()
            ? mfq_tensor_backend::cat({condition, bos}, 1).contiguous()
            : bos.contiguous();
    }

    mfq_tensor_backend::Tensor hidden_forward(mfq_tensor_backend::Tensor input_embeddings) {
        if (input_embeddings.dim() != 3 ||
                input_embeddings.size(2) != config.hidden_size) {
            throw std::runtime_error(
                "MiniCPM-o TTS embeddings must be [batch,tokens,768]");
        }
        const int64_t batch = input_embeddings.size(0);
        const int64_t tokens = input_embeddings.size(1);
        return mfq::models::minicpmo45::tts_forward(
            std::move(input_embeddings), batch, tokens, config.max_position_embeddings,
            cache_position, blocks,
            [&] { reset(batch); },
            [&](auto value) { return value.to(mfq_tensor_backend::kBFloat16).contiguous(); },
            [&] {
                auto positions = mfq_tensor_backend::arange(
                    cache_position, cache_position + tokens,
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                        .dtype(mfq_tensor_backend::kInt64));
                MfqOptional<mfq_tensor_backend::Tensor> sequence_length = mfq_nullopt;
                if (cache_position > 0) {
                    sequence_length = mfq_tensor_backend::full(
                        {batch}, cache_position + tokens,
                        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                            .dtype(mfq_tensor_backend::kInt64));
                }
                return std::make_pair(positions, sequence_length);
            },
            [&](auto& block, auto hidden, const auto& positions) {
                return block->forward(*execution, hidden, positions.first, cache_position,
                                      positions.second, rope);
            },
            [&](auto hidden) {
                auto flat = hidden.reshape(
                    {batch * tokens, config.hidden_size});
                auto normalized_f32 = flat.to(mfq_tensor_backend::kFloat32);
                normalized_f32 = normalized_f32 * mfq_tensor_backend::rsqrt(
                    mfq_tensor_backend::mean(normalized_f32.square(), -1, true) +
                    config.rms_norm_eps);
                return (
                    normalized_f32.to(flat.scalar_type()) *
                    output_norm.to(flat.scalar_type()))
                    .reshape({batch, tokens, config.hidden_size})
                    .contiguous();
            });
    }

    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const {
        return mfq_tensor_backend::matmul(
            hidden.to(code_head.scalar_type()),
            code_head.transpose(0, 1));
    }

    struct ChunkResult {
        mfq_tensor_backend::Tensor codes;
        bool finished = false;
    };

    mfq::StepSequence<ChunkResult> generate_duplex_chunk_steps(
            mfq_tensor_backend::Tensor condition_embeddings,
            int64_t max_new_tokens,
            int64_t minimum_new_tokens,
            int64_t eos_token,
            double temperature,
            double repetition_penalty) {
        if (condition_embeddings.dim() != 3 ||
                condition_embeddings.size(0) != 1 ||
                condition_embeddings.size(2) != config.hidden_size ||
                max_new_tokens <= 0 || minimum_new_tokens < 0 ||
                eos_token < 0 || eos_token >= code_embedding.out() ||
                temperature <= 0.0 || repetition_penalty <= 0.0) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS generation limits are invalid");
        }
        std::vector<mfq_tensor_backend::Tensor> sampled;
        sampled.reserve(static_cast<size_t>(max_new_tokens));
        auto current = condition_embeddings.to(mfq_tensor_backend::kBFloat16).contiguous();
        bool finished = false;
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            auto hidden = hidden_forward(current);
            auto step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(mfq_tensor_backend::kFloat32) / temperature;
            if (!sampled.empty() && repetition_penalty != 1.0) {
                const size_t begin = sampled.size() > 16
                    ? sampled.size() - 16 : 0;
                auto counts = mfq_tensor_backend::zeros_like(step_logits);
                for (size_t index = begin; index < sampled.size(); ++index) {
                    counts.scatter_add_(
                        1, sampled[index].reshape({1, 1}),
                        mfq_tensor_backend::ones({1, 1}, counts.options()));
                }
                auto alpha = mfq_tensor_backend::pow(
                    mfq_tensor_backend::full_like(counts, repetition_penalty), counts);
                step_logits = mfq_tensor_backend::where(
                    step_logits < 0,
                    step_logits * alpha,
                    step_logits / alpha);
            }
            if (step < minimum_new_tokens) {
                step_logits.index_put_(
                    {Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            auto token = mfq_tensor_backend::multinomial(
                mfq_tensor_backend::softmax(step_logits, -1), 1)
                .reshape({1}).to(mfq_tensor_backend::kInt64);
            sampled.push_back(token);
            finished = token.eq(eos_token).all().item<bool>();
            co_yield mfq::StepState::advanced;
            if (finished) break;
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        const int64_t returned = std::max<int64_t>(
            0, static_cast<int64_t>(sampled.size()) - 1);
        mfq_tensor_backend::Tensor codes;
        if (returned > 0) {
            codes = mfq_tensor_backend::stack(sampled, 1)
                .narrow(1, 0, returned)
                .unsqueeze(-1).contiguous();
        } else {
            codes = mfq_tensor_backend::empty(
                {1, 0, 1},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt64));
        }
        co_yield ChunkResult{codes, finished};
    }

    ChunkResult generate_duplex_chunk(mfq_tensor_backend::Tensor condition, int64_t maximum,
        int64_t minimum, int64_t eos, double temperature, double penalty) {
        return mfq::finish_steps(generate_duplex_chunk_steps(std::move(condition), maximum, minimum, eos,
            temperature, penalty));
    }

    mfq_tensor_backend::Tensor generate_official(
        mfq_tensor_backend::Tensor condition_embeddings, int64_t steps, int64_t eos_token = 6561,
        int64_t minimum_steps = 50, double temperature = 0.8, double top_p = 0.85,
        int64_t top_k = 25, double repetition_penalty = 1.05,
        std::vector<mfq_tensor_backend::Tensor> *logits_trace = nullptr, double min_p = 0.0,
        std::mt19937 *evaluator_rng = nullptr, int64_t evaluator_min_keep = 0) {
        const mfq::models::minicpmo45::TtsSampling config{eos_token,          minimum_steps, top_k,
                                                          evaluator_min_keep, temperature,   top_p,
                                                          repetition_penalty, min_p};
        config.validate(steps, 6562);
        if (condition_embeddings.dim() != 3 || condition_embeddings.size(0) != 1)
            throw std::runtime_error("official MiniCPM-o TTS generation requires batch size one");
        reset(1);
        using Tensor = mfq_tensor_backend::Tensor;
        std::vector<Tensor> generated;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        MiniCPMO45TtsSamplingOps sampling_ops{evaluator_rng};
        const auto result = mfq::engine::generate_tokens(
            steps,
            [&](int64_t step) {
                auto hidden = hidden_forward(current);
                auto raw_step_logits =
                    logits(hidden.index({Slice(), -1, Slice()})).to(mfq_tensor_backend::kFloat32);
                if (logits_trace)
                    logits_trace->push_back(raw_step_logits.clone());
                return mfq::models::minicpmo45::sample_tts(
                    sampling_ops, std::move(raw_step_logits), config,
                    std::span<const Tensor>(generated), step, evaluator_rng != nullptr);
            },
            [&](const Tensor &token, int64_t) { generated.push_back(token); },
            [&](const Tensor &token) { return token.eq(eos_token).all().item<bool>(); },
            [&](const Tensor &token) {
                current = minicpmo45_embedding(code_embedding, token.unsqueeze(1));
            });
        auto sampled = mfq_tensor_backend::stack(generated, 1);
        return sampled.narrow(1, 0, result.tokens - (result.hit_eos ? 1 : 0))
            .unsqueeze(-1)
            .contiguous();
    }
};
