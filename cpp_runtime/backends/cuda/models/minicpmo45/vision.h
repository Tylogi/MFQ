#pragma once
#include "models/common/transformer_layer.h"
#include "models/minicpmo45/causal_lm.h"

#include "architecture.h"

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

inline mfq_tensor_backend::Tensor load_dense_native_gpu(CudaExecutionContext &execution,
                                                        const mfq::ModelSource &mfq,
                                                        const std::string &name) {
    MfqCudaGuard guard(active_weight_load_device(execution));
    const auto &record = require_tensor(mfq, name);
    const auto blob = read_tensor(mfq, name);
    size_t offset = 0;
    const uint32_t dimensions = read_u32_from(blob, offset);
    std::vector<int64_t> shape(dimensions);
    for (uint32_t index = 0; index < dimensions; ++index) {
        shape[index] = read_i64_from(blob, offset);
    }
    mfq_tensor_backend::Tensor value;
    if (record.dtype == "BF16") {
        value = mfq_tensor_backend::from_blob(
                    const_cast<uint8_t *>(blob.data()) + offset, shape,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16))
                    .clone();
    } else if (record.dtype == "F16") {
        value = mfq_tensor_backend::from_blob(
                    const_cast<uint8_t *>(blob.data()) + offset, shape,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16))
                    .clone();
    } else if (record.dtype == "F32") {
        value = mfq_tensor_backend::from_blob(
                    const_cast<uint8_t *>(blob.data()) + offset, shape,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
                    .clone();
    } else if (record.dtype == "I64") {
        value = mfq_tensor_backend::from_blob(
                    const_cast<uint8_t *>(blob.data()) + offset, shape,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64))
                    .clone();
    } else if (record.dtype == "I32") {
        value = mfq_tensor_backend::from_blob(
                    const_cast<uint8_t *>(blob.data()) + offset, shape,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
                    .clone();
    } else {
        throw std::runtime_error("MiniCPM-o dense tensor has unsupported dtype: " + name +
                                 " dtype=" + record.dtype);
    }
    return value.to(mfq_tensor_backend::kCUDA).contiguous();
}

inline void minicpmo45_write_pickle_tensor(const mfq_tensor_backend::Tensor &value,
                                           const std::string &path) {
    auto bytes =
        mfq_tensor_backend::pickle_save(value.detach().to(mfq_tensor_backend::kCPU).contiguous());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create MiniCPM-o tensor file: " + path);
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("failed to write MiniCPM-o tensor file: " + path);
    }
}

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

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
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
        return mfq::models::minicpmo45::vision_encoder(
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
