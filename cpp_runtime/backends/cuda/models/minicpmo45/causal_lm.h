#pragma once

#include "../../engine/causal_lm.h"
#include "../../engine/cuda_transformer.h"
#include "models/include/minicpmo45.h"
#include "mfq_cuda_ops.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline constexpr const char * MINICPMO45_RESAMPLER_POS_EMBED_ASSET =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";

struct MiniCPMO45Linear {
    QuantLinear weight;
    mfq_tensor_backend::Tensor bias;

    static MiniCPMO45Linear load(
            const mfq::ModelSource & mfq,
            const std::string & prefix,
            bool with_bias = true) {
        MiniCPMO45Linear result;
        result.weight = load_quant_linear(mfq, prefix + ".weight");
        const std::string bias_name = prefix + ".bias";
        if (with_bias && has_tensor(mfq, bias_name)) {
            result.bias = load_dense_gpu(mfq, bias_name);
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input) const {
        const auto output_dtype = input.scalar_type();
        if (weight.is_dense()) {
            const auto dense_dtype = weight.dense.scalar_type();
            std::optional<mfq_tensor_backend::Tensor> dense_bias = std::nullopt;
            if (bias.defined()) {
                dense_bias = bias.to(dense_dtype);
            }
            return mfq_linear(
                input.to(dense_dtype), weight.dense, dense_bias)
                .to(output_dtype)
                .contiguous();
        }
        auto output = weight.forward(input);
        if (bias.defined()) {
            output = output + bias.to(output.scalar_type());
        }
        return output.to(output_dtype).contiguous();
    }
};

inline mfq_tensor_backend::Tensor load_dense_native_gpu(
        const mfq::ModelSource & mfq,
        const std::string & name) {
    MfqCudaGuard guard(active_weight_load_device());
    const auto & record = require_tensor(mfq, name);
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
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)).clone();
    } else if (record.dtype == "F16") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat16)).clone();
    } else if (record.dtype == "F32") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32)).clone();
    } else if (record.dtype == "I64") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone();
    } else if (record.dtype == "I32") {
        value = mfq_tensor_backend::from_blob(
            const_cast<uint8_t *>(blob.data()) + offset, shape,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32)).clone();
    } else {
        throw std::runtime_error(
            "MiniCPM-o dense tensor has unsupported dtype: " +
            name + " dtype=" + record.dtype);
    }
    return value.to(mfq_tensor_backend::kCUDA).contiguous();
}

inline void minicpmo45_write_pickle_tensor(
        const mfq_tensor_backend::Tensor & value,
        const std::string & path) {
    auto bytes = mfq_tensor_backend::pickle_save(
        value.detach().to(mfq_tensor_backend::kCPU).contiguous());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to create MiniCPM-o tensor file: " + path);
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error(
            "failed to write MiniCPM-o tensor file: " + path);
    }
}

inline mfq_tensor_backend::Tensor minicpmo45_embedding(
        const QuantLinear & embedding,
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::ScalarType output_dtype = mfq_tensor_backend::kBFloat16) {
    ids = ids.contiguous().to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64);
    auto output = quant_embedding_lookup(embedding, ids);
    return output.to(output_dtype).contiguous();
}

inline mfq_tensor_backend::Tensor minicpmo45_layer_norm(
        mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor & weight,
        const mfq_tensor_backend::Tensor & bias,
        double eps) {
    const auto dtype = input.scalar_type();
    auto output = mfq_tensor_backend::layer_norm(
        input, {input.size(-1)},
        weight.to(dtype), bias.to(dtype), eps);
    return output.to(dtype).contiguous();
}

inline mfq_tensor_backend::Tensor minicpmo45_attention(
        mfq_tensor_backend::Tensor query,
        mfq_tensor_backend::Tensor key,
        mfq_tensor_backend::Tensor value,
    MfqOptional<mfq_tensor_backend::Tensor> additive_mask,
        double scale) {
    auto scores = mfq_tensor_backend::matmul(
        query,
        key.transpose(-2, -1)) * scale;
    if (additive_mask.has_value()) {
        scores = scores + additive_mask.value().to(scores.scalar_type());
    }
    auto probabilities = mfq_tensor_backend::softmax(
        scores, -1, mfq_tensor_backend::kFloat32).to(value.scalar_type());
    return mfq_tensor_backend::matmul(probabilities, value).contiguous();
}

struct MiniCPMO45VisionAttention {
    MiniCPMO45Linear q;
    MiniCPMO45Linear k;
    MiniCPMO45Linear v;
    MiniCPMO45Linear output;
    int64_t heads = 16;
    int64_t head_dim = 72;

    static MiniCPMO45VisionAttention load(
            const mfq::ModelSource & mfq,
            const std::string & prefix) {
        MiniCPMO45VisionAttention result;
        result.q = MiniCPMO45Linear::load(mfq, prefix + ".query");
        result.k = MiniCPMO45Linear::load(mfq, prefix + ".key");
        result.v = MiniCPMO45Linear::load(mfq, prefix + ".value");
        result.output = MiniCPMO45Linear::load(
            mfq, prefix + ".output");
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](mfq_tensor_backend::Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim})
                .transpose(1, 2).contiguous();
        };
        auto attended = minicpmo45_attention(
            reshape(q.forward(input)),
            reshape(k.forward(input)),
            reshape(v.forward(input)),
            mask, 1.0 / std::sqrt(static_cast<double>(head_dim)));
        attended = attended.transpose(1, 2).reshape(
            {batch, tokens, heads * head_dim});
        return output.forward(attended);
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

    static MiniCPMO45VisionLayer load(
            const mfq::ModelSource & mfq,
            int index) {
        const std::string prefix =
            "vision.block." + std::to_string(index);
        MiniCPMO45VisionLayer result;
        result.attention = MiniCPMO45VisionAttention::load(
            mfq, prefix + ".attention");
        result.norm1_weight = load_dense_native_gpu(
            mfq, prefix + ".norm1.weight");
        result.norm1_bias = load_dense_native_gpu(
            mfq, prefix + ".norm1.bias");
        result.norm2_weight = load_dense_native_gpu(
            mfq, prefix + ".norm2.weight");
        result.norm2_bias = load_dense_native_gpu(
            mfq, prefix + ".norm2.bias");
        result.fc1 = MiniCPMO45Linear::load(mfq, prefix + ".mlp.up");
        result.fc2 = MiniCPMO45Linear::load(mfq, prefix + ".mlp.down");
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask) const {
        auto normalized = minicpmo45_layer_norm(
            input, norm1_weight, norm1_bias, 1e-6);
        auto hidden = input + attention.forward(normalized, mask);
        normalized = minicpmo45_layer_norm(
            hidden, norm2_weight, norm2_bias, 1e-6);
        auto mlp = fc2.forward(mfq_tensor_backend::gelu(fc1.forward(normalized), "tanh"));
        return (hidden + mlp).contiguous();
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

    static MiniCPMO45VisionEncoder load(const mfq::ModelSource & mfq) {
        MiniCPMO45VisionEncoder result;
        result.patch_weight = load_dense_native_gpu(
            mfq, "vision.patch_embedding.weight");
        result.patch_bias = load_dense_native_gpu(
            mfq, "vision.patch_embedding.bias");
        result.position_embedding = load_dense_native_gpu(
            mfq, "vision.position_embedding.weight");
        result.post_norm_weight = load_dense_native_gpu(
            mfq, "vision.output_norm.weight");
        result.post_norm_bias = load_dense_native_gpu(
            mfq, "vision.output_norm.bias");
        result.layers.reserve(27);
        for (int index = 0; index < 27; ++index) {
            result.layers.push_back(
                MiniCPMO45VisionLayer::load(mfq, index));
        }
        if (result.patch_weight.sizes().vec() !=
                std::vector<int64_t>({1152, 3, 14, 14}) ||
            result.position_embedding.sizes().vec() !=
                std::vector<int64_t>({4900, 1152})) {
            throw std::runtime_error(
                "MiniCPM-o SigLIP tensor shapes disagree with version 4.5");
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor pixels,
            mfq_tensor_backend::Tensor patch_mask,
            mfq_tensor_backend::Tensor target_sizes) const {
        if (patch_mask.dim() == 3) {
            patch_mask = patch_mask.flatten(1);
        }
        if (pixels.dim() != 4 || pixels.size(1) != 3 ||
                patch_mask.dim() != 2 || target_sizes.dim() != 2 ||
                target_sizes.size(1) != 2 ||
                pixels.size(0) != patch_mask.size(0) ||
                pixels.size(0) != target_sizes.size(0) ||
                target_sizes.device().is_cuda()) {
            throw std::runtime_error(
                "MiniCPM-o vision input geometry is invalid");
        }
        auto embedded = mfq_tensor_backend::conv2d(
            pixels.to(patch_weight.scalar_type()),
            patch_weight, patch_bias,
            std::vector<int64_t>{patch_size, patch_size},
            std::vector<int64_t>{0, 0},
            std::vector<int64_t>{1, 1}, 1)
            .flatten(2).transpose(1, 2).contiguous();
        if (embedded.size(1) != patch_mask.size(1)) {
            throw std::runtime_error(
                "MiniCPM-o patch mask length does not match patch convolution");
        }
        std::vector<int64_t> position_ids(
            static_cast<size_t>(embedded.size(0) * embedded.size(1)), 0);
        auto sizes = target_sizes.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        auto mask_cpu = patch_mask.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kBool).contiguous();
        const auto * size_data = sizes.data_ptr<int64_t>();
        const auto * mask_data = mask_cpu.data_ptr<bool>();
        bool all_patches_active = true;
        for (int64_t batch = 0; batch < embedded.size(0); ++batch) {
            const int64_t height = size_data[2 * batch];
            const int64_t width = size_data[2 * batch + 1];
            if (height <= 0 || width <= 0 ||
                    height * width > embedded.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o target patch size is invalid");
            }
            int64_t active = 0;
            for (int64_t patch = 0; patch < embedded.size(1); ++patch) {
                if (!mask_data[batch * embedded.size(1) + patch]) {
                    all_patches_active = false;
                    continue;
                }
                if (active >= height * width) {
                    throw std::runtime_error(
                        "MiniCPM-o patch mask has too many active entries");
                }
                const int64_t row = active / width;
                const int64_t column = active % width;
                const int64_t bucket_column = std::min<int64_t>(
                    position_side - 1, column * position_side / width);
                const int64_t bucket_row = std::min<int64_t>(
                    position_side - 1, row * position_side / height);
                position_ids[static_cast<size_t>(
                    batch * embedded.size(1) + patch)] =
                    bucket_row * position_side + bucket_column;
                ++active;
            }
            if (active != height * width) {
                throw std::runtime_error(
                    "MiniCPM-o patch mask active count disagrees with target size");
            }
        }
        auto ids = mfq_tensor_backend::from_blob(
            position_ids.data(),
            std::vector<int64_t>{embedded.size(0), embedded.size(1)},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).clone().to(mfq_tensor_backend::kCUDA);
        embedded = embedded + position_embedding.index_select(
            0, ids.reshape({-1})).reshape(embedded.sizes());
        MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt;
        if (!all_patches_active) {
            auto invalid = patch_mask.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool)
                .logical_not().unsqueeze(1).unsqueeze(2);
            attention_mask = mfq_tensor_backend::zeros(
                {embedded.size(0), 1, embedded.size(1), embedded.size(1)},
                embedded.options().dtype(mfq_tensor_backend::kFloat32))
                .masked_fill(invalid, -std::numeric_limits<float>::infinity());
        }
        for (const auto & layer : layers) {
            embedded = layer.forward(embedded, attention_mask);
        }
        embedded = minicpmo45_layer_norm(
            embedded, post_norm_weight, post_norm_bias, 1e-6);
        return embedded;
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

    static MiniCPMO45Resampler load(const mfq::ModelSource & mfq) {
        MiniCPMO45Resampler result;
        if (!mfq.has_asset(MINICPMO45_RESAMPLER_POS_EMBED_ASSET)) {
            throw std::runtime_error(
                "MiniCPM-o MFQ is missing the exact Resampler position asset; "
                "repack it with the current runtime assets tool");
        }
        auto position_blob = read_asset(mfq,
            MINICPMO45_RESAMPLER_POS_EMBED_ASSET);
        constexpr size_t position_header_bytes = 20;
        constexpr size_t position_value_bytes =
            static_cast<size_t>(70) * 70 * 4096 * sizeof(uint16_t);
        if (position_blob.size() !=
                position_header_bytes + position_value_bytes ||
            std::memcmp(position_blob.data(), "MFQRSPB1", 8) != 0) {
            throw std::runtime_error(
                "MiniCPM-o Resampler position asset has an invalid format");
        }
        size_t position_offset = 8;
        const uint32_t position_height = read_u32_from(
            position_blob, position_offset);
        const uint32_t position_width = read_u32_from(
            position_blob, position_offset);
        const uint32_t position_dim = read_u32_from(
            position_blob, position_offset);
        if (position_height != 70 || position_width != 70 ||
                position_dim != 4096 ||
                position_offset != position_header_bytes) {
            throw std::runtime_error(
                "MiniCPM-o Resampler position asset shape is not 70x70x4096");
        }
        {
            MfqCudaGuard guard(active_weight_load_device());
            result.position_embedding = mfq_tensor_backend::from_blob(
                position_blob.data() + position_offset,
                std::vector<int64_t>{70, 70, 4096},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16))
                .clone().to(mfq_tensor_backend::kCUDA).contiguous();
        }
        result.query = load_dense_native_gpu(mfq, "vision.resampler.query");
        result.kv_projection = MiniCPMO45Linear::load(
            mfq, "vision.resampler.key_value", false);
        result.q_norm_weight = load_dense_native_gpu(
            mfq, "vision.resampler.query_norm.weight");
        result.q_norm_bias = load_dense_native_gpu(
            mfq, "vision.resampler.query_norm.bias");
        result.kv_norm_weight = load_dense_native_gpu(
            mfq, "vision.resampler.key_value_norm.weight");
        result.kv_norm_bias = load_dense_native_gpu(
            mfq, "vision.resampler.key_value_norm.bias");
        result.post_norm_weight = load_dense_native_gpu(
            mfq, "vision.resampler.output_norm.weight");
        result.post_norm_bias = load_dense_native_gpu(
            mfq, "vision.resampler.output_norm.bias");
        result.in_projection_weight = load_dense_native_gpu(
            mfq, "vision.resampler.attention.qkv.weight");
        result.in_projection_bias = load_dense_native_gpu(
            mfq, "vision.resampler.attention.qkv.bias");
        result.out_projection_weight = load_dense_native_gpu(
            mfq, "vision.resampler.attention.output.weight");
        result.out_projection_bias = load_dense_native_gpu(
            mfq, "vision.resampler.attention.output.bias");
        result.final_projection = load_dense_native_gpu(
            mfq, "vision.resampler.output.weight");
        if (result.query.sizes().vec() !=
                std::vector<int64_t>({64, 4096}) ||
            result.in_projection_weight.sizes().vec() !=
                std::vector<int64_t>({12288, 4096})) {
            throw std::runtime_error(
                "MiniCPM-o Resampler tensor shapes disagree with version 4.5");
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            mfq_tensor_backend::Tensor target_sizes) const {
        if (input.dim() != 3 || input.size(2) != 1152 ||
                target_sizes.dim() != 2 || target_sizes.size(1) != 2 ||
                target_sizes.size(0) != input.size(0) ||
                target_sizes.device().is_cuda() ||
                input.scalar_type() != mfq_tensor_backend::kBFloat16) {
            throw std::runtime_error(
                "MiniCPM-o Resampler input geometry is invalid");
        }
        const int64_t batch = input.size(0);
        const int64_t length = input.size(1);
        auto sizes = target_sizes.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        const auto * size_data = sizes.data_ptr<int64_t>();
        std::vector<mfq_tensor_backend::Tensor> positions;
        positions.reserve(static_cast<size_t>(batch));
        std::vector<uint8_t> key_padding(
            static_cast<size_t>(batch * length), uint8_t{1});
        for (int64_t index = 0; index < batch; ++index) {
            const int64_t height = size_data[2 * index];
            const int64_t width = size_data[2 * index + 1];
            const int64_t patches = height * width;
            if (height <= 0 || width <= 0 ||
                    height > position_embedding.size(0) ||
                    width > position_embedding.size(1) ||
                    patches > length) {
                throw std::runtime_error(
                    "MiniCPM-o Resampler target size is invalid");
            }
            auto position = position_embedding
                .narrow(0, 0, height)
                .narrow(1, 0, width)
                .reshape({patches, 4096});
            if (patches < length) {
                position = mfq_tensor_backend::cat({
                    position,
                    mfq_tensor_backend::zeros(
                        {length - patches, 4096}, position.options())}, 0);
            }
            positions.push_back(position);
            for (int64_t patch = 0; patch < patches; ++patch) {
                key_padding[static_cast<size_t>(index * length + patch)] = 0;
            }
        }
        auto position = mfq_tensor_backend::stack(positions, 0);
        auto key_padding_mask = mfq_tensor_backend::from_blob(
            key_padding.data(), std::vector<int64_t>{batch, length},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8)).clone()
            .to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kBool);
        auto kv = kv_projection.forward(input);
        kv = minicpmo45_layer_norm(
            kv, kv_norm_weight, kv_norm_bias, 1e-6);
        auto normalized_query = minicpmo45_layer_norm(
            query, q_norm_weight, q_norm_bias, 1e-6);
        auto repeated_query = normalized_query.unsqueeze(0)
            .expand({batch, normalized_query.size(0), normalized_query.size(1)})
            .contiguous();

        const auto dtype = input.scalar_type();
        auto project = [&](mfq_tensor_backend::Tensor value, int64_t offset) {
            auto weight = in_projection_weight.narrow(0, offset, 4096);
            auto bias = in_projection_bias.narrow(0, offset, 4096);
            return mfq_linear(
                value.to(weight.scalar_type()), weight, bias)
                .to(dtype).contiguous();
        };
        auto q = project(repeated_query, 0)
            .reshape({batch, 64, heads, head_dim})
            .transpose(1, 2).contiguous();
        auto k = project(kv + position, 4096)
            .reshape({batch, length, heads, head_dim})
            .transpose(1, 2).contiguous();
        auto v = project(kv, 8192)
            .reshape({batch, length, heads, head_dim})
            .transpose(1, 2).contiguous();
        auto mask = mfq_tensor_backend::zeros(
            {batch, 1, 1, length},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(dtype))
            .masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -std::numeric_limits<float>::infinity())
            .expand({batch, heads, 1, length})
            .reshape({batch * heads, 1, length});
        auto q_flat = q.reshape({batch * heads, 64, head_dim});
        auto k_flat = k.reshape({batch * heads, length, head_dim});
        auto v_flat = v.reshape({batch * heads, length, head_dim});
        auto attention_weights = mfq_tensor_backend::baddbmm(
            mask,
            q_flat * std::sqrt(1.0 / static_cast<double>(head_dim)),
            k_flat.transpose(1, 2));
        attention_weights = mfq_tensor_backend::softmax(attention_weights, -1);
        auto attended = mfq_tensor_backend::bmm(attention_weights, v_flat)
            .reshape({batch, heads, 64, head_dim})
            .transpose(1, 2).reshape({batch, 64, 4096});
        attended = mfq_linear(
            attended.to(out_projection_weight.scalar_type()),
            out_projection_weight, out_projection_bias)
            .to(dtype).contiguous();
        attended = minicpmo45_layer_norm(
            attended, post_norm_weight, post_norm_bias, 1e-6);
        return mfq_tensor_backend::matmul(
            attended.to(final_projection.scalar_type()),
            final_projection).to(dtype).contiguous();
    }
};

struct MiniCPMO45WhisperAttention {
    MiniCPMO45Linear q;
    MiniCPMO45Linear k;
    MiniCPMO45Linear v;
    MiniCPMO45Linear output;
    int64_t heads = 16;
    int64_t head_dim = 64;

    static MiniCPMO45WhisperAttention load(
            const mfq::ModelSource & mfq,
            const std::string & prefix) {
        MiniCPMO45WhisperAttention result;
        result.q = MiniCPMO45Linear::load(mfq, prefix + ".query");
        result.k = MiniCPMO45Linear::load(
            mfq, prefix + ".key", false);
        result.v = MiniCPMO45Linear::load(mfq, prefix + ".value");
        result.output = MiniCPMO45Linear::load(
            mfq, prefix + ".output");
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask,
            mfq_tensor_backend::Tensor * key_cache = nullptr,
            mfq_tensor_backend::Tensor * value_cache = nullptr) const {
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto reshape = [&](mfq_tensor_backend::Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim})
                .transpose(1, 2).contiguous();
        };
        auto query_projection = q.forward(input);
        auto key_projection = k.forward(input);
        auto value_projection = v.forward(input);
        auto query = reshape(query_projection);
        auto key = reshape(key_projection);
        auto value = reshape(value_projection);
        if (key_cache != nullptr && value_cache != nullptr) {
            if (key_cache->defined()) {
                key = mfq_tensor_backend::cat({*key_cache, key}, 2).contiguous();
                value = mfq_tensor_backend::cat({*value_cache, value}, 2).contiguous();
            }
            *key_cache = key;
            *value_cache = value;
        }
        std::optional<mfq_tensor_backend::Tensor> attention_mask = std::nullopt;
        if (mask.has_value()) {
            attention_mask = mask.value().to(query.scalar_type());
        }
        auto attended = mfq_scaled_dot_product_attention(
            query, key, value, attention_mask,
            0.0, false, std::nullopt, false).contiguous();
        attended = attended.transpose(1, 2).reshape(
            {batch, tokens, heads * head_dim});
        return output.forward(attended);
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

    static MiniCPMO45WhisperLayer load(
            const mfq::ModelSource & mfq,
            int index) {
        const std::string prefix =
            "audio.block." + std::to_string(index);
        MiniCPMO45WhisperLayer result;
        result.attention = MiniCPMO45WhisperAttention::load(
            mfq, prefix + ".attention");
        result.attention_norm_weight = load_dense_native_gpu(
            mfq, prefix + ".attention.norm.weight");
        result.attention_norm_bias = load_dense_native_gpu(
            mfq, prefix + ".attention.norm.bias");
        result.final_norm_weight = load_dense_native_gpu(
            mfq, prefix + ".mlp.norm.weight");
        result.final_norm_bias = load_dense_native_gpu(
            mfq, prefix + ".mlp.norm.bias");
        result.fc1 = MiniCPMO45Linear::load(mfq, prefix + ".mlp.up");
        result.fc2 = MiniCPMO45Linear::load(mfq, prefix + ".mlp.down");
        return result;
    }

    void reset() {
        key_cache = mfq_tensor_backend::Tensor();
        value_cache = mfq_tensor_backend::Tensor();
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask,
            bool use_cache) {
        auto normalized = minicpmo45_layer_norm(
            input, attention_norm_weight, attention_norm_bias, 1e-5);
        auto attended = use_cache
            ? attention.forward(
                normalized, mask, &key_cache, &value_cache)
            : attention.forward(normalized, mask);
        auto hidden = (input + attended).contiguous();
        normalized = minicpmo45_layer_norm(
            hidden, final_norm_weight, final_norm_bias, 1e-5);
        auto fc1_output = fc1.forward(normalized);
        auto activation = mfq_tensor_backend::gelu(fc1_output);
        auto feed_forward = fc2.forward(activation);
        return (hidden + feed_forward).contiguous();
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

    static MiniCPMO45AudioEncoder load(const mfq::ModelSource & mfq) {
        MiniCPMO45AudioEncoder result;
        result.conv1_weight = load_dense_native_gpu(
            mfq, "audio.patch_embedding.conv1.weight");
        result.conv1_bias = load_dense_native_gpu(
            mfq, "audio.patch_embedding.conv1.bias");
        result.conv2_weight = load_dense_native_gpu(
            mfq, "audio.patch_embedding.conv2.weight");
        result.conv2_bias = load_dense_native_gpu(
            mfq, "audio.patch_embedding.conv2.bias");
        result.position_embedding = load_dense_native_gpu(
            mfq, "audio.position_embedding.weight");
        result.final_norm_weight = load_dense_native_gpu(
            mfq, "audio.output_norm.weight");
        result.final_norm_bias = load_dense_native_gpu(
            mfq, "audio.output_norm.bias");
        result.projector1 = MiniCPMO45Linear::load(
            mfq, "audio.projector.input");
        result.projector2 = MiniCPMO45Linear::load(
            mfq, "audio.projector.output");
        result.layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            result.layers.push_back(
                MiniCPMO45WhisperLayer::load(mfq, index));
        }
        if (result.conv1_weight.sizes().vec() !=
                std::vector<int64_t>({1024, 80, 3}) ||
            result.conv2_weight.sizes().vec() !=
                std::vector<int64_t>({1024, 1024, 3}) ||
            result.position_embedding.sizes().vec() !=
                std::vector<int64_t>({1500, 1024})) {
            throw std::runtime_error(
                "MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
        return result;
    }

    void reset() {
        for (auto & layer : layers) layer.reset();
    }

    int64_t cache_length() const {
        if (layers.empty() || !layers.front().key_cache.defined()) return 0;
        return layers.front().key_cache.size(2);
    }

    mfq_tensor_backend::Tensor forward_streaming(
            mfq_tensor_backend::Tensor features,
            int64_t prefix_extra_frames,
            int64_t suffix_extra_frames) {
        if (features.dim() != 3 || features.size(0) != 1 ||
                features.size(1) != 80 || prefix_extra_frames < 0 ||
                suffix_extra_frames < 0) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio expects [1,80,frames] and "
                "non-negative extra-frame counts");
        }
        const int64_t conv_tokens_before_crop =
            (features.size(2) - 1) / 2 + 1;
        if (cache_length() + conv_tokens_before_crop >=
                position_embedding.size(0)) {
            reset();
        }

        auto conv1 = mfq_tensor_backend::conv1d(
            features.to(conv1_weight.scalar_type()),
            conv1_weight, conv1_bias,
            std::vector<int64_t>{1},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        auto hidden = mfq_tensor_backend::gelu(conv1);
        auto conv2 = mfq_tensor_backend::conv1d(
            hidden, conv2_weight, conv2_bias,
            std::vector<int64_t>{2},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        hidden = mfq_tensor_backend::gelu(conv2);

        const int64_t prefix_to_remove =
            prefix_extra_frames > 0 ? (prefix_extra_frames + 1) / 2 : 0;
        const int64_t suffix_to_remove =
            suffix_extra_frames > 0 ? (suffix_extra_frames + 1) / 2 : 0;
        if (prefix_to_remove + suffix_to_remove >= hidden.size(2)) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio extra context removes every frame");
        }
        if (prefix_to_remove > 0) {
            hidden = hidden.narrow(
                2, prefix_to_remove, hidden.size(2) - prefix_to_remove);
        }
        if (suffix_to_remove > 0) {
            hidden = hidden.narrow(
                2, 0, hidden.size(2) - suffix_to_remove);
        }
        hidden = hidden.transpose(1, 2).contiguous();

        const int64_t past = cache_length();
        const int64_t tokens = hidden.size(1);
        if (past + tokens > position_embedding.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
        }
        hidden = hidden + position_embedding.narrow(0, past, tokens);
        auto attention_mask = mfq_tensor_backend::zeros(
            {1, 1, tokens, past + tokens},
            hidden.options());
        for (auto & layer : layers) {
            hidden = layer.forward(hidden, attention_mask, true);
        }
        hidden = minicpmo45_layer_norm(
            hidden, final_norm_weight, final_norm_bias, 1e-5);
        hidden = projector1.forward(hidden);
        hidden = mfq_tensor_backend::relu(hidden);
        hidden = projector2.forward(hidden);
        if (hidden.size(1) < 5) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio chunk is too short for stride-5 pooling");
        }
        return mfq_tensor_backend::avg_pool1d(
            hidden.transpose(1, 2),
            std::vector<int64_t>{5},
            std::vector<int64_t>{5})
            .transpose(1, 2).contiguous();
    }

    static std::vector<int64_t> pooled_lengths(
            mfq_tensor_backend::Tensor raw_lengths) {
        auto lengths = raw_lengths.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
        const auto * values = lengths.data_ptr<int64_t>();
        std::vector<int64_t> result(static_cast<size_t>(lengths.numel()));
        for (int64_t index = 0; index < lengths.numel(); ++index) {
            const int64_t after_conv = (values[index] - 1) / 2 + 1;
            result[static_cast<size_t>(index)] =
                (after_conv - 5) / 5 + 1;
            if (values[index] <= 0 || result[static_cast<size_t>(index)] <= 0) {
                throw std::runtime_error(
                    "MiniCPM-o audio length is too short");
            }
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor features,
            mfq_tensor_backend::Tensor raw_lengths,
            bool use_cache = false) {
        if (features.dim() != 3 || features.size(1) != 80 ||
                raw_lengths.dim() != 1 ||
                raw_lengths.size(0) != features.size(0)) {
            throw std::runtime_error(
                "MiniCPM-o audio input geometry is invalid");
        }
        auto conv1 = mfq_tensor_backend::conv1d(
            features.to(conv1_weight.scalar_type()),
            conv1_weight, conv1_bias,
            std::vector<int64_t>{1},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        auto hidden = mfq_tensor_backend::gelu(conv1);
        auto conv2 = mfq_tensor_backend::conv1d(
            hidden, conv2_weight, conv2_bias,
            std::vector<int64_t>{2},
            std::vector<int64_t>{1},
            std::vector<int64_t>{1}, 1);
        hidden = mfq_tensor_backend::gelu(conv2);
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
        auto length_tensor = raw_lengths.to(
            mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        auto key_positions = mfq_tensor_backend::arange(
            past + tokens,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kInt64));
        auto valid_keys = key_positions.unsqueeze(0) <
            (length_tensor + past).unsqueeze(1);
        auto query_positions = mfq_tensor_backend::arange(
            past, past + tokens,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kInt64));
        // Official MiniCPM-o 4.5 uses audio_chunk_length=1.0. Whisper emits
        // 50 encoder frames per second, and every query can see its current
        // 50-frame chunk plus all preceding chunks.
        auto chunk_visible = key_positions.unsqueeze(0) <
            ((query_positions / 50 + 1) * 50).unsqueeze(1);
        auto visible = valid_keys.unsqueeze(1) & chunk_visible.unsqueeze(0);
        auto attention_mask = mfq_tensor_backend::zeros(
            {features.size(0), 1, tokens, past + tokens},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                .dtype(mfq_tensor_backend::kFloat32))
            .masked_fill(
                visible.logical_not().unsqueeze(1),
                -std::numeric_limits<float>::infinity());
        for (auto & layer : layers) {
            hidden = layer.forward(hidden, attention_mask, use_cache);
        }
        hidden = minicpmo45_layer_norm(
            hidden, final_norm_weight, final_norm_bias, 1e-5);
        hidden = projector1.forward(hidden);
        hidden = mfq_tensor_backend::relu(hidden);
        hidden = projector2.forward(hidden);
        hidden = mfq_tensor_backend::avg_pool1d(
            hidden.transpose(1, 2),
            std::vector<int64_t>{5},
            std::vector<int64_t>{5})
            .transpose(1, 2).contiguous();
        return hidden;
    }
};

struct MiniCPMO45TtsDecoder {
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

    static mfq::models::ModelConfig make_config() {
        mfq::models::ModelConfig result;
        result.model_type = "minicpmtts";
        result.hidden_size = 768;
        result.intermediate_size = 3072;
        result.num_hidden_layers = 20;
        result.num_attention_heads = 12;
        result.num_key_value_heads = 12;
        result.head_dim = 64;
        result.rotary_dim = 64;
        result.max_position_embeddings = 4096;
        result.rope_base = 10000.0;
        result.rms_norm_eps = 1e-6;
        result.layer_types.assign(20, "full_attention");
        return result;
    }

    static MiniCPMO45TtsDecoder load(const mfq::ModelSource & mfq) {
        MiniCPMO45TtsDecoder result;
        result.config = make_config();
        result.rope = RopeCache(
            result.config.max_position_embeddings,
            result.config.rotary_dim,
            result.config.rope_base);
        result.text_embedding = load_quant_linear(
            mfq, "tts.text_embedding.weight");
        result.code_embedding = load_quant_linear(
            mfq, "tts.code_embedding.0.weight");
        result.semantic_projector1 = MiniCPMO45Linear::load(
            mfq, "tts.semantic_projector.input");
        result.semantic_projector2 = MiniCPMO45Linear::load(
            mfq, "tts.semantic_projector.output");
        result.speaker_projector1 = MiniCPMO45Linear::load(
            mfq, "tts.speaker_projector.input");
        result.speaker_projector2 = MiniCPMO45Linear::load(
            mfq, "tts.speaker_projector.output");
        result.output_norm = load_dense_native_gpu(
            mfq, "tts.output_norm.weight");
        auto head_g = load_dense_native_gpu(
            mfq, "tts.code_output.0.weight_norm.magnitude");
        auto head_v = load_dense_native_gpu(
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
            auto block = load_transformer_block(
                mfq, result.config, index, "full_attention", false, "tts");
            static_cast<FullBlock&>(*block).norm_weight_offset = 0.0;
            block->cuda_device = g_layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch) {
        cache_position = 0;
        for (auto & block : blocks) block->reset(batch);
    }

    mfq_tensor_backend::Tensor semantic_projection(mfq_tensor_backend::Tensor hidden) const {
        auto projected = semantic_projector1.forward(hidden);
        projected = mfq_tensor_backend::relu(projected);
        projected = semantic_projector2.forward(projected);
        auto norm = mfq_tensor_backend::sqrt(
            mfq_tensor_backend::sum(projected.to(mfq_tensor_backend::kFloat32).square(), -1, true))
            .clamp_min(1e-12);
        return (projected / norm.to(projected.scalar_type())).contiguous();
    }

    mfq_tensor_backend::Tensor speaker_projection(mfq_tensor_backend::Tensor hidden) const {
        return speaker_projector2.forward(
            mfq_tensor_backend::relu(speaker_projector1.forward(hidden)));
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
        if (cache_position == 0) reset(batch);
        if (cache_position + tokens > config.max_position_embeddings) {
            throw std::runtime_error(
                "MiniCPM-o TTS context exceeds 4096 tokens");
        }
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
        auto hidden = input_embeddings.to(mfq_tensor_backend::kBFloat16).contiguous();
        for (auto & block : blocks) {
            hidden = block->forward(
                hidden, positions, cache_position,
                sequence_length, rope);
        }
        cache_position += tokens;
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

    ChunkResult generate_duplex_chunk(
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
        return {codes, finished};
    }

    mfq_tensor_backend::Tensor generate_official(
            mfq_tensor_backend::Tensor condition_embeddings,
            int64_t steps,
            int64_t eos_token = 6561,
            int64_t minimum_steps = 50,
            double temperature = 0.8,
            double top_p = 0.85,
            int64_t top_k = 25,
            double repetition_penalty = 1.05,
            std::vector<mfq_tensor_backend::Tensor> * logits_trace = nullptr,
            double min_p = 0.0,
            std::mt19937 * evaluator_rng = nullptr,
            int64_t evaluator_min_keep = 0) {
        if (steps <= 0 || minimum_steps < 0 ||
                eos_token < 0 || eos_token >= 6562 ||
                temperature <= 0.0 || top_p <= 0.0 || top_p > 1.0 ||
                min_p < 0.0 || min_p > 1.0 ||
                top_k < 3 || top_k > 6562 || repetition_penalty <= 0.0 ||
                evaluator_min_keep < 0 || evaluator_min_keep > top_k) {
            throw std::runtime_error(
                "MiniCPM-o TTS generation limits are invalid");
        }
        if (condition_embeddings.size(0) != 1) {
            throw std::runtime_error(
                "official MiniCPM-o TTS generation requires batch size one");
        }
        reset(condition_embeddings.size(0));
        std::vector<mfq_tensor_backend::Tensor> generated;
        bool hit_eos = false;
        generated.reserve(static_cast<size_t>(steps));
        auto current = condition_embeddings;
        for (int64_t step = 0; step < steps; ++step) {
            auto hidden = hidden_forward(current);
            auto raw_step_logits = logits(
                hidden.index({Slice(), -1, Slice()}))
                .to(mfq_tensor_backend::kFloat32);
            if (logits_trace != nullptr) {
                logits_trace->push_back(raw_step_logits.clone());
            }
            auto step_logits = evaluator_rng != nullptr
                ? raw_step_logits / temperature
                : raw_step_logits;
            if (!generated.empty()) {
                if (repetition_penalty != 1.0) {
                    auto counts = mfq_tensor_backend::zeros_like(step_logits);
                    const size_t begin = generated.size() > 16
                        ? generated.size() - 16 : 0;
                    for (size_t index = begin; index < generated.size(); ++index) {
                        counts.scatter_add_(
                            1, generated[index].reshape({1, 1}),
                            mfq_tensor_backend::ones(
                                {1, 1}, counts.options()));
                    }
                    auto alpha = mfq_tensor_backend::pow(
                        mfq_tensor_backend::full_like(counts, repetition_penalty), counts);
                    step_logits = mfq_tensor_backend::where(
                        step_logits < 0,
                        step_logits * alpha,
                        step_logits / alpha);
                }
            }
            if (step < minimum_steps) {
                step_logits.index_put_({Slice(), eos_token},
                    -std::numeric_limits<float>::infinity());
            }
            if (evaluator_rng != nullptr) {
                auto cpu_logits = step_logits
                    .reshape({-1}).to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
                    .contiguous();
                const auto * values = cpu_logits.data_ptr<float>();
                const int64_t vocabulary = cpu_logits.numel();
                float maximum = values[0];
                for (int64_t index = 1; index < vocabulary; ++index) {
                    maximum = std::max(maximum, values[index]);
                }
                std::vector<std::pair<float, int64_t>> probabilities;
                probabilities.reserve(static_cast<size_t>(vocabulary));
                float sum = 0.0F;
                for (int64_t index = 0; index < vocabulary; ++index) {
                    const float probability = std::exp(values[index] - maximum);
                    probabilities.emplace_back(probability, index);
                    sum += probability;
                }
                for (auto & probability : probabilities) {
                    probability.first /= sum;
                }
                std::sort(probabilities.begin(), probabilities.end(),
                    [](const auto & left, const auto & right) {
                        return left.first > right.first;
                    });
                std::vector<float> kept_probabilities;
                std::vector<int64_t> kept_indices;
                kept_probabilities.reserve(static_cast<size_t>(top_k));
                kept_indices.reserve(static_cast<size_t>(top_k));
                float cumulative = 0.0F;
                for (const auto & probability : probabilities) {
                    if (static_cast<int64_t>(kept_probabilities.size()) <
                            evaluator_min_keep ||
                            (cumulative < static_cast<float>(top_p) &&
                             static_cast<int64_t>(kept_probabilities.size()) <
                                 top_k)) {
                        cumulative += probability.first;
                        kept_probabilities.push_back(probability.first);
                        kept_indices.push_back(probability.second);
                    } else {
                        break;
                    }
                }
                float kept_sum = std::accumulate(
                    kept_probabilities.begin(), kept_probabilities.end(), 0.0F);
                for (auto & probability : kept_probabilities) {
                    probability /= kept_sum;
                }
                std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
                const float sample = distribution(*evaluator_rng);
                float selected_sum = 0.0F;
                int64_t selected = kept_indices.back();
                for (size_t index = 0; index < kept_probabilities.size(); ++index) {
                    selected_sum += kept_probabilities[index];
                    if (sample <= selected_sum) {
                        selected = kept_indices[index];
                        break;
                    }
                }
                auto token = mfq_tensor_backend::tensor(
                    std::vector<int64_t>{selected},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                        .dtype(mfq_tensor_backend::kInt64));
                generated.push_back(token);
                if (token.eq(eos_token).all().item<bool>()) {
                    hit_eos = true;
                    break;
                }
                current = minicpmo45_embedding(
                    code_embedding, token.unsqueeze(1));
                continue;
            }
            if (top_k < step_logits.size(1)) {
                auto top_values = std::get<0>(
                    mfq_tensor_backend::topk(
                        step_logits, top_k, -1, true, true));
                auto threshold = top_values.select(1, top_k - 1)
                    .unsqueeze(1);
                step_logits = step_logits.masked_fill(
                    step_logits < threshold,
                    -std::numeric_limits<float>::infinity());
            }
            if (top_p < 1.0) {
                auto sorted = mfq_tensor_backend::sort(
                    step_logits, -1, true);
                auto sorted_logits = std::get<0>(sorted);
                auto sorted_indices = std::get<1>(sorted);
                auto cumulative = mfq_tensor_backend::cumsum(
                    mfq_tensor_backend::softmax(sorted_logits, -1), -1);
                auto keep = cumulative <= top_p;
                keep.narrow(1, 1, keep.size(1) - 1).copy_(
                    keep.narrow(1, 0, keep.size(1) - 1).clone());
                keep.index_put_({Slice(), 0}, true);
                sorted_logits = sorted_logits.masked_fill(
                    keep.logical_not(),
                    -std::numeric_limits<float>::infinity());
                step_logits = mfq_tensor_backend::full_like(
                    step_logits,
                    -std::numeric_limits<float>::infinity());
                step_logits.scatter_(1, sorted_indices, sorted_logits);
            }
            if (min_p > 0.0) {
                auto max_logits = std::get<0>(
                    mfq_tensor_backend::max(step_logits, -1, true));
                auto remove =
                    step_logits < max_logits + std::log(min_p);
                step_logits = step_logits.masked_fill(
                    remove,
                    -std::numeric_limits<float>::infinity());
            }
            step_logits = step_logits / temperature;
            auto token = mfq_tensor_backend::multinomial(
                mfq_tensor_backend::softmax(step_logits, -1), 1)
                .reshape({1}).to(mfq_tensor_backend::kInt64);
            generated.push_back(token);
            if (token.eq(eos_token).all().item<bool>()) {
                hit_eos = true;
                break;
            }
            current = minicpmo45_embedding(
                code_embedding, token.unsqueeze(1));
        }
        auto sampled = mfq_tensor_backend::stack(generated, 1);
        const int64_t returned = sampled.size(1) - (hit_eos ? 1 : 0);
        return sampled.narrow(1, 0, returned)
            .unsqueeze(-1).contiguous();
    }
};

struct MiniCPMO45Bounds {
    int64_t batch = 0;
    int64_t source = 0;
    int64_t begin = 0;
    int64_t end = 0;
};

inline std::vector<MiniCPMO45Bounds> minicpmo45_parse_bounds(
        mfq_tensor_backend::Tensor bounds,
        const char * label) {
    if (!bounds.defined() || bounds.numel() == 0) return {};
    bounds = bounds.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
    if (bounds.dim() != 2 || bounds.size(1) != 4) {
        throw std::runtime_error(
            std::string("MiniCPM-o ") + label +
            " bounds must have shape [count,4]");
    }
    const auto * values = bounds.data_ptr<int64_t>();
    std::vector<MiniCPMO45Bounds> result;
    result.reserve(static_cast<size_t>(bounds.size(0)));
    for (int64_t row = 0; row < bounds.size(0); ++row) {
        MiniCPMO45Bounds bound;
        bound.batch = values[row * 4];
        bound.source = values[row * 4 + 1];
        bound.begin = values[row * 4 + 2];
        bound.end = values[row * 4 + 3];
        if (bound.batch < 0 || bound.source < 0 ||
                bound.begin < 0 || bound.end <= bound.begin) {
            throw std::runtime_error(
                std::string("MiniCPM-o ") + label +
                " bounds contain an invalid row");
        }
        result.push_back(bound);
    }
    return result;
}

struct MiniCPMO45ForwardResult {
    mfq_tensor_backend::Tensor vision_states;
    mfq_tensor_backend::Tensor image_embeddings;
    mfq_tensor_backend::Tensor audio_embeddings;
    mfq_tensor_backend::Tensor input_embeddings;
    mfq_tensor_backend::Tensor hidden_states;
    mfq_tensor_backend::Tensor logits;
};

struct MiniCPMO45Runtime {
    mfq::cuda::MiniCPMO45CausalLm language;
    MiniCPMO45VisionEncoder vision;
    MiniCPMO45Resampler resampler;
    MiniCPMO45AudioEncoder audio;
    MiniCPMO45TtsDecoder tts;

    static MiniCPMO45Runtime load(
            const std::string & model_path,
            const std::string & config_path,
            int64_t context_size) {
        return load_with_language(
            mfq::cuda::load_causal_lm<
                mfq::cuda::CudaBackbone::minicpmo45>(
                    model_path, config_path, context_size));
    }

    static MiniCPMO45Runtime load_with_language(
            mfq::cuda::MiniCPMO45CausalLm language) {
        MiniCPMO45Runtime result;
        result.language = std::move(language);
        g_layer_placement.load_device =
            g_layer_placement.primary_device();
        MfqCudaGuard guard(
            g_layer_placement.primary_device());
        const auto& mfq = *result.language.source;
        result.vision = MiniCPMO45VisionEncoder::load(mfq);
        result.resampler = MiniCPMO45Resampler::load(mfq);
        result.audio = MiniCPMO45AudioEncoder::load(mfq);
        result.tts = MiniCPMO45TtsDecoder::load(mfq);
        return result;
    }

    MiniCPMO45ForwardResult forward(
            mfq_tensor_backend::Tensor input_ids,
            mfq_tensor_backend::Tensor position_ids,
            mfq_tensor_backend::Tensor attention_mask,
            mfq_tensor_backend::Tensor pixels,
            mfq_tensor_backend::Tensor patch_mask,
            mfq_tensor_backend::Tensor target_sizes,
            mfq_tensor_backend::Tensor image_bounds,
            mfq_tensor_backend::Tensor audio_features,
            mfq_tensor_backend::Tensor audio_lengths,
            mfq_tensor_backend::Tensor audio_bounds) {
        if (input_ids.dim() == 1) input_ids = input_ids.unsqueeze(0);
        if (input_ids.dim() != 2) {
            throw std::runtime_error(
                "MiniCPM-o input_ids must have shape [batch,tokens]");
        }
        input_ids = input_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        MiniCPMO45ForwardResult result;
        result.input_embeddings = language.embed_forward(input_ids);
        const auto images = minicpmo45_parse_bounds(
            image_bounds, "image");
        if (!images.empty()) {
            if (!pixels.defined() || !patch_mask.defined() ||
                    !target_sizes.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o image bounds require image tensors");
            }
            result.vision_states = vision.forward(
                pixels.to(mfq_tensor_backend::kCUDA), patch_mask, target_sizes);
            result.image_embeddings = resampler.forward(
                result.vision_states, target_sizes);
            for (const auto & bound : images) {
                if (bound.batch >= input_ids.size(0) ||
                        bound.source >= result.image_embeddings.size(0) ||
                        bound.end > input_ids.size(1) ||
                        bound.end - bound.begin !=
                            result.image_embeddings.size(1)) {
                    throw std::runtime_error(
                        "MiniCPM-o image bound does not match 64 resampler queries");
                }
                result.input_embeddings.index({
                    bound.batch, Slice(bound.begin, bound.end), Slice()})
                    .copy_(result.image_embeddings.index({bound.source})
                        .to(result.input_embeddings.scalar_type()));
            }
        }
        const auto audios = minicpmo45_parse_bounds(
            audio_bounds, "audio");
        if (!audios.empty()) {
            if (!audio_features.defined() || !audio_lengths.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o audio bounds require audio tensors");
            }
            audio.reset();
            result.audio_embeddings = audio.forward(
                audio_features.to(mfq_tensor_backend::kCUDA), audio_lengths, false);
            const auto valid_lengths =
                MiniCPMO45AudioEncoder::pooled_lengths(audio_lengths);
            for (const auto & bound : audios) {
                if (bound.batch >= input_ids.size(0) ||
                        bound.source >= result.audio_embeddings.size(0) ||
                        bound.source >=
                            static_cast<int64_t>(valid_lengths.size()) ||
                        bound.end > input_ids.size(1) ||
                        bound.end - bound.begin !=
                            valid_lengths[static_cast<size_t>(bound.source)]) {
                    throw std::runtime_error(
                        "MiniCPM-o audio bound does not match pooled audio length");
                }
                result.input_embeddings.index({
                    bound.batch, Slice(bound.begin, bound.end), Slice()})
                    .copy_(result.audio_embeddings.index({
                        bound.source,
                        Slice(0, bound.end - bound.begin), Slice()})
                        .to(result.input_embeddings.scalar_type()));
            }
        }
        language.reset(input_ids.size(0));
        MfqOptional<mfq_tensor_backend::Tensor> positions = mfq_nullopt;
        if (position_ids.defined()) positions = position_ids;
        MfqOptional<mfq_tensor_backend::Tensor> mask = mfq_nullopt;
        if (attention_mask.defined()) mask = attention_mask;
        result.hidden_states = language.hidden_forward_inputs(
            input_ids, result.input_embeddings,
            positions, mfq_nullopt, nullptr, mask,
            position_ids.defined());
        result.logits = language.logits_from_hidden(result.hidden_states);
        return result;
    }
};

struct MiniCPMO45DuplexSpecialIds {
    int64_t unit_start = -1;
    int64_t unit_end = -1;
    int64_t image_start = -1;
    int64_t image_end = -1;
    int64_t slice_start = -1;
    int64_t slice_end = -1;
    int64_t listen = -1;
    int64_t speak = -1;
    int64_t tts_bos = -1;
    int64_t tts_eos = -1;
    int64_t chunk_eos = -1;
    int64_t chunk_tts_eos = -1;
    int64_t turn_eos = -1;
    int64_t tts_pad = -1;
    int64_t audio_bos = -1;

    static MiniCPMO45DuplexSpecialIds from_tensor(mfq_tensor_backend::Tensor value) {
        value = value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
        if (value.numel() != 15) {
            throw std::runtime_error(
                "MiniCPM-o duplex special_ids must contain 15 token IDs");
        }
        const auto * ids = value.data_ptr<int64_t>();
        MiniCPMO45DuplexSpecialIds result;
        result.unit_start = ids[0];
        result.unit_end = ids[1];
        result.image_start = ids[2];
        result.image_end = ids[3];
        result.slice_start = ids[4];
        result.slice_end = ids[5];
        result.listen = ids[6];
        result.speak = ids[7];
        result.tts_bos = ids[8];
        result.tts_eos = ids[9];
        result.chunk_eos = ids[10];
        result.chunk_tts_eos = ids[11];
        result.turn_eos = ids[12];
        result.tts_pad = ids[13];
        result.audio_bos = ids[14];
        const int64_t minimum = *std::min_element(ids, ids + 15);
        if (minimum < 0) {
            throw std::runtime_error(
                "MiniCPM-o duplex special token IDs must be non-negative");
        }
        return result;
    }

    bool is_chunk_terminator(int64_t token) const {
        return token == listen || token == chunk_eos ||
            token == chunk_tts_eos;
    }

    bool is_special(int64_t token) const {
        const std::array<int64_t, 15> values = {
            unit_start, unit_end, image_start, image_end,
            slice_start, slice_end, listen, speak, tts_bos,
            tts_eos, chunk_eos, chunk_tts_eos, turn_eos,
            tts_pad, audio_bos};
        return std::find(values.begin(), values.end(), token) != values.end();
    }
};

struct MiniCPMO45DuplexStepResult {
    mfq_tensor_backend::Tensor decision_logits;
    mfq_tensor_backend::Tensor audio_embeddings;
    mfq_tensor_backend::Tensor generated_ids;
    mfq_tensor_backend::Tensor tts_codes;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
};

struct MiniCPMO45DuplexSession {
    MiniCPMO45Runtime & runtime;
    MiniCPMO45DuplexSpecialIds ids;
    std::vector<int64_t> forbidden_ids;
    std::vector<int64_t> generated_text_ids;
    int64_t audio_chunk_index = 0;
    int64_t tts_text_start_position = 0;
    bool current_turn_ended = true;
    bool greedy = false;
    double temperature = 0.7;
    int64_t top_k = 100;
    double top_p = 0.8;
    double listen_probability_scale = 1.0;
    double repetition_penalty = 1.05;
    int64_t repetition_window = 512;
    double length_penalty = 1.0;
    double tts_temperature = 0.8;
    double tts_repetition_penalty = 1.05;

    MiniCPMO45DuplexSession(
            MiniCPMO45Runtime & runtime_,
            MiniCPMO45DuplexSpecialIds ids_,
            std::vector<int64_t> forbidden_ids_,
            bool greedy_)
        : runtime(runtime_), ids(ids_),
          forbidden_ids(std::move(forbidden_ids_)), greedy(greedy_) {
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.chunk_eos) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.chunk_eos);
        }
        if (std::find(forbidden_ids.begin(), forbidden_ids.end(),
                ids.tts_pad) == forbidden_ids.end()) {
            forbidden_ids.push_back(ids.tts_pad);
        }
    }

    void prepare(
            mfq_tensor_backend::Tensor system_prefix_ids,
            mfq_tensor_backend::Tensor reference_audio_features = mfq_tensor_backend::Tensor(),
            mfq_tensor_backend::Tensor system_suffix_ids = mfq_tensor_backend::Tensor()) {
        runtime.language.reset(1);
        runtime.audio.reset();
        runtime.tts.reset(1);
        generated_text_ids.clear();
        audio_chunk_index = 0;
        tts_text_start_position = 0;
        current_turn_ended = true;
        if (system_prefix_ids.defined() &&
                system_prefix_ids.numel() > 0) {
            feed_ids(system_prefix_ids);
        }
        if (reference_audio_features.defined()) {
            if (reference_audio_features.dim() != 3 ||
                    reference_audio_features.size(0) != 1 ||
                    reference_audio_features.size(1) != 80 ||
                    reference_audio_features.size(2) < 3) {
                throw std::runtime_error(
                    "MiniCPM-o duplex reference audio expects [1,80,frames]");
            }
            auto raw_lengths = mfq_tensor_backend::tensor(
                std::vector<int64_t>{reference_audio_features.size(2)},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64));
            auto embeddings = runtime.audio.forward(
                reference_audio_features.to(mfq_tensor_backend::kCUDA),
                raw_lengths, false);
            feed_embeddings(embeddings);
            runtime.audio.reset();
        }
        if (system_suffix_ids.defined() &&
                system_suffix_ids.numel() > 0) {
            feed_ids(system_suffix_ids);
        }
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_embeddings(
            mfq_tensor_backend::Tensor embeddings,
            mfq_tensor_backend::Tensor token_ids = mfq_tensor_backend::Tensor()) {
        if (embeddings.dim() == 2) embeddings = embeddings.unsqueeze(0);
        if (embeddings.dim() != 3 || embeddings.size(0) != 1 ||
                embeddings.size(2) != runtime.language.hidden_size()) {
            throw std::runtime_error(
                "MiniCPM-o duplex embeddings must have shape [1,tokens,4096]");
        }
        if (!token_ids.defined()) {
            token_ids = mfq_tensor_backend::zeros(
                {1, embeddings.size(1)},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
        } else {
            if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
            if (token_ids.dim() != 2 || token_ids.size(0) != 1 ||
                    token_ids.size(1) != embeddings.size(1)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex token IDs do not match embeddings");
            }
            token_ids = token_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        }
        auto hidden = runtime.language.hidden_forward_inputs(
            token_ids, embeddings.to(mfq_tensor_backend::kBFloat16).contiguous());
        auto logits = runtime.language.logits_from_hidden(
            hidden.index({Slice(), -1, Slice()}));
        return {logits, hidden};
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_ids(mfq_tensor_backend::Tensor token_ids) {
        if (token_ids.dim() == 0) token_ids = token_ids.reshape({1});
        if (token_ids.dim() == 1) token_ids = token_ids.unsqueeze(0);
        token_ids = token_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        return feed_embeddings(
            runtime.language.embed_forward(token_ids), token_ids);
    }

    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> feed_id(int64_t token) {
        return feed_ids(mfq_tensor_backend::tensor(
            std::vector<int64_t>{token},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA)));
    }

    mfq_tensor_backend::Tensor apply_text_sampling_filters(mfq_tensor_backend::Tensor logits) const {
        logits = logits.clone().to(mfq_tensor_backend::kFloat32);
        if (!generated_text_ids.empty() && repetition_penalty != 1.0) {
            const size_t begin = generated_text_ids.size() >
                    static_cast<size_t>(repetition_window)
                ? generated_text_ids.size() -
                    static_cast<size_t>(repetition_window)
                : 0;
            std::vector<int64_t> recent(
                generated_text_ids.begin() +
                    static_cast<std::ptrdiff_t>(begin),
                generated_text_ids.end());
            std::sort(recent.begin(), recent.end());
            recent.erase(std::unique(recent.begin(), recent.end()), recent.end());
            auto recent_tensor = mfq_tensor_backend::tensor(
                recent, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(logits.device()));
            auto selected = logits.index_select(1, recent_tensor);
            selected = repetition_penalty > 1.0
                ? selected / repetition_penalty
                : selected * (1.0 / repetition_penalty);
            logits.index_copy_(1, recent_tensor, selected);
        }
        if (length_penalty != 1.0) {
            auto selected = logits.index({Slice(), ids.turn_eos});
            selected = mfq_tensor_backend::where(
                selected > 0,
                selected / length_penalty,
                selected * length_penalty);
            logits.index_put_({Slice(), ids.turn_eos}, selected);
        }
        if (listen_probability_scale != 1.0) {
            logits.index_put_(
                {Slice(), ids.listen},
                logits.index({Slice(), ids.listen}) *
                    listen_probability_scale);
        }
        if (!forbidden_ids.empty()) {
            auto forbidden = mfq_tensor_backend::tensor(
                forbidden_ids,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(logits.device()));
            logits.index_fill_(
                1, forbidden,
                -std::numeric_limits<float>::infinity());
        }
        if (greedy) return logits;
        logits = logits / temperature;
        if (top_k > 0 && top_k < logits.size(1)) {
            auto values = std::get<0>(
                mfq_tensor_backend::topk(
                    logits, top_k, -1, true, true));
            auto threshold = values.select(1, top_k - 1).unsqueeze(1);
            logits = logits.masked_fill(
                logits < threshold,
                -std::numeric_limits<float>::infinity());
        }
        if (top_p < 1.0) {
            auto sorted = mfq_tensor_backend::sort(logits, -1, true);
            auto sorted_logits = std::get<0>(sorted);
            auto sorted_indices = std::get<1>(sorted);
            auto cumulative = mfq_tensor_backend::cumsum(
                mfq_tensor_backend::softmax(sorted_logits, -1), -1);
            auto remove = cumulative > top_p;
            auto shifted = mfq_tensor_backend::zeros_like(remove);
            shifted.narrow(1, 1, remove.size(1) - 1)
                .copy_(remove.narrow(1, 0, remove.size(1) - 1));
            sorted_logits = sorted_logits.masked_fill(
                shifted, -std::numeric_limits<float>::infinity());
            logits = mfq_tensor_backend::full_like(
                logits, -std::numeric_limits<float>::infinity());
            logits.scatter_(1, sorted_indices, sorted_logits);
        }
        return logits;
    }

    int64_t sample_text_token(mfq_tensor_backend::Tensor logits, bool force_listen) const {
        if (force_listen) return ids.listen;
        auto original = logits.to(mfq_tensor_backend::kFloat32);
        auto first = greedy
            ? mfq_tensor_backend::argmax(original, -1)
            : mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(original, -1), 1)
                .reshape({1});
        const int64_t first_id = first.item<int64_t>();
        if (first_id == ids.chunk_eos) return first_id;
        auto filtered = apply_text_sampling_filters(original);
        auto sampled = greedy
            ? mfq_tensor_backend::argmax(filtered, -1)
            : mfq_tensor_backend::multinomial(mfq_tensor_backend::softmax(filtered, -1), 1)
                .reshape({1});
        return sampled.item<int64_t>();
    }

    MiniCPMO45DuplexStepResult run_step(
            mfq_tensor_backend::Tensor pixels,
            mfq_tensor_backend::Tensor patch_mask,
            mfq_tensor_backend::Tensor target_sizes,
            mfq_tensor_backend::Tensor image_slice_counts,
            mfq_tensor_backend::Tensor audio_features,
            int64_t audio_prefix_extra_frames,
            int64_t audio_suffix_extra_frames,
            mfq_tensor_backend::Tensor text_ids,
            int64_t max_new_speak_tokens,
            bool force_listen,
            bool force_speak = false) {
        if (max_new_speak_tokens < 2) {
            throw std::runtime_error(
                "MiniCPM-o duplex generation requires at least two token slots");
        }
        auto pending = feed_id(ids.unit_start);
        mfq_tensor_backend::Tensor generation_logits;
        bool has_content = false;
        mfq_tensor_backend::Tensor audio_embeddings;

        if (pixels.defined()) {
            if (!patch_mask.defined() || !target_sizes.defined()) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image pixels require patch mask and target sizes");
            }
            auto vision_states = runtime.vision.forward(
                pixels.to(mfq_tensor_backend::kCUDA), patch_mask, target_sizes);
            auto image_embeddings = runtime.resampler.forward(
                vision_states, target_sizes);
            std::vector<int64_t> counts;
            if (image_slice_counts.defined()) {
                auto count_tensor = image_slice_counts
                    .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous().reshape({-1});
                const auto * values = count_tensor.data_ptr<int64_t>();
                counts.assign(values, values + count_tensor.numel());
            } else {
                counts.assign(
                    static_cast<size_t>(image_embeddings.size(0)), 1);
            }
            int64_t offset = 0;
            for (const int64_t count : counts) {
                if (count <= 0 || offset + count > image_embeddings.size(0)) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex image slice counts are invalid");
                }
                pending = feed_id(ids.image_start);
                pending = feed_embeddings(image_embeddings.index({offset}));
                pending = feed_id(ids.image_end);
                ++offset;
                for (int64_t slice = 1; slice < count; ++slice) {
                    pending = feed_id(ids.slice_start);
                    pending = feed_embeddings(image_embeddings.index({offset}));
                    pending = feed_id(ids.slice_end);
                    ++offset;
                }
            }
            if (offset != image_embeddings.size(0)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image slice counts do not cover embeddings");
            }
            generation_logits = pending.first;
            has_content = true;
        }

        if (audio_features.defined()) {
            audio_embeddings = runtime.audio.forward_streaming(
                audio_features.to(mfq_tensor_backend::kCUDA),
                audio_prefix_extra_frames,
                audio_suffix_extra_frames);
            pending = feed_embeddings(audio_embeddings);
            generation_logits = pending.first;
            ++audio_chunk_index;
            has_content = true;
        }

        if (text_ids.defined() && text_ids.numel() > 0) {
            pending = feed_ids(text_ids);
            if (!generation_logits.defined()) {
                generation_logits = pending.first;
            }
            has_content = true;
        }
        if (!has_content) {
            throw std::runtime_error(
                "MiniCPM-o duplex step contains no image, audio, or text input");
        }
        if (pixels.defined() && !audio_features.defined()) {
            ++audio_chunk_index;
        }

        MiniCPMO45DuplexStepResult result;
        result.decision_logits = generation_logits;
        result.audio_embeddings = audio_embeddings;
        std::vector<int64_t> generated;
        std::vector<int64_t> spoken_ids;
        std::vector<mfq_tensor_backend::Tensor> spoken_hidden;
        auto logits = generation_logits;
        bool force_current = force_listen;
        for (int64_t index = 0; index < max_new_speak_tokens; ++index) {
            if (index == max_new_speak_tokens - 1) {
                feed_id(ids.chunk_eos);
                generated.push_back(ids.chunk_eos);
                break;
            }
            const bool forced_decision = force_current;
            int64_t token = sample_text_token(logits, force_current);
            force_current = false;
            if (!forced_decision && token != ids.chunk_eos) {
                generated_text_ids.push_back(token);
            }
            if (!forced_decision && token == ids.listen &&
                    (!current_turn_ended || force_speak)) {
                token = ids.tts_bos;
            }
            generated.push_back(token);
            result.is_listen = token == ids.listen;
            if (ids.is_chunk_terminator(token)) {
                pending = feed_id(token);
                break;
            }

            current_turn_ended = false;
            pending = feed_id(token);
            logits = pending.first;
            result.end_of_turn = token == ids.turn_eos;
            if (result.end_of_turn) current_turn_ended = true;
            if (index != 0) {
                spoken_ids.push_back(token);
                spoken_hidden.push_back(
                    pending.second.index({Slice(), -1, Slice()}));
            }
        }
        feed_id(ids.unit_end);

        result.generated_ids = mfq_tensor_backend::tensor(
            generated,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA));
        if (!result.is_listen) {
            mfq_tensor_backend::Tensor text_id_tensor;
            mfq_tensor_backend::Tensor hidden_tensor;
            if (spoken_ids.empty()) {
                text_id_tensor = mfq_tensor_backend::empty(
                    {1, 0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                        .device(mfq_tensor_backend::kCUDA));
                hidden_tensor = mfq_tensor_backend::empty(
                    {1, 0, runtime.language.hidden_size()},
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kBFloat16)
                        .device(mfq_tensor_backend::kCUDA));
            } else {
                text_id_tensor = mfq_tensor_backend::tensor(
                    spoken_ids,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                        .device(mfq_tensor_backend::kCUDA)).unsqueeze(0);
                hidden_tensor = mfq_tensor_backend::stack(spoken_hidden, 1)
                    .to(mfq_tensor_backend::kBFloat16).contiguous();
            }
            auto condition = runtime.tts.duplex_condition(
                text_id_tensor, hidden_tensor, ids.audio_bos);
            const bool first_tts_chunk = tts_text_start_position == 0;
            if (first_tts_chunk) {
                runtime.tts.reset(1);
                result.tts_force_flush = true;
            }
            if (runtime.tts.cache_position != tts_text_start_position) {
                throw std::runtime_error(
                    "MiniCPM-o duplex TTS cache position is inconsistent");
            }
            const int64_t minimum_codes =
                result.end_of_turn || first_tts_chunk ? 0 : 26;
            auto tts_result = runtime.tts.generate_duplex_chunk(
                condition, 26, minimum_codes, 6561,
                tts_temperature, tts_repetition_penalty);
            result.tts_codes = tts_result.codes;
            if (result.end_of_turn) {
                runtime.tts.reset(1);
                tts_text_start_position = 0;
            } else {
                tts_text_start_position +=
                    condition.size(1) + result.tts_codes.size(1);
                if (runtime.tts.cache_position != tts_text_start_position) {
                    throw std::runtime_error(
                        "MiniCPM-o duplex TTS cache did not advance exactly");
                }
            }
        } else {
            result.tts_codes = mfq_tensor_backend::empty(
                {1, 0, 1},
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
        }
        return result;
    }
};

inline mfq_tensor_backend::Tensor minicpmo45_load_tensor(
        const std::string & path,
        bool required) {
    if (!std::filesystem::is_regular_file(path)) {
        if (required) {
            throw std::runtime_error(
                "MiniCPM-o tensor file is missing: " + path);
        }
        return mfq_tensor_backend::Tensor();
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "failed to open MiniCPM-o tensor file: " + path);
    }
    const auto end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error(
            "MiniCPM-o tensor file is empty: " + path);
    }
    std::vector<char> bytes(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error(
            "failed to read MiniCPM-o tensor file: " + path);
    }
    auto value = mfq_tensor_backend::pickle_load(bytes);
    if (!value.isTensor()) {
        throw std::runtime_error(
            "MiniCPM-o input is not a tensor: " + path);
    }
    return value.toTensor();
}

inline void minicpmo45_save_tensor(
        const mfq_tensor_backend::Tensor & value,
        const std::string & path) {
    if (!value.defined()) return;
    const std::filesystem::path output(path);
    if (output.has_parent_path()) {
        std::filesystem::create_directories(output.parent_path());
    }
    minicpmo45_write_pickle_tensor(value, path);
}

inline int64_t minicpmo45_load_optional_scalar(
        const std::string & path,
        int64_t fallback) {
    auto value = minicpmo45_load_tensor(path, false);
    if (!value.defined()) return fallback;
    if (value.numel() != 1) {
        throw std::runtime_error(
            "MiniCPM-o scalar tensor must contain one value: " + path);
    }
    return value.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).item<int64_t>();
}

inline std::string minicpmo45_duplex_step_prefix(
        const std::string & prefix,
        int64_t step) {
    std::ostringstream stream;
    stream << prefix << ".step" << std::setfill('0')
           << std::setw(4) << step;
    return stream.str();
}

int run_minicpmo45_duplex(
    const std::string& model_path,
    const std::string& config_path,
    const std::string& input_prefix,
    const std::string& output_prefix,
    std::int64_t context_size,
    std::int64_t steps,
    std::int64_t max_speak_tokens,
    bool greedy,
    std::int64_t seed);
int run_minicpmo45_eval_batch(
    const std::string& model_path,
    const std::string& config_path,
    std::int64_t context_size,
    std::int64_t vision_batch_size);
int run_minicpmo45_composite(
    const std::string& model_path,
    const std::string& config_path,
    const std::string& input_prefix,
    const std::string& output_prefix,
    std::int64_t context_size,
    std::int64_t tts_steps);
