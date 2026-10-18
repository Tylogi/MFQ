#pragma once

#include "storage/weight_loader.h"
#include "runtime/execution_options.h"
#include "storage/moe_expert_cache.h"
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

namespace mfq::cuda::vision_runtime {

using Tensor = mfq_tensor_backend::Tensor;

namespace detail {

template<class Visitor> void visit_linear(QuantLinear& w,Visitor visit) {
    MFQ_RUNTIME_CHECK(!w.tensor_parallel(),"streamed vision weights require one CUDA device");
    auto& n=w.nint;auto& v=w.nvq;
    for(auto* tensor:{&w.dense,&n.q_packed,&n.row_q_bits,&n.row_q_bit_offsets,&n.q8_zero_scale,
            &n.sub_scale,&n.sub_min,&n.neuron_scale,&n.neuron_min,&n.route_metadata,
            &v.indices_packed,&v.aux_packed,&v.sub_scale_packed,&v.neuron_scale,&v.codebook,
            &v.integer_codebook,&v.decode_records,&w.mxfp4.weight.values,&w.mxfp4.weight.scales,
            &w.mxfp8.weight.values,&w.mxfp8.weight.scales,&w.mxfp4_sq.weight.blob,
            &w.mxfp4_sq.weight.row_q,&w.mxfp4_sq.weight.row_symbol_byte_offsets,&w.mxfp4_sq.weight.row_auxiliary,
            &w.fp8_sq.weight.blob,&w.fp8_sq.weight.row_q,&w.fp8_sq.weight.row_symbol_byte_offsets})
        if(tensor->defined())visit(*tensor);
    n.workspaces.clear();v.workspaces.clear();
}

template<class Weights> Weights to_cuda(Weights weights) {
    weights.visit_tensors([](Tensor& t){t=t.to(mfq_tensor_backend::kCUDA);});
    return weights;
}
template<class Weights> std::size_t weight_bytes(Weights weights) {
    std::size_t bytes=0;
    weights.visit_tensors([&](Tensor& t){bytes+=t.nbytes();});
    return bytes;
}

// Finish each loan before yielding to the scheduler. Another request can then
// decode using the restored experts, including if image preparation is cancelled.
template<class Forward> Tensor vision_workspace(CudaExecutionContext& execution,std::size_t bytes,Forward forward) {
    MoePrefillWorkspaceLoan loan(execution.moe_expert_cache);
    const bool borrowed=loan.begin(bytes)!=0;
    auto result=forward();
    if(borrowed) {
        loan.suspend();
        result=result.clone();
        loan.finish();
    }
    return result;
}

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
    template<class Visitor> void visit_tensors(Visitor visit) {
        visit_linear(linear,visit);if(bias.defined())visit(bias);
    }

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
    template<class Visitor> void visit_tensors(Visitor visit) { visit(weight);visit(bias); }

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


struct LoadingScope {
    bool& value;
    bool previous;
    LoadingScope(CudaExecutionContext& execution)
        : value(execution.loading_cpu_layer), previous(value) {
        value = !runtime_options::vision_resident() && !execution.tensor_parallel.enabled();
    }
    ~LoadingScope() { value = previous; }
};

inline std::string cache_key(const MfqMultimodalInput& media, const std::string& contract,
                             const std::string& policy) {
    auto key = contract + '|' + policy + '|' + mfq::cache::block_hash_hex(mfq::cache::sha256(
        media.pixel_values.data(), media.pixel_values.size() * sizeof(float)));
    for (const auto& values : {media.pixel_shape, std::vector<int64_t>(media.vision_grid.begin(), media.vision_grid.end())})
        for (const auto value : values) key += '|' + std::to_string(value);
    return key;
}

} // namespace detail
} // namespace mfq::cuda::vision_runtime
