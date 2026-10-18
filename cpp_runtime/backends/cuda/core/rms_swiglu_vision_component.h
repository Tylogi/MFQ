#pragma once

#include "vision_component.h"
#include "core/attention.h"
#include "mfq_weight_memory.h"
#include "models/common/rms_vision_config.h"
#include "models/common/vision_prompt.h"

namespace mfq::cuda::vision_runtime {

using mfq::models::RmsVisionConfig;

namespace detail {

inline Tensor vision_vector(CudaExecutionContext& execution, const ModelSource& source,
                            const std::string& name, int64_t width) {
    auto value = load_dense_gpu(execution, source, name);
    if (value.dim() != 1 || value.size(0) != width)
        throw std::runtime_error("vision vector shape mismatch: " + name);
    return value;
}

inline Affine vision_affine(CudaExecutionContext& execution, const ModelSource& source,
                            const std::string& prefix, int64_t output, int64_t input,
                            bool bias, const std::vector<int64_t>& convolution = {}) {
    const auto name = prefix + ".weight";
    const auto& record = require_tensor(source, name);
    QuantLinear linear;
    if (!convolution.empty() && (record.dtype == "BF16" || record.dtype == "F16" || record.dtype == "F32")) {
        const mfq::cuda::WeightAllocationScope weight_memory;
        auto weight = load_dense_cpu(execution, source, name);
        if (!execution.loading_cpu_layer)
            weight = weight.to(mfq_tensor_backend::kCUDA);
        if (weight.sizes().vec() != convolution)
            throw std::runtime_error("vision convolution shape mismatch: " + name);
        linear.kind = QuantLinearKind::Dense;
        linear.dense = weight.reshape({output, input});
        linear.logical_out = output;
        linear.logical_neuron_len = input;
    } else {
        linear = load_quant_linear(execution, source, name, TensorParallelAxis::Mirrored);
    }
    if (linear.out() != output || linear.neuron_len() != input)
        throw std::runtime_error("vision projection shape mismatch: " + name);
    return {std::move(linear), bias ? vision_vector(execution, source, prefix + ".bias", output) : Tensor{}};
}

inline Tensor bounded_mlp(CudaExecutionContext& execution, const Tensor& input,
                          const Affine& gate, const Affine& up, const Affine& down, double limit) {
    auto hidden = mfq_tensor_backend::silu(gate(execution, input).clamp_max(limit)) *
                  up(execution, input).clamp(-limit, limit);
    return down(execution, hidden);
}

struct RmsVisionBlock {
    Tensor norm1, norm2, query_norm, key_norm;
    Affine qkv, output, gate, up, down;
    template<class Visitor> void visit_tensors(Visitor visit) {
        for (auto* t : {&norm1, &norm2, &query_norm, &key_norm}) visit(*t);
        qkv.visit_tensors(visit); output.visit_tensors(visit);
        gate.visit_tensors(visit); up.visit_tensors(visit); down.visit_tensors(visit);
    }
    Tensor operator()(CudaExecutionContext& execution, const Tensor& input,
                      const GridVisionLayout& layout, const RmsVisionConfig& config) const {
        const auto& g = config.grid;
        const auto norm = [&](const Tensor& x, const Tensor& w) {
            return attention_ops::rms_norm(x, w, config.epsilon);
        };
        const auto count = input.size(0);
        auto projected = qkv(execution, norm(input, norm1)).reshape({count, 3, g.num_heads, g.head_dim()});
        auto query = apply_axial_rope(norm(projected.select(1, 0), query_norm), layout.positions, g.rope_theta);
        auto key = apply_axial_rope(norm(projected.select(1, 1), key_norm), layout.positions, g.rope_theta);
        auto attended = segmented_attention(query, key, projected.select(1, 2), layout.segment_lengths);
        auto hidden = input + output(execution, attended.reshape({count, g.hidden_size})).to(input.scalar_type());
        auto normalized = norm(hidden, norm2);
        return hidden + bounded_mlp(execution, normalized, gate, up, down, config.limit).to(hidden.scalar_type());
    }
};

struct RmsVisionMerger {
    Tensor norm;
    Affine downsample, projection, gate, up, down;
    LayerNorm projection_norm;
    template<class Visitor> void visit_tensors(Visitor visit) {
        visit(norm); downsample.visit_tensors(visit); projection.visit_tensors(visit);
        projection_norm.visit_tensors(visit);
        gate.visit_tensors(visit); up.visit_tensors(visit); down.visit_tensors(visit);
    }
    Tensor operator()(CudaExecutionContext& execution, const Tensor& patches,
                      const RmsVisionConfig& config) const {
        const auto& g = config.grid;
        const auto merge = g.spatial_merge_size;
        auto normalized = attention_ops::rms_norm(patches, norm, config.epsilon);
        // The convolution sees [channel, row, column], while preprocessing
        // emits [row, column, channel] within each merge group.
        auto grouped = normalized.reshape({-1, merge, merge, g.hidden_size})
            .permute({0, 3, 1, 2}).contiguous().reshape({-1, g.hidden_size * merge * merge});
        auto projected = projection(execution, downsample(execution, grouped));
        projected = mfq_tensor_backend::gelu(projection_norm(projected), "none");
        return bounded_mlp(execution, projected, gate, up, down, config.limit);
    }
};

} // namespace detail

class CudaRmsSwiGluVisionPromptComponent {
  public:
    static CudaRmsSwiGluVisionPromptComponent load(CudaExecutionContext& execution,
            const ModelSource& source, RmsVisionConfig config, int64_t image, int64_t video,
            const std::string& contract, const std::string& policy) {
        if (contract != kMfqGridVisionInputContract || policy != kMfqSequentialPositionPolicy ||
            image < 0 || video < 0 || image == video)
            throw std::invalid_argument("invalid RMS-SwiGLU vision prompt contract");
        detail::LoadingScope loading(execution);
        CudaRmsSwiGluVisionPromptComponent result;
        result.config_ = config; result.image_ = image; result.video_ = video;
        result.contract_ = contract; result.policy_ = policy;
        const auto& g = config.grid;
        const auto h = g.hidden_size, out = g.out_hidden_size, merge = g.spatial_merge_size;
        const auto vector = [&](const std::string& name, int64_t width) {
            return detail::vision_vector(execution, source, "vision." + name + ".weight", width);
        };
        const auto affine = [&](const std::string& name, int64_t rows, int64_t cols, bool bias = true) {
            return detail::vision_affine(execution, source, "vision." + name, rows, cols, bias);
        };
        result.patch_ = detail::vision_affine(execution, source, "vision.patch_embedding", h,
            g.patch_width(), true, {h, g.in_channels, g.temporal_patch_size, g.patch_size, g.patch_size});
        for (int64_t index = 0; index < g.depth; ++index) {
            const auto p = "block." + std::to_string(index) + '.';
            result.blocks_.push_back({vector(p + "norm1", h), vector(p + "norm2", h),
                vector(p + "attention.query_norm", g.head_dim()), vector(p + "attention.key_norm", g.head_dim()),
                affine(p + "attention.qkv", 3*h, h), affine(p + "attention.output", h, h),
                affine(p + "mlp.gate", g.intermediate_size, h), affine(p + "mlp.up", g.intermediate_size, h),
                affine(p + "mlp.down", h, g.intermediate_size)});
        }
        auto& m = result.merger_;
        m.norm = vector("output_norm", h);
        m.downsample = detail::vision_affine(execution, source, "vision.downsample", out, h*merge*merge,
            true, {out, h, merge, merge});
        m.projection = affine("merger.projection", out, out, false);
        m.projection_norm = {vector("merger.norm", out),
            detail::vision_vector(execution, source, "vision.merger.norm.bias", out), 1e-5};
        m.gate = affine("merger.mlp.gate", config.projection_intermediate, out, false);
        m.up = affine("merger.mlp.up", config.projection_intermediate, out, false);
        m.down = affine("merger.mlp.down", out, config.projection_intermediate, false);
        return result;
    }

    template<class Model>
    mfq::StepSequence<CudaPreparedPrompt> prepare(Model& language,
            const std::vector<int64_t>& ids, const MfqMultimodalInput& media) const {
        const auto grids = validate_media(media);
        const auto key = detail::cache_key(media, contract_, policy_);
        auto& execution = *language.execution;
        Tensor merged;
        if (cached_key_ == key && cached_.defined()) merged = cached_;
        else {
            auto pixels = mfq_tensor_backend::tensor(media.pixel_values,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU))
                .reshape(media.pixel_shape).to(mfq_tensor_backend::kCUDA);
            const auto layout = make_grid_vision_layout(grids, static_cast<int32_t>(config_.grid.spatial_merge_size));
            auto hidden = resident_forward(execution, patch_, pixels, config_.grid.hidden_size, [&](const auto& patch) { return patch(execution, pixels); });
            co_yield mfq::StepState::advanced;
            for (const auto& block : blocks_) {
                hidden = resident_forward(execution, block, hidden, std::max(3*config_.grid.hidden_size, config_.grid.intermediate_size),
                    [&](const auto& weights) { return weights(execution, hidden, layout, config_); });
                co_yield mfq::StepState::advanced;
            }
            merged = resident_forward(execution, merger_, hidden, config_.projection_intermediate,
                [&](const auto& weights) { return weights(execution, hidden, config_); });
            cached_key_ = key; cached_ = merged;
            co_yield mfq::StepState::advanced;
        }
        auto options = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
        auto embeddings = mfq::models::inject_vision_embeddings(ids, merged, merged.size(0), image_, video_, true,
            [&](const auto& tokens) {
                return language.embed_forward(mfq_tensor_backend::tensor(tokens, options).reshape({1, -1})).contiguous();
            },
            [&](Tensor& output, const Tensor& vision, const std::vector<int64_t>& placeholders) {
                output.index_copy_(1, mfq_tensor_backend::tensor(placeholders, options),
                    vision.to(output.scalar_type()).unsqueeze(0));
            });
        // Sequential policy: no position override. The text model advances its
        // own contiguous causal cache positions (GLM rejects explicit overrides).
        co_yield CudaPreparedPrompt{ids, std::move(embeddings), {}, 0, key};
    }

  private:
    std::vector<GridShape> validate_media(const MfqMultimodalInput& media) const {
        if (media.processor != MfqMultimodalProcessor::grid_vision || media.processor_name != contract_ ||
            media.vision_types.empty() || media.vision_grid_shape != std::vector<int64_t>{static_cast<int64_t>(media.vision_types.size()), 3} ||
            media.vision_grid.size() != 3*media.vision_types.size())
            throw std::invalid_argument("RMS-SwiGLU vision requires the declared grid media contract");
        std::vector<GridShape> grids;
        std::vector<int32_t> images, videos;
        for (size_t i = 0; i < media.vision_types.size(); ++i) {
            const auto t = media.vision_types[i];
            const auto start = media.vision_grid.begin() + static_cast<std::ptrdiff_t>(3*i);
            if ((t != 1 && t != 2) || (t == 1 && *start != 1))
                throw std::invalid_argument("invalid vision media type or image temporal grid");
            auto& split = t == 1 ? images : videos;
            split.insert(split.end(), start, start+3);
            grids.push_back({start[0], start[1], start[2]});
        }
        const auto layout = make_grid_vision_layout(grids, static_cast<int32_t>(config_.grid.spatial_merge_size));
        if (images != media.image_grid || videos != media.video_grid ||
            media.image_grid_shape != std::vector<int64_t>{static_cast<int64_t>(images.size()/3), 3} ||
            media.video_grid_shape != std::vector<int64_t>{static_cast<int64_t>(videos.size()/3), 3} ||
            media.pixel_shape != std::vector<int64_t>{layout.patch_count, config_.grid.patch_width()} ||
            media.pixel_values.size() != static_cast<size_t>(layout.patch_count*config_.grid.patch_width()) ||
            !std::all_of(media.pixel_values.begin(), media.pixel_values.end(), [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("vision grids disagree with pixels or modality split");
        return grids;
    }

    template<class Weights, class Forward>
    Tensor resident_forward(CudaExecutionContext& execution, const Weights& weights,
                            const Tensor& input, int64_t intermediate, Forward forward) const {
        if (blocks_.front().norm1.is_cuda()) return forward(weights);
        const auto& g = config_.grid;
        const auto scratch = input.nbytes()*16 + static_cast<size_t>(input.size(0)*intermediate)*16 +
            static_cast<size_t>(std::max(g.hidden_size, g.out_hidden_size)*intermediate)*2 + (std::size_t(16)<<20);
        return detail::vision_workspace(execution, detail::weight_bytes(weights)+scratch, [&] {
            auto resident = detail::to_cuda(weights);
            return forward(resident);
        });
    }
    RmsVisionConfig config_;
    detail::Affine patch_;
    std::vector<detail::RmsVisionBlock> blocks_;
    detail::RmsVisionMerger merger_;
    int64_t image_ = -1, video_ = -1;
    std::string contract_, policy_;
    mutable std::string cached_key_;
    mutable Tensor cached_;
};

} // namespace mfq::cuda::vision_runtime
