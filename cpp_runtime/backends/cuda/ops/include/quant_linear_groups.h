#pragma once

#include "quant_linear_weight.h"

struct MfeWeight;

struct QuantLinearGroup {
    bool nint_grouped = false;
    bool nvq_prefix2 = false;
    bool decode_branch_parallel = true;
    NintLinearGroup nint;
    std::vector<QuantLinear> layers;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();

    QuantLinearProjectionRefs tensor_parallel_output_projections() const {
        QuantLinearProjectionRefs projections;
        projections.reserve(layers.size());
        for (const auto & layer : layers) {
            projections.push_back(&layer);
        }
        return projections;
    }

    bool tensor_parallel_output_compatible() const {
        return tensor_parallel_output_projections_compatible(
            tensor_parallel_output_projections());
    }

    std::vector<mfq_tensor_backend::Tensor>
    forward_tensor_parallel_output_group(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const {
        return forward_tensor_parallel_output_projections(
            execution, x, tensor_parallel_output_projections());
    }

    std::vector<mfq_tensor_backend::Tensor> forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const {
        const bool default_mmq =
            execution.kl_mmq.mode == KlMmqMode::Default;
        if (!x.is_cuda()) {
            MFQ_RUNTIME_CHECK(
                !nint_grouped,
                "CPU dense offload requires separate compact linear weights");
            std::vector<mfq_tensor_backend::Tensor> result;
            result.reserve(layers.size());
            for (const auto & layer : layers) {
                result.push_back(layer.forward(execution, x));
            }
            return result;
        }
        if (execution.config.tensor_parallel_grouped_projections &&
                tensor_parallel_output_compatible()) {
            return forward_tensor_parallel_output_group(execution, x);
        }
        if (nint_grouped) {
            return nint.forward(
                execution.profiler, execution.kl_mmq,
                execution.config, execution.decode_graph_serial_branches, x);
        }
        if (default_mmq && nvq_prefix2 && nvq_fusion_enabled(execution.config)) {
            auto shape = x.sizes().vec();
            const auto flat =
                x.reshape({-1, x.size(-1)});
            std::vector<mfq_tensor_backend::Tensor> branch_outputs;
            const bool parallel =
                decode_branch_parallel &&
                decode_branch_parallel_enabled(
                    execution.config, execution.decode_graph_serial_branches, flat.size(0)) &&
                layers.size() > 2 &&
                branch_executor->run(
                    layers.size() - 1,
                    [&](size_t branch) {
                        if (branch == 0) {
                            return nvq_matmul_multi2(
                                execution.profiler, layers[0].nvq,
                                layers[1].nvq, flat);
                        }
                        return layers[branch + 1].forward(execution, x);
                    },
                    branch_outputs);
            auto combined = parallel
                ? branch_outputs[0]
                : nvq_matmul_multi2(
                    execution.profiler, layers[0].nvq,
                    layers[1].nvq, flat);
            auto pair = combined.split_with_sizes({outs[0], outs[1]}, -1);
            std::vector<mfq_tensor_backend::Tensor> result;
            result.reserve(layers.size());
            for (size_t i = 0; i < 2; ++i) {
                auto part_shape = shape;
                part_shape.back() = outs[i];
                result.push_back(pair[i].reshape(part_shape));
            }
            for (size_t i = 2; i < layers.size(); ++i) {
                result.push_back(
                    parallel
                        ? branch_outputs[i - 1]
                        : layers[i].forward(execution, x));
            }
            return result;
        }
        std::vector<mfq_tensor_backend::Tensor> result;
        if (default_mmq && decode_branch_parallel &&
                decode_branch_parallel_enabled(
                    execution.config, execution.decode_graph_serial_branches, x.numel() / x.size(-1)) &&
                branch_executor->run(
                    layers.size(),
                    [&](size_t index) {
                        return layers[index].forward(execution, x);
                    },
                    result)) {
            return result;
        }
        result.reserve(layers.size());
        for (const auto & layer : layers) {
            result.push_back(layer.forward(execution, x));
        }
        return result;
    }
    mfq_tensor_backend::Tensor forward_swiglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const {
        const bool default_mmq =
            execution.kl_mmq.mode == KlMmqMode::Default;
        if (default_mmq && nint_grouped && nint.split_w.empty() &&
                x.numel() / x.size(-1) >= 1 && x.numel() / x.size(-1) <= 6) {
            return nint.forward_swiglu(execution.profiler, x);
        }
        if (default_mmq && nvq_prefix2 && layers.size() == 2 &&
                nvq_fusion_enabled(execution.config)) {
            auto shape = x.sizes().vec();
            auto y = nvq_matmul_swiglu(
                execution.profiler, layers[0].nvq, layers[1].nvq,
                x.reshape({-1, x.size(-1)}));
            shape.back() = y.size(-1);
            return y.reshape(shape);
        }
        if (outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("SwiGLU requires equal gate/up output widths");
        }
        auto parts = forward(execution, x);
        return mfq_tensor_backend::silu(parts[0]) * parts[1];
    }
    mfq_tensor_backend::Tensor forward_geglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const {
        if (execution.kl_mmq.mode == KlMmqMode::Default &&
                nint_grouped && nint.split_w.empty() &&
                x.numel() / x.size(-1) == 1) {
            return nint.forward_geglu(execution.profiler, x);
        }
        if (outs.size() != 2 || outs[0] != outs[1]) {
            throw std::runtime_error("GeGLU requires equal gate/up output widths");
        }
        auto parts = forward(execution, x);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    }
};

struct DenseLinearGroup {
    mfq_tensor_backend::Tensor w;
    std::vector<int64_t> outs;

    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = mfq_tensor_backend::matmul(x.reshape({-1, x.size(-1)}).to(mfq_tensor_backend::kFloat32), w.transpose(0, 1));
        auto parts = y.split_with_sizes(outs, -1);
        for (auto & p : parts) {
            auto s = shape;
            s.back() = p.size(-1);
            p = p.reshape(s);
        }
        return parts;
    }
};

bool has_tensor(const mfq::ModelSource& source, std::string_view name) noexcept;
bool has_tensor_prefix(const mfq::ModelSource& source, std::string_view prefix);
std::vector<std::uint8_t> read_asset(
    const mfq::ModelSource& source, std::string_view name);
std::string read_asset_text(
    const mfq::ModelSource& source, std::string_view name);

MoeRoutePlan build_moe_route_plan(
    mfq_tensor_backend::Tensor ids, int n_experts);
MfeWeight load_mfe_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name,
    bool cacheable = false,
    int layer_id = -1,
    const std::string& projection_role = {});
std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
    const mfq::ModelSource& source,
    const std::string& name);
MfeWeight cpu_mixed_moe_metadata(
    const std::shared_ptr<MixedMoeRuntime>& runtime);
mfq_tensor_backend::Tensor load_dense_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name);
QuantLinear load_quant_linear(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name,
    std::optional<TensorParallelAxis> axis_override = std::nullopt,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr);
QuantLinearGroup load_quant_group(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    size_t required_compatible_prefix = 0,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr,
    bool preserve_projection_boundaries = false);
QuantLinearGroup load_paired_gate_up(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    const QuantLinear& down,
    size_t required_compatible_prefix = 2,
    bool preserve_projection_boundaries = false);
mfq_tensor_backend::Tensor quant_embedding_lookup(
    const QuantLinear& weight, mfq_tensor_backend::Tensor token_ids);
DenseLinearGroup make_fp32_quant_group(
    CudaExecutionContext& execution,
    QuantLinearGroup group);

MfeWeight stage_cpu_mixed_moe(
    const std::shared_ptr<MixedMoeRuntime>& runtime,
    const CudaExecutionConfig& config = {});
bool prefetch_cached_moe_projection_bundle(
    const MfeWeight& gate,
    const MfeWeight& up,
    const MfeWeight& down,
    const MoeRoutePlan& route);
bool prefetch_cached_moe_projection_bundle(
    const MfeWeight& gate_up,
    const MfeWeight& down,
    const MoeRoutePlan& route);
mfq_tensor_backend::Tensor nvq_ffn_swiglu_down(
    CudaProfiler& profiler,
    const NvqWeight& gate,
    const NvqWeight& up,
    const NvqWeight& down,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> residual = mfq_nullopt);
bool nvq_fused_residual_format(std::int64_t kernel_format);
mfq_tensor_backend::Tensor nint_matmul_groupwise_u8(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    std::int64_t groups);
bool is_quant_dtype(const std::string& dtype);
QuantLinearGroup make_quant_group(
    CudaExecutionContext& execution,
    std::vector<QuantLinear> layers,
    bool preserve_projection_boundaries = false);
DenseLinearGroup make_dense_group(
    const std::vector<mfq_tensor_backend::Tensor>& weights);
mfq_tensor_backend::Tensor quant_linear_reference_weight(
    const QuantLinear& linear);
mfq_tensor_backend::Tensor mfe_dense_reference(
    const mfq::ModelSource& source,
    const std::string& name,
    mfq_tensor_backend::Tensor input,
    const std::vector<std::int32_t>& expert_ids,
    int tokens,
    int routes,
    bool routed_input);
mfq_tensor_backend::Tensor materialize_mfe_dense(
    const mfq::ModelSource& source,
    const std::string& name);
