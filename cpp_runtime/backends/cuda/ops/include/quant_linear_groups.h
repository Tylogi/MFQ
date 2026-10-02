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

    QuantLinearProjectionRefs tensor_parallel_output_projections() const;

    bool tensor_parallel_output_compatible() const;

    std::vector<mfq_tensor_backend::Tensor>
    forward_tensor_parallel_output_group(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;

    std::vector<mfq_tensor_backend::Tensor> forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_swiglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_geglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
};

struct DenseLinearGroup {
    mfq_tensor_backend::Tensor w;
    std::vector<int64_t> outs;

    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor x) const;
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
