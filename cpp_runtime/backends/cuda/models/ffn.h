#pragma once

#include "quant_linear.h"
#include "mfe_weight.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace mfq {
class ModelSource;
namespace models { struct ModelConfig; }
}

struct FFN {
    QuantLinearGroup gate_up;
    QuantLinear down;
    std::unique_ptr<FFN> important_neurons;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        important_neuron_executor =
            std::make_shared<CudaIndependentBranchExecutor>();
    bool geglu = false;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    MfeWeight moe_gate_up;
    MfeWeight moe_gate;
    MfeWeight moe_up;
    MfeWeight moe_down;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_down;
    mfq_tensor_backend::Tensor moe_router;
    mfq_tensor_backend::Tensor moe_shared_gate;
    mfq_tensor_backend::Tensor moe_router_bias;
    mfq_tensor_backend::Tensor moe_hash_ids;
    std::unique_ptr<FFN> shared;
    int moe_top_k = 0;
    bool moe_use_sigmoid = false;
    bool moe_use_sqrt_softplus = false;
    bool moe_normalize = false;
    bool moe_delayed_softmax = true;
    bool moe_shared_ungated = false;
    double moe_router_scale = 1.0;
    double swiglu_limit = 0.0;
    int moe_layer = -1;

    bool uses_moe_expert_cache() const;
    bool tensor_parallel_dense_compatible() const;
    mfq_tensor_backend::Tensor forward_tensor_parallel_dense(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const;
    bool expert_parallel_moe_compatible() const;
    mfq_tensor_backend::Tensor forward_expert_parallel_moe(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        const MoeRoutePlan& route,
        mfq_tensor_backend::Tensor route_weights) const;
    mfq_tensor_backend::Tensor forward_dense_f32_down_kld(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_impl(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids,
        bool allow_important_neurons) const;
    bool can_forward_fused_residual(
        const CudaExecutionConfig& config,
        const mfq_tensor_backend::Tensor& x,
        const mfq_tensor_backend::Tensor& residual) const;
    mfq_tensor_backend::Tensor forward_fused_residual(
        CudaProfiler& profiler,
        const CudaExecutionConfig& config,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor residual) const;
    mfq_tensor_backend::Tensor forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt) const;
};

struct MoeWeightLoadOptions {
    int layer;
    bool cpu_offloaded = false;
    bool router_bias_required = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
};

void prepare_ffn_workspaces(CudaExecutionContext& execution, FFN& ffn);
FFN load_moe_weights(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    std::string_view prefix,
    const MoeWeightLoadOptions& options);
FFN load_ffn(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const mfq::models::ModelConfig& config,
    int layer,
    bool minicpmo45 = false,
    std::string_view tensor_root = "model");
void load_important_neuron_branch(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    int64_t hidden_size,
    int64_t intermediate_size,
    FFN& ffn,
    const std::string& down_name,
    const std::string& gate_name,
    const std::string& up_name);
