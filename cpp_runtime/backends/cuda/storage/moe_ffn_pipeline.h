#pragma once
#include "mfe_weight.h"
#include "mfq/moe_dispatch_plan.h"
#include "runtime/decode_window.h"
#include <array>
using MoeFfnShared=std::function<mfq_tensor_backend::Tensor(CudaExecutionContext&,const mfq_tensor_backend::Tensor&)>;
using MoeFfnForward=std::function<mfq_tensor_backend::Tensor(CudaExecutionContext&,const mfq_tensor_backend::Tensor&,
    const mfq_tensor_backend::Tensor&,const mfq_tensor_backend::Tensor&)>;
using MoeFfnDispatchObserver=std::function<void(const mfq::MoeDispatchPlan&)>;
using MoeFfnPrefetch=std::function<mfq::cuda::DecodeWindow::Task(const mfq_tensor_backend::Tensor&)>;
struct QuantLinear;
struct MoeFfnSharedWeights {
    std::array<std::shared_ptr<const QuantLinear>,3> projections;
    MoeFfnShared gate;
};
MoeFfnForward make_moe_ffn_pipeline(const std::vector<std::shared_ptr<MfeWeight>>& gate_up,
    const std::vector<std::shared_ptr<MfeWeight>>& down,MoeFfnShared shared={},MoeFfnDispatchObserver observer={},
    MoeFfnSharedWeights shared_weights={},MoeFfnPrefetch* prefetch=nullptr);
