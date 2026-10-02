#pragma once

#include "quant_linear.h"
#include "mfe_weight.h"
#include "storage/moe_expert_cache.h"
#include "models/include/qwen4_exp.h"
#include "runtime.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::qwen4_exp {

using Routed = std::function<Tensor(
    CudaExecutionContext&, const Tensor&, const Tensor&)>;

inline Linear linear(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    auto weight=std::make_shared<QuantLinear>(load_quant_linear(execution, file,name));
    return [weight](CudaExecutionContext& execution, const Tensor& x) {
        return weight->forward(execution, x);
    };
}

inline Tensor dense(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    const auto& dtype=require_tensor(file, name).dtype;
    MFQ_RUNTIME_CHECK(dtype=="F32" || dtype=="F16" || dtype=="BF16",
        "Qwen requires a dense parameter: ",name);
    auto value=load_dense_gpu(execution, file,name);
    return value.to(dtype=="F16" ? tb::kFloat16 : dtype=="BF16" ? tb::kBFloat16 : tb::kFloat32);
}

inline Routed routed(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input) {
    const auto& dtype=require_tensor(file, name).dtype;
    if (dtype=="F32" || dtype=="F16" || dtype=="BF16") {
        MFQ_RUNTIME_CHECK(!moe_parallel_config(execution).enabled(),
            "expert parallelism requires packed routed tensors: ",name);
        auto w=dense(execution,file,name);
        MFQ_RUNTIME_CHECK(w.sizes().vec()==std::vector<int64_t>({experts,output,input}),
            "Qwen dense expert tensor shape mismatch: ",name);
        return [w,output,input](CudaExecutionContext&, const Tensor& x,const Tensor& ids) {
            const auto rows=ids.size(0), routes=ids.size(1);
            auto selected=w.index_select(0,ids.reshape({-1}).to(tb::kInt64)).reshape({rows,routes,output,input});
            auto source=x.dim()==2 ? x.unsqueeze(1).expand({rows,routes,input}) : x;
            return tb::matmul(selected,source.to(w.scalar_type()).unsqueeze(-1)).squeeze(-1);
        };
    }
    auto w=std::make_shared<MfeWeight>(load_mfe_gpu(execution, file,name,true,layer,"qwen4_exp"));
    MFQ_RUNTIME_CHECK(w->n_experts==experts && w->out_per_expert==output && w->neuron_len==input,
        "Qwen routed tensor shape mismatch: ",name);
    return [w,experts](CudaExecutionContext& execution, const Tensor& x,const Tensor& ids) {
        auto route=build_moe_route_plan(ids.to(tb::kInt32).contiguous(),int(experts));
        return w->forward(execution,x.contiguous(),route);
    };
}

inline Routed routed_gate_up(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& mlp_prefix,
    int layer, int64_t experts, int64_t width, int64_t input) {
    const auto base=mlp_prefix+".experts";
    const auto gate_name=base+".gate.weight",up_name=base+".up.weight";
    const bool has_gate=has_tensor(file, gate_name),has_up=has_tensor(file, up_name);
    MFQ_RUNTIME_CHECK(has_gate==has_up,"incomplete routed Gate/Up pair under ",base);
    if (!has_gate) return routed(execution,file,base+".gate_up.weight",layer,experts,2*width,input);
    auto gate=routed(execution,file,gate_name,layer,experts,width,input);
    auto up=routed(execution,file,up_name,layer,experts,width,input);
    return [gate=std::move(gate),up=std::move(up)](CudaExecutionContext& execution, const Tensor& x,const Tensor& ids) {
        return tb::cat({gate(execution,x,ids),up(execution,x,ids)},-1);
    };
}

inline void validate_load_options(
        const CudaExecutionContext& execution) {
    if (execution.tensor_parallel.enabled() ||
            execution.layer_placement.enabled() ||
            execution.n_gpu_layers >= 0 || execution.moe_expert_cache) {
        throw std::runtime_error(
            "Qwen native adapter supports expert parallelism, but "
            "tensor/layer parallelism and offload require a different placement path");
    }
}

} // namespace mfq::cuda::qwen4_exp
