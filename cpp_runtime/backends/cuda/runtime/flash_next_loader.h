#pragma once

#include "cuda_transformer.h"
#include "flash_next_common.h"
#include "moe_expert_cache.h"
#include "models/flash_next.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Flash-Next layers use the shared CUDA model loader and the common Block
// context. Architecture-specific execution stays out of the CLI/model loop.
namespace flash_runtime {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;
using Linear = mfq::flash_next::Linear;
using Routed = std::function<Tensor(const Tensor&, const Tensor&)>;

inline Linear linear(const mfq::ModelSource& file, const std::string& name) {
    auto weight=std::make_shared<QuantLinear>(load_quant_linear(file,name));
    return [weight](const Tensor& x) { return weight->forward(x); };
}

inline Tensor dense(const mfq::ModelSource& file, const std::string& name) {
    const auto& dtype=require_tensor(file, name).dtype;
    MFQ_RUNTIME_CHECK(dtype=="F32" || dtype=="F16" || dtype=="BF16",
        "Flash-Next requires a dense parameter: ",name);
    auto value=load_dense_gpu(file,name);
    return value.to(dtype=="F16" ? tb::kFloat16 : dtype=="BF16" ? tb::kBFloat16 : tb::kFloat32);
}

inline Routed routed(const mfq::ModelSource& file, const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input) {
    const auto& dtype=require_tensor(file, name).dtype;
    if (dtype=="F32" || dtype=="F16" || dtype=="BF16") {
        MFQ_RUNTIME_CHECK(!moe_parallel_config().enabled(),
            "expert parallelism requires packed routed tensors: ",name);
        auto w=dense(file,name);
        MFQ_RUNTIME_CHECK(w.sizes().vec()==std::vector<int64_t>({experts,output,input}),
            "Flash-Next dense expert tensor shape mismatch: ",name);
        return [w,output,input](const Tensor& x,const Tensor& ids) {
            const auto rows=ids.size(0), routes=ids.size(1);
            auto selected=w.index_select(0,ids.reshape({-1}).to(tb::kInt64)).reshape({rows,routes,output,input});
            auto source=x.dim()==2 ? x.unsqueeze(1).expand({rows,routes,input}) : x;
            return tb::matmul(selected,source.to(w.scalar_type()).unsqueeze(-1)).squeeze(-1);
        };
    }
    auto w=std::make_shared<MfeWeight>(load_mfe_gpu(file,name,true,layer,"flash_next"));
    MFQ_RUNTIME_CHECK(w->n_experts==experts && w->out_per_expert==output && w->neuron_len==input,
        "Flash-Next routed tensor shape mismatch: ",name);
    return [w,experts](const Tensor& x,const Tensor& ids) {
        auto route=build_moe_route_plan(ids.to(tb::kInt32).contiguous(),int(experts));
        return w->forward(x.contiguous(),route);
    };
}

inline Routed routed_gate_up(const mfq::ModelSource& file, const std::string& mlp_prefix,
    int layer, int64_t experts, int64_t width, int64_t input) {
    const auto base=mlp_prefix+".experts";
    const auto gate_name=base+".gate.weight",up_name=base+".up.weight";
    const bool has_gate=has_tensor(file, gate_name),has_up=has_tensor(file, up_name);
    MFQ_RUNTIME_CHECK(has_gate==has_up,"incomplete routed Gate/Up pair under ",base);
    if (!has_gate) return routed(file,base+".gate_up.weight",layer,experts,2*width,input);
    auto gate=routed(file,gate_name,layer,experts,width,input);
    auto up=routed(file,up_name,layer,experts,width,input);
    return [gate=std::move(gate),up=std::move(up)](const Tensor& x,const Tensor& ids) {
        return tb::cat({gate(x,ids),up(x,ids)},-1);
    };
}

inline Linear headwise(Routed projection,int64_t heads,int64_t output) {
    return [projection=std::move(projection),heads,output](const Tensor& x) {
        MFQ_RUNTIME_CHECK(x.dim()==4 && x.size(2)==heads,"Flash-Next head-wise projection shape mismatch");
        const auto b=x.size(0),t=x.size(1),rows=b*t*heads;
        auto ids=tb::arange(rows,x.options().dtype(tb::kInt32)).remainder(heads).reshape({rows,1});
        return projection(x.reshape({rows,x.size(-1)}),ids).reshape({b,t,heads,output});
    };
}

inline Linear dense_ffn(const mfq::ModelSource& file,const std::string& p,double limit) {
    auto gate=linear(file,p+".gate.weight"),up=linear(file,p+".up.weight"),down=linear(file,p+".down.weight");
    return [gate,up,down,limit](const Tensor& x) {
        auto g=tb::clamp_max(gate(x),limit),u=tb::clamp(up(x),-limit,limit);
        return down((g*tb::sigmoid(g))*u);
    };
}
} // namespace flash_runtime

namespace mfq::cuda::flash_next {
inline void validate_load_options() {
    if (g_tensor_parallel.enabled() || g_layer_placement.enabled() ||
            g_n_gpu_layers >= 0 || g_moe_expert_cache) {
        throw std::runtime_error(
            "Flash-Next native adapter supports expert parallelism, but "
            "tensor/layer parallelism and offload still require a different placement path");
    }
}
} // namespace mfq::cuda::flash_next
