#include "timing.h"
#include "cuda_execution.h"
#include "storage/weight_loader.h"
#include "mfq/model_source.h"
#include "mfq_cuda_shared_gate.h"
#include "cpp_runtime/backends/cuda/kernels/shared_gate_fused.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <array>
#include <cstring>
#include <random>

namespace tb=mfq_tensor_backend;
namespace fb=format_bench;
namespace {
using mfq::cuda::shared_gate_detail::project_sigmoid;
struct Outputs {tb::Tensor dot,gate;};
void launch(int variant,const tb::Tensor& x,const tb::Tensor& w,Outputs& y) {
    const int rows=int(x.size(0)),width=int(x.size(1));
    const auto stream=mfq_current_cuda_stream();
    const auto* in=static_cast<const __half*>(x.data_ptr());
    const auto* weights=static_cast<const __nv_bfloat16*>(w.data_ptr());
    auto* dot=static_cast<__nv_bfloat16*>(y.dot.data_ptr());
    auto* gate=static_cast<__nv_bfloat16*>(y.gate.data_ptr());
    switch(variant) {
        case 0: project_sigmoid<128,false,false><<<rows,128,0,stream>>>(in,weights,dot,gate,width);break;
        case 1: project_sigmoid<128,true,false><<<rows,128,0,stream>>>(in,weights,dot,gate,width);break;
        case 2: project_sigmoid<256,true,false><<<rows,256,0,stream>>>(in,weights,dot,gate,width);break;
        case 3: project_sigmoid<256,false,true><<<rows,256,0,stream>>>(in,weights,dot,gate,width);break;
        default: throw std::runtime_error("invalid shared gate variant");
    }
    fb::check(cudaGetLastError());
}
std::size_t differences(const tb::Tensor& actual,const tb::Tensor& expected) {
    const auto cpu=actual.cpu().contiguous();
    const auto* a=static_cast<const std::uint16_t*>(cpu.data_ptr());
    const auto* b=static_cast<const std::uint16_t*>(expected.data_ptr());
    std::size_t count=0;
    for(std::int64_t i=0;i<cpu.numel();++i)count+=a[i]!=b[i];
    return count;
}
}
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("usage: mfq-cuda-shared-gate-bench MODEL.mfq");
    const auto model=mfq::open_model_source(argv[1]);
    CudaExecutionContext execution;
    std::vector<QuantLinear> weights;
    for(int layer:{0,8,16,24,32,40}) {
        auto w=load_quant_linear(execution,*model,"model.block."+std::to_string(layer)+".mlp.shared_expert.router.weight");
        if(!w.is_dense() || w.dense.scalar_type()!=tb::kBFloat16 || w.dense.size(0)!=1)
            throw std::runtime_error("shared scalar gate requires one BF16 weight row");
        weights.push_back(std::move(w));
    }
    const int width=int(weights.front().dense.size(1));
    std::mt19937 random(0x5aa913u);
    std::normal_distribution<float> normal(0.f,1.f);
    std::array<std::size_t,4> dot_diff{},gate_diff{};
    std::size_t production_gate_diff=0;
    std::size_t compared=0;
    for(int rows:{1,3,6,8}) {
        auto x=tb::empty({rows,width},weights.front().dense.options().dtype(tb::kFloat16));
        std::array<Outputs,4> actual;
        for(auto& y:actual) {
            y.dot=tb::empty({rows,1},weights.front().dense.options());
            y.gate=tb::empty({rows,1},weights.front().dense.options());
        }
        MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);
        auto warm=try_shared_gate_sigmoid_cuda(x,weights.front().dense);
        if(!warm)throw std::runtime_error("production shared gate unexpectedly rejected its supported view");
        fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));warm.reset();
        tb::Tensor captured_gate;
        graph.capture_begin();
        for(int variant=0;variant<4;++variant) {
            launch(variant,x,weights.front().dense,actual[variant]);
        }
        captured_gate=*try_shared_gate_sigmoid_cuda(x,weights.front().dense);
        graph.capture_end();
        for(int step=0;step<1024;++step) {
            std::vector<float> values(std::size_t(rows)*width);
            const float amplitude=std::array<float,8>{0.f,.03125f,.125f,.5f,1.f,2.f,8.f,32.f}[step%8];
            for(auto& value:values)value=normal(random)*amplitude;
            x.copy_(tb::tensor(values).reshape({rows,width}).to(tb::kCUDA,tb::kFloat16));
            for(std::size_t layer=0;layer<weights.size();++layer) {
                // dense_projection keeps 1..6 rows on the scalar decode path;
                // rows=8 are also compared individually here.
                std::vector<tb::Tensor> dots;
                for(int row=0;row<rows;++row)dots.push_back(weights[layer].forward(execution,x.narrow(0,row,1)));
                const auto reference_dot=tb::cat(dots,0);
                const auto reference_gate=tb::sigmoid(reference_dot);
                const auto saved_dot=reference_dot.cpu().contiguous();
                const auto saved_gate=reference_gate.cpu().contiguous();
                if(layer==0)graph.replay();
                for(int variant=0;variant<4;++variant) {
                    if(layer!=0)launch(variant,x,weights[layer].dense,actual[variant]);
                    dot_diff[variant]+=differences(actual[variant].dot,saved_dot);
                    gate_diff[variant]+=differences(actual[variant].gate,saved_gate);
                }
                const auto production=layer==0?captured_gate:*try_shared_gate_sigmoid_cuda(x,weights[layer].dense);
                production_gate_diff+=differences(production,saved_gate);
                compared+=rows;
            }
        }
    }
    for(int variant=0;variant<4;++variant)
        std::cout<<"CHECK variant="<<variant<<" values="<<compared<<" dot_bit_differences="<<dot_diff[variant]
            <<" sigmoid_bit_differences="<<gate_diff[variant]<<'\n';
    std::cout<<"CHECK production_values="<<compared<<" sigmoid_bit_differences="<<production_gate_diff<<'\n';
    auto x=tb::tensor(fb::input(1,width)).reshape({1,width}).to(tb::kCUDA,tb::kFloat16);
    for(int variant=-1;variant<5;++variant) {
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        std::vector<Outputs> outputs(weights.size());
        if(variant>=0)for(auto& y:outputs) {
            y.dot=tb::empty({1,1},weights.front().dense.options());
            y.gate=tb::empty({1,1},weights.front().dense.options());
        }
        auto body=[&] {
            for(std::size_t layer=0;layer<weights.size();++layer) {
                if(variant<0) {
                    outputs[layer].dot=weights[layer].forward(execution,x);
                    outputs[layer].gate=tb::sigmoid(outputs[layer].dot);
                } else if(variant==4)outputs[layer].gate=*try_shared_gate_sigmoid_cuda(x,weights[layer].dense);
                else launch(variant,x,weights[layer].dense,outputs[layer]);
            }
        };
        body();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(variant<0 || variant==4)for(auto& y:outputs)y={};
        graph.capture_begin();body();graph.capture_end();
        const auto samples=fb::measure([&]{graph.replay();},mfq_current_cuda_stream(),weights.size());
        std::cout<<std::setprecision(10)<<"BENCH variant="<<variant<<" graph_nodes="<<graph.nodes()
            <<" calls="<<weights.size()<<" us="<<fb::median(samples)<<" samples_us=";
        for(auto value:samples)std::cout<<value<<',';
        std::cout<<'\n'<<std::flush;
    }
    return production_gate_diff==0?0:2;
} catch(const std::exception& error) {
    std::cerr<<"shared gate benchmark: "<<error.what()<<'\n';return 1;
}
