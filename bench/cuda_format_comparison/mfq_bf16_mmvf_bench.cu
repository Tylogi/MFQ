#include "timing.h"
#include "cuda_execution.h"
#include "storage/weight_loader.h"
#include "mfq/model_source.h"
#include "cpp_runtime/backends/cuda/kernels/bf16_mmvf.cuh"
#include <array>
#include <cstring>
#include <random>

namespace tb=mfq_tensor_backend;
namespace fb=format_bench;
namespace {
template<int Threads,int Rows>
void launch(const tb::Tensor& x,const tb::Tensor& weight,tb::Tensor& y) {
    mfq::cuda::bf16_mmvf_detail::project_rows<Threads,Rows><<<
        dim3(unsigned((weight.size(0)+Rows-1)/Rows),unsigned(x.size(0))),Threads,0,
        mfq_current_cuda_stream()>>>(static_cast<const __nv_bfloat16*>(x.data_ptr()),
            static_cast<const __nv_bfloat16*>(weight.data_ptr()),
            static_cast<__nv_bfloat16*>(y.data_ptr()),int(weight.size(1)),int(weight.size(0)));
    fb::check(cudaGetLastError());
}
struct Variant {int threads,rows;void (*launch)(const tb::Tensor&,const tb::Tensor&,tb::Tensor&);};
constexpr std::array<Variant,9> variants{{
    {32,4,launch<32,4>},{64,4,launch<64,4>},{128,1,launch<128,1>},
    {128,2,launch<128,2>},{128,4,launch<128,4>},{128,8,launch<128,8>},
    {256,2,launch<256,2>},{256,4,launch<256,4>},{256,8,launch<256,8>}
}};
float value(std::uint16_t bits) {
    const std::uint32_t wide=std::uint32_t(bits)<<16;
    float result;std::memcpy(&result,&wide,sizeof(result));return result;
}
struct Check {std::size_t different=0,values=0,argmax_different=0;double squared_error=0,squared_ref=0,max_error=0;};
void compare(Check& check,const tb::Tensor& actual,const tb::Tensor& expected) {
    const auto saved=actual.cpu().contiguous();
    const auto* a=static_cast<const std::uint16_t*>(saved.data_ptr());
    const auto* b=static_cast<const std::uint16_t*>(expected.data_ptr());
    for(std::int64_t i=0;i<saved.numel();++i) {
        check.different+=a[i]!=b[i];++check.values;
        const double av=value(a[i]),bv=value(b[i]),error=av-bv;
        check.squared_error+=error*error;check.squared_ref+=bv*bv;
        check.max_error=std::max(check.max_error,std::abs(error));
    }
    for(std::int64_t row=0;row<saved.size(0);++row) {
        std::int64_t ai=0,bi=0;
        const auto n=saved.size(1);
        for(std::int64_t col=1;col<n;++col) {
            if(value(a[row*n+col])>value(a[row*n+ai]))ai=col;
            if(value(b[row*n+col])>value(b[row*n+bi]))bi=col;
        }
        check.argmax_different+=ai!=bi;
    }
}
}
int main(int argc,char** argv) try {
    if(argc!=2)throw std::runtime_error("usage: mfq-cuda-bf16-mmvf-bench MODEL.mfq");
    const auto model=mfq::open_model_source(argv[1]);
    CudaExecutionContext execution;
    const auto weight=load_quant_linear(execution,*model,"model.output.weight");
    if(!weight.is_dense() || weight.dense.scalar_type()!=tb::kBFloat16 ||
        weight.dense.dim()!=2 || weight.dense.size(1)%2)
        throw std::runtime_error("MMVF probe requires a contiguous even-width BF16 output matrix");
    const int width=int(weight.dense.size(1)),outputs=int(weight.dense.size(0));
    const auto bytes=weight.dense.nbytes();
    int l2=0;fb::check(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,mfq_current_cuda_device()));
    if(bytes<4*std::size_t(l2))throw std::runtime_error("MMVF weight must exceed four times L2");
    std::array<Check,variants.size()> checks{};
    std::mt19937 random(0x865419u);std::normal_distribution<float> normal(0.f,1.f);
    for(int tokens:{1,3,6}) {
        auto x=tb::empty({tokens,width},weight.dense.options());
        std::array<tb::Tensor,variants.size()> y;
        for(auto& output:y)output=tb::empty({tokens,outputs},weight.dense.options());
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        graph.capture_begin();
        for(std::size_t i=0;i<variants.size();++i)variants[i].launch(x,weight.dense,y[i]);
        graph.capture_end();
        for(int step=0;step<24;++step) {
            std::vector<float> input(std::size_t(tokens)*width);
            const float amplitude=std::array<float,8>{0.f,.03125f,.125f,.5f,1.f,2.f,8.f,32.f}[step%8];
            for(auto& v:input)v=normal(random)*amplitude;
            x.copy_(tb::tensor(input).reshape({tokens,width}).to(tb::kCUDA,tb::kBFloat16));
            const auto expected=weight.forward(execution,x).cpu().contiguous();
            graph.replay();
            for(std::size_t i=0;i<variants.size();++i)compare(checks[i],y[i],expected);
        }
    }
    for(std::size_t i=0;i<variants.size();++i) {
        const auto& check=checks[i];
        std::cout<<std::setprecision(10)<<"CHECK threads="<<variants[i].threads<<" rows="<<variants[i].rows
            <<" values="<<check.values<<" bit_differences="<<check.different
            <<" argmax_differences="<<check.argmax_different
            <<" relative_l2="<<std::sqrt(check.squared_error/std::max(check.squared_ref,1e-300))
            <<" max_abs="<<check.max_error<<'\n'<<std::flush;
    }
    auto x=tb::tensor(fb::input(1,width)).reshape({1,width}).to(tb::kCUDA,tb::kBFloat16);
    for(int i=-1;i<int(variants.size());++i) {
        MfqCudaGraph graph;mfq_prepare_cuda_graph_memory(graph);
        auto output=tb::empty({1,outputs},weight.dense.options());
        const auto body=[&] {
            if(i<0)output=weight.forward(execution,x);
            else variants[i].launch(x,weight.dense,output);
        };
        body();fb::check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        if(i<0)output={};
        graph.capture_begin();body();graph.capture_end();
        const auto samples=fb::measure([&]{graph.replay();},mfq_current_cuda_stream(),1);
        const double us=fb::median(samples);
        std::cout<<std::setprecision(10)<<"BENCH variant="<<i
            <<" threads="<<(i<0?0:variants[i].threads)<<" rows="<<(i<0?0:variants[i].rows)
            <<" width="<<width<<" outputs="<<outputs<<" weight_bytes="<<bytes
            <<" us="<<us<<" effective_GBs="<<bytes/(us*1000.)<<" samples_us=";
        for(auto sample:samples)std::cout<<sample<<',';
        std::cout<<'\n'<<std::flush;
    }
    // This is a probe: report rounding differences without labelling them exact.
    // Argmax changes are a failed decode candidate even when relative error is small.
    for(const auto& check:checks)if(check.argmax_different)return 2;
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"BF16 MMVF benchmark: "<<error.what()<<'\n';return 1;
}
