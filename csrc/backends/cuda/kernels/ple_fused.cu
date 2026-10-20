#include "ple_fused.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace mfq_qwen4_exp {
namespace tb=mfq_tensor_backend;
namespace {
// Match the native serial score sum exactly. Coalesced product loads replace
// the old four-thread reduction's repeated global loads; no reassociation.
template<class Key,class Query,class Value>
__global__ void ple_gate_norm_kernel(const Key* key,const Query* query,const Value* value,
        const float* norm,float* gated,float* normalized,int hidden,int streams,double divisor,float eps) {
    extern __shared__ float products[];
    __shared__ float partial[256],gate,inverse;
    const int tid=threadIdx.x,row=blockIdx.x,s=row%streams;
    const size_t base=size_t(row)*hidden,value_base=size_t(row/streams)*hidden;
    for(int column=tid;column<hidden;column+=256)
        products[column]=__fmul_rn(float(key[base+column]),float(query[base+column]));
    __syncthreads();
    if(tid==0) {
        float score=0.f;
        for(int column=0;column<hidden;++column)score=__fadd_rn(score,products[column]);
        // The native scalar fallback divides by the original double constant.
        score=float(double(score)/divisor);
        const float sign=float(score>0.f)-float(score<0.f);
        const float root=__fmul_rn(sign,sqrtf(fmaxf(fabsf(score),1e-6f)));
        gate=1.f/(1.f+expf(-root));
    }
    __syncthreads();
    float sum=0.f;
    for(int column=tid;column<hidden;column+=256) {
        const float v=__fmul_rn(gate,float(value[value_base+column]));
        gated[base+column]=v;
        sum=__fadd_rn(sum,__fmul_rn(v,v));
    }
    partial[tid]=sum;__syncthreads();
    for(int stride=128;stride;stride>>=1) {
        if(tid<stride)partial[tid]=__fadd_rn(partial[tid],partial[tid+stride]);
        __syncthreads();
    }
    if(tid==0)inverse=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(partial[0],float(hidden)),eps)));
    __syncthreads();
    for(int column=tid;column<hidden;column+=256)
        normalized[base+column]=__fmul_rn(__fmul_rn(gated[base+column],inverse),
            __fadd_rn(norm[s*hidden+column],1.f));
}

template<class Output>
__global__ void ple_conv_add_kernel(const float* input,const float* weight,const float* previous,
        const float* gated,Output* output,float* next,int batch,int tokens,int channels,int taps,int dilation) {
    const int history=(taps-1)*dilation;
    const size_t out_count=size_t(batch)*tokens*channels,next_count=size_t(batch)*history*channels;
    const size_t count=out_count>next_count?out_count:next_count;
    for(size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;at<count;at+=size_t(blockDim.x)*gridDim.x) {
        if(at<out_count) {
            const int channel=at%channels,t=(at/channels)%tokens,b=at/(size_t(channels)*tokens);
            float sum=0.f;
            for(int tap=0;tap<taps;++tap) {
                const int position=t+tap*dilation;
                const float x=position<history?
                    (previous?previous[(size_t(b)*history+position)*channels+channel]:0.f):
                    input[(size_t(b)*tokens+position-history)*channels+channel];
                sum=__fadd_rn(sum,__fmul_rn(x,weight[channel*taps+tap]));
            }
            const float convolved=__fmul_rn(sum,1.f/(1.f+expf(-sum)));
            output[at]=static_cast<Output>(__fadd_rn(gated[at],convolved));
        }
        if(at<next_count) {
            const int channel=at%channels,p=(at/channels)%history,b=at/(size_t(channels)*history);
            const int position=tokens+p;
            next[at]=position<history?
                (previous?previous[(size_t(b)*history+position)*channels+channel]:0.f):
                input[(size_t(b)*tokens+position-history)*channels+channel];
        }
    }
}
}

std::vector<Tensor> ple_gate_norm_cuda(const Tensor& key,const Tensor& query,const Tensor& value,
        const Tensor& norm,int64_t hidden,int64_t streams,double eps) {
    const auto valid=[](const Tensor& x) {return x.is_cuda() && x.is_contiguous() &&
        (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32);};
    MFQ_RUNTIME_CHECK(valid(key) && valid(query) && valid(value) && key.sizes()==query.sizes() &&
        hidden>0 && hidden<=8192 && streams>0 && key.numel()==value.numel()*streams &&
        value.numel()%hidden==0 && key.device()==query.device() && key.device()==value.device() &&
        norm.is_cuda() && norm.is_contiguous() && norm.scalar_type()==tb::kFloat32 &&
        norm.numel()==hidden*streams && norm.device()==key.device(),"PLE fused gate geometry/dtype invalid");
    MfqCudaGuard guard(key.device());
    auto gated=tb::empty(key.sizes(),key.options().dtype(tb::kFloat32));
    auto normalized=tb::empty(key.sizes(),key.options().dtype(tb::kFloat32));
    const auto launch=[&](auto k) {
      const auto launch_query=[&](auto q) {
        const auto launch_value=[&](auto v) {
            using K=decltype(k);using Q=decltype(q);using V=decltype(v);
            ple_gate_norm_kernel<K,Q,V><<<unsigned(key.numel()/hidden),256,unsigned(hidden*sizeof(float)),mfq_current_cuda_stream()>>>(
                reinterpret_cast<const K*>(key.data_ptr()),reinterpret_cast<const Q*>(query.data_ptr()),
                reinterpret_cast<const V*>(value.data_ptr()),norm.data_ptr<float>(),gated.data_ptr<float>(),
                normalized.data_ptr<float>(),int(hidden),int(streams),std::sqrt(double(hidden)),float(eps));
        };
        if(value.scalar_type()==tb::kFloat16)launch_value(__half{});else launch_value(float{});
      };
      if(query.scalar_type()==tb::kFloat16)launch_query(__half{});else launch_query(float{});
    };
    if(key.scalar_type()==tb::kFloat16)launch(__half{});else launch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());
    return {gated,normalized};
}

std::vector<Tensor> ple_conv_add_cuda(const Tensor& input,const Tensor& source_weight,
        const std::optional<Tensor>& previous,int64_t dilation,const Tensor& gated,bool half_output) {
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.dim()==3 &&
        input.scalar_type()==tb::kFloat32 && gated.is_contiguous() && gated.scalar_type()==tb::kFloat32 &&
        gated.sizes()==input.sizes() && gated.device()==input.device() && dilation>0,
        "PLE fused convolution input invalid");
    auto weight=source_weight;
    if(weight.dim()==3 && weight.size(1)==1)weight=weight.select(1,0);
    MFQ_RUNTIME_CHECK(weight.is_cuda() && weight.dim()==2 && weight.size(0)==input.size(2) &&
        weight.size(1)>0 && weight.size(1)<=32 && weight.device()==input.device(),"PLE convolution weights invalid");
    const int64_t history=(weight.size(1)-1)*dilation;
    const std::vector<int64_t> shape{input.size(0),history,input.size(2)};
    MFQ_RUNTIME_CHECK(!previous || (previous->is_cuda() && previous->is_contiguous() &&
        previous->scalar_type()==tb::kFloat32 && previous->sizes().vec()==shape &&
        previous->device()==input.device()),"PLE convolution state invalid");
    MfqCudaGuard guard(input.device());weight=weight.to(tb::kFloat32).contiguous();
    auto output=tb::empty(input.sizes(),input.options().dtype(half_output?tb::kFloat16:tb::kFloat32));
    auto next=tb::empty(shape,input.options());
    const int64_t count=std::max(input.numel(),next.numel());
    const auto launch=[&](auto out) {
        ple_conv_add_kernel<decltype(out)><<<unsigned((count+255)/256),256,0,mfq_current_cuda_stream()>>>(
            input.data_ptr<float>(),weight.data_ptr<float>(),previous?previous->data_ptr<float>():nullptr,
            gated.data_ptr<float>(),reinterpret_cast<decltype(out)*>(output.data_ptr()),next.data_ptr<float>(),
            int(input.size(0)),int(input.size(1)),int(input.size(2)),int(weight.size(1)),int(dilation));
    };
    if(half_output)launch(__half{});else launch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());return {output,next};
}
}
