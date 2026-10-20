#include "native/env.h"
#include "shared_gate.h"
#include "native/context.h"
#include "shared_gate_fused.cuh"
#include "native/nint_weight.h"
#include <cstdlib>

namespace {
template<bool Ordered>
__global__ void nint_gate_prepared(const int8_t* activation,const float* activation_scale,
        const uint8_t* packed,const int64_t* offset,const uint8_t* scales,const uint8_t* minima,
        const float* outer_scale,const float* outer_minimum,__half* output,int groups) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const auto* x=activation+size_t(blockIdx.x)*groups*48;
    const auto* xs=activation_scale+size_t(blockIdx.x)*groups;
    const auto* w=packed+(uint64_t(offset[0])>>3);
    __shared__ float terms[64][12],partial[4];
    float accumulator=0.f;
    if constexpr(Ordered) {
        const int relative=lane/12,chunk=lane%12;
        if(relative<2)for(int group=warp*2+relative;group<groups;group+=8) {
            const int column=group*48+chunk*4;
            const int q=*reinterpret_cast<const int*>(x+column);
            const int code=*reinterpret_cast<const int*>(w+column);
            const int sum=__dp4a(0x01010101,q,0);
            const int dot=__dp4a(code^int(0x80808080u),q,0)+128*sum;
            terms[group][chunk]=outer_scale[0]*float(scales[group])*float(dot)-
                outer_minimum[0]*float(minima[group])*float(sum);
        }
        __syncthreads();
        if(warp)return;
        if(relative<2)for(int group=relative;group<groups;group+=2)
            accumulator+=xs[group]*terms[group][chunk];
        #pragma unroll
        for(int step=16;step;step>>=1)accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,step);
    } else {
        const int member=lane&3;
        for(int base=warp*8;base<groups;base+=32) {
            const int group=base+(lane>>2);const bool active=group<groups;
            int dot=0,sum=0;
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk) {
                const int column=group*48+(chunk*4+member)*4;
                const int code=active?*reinterpret_cast<const int*>(w+column):0;
                const int q=active?*reinterpret_cast<const int*>(x+column):0;
                asm("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(dot) : "r"(code),"r"(q));
                sum=__dp4a(0x01010101,q,sum);
            }
            dot+=__shfl_xor_sync(0xffffffffu,dot,2,4);dot+=__shfl_xor_sync(0xffffffffu,dot,1,4);
            sum+=__shfl_xor_sync(0xffffffffu,sum,2,4);sum+=__shfl_xor_sync(0xffffffffu,sum,1,4);
            if(active && member==0)accumulator+=xs[group]*(outer_scale[0]*float(scales[group])*float(dot)-
                outer_minimum[0]*float(minima[group])*float(sum));
        }
        #pragma unroll
        for(int step=16;step;step>>=1)accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,step);
        if(!lane)partial[warp]=accumulator;
        __syncthreads();
        if(warp)return;
        accumulator=lane<4?partial[lane]:0.f;
        #pragma unroll
        for(int step=16;step;step>>=1)accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,step);
    }
    if(!lane)output[blockIdx.x]=__float2half_rn(1.f/(1.f+expf(-__half2float(__float2half_rn(accumulator)))));
}

// One packed GS48 NINT8 row. Keep the ordinary projection's activation
// quantization, affine/FMA order, Half dot boundary and Half sigmoid result.
template<bool Ordered>
__global__ void nint_gate_sigmoid(const __half* input,const uint8_t* packed,const int64_t* offset,
        const uint8_t* subgroup_scale,const uint8_t* subgroup_minimum,
        const float* outer_scale,const float* outer_minimum,__half* output,int width,int groups) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const auto* x=input+size_t(blockIdx.x)*width;
    const auto* weights=packed+(uint64_t(offset[0])>>3);
    __shared__ float terms[64][12],scales[64],partial[4];
    for(int group=warp;group<groups;group+=blockDim.x/32) {
        const int column=group*48+lane;
        const float a=column<width?__half2float(x[column]):0.f;
        const float b=lane<16 && column+32<width?__half2float(x[column+32]):0.f;
        float maximum=fmaxf(fabsf(a),fabsf(b));
        #pragma unroll
        for(int step=16;step;step>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,step));
        const float scale=maximum>0.f?maximum/127.f:1.f;
        const int qa=max(-127,min(127,int(roundf(a/scale))));
        const int qb=max(-127,min(127,int(roundf(b/scale))));
        // All lanes participate in every shuffle; twelve lanes consume one
        // original packed four-byte chunk each.
        uint32_t q=0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int index=(lane%12)*4+j;
            const int first=__shfl_sync(0xffffffffu,qa,index&31);
            const int second=__shfl_sync(0xffffffffu,qb,index&31);
            q|=uint32_t(uint8_t(index<32?first:second))<<(j*8);
        }
        int dot=0,sum=0;
        if(lane<12) {
            const int w=*reinterpret_cast<const int*>(weights+group*48+lane*4);
            sum=__dp4a(0x01010101,int(q),0);
            dot=__dp4a(w^int(0x80808080u),int(q),0)+128*sum;
        }
        const float factor=outer_scale[0]*float(subgroup_scale[group]);
        const float minimum=outer_minimum[0]*float(subgroup_minimum[group]);
        if constexpr(Ordered) {
            if(lane<12)terms[group][lane]=factor*float(dot)-minimum*float(sum);
        } else {
            #pragma unroll
            for(int step=16;step;step>>=1) {
                dot+=__shfl_xor_sync(0xffffffffu,dot,step);
                sum+=__shfl_xor_sync(0xffffffffu,sum,step);
            }
            if(!lane)terms[group][0]=factor*float(dot)-minimum*float(sum);
        }
        if(!lane)scales[group]=scale;
    }
    __syncthreads();
    float value=0.f;
    if constexpr(Ordered) {
        if(warp)return;
        if(lane/12<2)for(int group=lane/12;group<groups;group+=2)
            value+=scales[group]*terms[group][lane%12];
        #pragma unroll
        for(int step=16;step;step>>=1)value+=__shfl_xor_sync(0xffffffffu,value,step);
    } else {
        if(warp<4) {
            if((lane&3)==0)for(int group=warp*8+(lane>>2);group<groups;group+=32)
                value+=scales[group]*terms[group][0];
            #pragma unroll
            for(int step=16;step;step>>=1)value+=__shfl_xor_sync(0xffffffffu,value,step);
            if(!lane)partial[warp]=value;
        }
        __syncthreads();
        if(warp)return;
        value=lane<4?partial[lane]:0.f;
        #pragma unroll
        for(int step=16;step;step>>=1)value+=__shfl_xor_sync(0xffffffffu,value,step);
    }
    if(!lane) {
        const float dot=__half2float(__float2half_rn(value));
        output[blockIdx.x]=__float2half_rn(1.f/(1.f+expf(-dot)));
    }
}
}

std::optional<mfq_tensor_backend::Tensor> try_nint_shared_gate_sigmoid_cuda(
        const mfq_tensor_backend::Tensor& input,const NintWeight& w,
        const mfq_tensor_backend::Tensor& quantized,const mfq_tensor_backend::Tensor& scales) {
    namespace tb=mfq_tensor_backend;
    if(!input.defined() || !input.is_cuda() || !input.is_contiguous() || input.dim()!=2 ||
            input.scalar_type()!=tb::kFloat16 || input.size(0)<1 || input.size(0)>6 ||
            !w.aligned_q8 || w.q8_zero || w.out!=1 || w.gs!=48 || w.ng<1 || w.ng>64 ||
            input.size(1)!=w.neuron_len || w.neuron_len>w.ng*48 ||
            !w.q_packed.is_cuda() || w.q_packed.device()!=input.device())return std::nullopt;
    // Explicit changes to the reference arithmetic keep their original path.
    if(const auto* option=mfq::cuda::env_lookup("MFQ_NINT_Q8_DECODE"))if(std::atoi(option)!=2)return std::nullopt;
    if(input.size(0)==1) {
        if(const auto* option=mfq::cuda::env_lookup("MFQ_NINT_Q8_ORDERED"))if(option[0]=='0')return std::nullopt;
    } else if(const auto* option=mfq::cuda::env_lookup("MFQ_NINT_Q8_SMALL_WARPS"))if(option[0]=='0')return std::nullopt;
    for(const auto* tensor:{&w.q_packed,&w.row_q_bit_offsets,&w.sub_scale,&w.sub_min,&w.neuron_scale,&w.neuron_min})
        if(!tensor->is_cuda() || !tensor->is_contiguous() || tensor->device()!=input.device())return std::nullopt;
    if(quantized.defined() && (!scales.defined() || quantized.scalar_type()!=tb::kInt8 ||
            scales.scalar_type()!=tb::kFloat32 || !quantized.is_contiguous() || !scales.is_contiguous() ||
            quantized.device()!=input.device() || scales.device()!=input.device() ||
            quantized.numel()!=input.size(0)*w.ng*48 || scales.numel()!=input.size(0)*w.ng))return std::nullopt;
    auto output=tb::empty({input.size(0),1},input.options());
    mfq::cuda::DeviceGuard guard(input.get_device());
    const auto launch=[&](auto ordered) {
        if(quantized.defined()) {
            nint_gate_prepared<decltype(ordered)::value><<<int(input.size(0)),128,0,mfq_current_cuda_stream()>>>(
                quantized.data_ptr<int8_t>(),scales.data_ptr<float>(),w.q_packed.data_ptr<uint8_t>(),
                w.row_q_bit_offsets.data_ptr<int64_t>(),w.sub_scale.data_ptr<uint8_t>(),w.sub_min.data_ptr<uint8_t>(),
                w.neuron_scale.data_ptr<float>(),w.neuron_min.data_ptr<float>(),static_cast<__half*>(output.data_ptr()),int(w.ng));
            return;
        }
        nint_gate_sigmoid<decltype(ordered)::value><<<int(input.size(0)),1024,0,mfq_current_cuda_stream()>>>(
            static_cast<const __half*>(input.data_ptr()),w.q_packed.data_ptr<uint8_t>(),
            w.row_q_bit_offsets.data_ptr<int64_t>(),w.sub_scale.data_ptr<uint8_t>(),w.sub_min.data_ptr<uint8_t>(),
            w.neuron_scale.data_ptr<float>(),w.neuron_min.data_ptr<float>(),
            static_cast<__half*>(output.data_ptr()),int(input.size(1)),int(w.ng));
    };
    if(input.size(0)==1)launch(std::true_type{});else launch(std::false_type{});
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());return output;
}

std::optional<mfq_tensor_backend::Tensor> try_shared_gate_sigmoid_cuda(
        const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight) {
    namespace tb=mfq_tensor_backend;
    if(!input.defined() || !weight.defined() || !input.is_cuda() || !weight.is_cuda() ||
        input.get_device()!=weight.get_device() || !input.is_contiguous() || !weight.is_contiguous() ||
        input.scalar_type()!=tb::kFloat16 || weight.scalar_type()!=tb::kBFloat16 ||
        input.dim()!=2 || weight.dim()!=2 || weight.size(0)!=1 ||
        input.size(0)<1 || input.size(0)>8 || input.size(1)<1 || input.size(1)>4096 ||
        input.size(1)!=weight.size(1))return std::nullopt;
    auto output=tb::empty({input.size(0),1},weight.options());
    mfq::cuda::DeviceGuard guard(input.get_device());
    const auto stream=mfq::cuda::current_stream(input.get_device()).stream();
    mfq::cuda::shared_gate_detail::project_sigmoid<128,true,false>
        <<<int(input.size(0)),128,0,stream>>>(
            static_cast<const __half*>(input.data_ptr()),
            static_cast<const __nv_bfloat16*>(weight.data_ptr()),nullptr,
            static_cast<__nv_bfloat16*>(output.data_ptr()),int(input.size(1)));
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}
