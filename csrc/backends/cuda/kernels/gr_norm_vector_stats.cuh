#pragma once

// Coalesced vector I/O, followed by the original 256-thread RMS tree.
// Only activation staging changes; all half rounding points are retained.
template<bool Half>
__device__ __forceinline__ float4 gr_norm_read4(const void* source,size_t at) {
    if constexpr(Half) {
        const auto bits=*reinterpret_cast<const uint2*>(static_cast<const __half*>(source)+at);
        const auto a=__half22float2(*reinterpret_cast<const __half2*>(&bits.x));
        const auto b=__half22float2(*reinterpret_cast<const __half2*>(&bits.y));
        return make_float4(a.x,a.y,b.x,b.y);
    }else return *reinterpret_cast<const float4*>(static_cast<const float*>(source)+at);
}
template<bool Half>
__device__ __forceinline__ void gr_norm_write4(void* destination,size_t at,float4 value) {
    if constexpr(Half) {
        const auto a=__floats2half2_rn(value.x,value.y),b=__floats2half2_rn(value.z,value.w);
        *reinterpret_cast<uint2*>(static_cast<__half*>(destination)+at)=make_uint2(
            *reinterpret_cast<const unsigned*>(&a),*reinterpret_cast<const unsigned*>(&b));
    }else *reinterpret_cast<float4*>(static_cast<float*>(destination)+at)=value;
}

template<bool Half,bool After,bool BranchHalf>
__global__ void __launch_bounds__(256) gated_residual_norm_vector_stats_kernel(
        MhcInput input,void* updated,float* inverses,float eps) {
    const int tid=threadIdx.x,sample=blockIdx.y,stream=blockIdx.x;
    __shared__ __align__(16) float values[2560];
    __shared__ float partial[256];
    const float gate=After?mhc_read(input.injection,size_t(sample)*4+stream,input.injection_half):0.f;
    const size_t base=(size_t(sample)*4+stream)*2560;
    #pragma unroll
    for(int item=0;item<3;++item) {
        const int column=tid*4+item*1024;
        if(column<2560) {
            auto value=gr_norm_read4<Half>(input.residual,base+column);
            if constexpr(After) {
                const auto branch=gr_norm_read4<BranchHalf>(input.branch,size_t(sample)*2560+column);
                const auto add=[&](float r,float b) {
                    return mhc_round(__fadd_rn(r,mhc_round(__fmul_rn(b,gate),BranchHalf && input.injection_half)),Half);
                };
                value=make_float4(add(value.x,branch.x),add(value.y,branch.y),
                    add(value.z,branch.z),add(value.w,branch.w));
            }
            *reinterpret_cast<float4*>(values+column)=value;
            gr_norm_write4<Half>(updated,base+column,value);
        }
    }
    __syncthreads();
    float sum=0.f;
    #pragma unroll
    for(int item=0;item<10;++item) {
        const float value=values[tid+item*256];
        sum=__fadd_rn(sum,__fmul_rn(value,value));
    }
    partial[tid]=sum;
    __syncthreads();
    if(tid<32) {
        const float a=__fadd_rn(partial[tid],partial[tid+128]);
        const float b=__fadd_rn(partial[tid+64],partial[tid+192]);
        const float c=__fadd_rn(partial[tid+32],partial[tid+160]);
        const float d=__fadd_rn(partial[tid+96],partial[tid+224]);
        float value=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
        #pragma unroll
        for(int shift=16;shift;shift>>=1) {
            const float other=__shfl_down_sync(0xffffffffu,value,shift);
            if(tid<shift)value=__fadd_rn(value,other);
        }
        if(tid==0)inverses[size_t(sample)*4+stream]=mhc_norm_inverse(value,2560,eps,input.fast_norm);
    }
}
