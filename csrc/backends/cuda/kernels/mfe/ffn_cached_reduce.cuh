#pragma once
#include "kernels/mfe/ffn.h"
#include <cuda_bf16.h>

namespace mfq::cuda { namespace {
// The transfer wait proves that every routed projection already completed.
// Consecutive lanes combine consecutive output columns in the original FMA order.
template<int Tile>
__device__ __forceinline__ bool mfe_ffn_cached_reduce(const MfeFfnBatch& b,const uint32_t* hot) {
    if(!hot || !*hot)return false;
    const int token=int(blockIdx.y);
    const int col=Tile==32?int(blockIdx.x)*32+int(threadIdx.x):
        int(blockIdx.x)*int(blockDim.x*blockDim.y)+int(threadIdx.y*blockDim.x+threadIdx.x);
    if((Tile!=32 || threadIdx.y==0) && col<b.output_width) {
        const int all_routes=b.routes+int(b.shared!=nullptr);
        float sum=0.f;
        for(int route=0;route<b.routes;++route) {
            const int expert=b.ids[token*b.routes+route];
            const auto value=unsigned(expert)<unsigned(b.experts)?
                b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+col]:__float2half(0.f);
            sum+=b.route_weights[token*b.routes+route]*__half2float(value);
        }
        const float routed=__half2float(__float2half(sum));
        const float gate=b.shared?(b.shared_product_half?
            __half2float(static_cast<const __half*>(b.shared_gate)[token]):b.shared_gate_bfloat?
            __bfloat162float(static_cast<const __nv_bfloat16*>(b.shared_gate)[token]):
            static_cast<const float*>(b.shared_gate)[token]):0.f;
        float contribution=b.shared?__fmul_rn(__half2float(b.resident_pairs[(int64_t(token)*all_routes+b.routes)*b.output_width+col]),gate):0.f;
        if(b.shared_product_half)contribution=__half2float(__float2half(contribution));
        const float value=b.shared?__fadd_rn(routed,contribution):routed;
        const auto at=int64_t(token)*b.output_width+col;
        if(b.output_float)static_cast<float*>(b.output)[at]=value;
        else static_cast<__half*>(b.output)[at]=__float2half(value);
    }
    return true;
}
}
}
