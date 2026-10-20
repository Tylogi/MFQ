#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace {
template<bool AlignedQ8,bool Zero>
__device__ __forceinline__ void gr_prefill_decode_four(
        float (&values)[4],const uint8_t* packed,const uint8_t* row_bits,
        const int64_t* row_offsets,const uint8_t* scales,const uint8_t* minima,
        const float* row_scales,const float* row_minima,const __half* zero_scales,
        int neuron,int column,int width,int groups,int gs) {
    const int bits=AlignedQ8 || Zero?8:row_bits[neuron];
    const int64_t first_meta=int64_t(neuron)*groups+column/gs;
    const float outer_scale=Zero?0.f:row_scales[neuron];
    const float outer_minimum=Zero?0.f:row_minima[neuron];
    if((gs&3)==0 && column+4<=width) {
        const float scale=Zero?__half2float(zero_scales[first_meta]):
            __fmul_rn(outer_scale,float(scales[first_meta]));
        const float minimum=Zero?0.f:__fmul_rn(outer_minimum,float(minima[first_meta]));
        uint64_t codes=0;
        if constexpr(Zero) {
            codes=*reinterpret_cast<const uint32_t*>(packed+first_meta*gs+column%gs);
        } else if constexpr(AlignedQ8) {
            codes=*reinterpret_cast<const uint32_t*>(packed+(row_offsets[neuron]>>3)+column);
        } else {
            const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
            const int shift=int(bit&7),bytes=(shift+bits*4+7)/8;
            #pragma unroll
            for(int b=0;b<5;++b)if(b<bytes)codes|=uint64_t(packed[(bit>>3)+b])<<(8*b);
            codes>>=shift;
        }
        #pragma unroll
        for(int j=0;j<4;++j) {
            const unsigned code=unsigned(codes>>(bits*j))&((1u<<bits)-1);
            const float q=Zero?float(static_cast<int8_t>(code)):float(code);
            values[j]=__fsub_rn(__fmul_rn(scale,q),minimum);
        }
        return;
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        float value=0;
        const int c=column+j;
        if(c<width) {
            const int64_t meta=int64_t(neuron)*groups+c/gs;
            float q,scale,minimum;
            if constexpr(Zero) {
                q=float(static_cast<int8_t>(packed[meta*gs+c%gs]));
                scale=__half2float(zero_scales[meta]);minimum=0;
            } else {
                const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(c)*bits;
                const int shift=int(bit&7);
                unsigned code=packed[bit>>3];
                if(shift+bits>8)code|=unsigned(packed[(bit>>3)+1])<<8;
                q=float((code>>shift)&((1u<<bits)-1));
                scale=__fmul_rn(outer_scale,float(scales[meta]));
                minimum=__fmul_rn(outer_minimum,float(minima[meta]));
            }
            value=__fsub_rn(__fmul_rn(scale,q),minimum);
        }
        values[j]=value;
    }
}
}
