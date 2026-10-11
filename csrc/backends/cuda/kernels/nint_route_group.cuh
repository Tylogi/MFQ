#pragma once
#include "packed_nint.cuh"

namespace mfq::cuda::packed_nint {

// A checked fixed-width row can load its entire subgroup once. Normalizing
// the word window also keeps byte/bit cursor arithmetic out of the dot loop.
template<int GroupSize,int Bits,int Warps,int Rows,bool LateScale=false,bool PackedRows=false>
__device__ __forceinline__ void nint_fixed_group_rows(
        const uint8_t* stream,const int64_t* offsets,const uint8_t* subscale,
        const uint8_t* submin,const float* outer_scale,const float* outer_min,
        const int8_t* input,const float* input_scale,int local,int first,
        int output_rows,int groups,int stride,float* result,bool multi_sum=false,
        const uint4* row_metadata=nullptr) {
    constexpr int chunks=GroupSize/4,words=(GroupSize*Bits+31)/32;
    const auto* packed=stream+size_t(local)*stride;
    const uint8_t* rows[Rows]{};
    uint32_t starts[Rows]{};
    int available[Rows]{};
    float scales[Rows]{},minima[Rows]{},partial[Rows]{},minimum_partial[Rows]{};
#pragma unroll
    for(int r=0;r<Rows;++r)if(first+r<output_rows) {
        const int neuron=local*output_rows+first+r;
        uint64_t offset;
        if constexpr(PackedRows) {
            const uint4 metadata=__ldg(row_metadata+neuron);
            offset=uint64_t(metadata.x)|(uint64_t(metadata.y)<<32);
            scales[r]=__uint_as_float(metadata.z);minima[r]=__uint_as_float(metadata.w);
        } else {
            offset=uint64_t(offsets[neuron]);
            scales[r]=outer_scale[neuron];minima[r]=outer_min[neuron];
        }
        const uint64_t byte=(offset>>3)&~uint64_t(3);
        rows[r]=packed+byte;starts[r]=uint32_t(offset&31);
        available[r]=stride-int(byte);
    }
    for(int group=int(threadIdx.x)+(Warps==1?0:int(threadIdx.y)*32);
            group<groups;group+=32*Warps) {
        int x[chunks],sum=0;
#pragma unroll
        for(int chunk=0;chunk<chunks;++chunk) {
            x[chunk]=load_i8x4(input+group*GroupSize+chunk*4);
            sum=__dp4a(0x01010101,x[chunk],sum);
        }
#pragma unroll
        for(int r=0;r<Rows;++r)if(first+r<output_rows) {
            const uint32_t bit=starts[r]+uint32_t(group)*GroupSize*Bits;
            const uint32_t byte=(bit>>5)*4;
            int dot=0;
            if((reinterpret_cast<uintptr_t>(rows[r])&3u)==0 &&
                    uint64_t(byte)+(words+1)*4<=uint64_t(available[r])) {
                const auto* source=reinterpret_cast<const uint32_t*>(rows[r]+byte);
                uint32_t packed_words[words+1],normalized[words];
#pragma unroll
                for(int w=0;w<=words;++w)packed_words[w]=source[w];
#pragma unroll
                for(int w=0;w<words;++w)
                    normalized[w]=__funnelshift_r(packed_words[w],packed_words[w+1],bit&31);
#pragma unroll
                for(int chunk=0;chunk<chunks;++chunk) {
                    const int index=chunk*4*Bits/32,shift=chunk*4*Bits%32;
                    uint32_t value=normalized[index]>>shift;
                    if(shift+4*Bits>32)value|=normalized[index+1]<<(32-shift);
                    uint32_t codes;
                    if constexpr(Bits==4)codes=__byte_perm(value,value>>4,0x5140)&0x0f0f0f0fu;
                    else {
                        const uint32_t pair=__byte_perm(value,value>>(2*Bits),0x5410);
                        codes=__byte_perm(pair,pair>>Bits,0x6240)&(((1u<<Bits)-1)*0x01010101u);
                    }
                    dot=__dp4a(int(codes),x[chunk],dot);
                }
            } else {
#pragma unroll
                for(int chunk=0;chunk<chunks;++chunk)
                    dot=__dp4a(unpack_nint_codes4_aligned(rows[r],uint64_t(bit)+chunk*4*Bits,
                        Bits,uint64_t(available[r])),x[chunk],dot);
            }
            const int64_t metadata=int64_t(local*output_rows+first+r)*groups+group;
            if constexpr(LateScale) {
                partial[r]+=input_scale[group]*float(subscale[metadata])*float(dot);
                minimum_partial[r]+=input_scale[group]*float(submin[metadata])*float(sum);
            }else partial[r]+=input_scale[group]*(scales[r]*float(subscale[metadata])*float(dot)-
                minima[r]*float(submin[metadata])*float(sum));
        }
    }
    if(multi_sum) {
        warp_multi_sum::broadcast(partial,int(threadIdx.x));
        if constexpr(LateScale)warp_multi_sum::broadcast(minimum_partial,int(threadIdx.x));
    }
    else {
#pragma unroll
        for(int r=0;r<Rows;++r) {
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)partial[r]+=__shfl_xor_sync(0xffffffffu,partial[r],offset);
            if constexpr(LateScale) {
#pragma unroll
                for(int offset=16;offset>0;offset>>=1)
                    minimum_partial[r]+=__shfl_xor_sync(0xffffffffu,minimum_partial[r],offset);
            }
        }
    }
#pragma unroll
    for(int r=0;r<Rows;++r) {
        if constexpr(LateScale) {
            result[r]=scales[r]*partial[r]-minima[r]*minimum_partial[r];
        }else result[r]=partial[r];
    }
}
}
