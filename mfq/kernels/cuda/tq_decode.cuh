// Fixed Gaussian book; independent 1..8-bit row precision.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

template<class Kernel> int launchable(Kernel kernel,int threads) {
    int resident=0;cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,threads,0);
    return e!=cudaSuccess?int(e):resident?0:-2;
}

__device__ __forceinline__ uint32_t select_word(const uint32_t* chunks,int count,int index) {
    uint32_t value=0;
    #pragma unroll
    for(int j=0;j<6;++j) if(j<count && j==index)value=chunks[j];
    return value;
}

template<int V,bool Lookup,bool Columns,int Bits=0>
__device__ __forceinline__ void decode_vector(const uint32_t* words,uint64_t descriptor,
        const void* selectors,const half* relative,const half* lut,const half* coefficients,
        int row,int col,int width,int gslog,half (&values)[V]) {
    int bits=Bits?Bits:int((descriptor>>32)&7)+1;
    float anchor=__half2float(__ushort_as_half(uint16_t(descriptor>>40)));
    int ringwords=8*bits;
    int minimum=(255-((col+V-1)&255))*bits;
    int first=minimum>>5;
    const uint32_t* ring=words+uint32_t(descriptor)+(col>>8)*ringwords;
    constexpr int NW=(31+(V-1)*(Bits?Bits:8)+16+31)/32;
    uint32_t chunks[6]={};
    #pragma unroll
    for(int j=0;j<NW;++j) {
        int at=first+j;if(at>=ringwords)at-=ringwords;
        chunks[j]=__ldg(ring+at);
    }
    #ifdef TRELLIS_NORMALIZED_WINDOW
    constexpr int NORM=((V-1)*(Bits?Bits:8)+16+31)/32;
    uint32_t normalized[NORM];
    if constexpr(Bits>0) {
        #pragma unroll
        for(int j=0;j<NORM;++j)normalized[j]=__funnelshift_r(chunks[j],j+1<NW?chunks[j+1]:0,minimum&31);
    }
    #endif
    float scale=anchor;
    if constexpr(!Columns) {
        int group=(row*((width+(1<<gslog)-1)>>gslog))+(col>>gslog);
        uint8_t packed=__ldg(static_cast<const uint8_t*>(selectors)+(group>>1));
        int code=(packed>>((group&1)*4))&15;
        scale=__fmul_rn(anchor,__half2float(__ldg(relative+code)));
    }
    float column_scales[V];
    if constexpr(Columns) {
        uint32_t packed_scale[V/2];
        if constexpr(V==4) {
            uint2 b=__ldg(reinterpret_cast<const uint2*>(static_cast<const half*>(selectors)+col));
            packed_scale[0]=b.x;packed_scale[1]=b.y;
        } else {
            #pragma unroll
            for(int j=0;j<V/8;++j) {
                uint4 b=__ldg(reinterpret_cast<const uint4*>(static_cast<const half*>(selectors)+col)+j);
                packed_scale[4*j]=b.x;packed_scale[4*j+1]=b.y;packed_scale[4*j+2]=b.z;packed_scale[4*j+3]=b.w;
            }
        }
        #pragma unroll
        for(int j=0;j<V;++j)column_scales[j]=__fmul_rn(anchor,__half2float(__ushort_as_half(uint16_t(packed_scale[j/2]>>((j&1)*16)))));
    }
    #ifdef TRELLIS_PAIR_BOOK
    #pragma unroll
    for(int j=0;j<V;j+=2) {
        uint32_t sum[2];
        #pragma unroll
        for(int t=0;t<2;++t) {
            uint32_t state;
            if constexpr(Bits>0) {
                int pos=(V-1-j-t)*Bits,wi=pos/32,shift=pos&31;
                state=__funnelshift_r(normalized[wi],wi+1<NORM?normalized[wi+1]:0,shift)&65535;
            } else {
                int pos=(255-((col+j+t)&255))*bits-first*32,wi=pos/32;
                state=__funnelshift_r(select_word(chunks,NW,wi),select_word(chunks,NW,wi+1),pos&31)&65535;
            }
            sum[t]=__dp4a(state*0x83dcd12du,0x01010101u,0u);
        }
        half2 base=__halves2half2(__ushort_as_half(0x6400+sum[0]),__ushort_as_half(0x6400+sum[1]));
        half2 q=__hfma2(base,__half2half2(__ushort_as_half(0x1eee)),__half2half2(__ushort_as_half(0xc931)));
        float2 z=__half22float2(q);
        float s0=scale,s1=scale;
        if constexpr(Columns){s0=column_scales[j];s1=column_scales[j+1];}
        half2 w=__floats2half2_rn(__fmul_rn(z.x,s0),__fmul_rn(z.y,s1));
        values[j]=__low2half(w);values[j+1]=__high2half(w);
    }
    #endif
}
