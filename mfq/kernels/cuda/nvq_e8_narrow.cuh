#pragma once
#include "packed_nvq.cuh"
#include <climits>

namespace mfq::cuda {

// Prove every row/group bit offset, including the padded last group, before
// selecting signed 32-bit metadata arithmetic. Device pointers remain 64-bit.
inline bool nvq_e8_narrow_eligible(const packed_nvq::NvqDeviceWeight& w,
        int rows,int experts,int format) {
    if((format!=5 && format!=13 && format!=14) || rows<=0 || experts<=0 ||
            w.N<=0 || w.ng<=0 || w.nvec<=0 || w.nvec!=w.nsign || w.sub_bits!=4 ||
            int64_t(w.ng)!=(int64_t(w.nvec)+2)/3)return false;
    const int64_t total_rows=int64_t(rows)*experts;
    if(total_rows>w.N)return false;
    const int bits=packed_nvq::format_index_bits(format);
    const int64_t first=(total_rows-1)*w.nvec+int64_t(w.ng-1)*3;
    // Reserve space for the complete three-vector window and byte guards.
    if(first>(INT_MAX-64)/bits || first>(INT_MAX-64)/7)return false;
    const int64_t states=total_rows*w.ng;
    if(states>INT_MAX)return false;
    if(w.indices_nbytes<=0 || w.indices_nbytes>INT_MAX ||
            w.aux_nbytes<=0 || w.aux_nbytes>INT_MAX ||
            w.sub_scale_nbytes<=0 || w.sub_scale_nbytes>INT_MAX)return false;
    return w.indices_nbytes>=(total_rows*w.nvec*bits+7)/8 &&
           w.aux_nbytes>=(total_rows*w.nsign*7+7)/8 &&
           w.sub_scale_nbytes>=(states+1)/2;
}

__device__ __forceinline__ uint32_t load_e8_state32(const uint8_t* data,int linear) {
    return (data[linear>>1]>>((linear&1)*4))&15u;
}

__device__ __forceinline__ uint32_t load_e8_signs32(const uint8_t* data,int bit,int nbytes) {
    const int byte=bit>>3;
    uint32_t word=0;
#pragma unroll
    for(int i=0;i<4;++i)if(byte+i<nbytes)word|=uint32_t(data[byte+i])<<(i*8);
    return (word>>(bit&7))&0x1fffffu;
}

template<int IndexBits,bool WordWindow>
__device__ __forceinline__ uint64_t load_e8_indices32(const uint8_t* data,int first_vector,int nbytes) {
    static_assert(IndexBits==8 || IndexBits==10 || IndexBits==12);
    constexpr int Bits=3*IndexBits;
    const int bit=first_vector*IndexBits,byte=bit>>3;
    const int aligned=byte-int((uint32_t(reinterpret_cast<uintptr_t>(data))+uint32_t(byte))&3u);
    if(aligned>=0 && aligned+(WordWindow?8:12)<=nbytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(data+aligned);
        const int shift=(byte-aligned)*8+(bit&7);
        const uint32_t a=words[0],b=words[1];
        if constexpr(WordWindow) {
            if constexpr(Bits<=32)return __funnelshift_r(a,b,shift)&((1u<<Bits)-1u);
            else return ((uint64_t(a)|(uint64_t(b)<<32))>>shift)&((uint64_t(1)<<Bits)-1);
        }else {
            const uint32_t c=words[2];
            return (uint64_t(__funnelshift_r(a,b,shift))|
                (uint64_t(__funnelshift_r(b,c,shift))<<32))&((uint64_t(1)<<Bits)-1);
        }
    }
    uint64_t value=0;
#pragma unroll
    for(int i=0;i<(Bits+14)/8;++i)if(byte+i<nbytes)value|=uint64_t(data[byte+i])<<(i*8);
    return (value>>(bit&7))&((uint64_t(1)<<Bits)-1);
}

}
