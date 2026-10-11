#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda {
__device__ __forceinline__ uint32_t nvq_checked_word(const uint8_t* data,int64_t byte,int64_t bytes) {
    if(byte+4<=bytes)return *reinterpret_cast<const uint32_t*>(data+byte);
    uint32_t value=0;
#pragma unroll
    for(int i=0;i<4;++i)if(byte+i<bytes)value|=uint32_t(data[byte+i])<<(i*8);
    return value;
}
template<int Bits,int Lanes>
__device__ __forceinline__ uint64_t nvq_cooperative_window(
        const uint8_t* data,int64_t bytes,int64_t base_bit,int lane) {
    static_assert(Bits>0 && Bits<64 && (Lanes==16 || Lanes==32));
    const int64_t byte=base_bit>>3;
    const int64_t aligned=byte-int64_t((reinterpret_cast<uintptr_t>(data)+byte)&3u);
    if(aligned<0)return packed_nvq::load_packed_group_window<Bits>(data,base_bit+int64_t(lane)*Bits,bytes);
    const uint32_t first=nvq_checked_word(data,aligned+int64_t(lane)*4,bytes);
    [[maybe_unused]] uint32_t second=0;
    if constexpr(Bits>=32)second=nvq_checked_word(data,aligned+int64_t(lane+Lanes)*4,bytes);
    const unsigned active=__activemask();
    const auto word=[&](int index) {
        const uint32_t a=__shfl_sync(active,first,index&(Lanes-1),Lanes);
        if constexpr(Bits>=32) {
            const uint32_t b=__shfl_sync(active,second,index&(Lanes-1),Lanes);
            return index<Lanes?a:b;
        }
        return a;
    };
    const int bit=int((byte-aligned)*8+(base_bit&7))+lane*Bits;
    const int index=bit>>5,shift=bit&31;
    const uint32_t a=word(index),b=word(index+1);
    uint64_t value=__funnelshift_r(a,b,shift);
    if constexpr(Bits>32)value|=uint64_t(__funnelshift_r(b,word(index+2),shift))<<32;
    return value&((uint64_t(1)<<Bits)-1);
}
}
