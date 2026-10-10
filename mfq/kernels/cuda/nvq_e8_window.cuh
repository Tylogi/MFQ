#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda {

// Three canonical E8 indices fit in two aligned words. In the 36-bit case,
// first_vector*12 and the pointer alignment make shift a multiple of four,
// so shift <= 28. Accept a vector number to preserve that precondition.
template<int IndexBits>
__device__ __forceinline__ uint64_t load_e8_index_window(
        const uint8_t* data,int64_t first_vector,int64_t nbytes) {
    static_assert(IndexBits==8 || IndexBits==10 || IndexBits==12);
    constexpr int Bits=3*IndexBits;
    const int64_t bit=first_vector*IndexBits;
    const int64_t byte=bit>>3;
    const int64_t aligned=byte-int64_t((reinterpret_cast<uintptr_t>(data)+byte)&3u);
    if(aligned>=0 && aligned+8<=nbytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(data+aligned);
        const int shift=int((byte-aligned)*8+(bit&7));
        const uint32_t a=words[0],b=words[1];
        if constexpr(Bits<=32)
            return __funnelshift_r(a,b,shift)&((uint32_t(1)<<Bits)-1u);
        else
            return ((uint64_t(a)|(uint64_t(b)<<32))>>shift)&((uint64_t(1)<<Bits)-1);
    }
    // Keep only bounded byte reads here. Calling the generic window helper
    // retains a second aligned fast branch in generated code.
    uint64_t value=0;
#pragma unroll
    for(int i=0;i<(Bits+14)/8;++i)
        if(byte+i<nbytes)value|=uint64_t(data[byte+i])<<(i*8);
    return (value>>(bit&7))&((uint64_t(1)<<Bits)-1);
}

}
