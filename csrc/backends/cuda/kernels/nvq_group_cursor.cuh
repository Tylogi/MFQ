#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda {
struct NvqPackedCursor {
    int64_t byte;
    int shift;
    __device__ explicit NvqPackedCursor(int64_t bit=0):byte(bit>>3),shift(int(bit&7)) {}
};

// A lane's byte stride is integral when its group stride is a multiple of 8.
// Bounds use the original payload; no padding or execution records are added.
template<int Bits>
__device__ __forceinline__ uint64_t nvq_cursor_window(
        const uint8_t* data,int64_t nbytes,const NvqPackedCursor& cursor) {
    static_assert(Bits>0 && Bits<64);
    const int64_t aligned=cursor.byte-int64_t((reinterpret_cast<uintptr_t>(data)+cursor.byte)&3u);
    const int shift=int((cursor.byte-aligned)*8)+cursor.shift;
    const int words=(Bits+shift+31)/32;
    if(aligned>=0 && aligned+words*4<=nbytes) {
        const auto* source=reinterpret_cast<const uint32_t*>(data+aligned);
        const uint32_t a=source[0],b=Bits+shift>32?source[1]:0;
        const uint32_t low=__funnelshift_r(a,b,shift);
        uint64_t value=low;
        if constexpr(Bits>32) {
            const uint32_t c=Bits+shift>64?source[2]:0;
            value|=uint64_t(__funnelshift_r(b,c,shift))<<32;
        }
        return value&((uint64_t(1)<<Bits)-1);
    }
    return packed_nvq::load_packed_group_window<Bits>(data,cursor.byte*8+cursor.shift,nbytes);
}
}
