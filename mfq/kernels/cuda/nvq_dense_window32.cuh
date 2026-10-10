#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace mfq::cuda::packed_nvq {
// Requires a proven nonnegative local bit offset <= INT_MAX-64 and a local
// byte count representable as int. The pointer itself remains 64-bit. Supports
// unaligned allocations and allocations without readable trailing padding.
template<int Bits>
__device__ __forceinline__ uint3 load_dense_window32(const uint8_t* data,int bytes,int bit) {
    static_assert(Bits>32 && Bits<=89);
    const int byte=bit>>3;
    const int aligned=byte-int((uint32_t(reinterpret_cast<uintptr_t>(data))+uint32_t(byte))&3u);
    constexpr int read_bytes=Bits<=64?12:16;
    if(aligned>=0 && aligned+read_bytes<=bytes) {
        const auto* w=reinterpret_cast<const uint32_t*>(data+aligned);
        const int shift=(byte-aligned)*8+(bit&7);
        const uint32_t a=w[0],b=w[1],c=w[2];
        if constexpr(Bits<=64)
            return make_uint3(__funnelshift_r(a,b,shift),__funnelshift_r(b,c,shift),0);
        else return make_uint3(__funnelshift_r(a,b,shift),__funnelshift_r(b,c,shift),
                               __funnelshift_r(c,w[3],shift));
    }
    uint32_t w[3]{};
#pragma unroll
    for(int i=0;i<(Bits+14)/8;++i)if(byte+i<bytes)
        w[i/4]|=uint32_t(data[byte+i])<<((i&3)*8);
    const int shift=bit&7;
    return make_uint3(__funnelshift_r(w[0],w[1],shift),
                      __funnelshift_r(w[1],w[2],shift),w[2]>>shift);
}
}
