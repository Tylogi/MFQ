#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace mfq::cuda {
__device__ __forceinline__ int nvq_apply_byte_signs4(int values,uint32_t signs,int base) {
    const uint32_t nibble=(signs>>base)&15u;
    const uint32_t ones=(nibble*0x00204081u)&0x01010101u;
    const uint32_t inverted=uint32_t(values)^(ones*255u);
    // Each low byte is at most 127 + 1, so the addition cannot carry
    // between bytes. XOR restores the high bit after the per-byte +1.
    return int(((inverted&0x7f7f7f7fu)+ones)^(inverted&0x80808080u));
}
__device__ __forceinline__ int2 nvq_apply_byte_signs8(int2 values,uint32_t signs) {
    return make_int2(nvq_apply_byte_signs4(values.x,signs,0),nvq_apply_byte_signs4(values.y,signs,4));
}
}
