#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace mfq::cuda::mx {
__device__ __forceinline__ float decode_e8m0(std::uint8_t code) {
    if(code==255)return __int_as_float(0x7fffffff);
    return code==0?__int_as_float(0x00400000):__int_as_float(unsigned(code)<<23);
}
__device__ __forceinline__ float decode_mxfp4_e2m1(std::uint8_t code) {
    const unsigned magnitude=code&7u;
    const unsigned bits=magnitude<2u?magnitude*0x3f000000u:
        ((126u+(magnitude>>1u))<<23u)|((magnitude&1u)<<22u);
    return __uint_as_float(bits|((unsigned(code)&8u)<<28u));
}
__device__ __forceinline__ float decode_e4m3fn(std::uint8_t raw) {
    const unsigned exponent=(unsigned(raw)>>3u)&15u,mantissa=unsigned(raw)&7u;
    if(exponent==15u && mantissa==7u)return __int_as_float(0x7fffffff);
    const float magnitude=exponent==0u?float(mantissa)*0.001953125f:
        __int_as_float(((exponent+120u)<<23u)|(mantissa<<20u));
    return (raw&128u)==0u?magnitude:-magnitude;
}
}
