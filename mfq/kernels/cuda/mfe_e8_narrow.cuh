#pragma once
#include "nvq_e8_narrow.cuh"

namespace mfq::cuda {
__device__ __forceinline__ uint32_t load_mfe_e8_signs_words32(
        const uint8_t* data,int bit,int nbytes) {
    const int aligned=(bit>>3)&~3;
    if((reinterpret_cast<uintptr_t>(data)&3u)==0 && aligned+8<=nbytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(data+aligned);
        return __funnelshift_r(words[0],words[1],bit&31)&0x1fffffu;
    }
    return load_e8_signs32(data,bit,nbytes);
}
}
