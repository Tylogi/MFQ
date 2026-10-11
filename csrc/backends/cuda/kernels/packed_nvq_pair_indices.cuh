#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda::packed_nvq {
template<int Bits>
__device__ __forceinline__ uint2 load_d4_index_pair(const uint8_t* data,
        int64_t index,int64_t nbytes,bool complete) {
    static_assert(Bits==9 || Bits==10);
    constexpr uint32_t mask=(1u<<Bits)-1u;
    if(!complete)return make_uint2(load_packed_bits(data,index*Bits,Bits,nbytes),0);
    // 18/20 packed bits can span four bytes. The bounded 21-bit reader also
    // covers that fourth byte; the ordinary scalar reader covers three only.
    const auto word=load_packed_sign_group3_words(data,index*Bits,nbytes);
    return make_uint2(word&mask,(word>>Bits)&mask);
}
}
