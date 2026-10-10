#pragma once
#include "mfq/kernels/cuda/nvq_e8_dense_group.cuh"

namespace e8_cache_policy {
namespace pn=mfq::cuda::packed_nvq;

// Strata's packed Q8 kernel uses __ldcs for its one-pass weight plane.
// Keep the exact compact E8 address and boundary rules while testing whether
// that policy leaves more cache capacity for the repeatedly used learned grid.
template<int Bits,int Policy>
__device__ __forceinline__ uint64_t window(const uint8_t* data,int64_t bit,int64_t nbytes) {
    static_assert(Bits>0 && Bits<64);
    const int64_t byte=bit>>3;
    const int64_t aligned=byte-int64_t((reinterpret_cast<uintptr_t>(data)+byte)&3u);
    uint64_t value=0;
    if(aligned>=0 && aligned+12<=nbytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(data+aligned);
        const int shift=int((byte-aligned)*8+(bit&7));
        const uint32_t a=Policy==1?__ldcs(words):Policy==2?__ldca(words):words[0];
        const uint32_t b=Policy==1?__ldcs(words+1):Policy==2?__ldca(words+1):words[1];
        const uint32_t c=Policy==1?__ldcs(words+2):Policy==2?__ldca(words+2):words[2];
        value=uint64_t(__funnelshift_r(a,b,shift)) |
            (uint64_t(__funnelshift_r(b,c,shift))<<32);
    } else {
        constexpr int bytes=(Bits+14)/8;
        uint32_t upper=0;
        #pragma unroll
        for(int i=0;i<bytes;++i)if(byte+i<nbytes) {
            if constexpr(bytes>8) {
                if(i==8)upper=data[byte+i];
                else value|=uint64_t(data[byte+i])<<(i*8);
            }else value|=uint64_t(data[byte+i])<<(i*8);
        }
        const int shift=int(bit&7);
        if(shift)value=(value>>shift)|(uint64_t(upper)<<(64-shift));
    }
    return value&((uint64_t(1)<<Bits)-1);
}

template<int Bits,int Policy>
__device__ __forceinline__ pn::DenseE8Group<Bits> group(const uint8_t* data,int64_t bytes,
        int row,int group_index,int vectors,int groups) {
    const int row_bits=vectors*(Bits+7)+groups*4;
    const int64_t bit=int64_t(row)*row_bits+group_index*(4+3*(Bits+7));
    return {window<4+3*(Bits+7),Policy>(data,bit,bytes)};
}

template<int Segment,int Bits,bool RetainBook>
__device__ __forceinline__ int2 decode(pn::DenseE8Group<Bits> packed,const int8_t* bank,int sign_mode) {
    const auto index=packed.template index<Segment>();
    const auto signs=packed.template signs<Segment>();
    const auto parity=(__popc(signs)&1)^((sign_mode&1)?((index>>7)&1u):0u);
    int2 digits;
    if constexpr(RetainBook) {
        // A cache hint only: no different values, cache allocation or table copy.
        asm("ld.global.L1::evict_last.v2.u32 {%0, %1}, [%2];"
            :"=r"(digits.x),"=r"(digits.y):"l"(bank+size_t(index)*8));
    }else digits=reinterpret_cast<const int2*>(bank)[index];
    return pn::apply_sign8(digits,signs|(parity<<7));
}
}
