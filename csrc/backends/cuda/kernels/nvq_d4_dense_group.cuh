#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda::packed_nvq {
// Byte-preserving D4 records: state[4], then three pairs of indices and
// sign[7]. Full groups occupy 73/79/85 bits. The final group retains only
// valid pairs, without rounding each group or each row to a byte boundary.
// This variant requires an even number of D4 vectors per row.
template<int IndexBits>
struct DenseD4Group {
    static_assert(IndexBits==8 || IndexBits==9 || IndexBits==10);
    uint32_t lo,mid,hi;
    template<int Bit,int Bits> __device__ __forceinline__ uint32_t field() const {
        static_assert(Bit>=0 && Bit+Bits<=96 && Bits>0 && Bits<32);
        constexpr int shift=Bit&31,word=Bit/32;
        const uint32_t a=word==0?lo:word==1?mid:hi;
        if constexpr(shift+Bits<=32)return (a>>shift)&((1u<<Bits)-1);
        else {
            const uint32_t b=word==0?mid:hi;
            return __funnelshift_r(a,b,shift)&((1u<<Bits)-1);
        }
    }
    __device__ uint32_t state() const {return lo&15u;}
    template<int Segment,int Part> __device__ uint32_t index() const {
        return field<4+Segment*(2*IndexBits+7)+Part*IndexBits,IndexBits>();
    }
    template<int Segment> __device__ uint32_t signs() const {
        return field<4+Segment*(2*IndexBits+7)+2*IndexBits,7>();
    }
};

template<int IndexBits>
__device__ __forceinline__ DenseD4Group<IndexBits> load_dense_d4_group(
        const uint8_t* stream,int64_t bytes,int row,int group,int pairs,int groups) {
    constexpr int group_bits=4+3*(2*IndexBits+7);
    const int row_bits=pairs*(2*IndexBits+7)+groups*4;
    const int64_t bit=int64_t(row)*row_bits+group*group_bits,byte=bit>>3;
    const int64_t aligned=byte-int((reinterpret_cast<uintptr_t>(stream)+byte)&3u);
    if(aligned>=0 && aligned+16<=bytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(stream+aligned);
        const int shift=int((byte-aligned)*8+(bit&7));
        return {__funnelshift_r(words[0],words[1],shift),
                __funnelshift_r(words[1],words[2],shift),
                __funnelshift_r(words[2],words[3],shift)};
    }
    uint32_t words[3]{};
#pragma unroll
    for(int i=0;i<(group_bits+14)/8;++i)if(byte+i<bytes)
        words[i/4]|=uint32_t(stream[byte+i])<<((i&3)*8);
    const int shift=int(bit&7);
    return {__funnelshift_r(words[0],words[1],shift),
            __funnelshift_r(words[1],words[2],shift),words[2]>>shift};
}

template<int Segment,int IndexBits>
__device__ __forceinline__ int2 decode_dense_d4_vector(DenseD4Group<IndexBits> group,const int8_t* bank) {
    const auto* codes=reinterpret_cast<const int*>(bank);
    const int2 values=make_int2(codes[group.template index<Segment,0>()],
                               codes[group.template index<Segment,1>()]);
    const uint32_t signs=group.template signs<Segment>();
    return apply_sign8(values,signs|((__popc(signs)&1u)<<7));
}
}
