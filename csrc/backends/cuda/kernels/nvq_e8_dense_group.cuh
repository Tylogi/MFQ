#pragma once
#include "packed_nvq.cuh"

namespace mfq::cuda::packed_nvq {
// Lossless E8 group stream: state[4], then (index[B], sign[7]) for
// each valid vector. The final group has no padding vectors. Rows need
// not start on a byte boundary; the backing allocation has no read padding.
template<int IndexBits>
struct DenseE8Group {
    static_assert(IndexBits==8 || IndexBits==10 || IndexBits==12);
    uint64_t bits;
    __device__ uint32_t state() const { return uint32_t(bits)&15u; }
    template<int Segment> __device__ uint32_t index() const {
        return uint32_t(bits>>(4+Segment*(IndexBits+7)))&((1u<<IndexBits)-1);
    }
    template<int Segment> __device__ uint32_t signs() const {
        return uint32_t(bits>>(4+Segment*(IndexBits+7)+IndexBits))&127u;
    }
};

template<int IndexBits>
__device__ __forceinline__ DenseE8Group<IndexBits> load_dense_e8_group(
        const uint8_t* stream,int64_t bytes,int row,int group,int vectors,int groups) {
    constexpr int group_bits=4+3*(IndexBits+7);
    const int row_bits=vectors*(IndexBits+7)+groups*4;
    const int64_t bit=int64_t(row)*row_bits+group*group_bits;
    // Unused last-group fields may overlap the next row. Callers mask these
    // vectors or use the existing zero-padded GS24 activation contract.
    return {load_packed_group_window<group_bits>(stream,bit,bytes)};
}

template<int Segment,int IndexBits>
__device__ __forceinline__ int2 decode_dense_e8_vector(
        DenseE8Group<IndexBits> group,const int8_t* bank,int sign_mode) {
    const auto index=group.template index<Segment>();
    const auto signs=group.template signs<Segment>();
    const auto parity=(__popc(signs)&1)^((sign_mode&1)?((index>>7)&1u):0u);
    return apply_sign8(reinterpret_cast<const int2*>(bank)[index],signs|(parity<<7));
}
}
