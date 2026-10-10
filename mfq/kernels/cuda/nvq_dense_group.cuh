#pragma once
#include "nvq_e8_dense_group.cuh"
#include "nvq_d4_dense_group.cuh"

namespace mfq::cuda::packed_nvq {
__host__ __device__ constexpr bool dense_nvq_format(int format) {
    return format==5 || format==10 || format==11 || format==12 ||
        format==13 || format==14 || format==15;
}
struct DenseNvqGroup {
    uint32_t lo=0,mid=0,hi=0;
    __device__ uint32_t state() const {return lo&15u;}
};
template<int Format>
__device__ __forceinline__ DenseNvqGroup load_dense_nvq_group(
        const uint8_t* stream,int64_t bytes,int row,int group,int pairs,int groups) {
    static_assert(dense_nvq_format(Format));
    constexpr int bits=format_index_bits(Format);
    if constexpr(is_d4_format(Format)) {
        const auto r=load_dense_d4_group<bits>(stream,bytes,row,group,pairs,groups);
        return {r.lo,r.mid,r.hi};
    }else {
        const auto r=load_dense_e8_group<bits>(stream,bytes,row,group,pairs,groups);
        return {uint32_t(r.bits),uint32_t(r.bits>>32),0};
    }
}
template<int Format,int Segment>
__device__ __forceinline__ int2 decode_dense_nvq_vector(
        DenseNvqGroup r,const int8_t* bank,int sign_mode) {
    static_assert(dense_nvq_format(Format));
    constexpr int bits=format_index_bits(Format);
    if constexpr(is_d4_format(Format))
        return decode_dense_d4_vector<Segment>(DenseD4Group<bits>{r.lo,r.mid,r.hi},bank);
    else return decode_dense_e8_vector<Segment>(DenseE8Group<bits>{uint64_t(r.lo)|(uint64_t(r.mid)<<32)},bank,sign_mode);
}
template<int Format>
__device__ __forceinline__ int2 decode_dense_nvq_vector(
        DenseNvqGroup r,const int8_t* bank,int segment,int sign_mode) {
    if(segment==0)return decode_dense_nvq_vector<Format,0>(r,bank,sign_mode);
    if(segment==1)return decode_dense_nvq_vector<Format,1>(r,bank,sign_mode);
    return decode_dense_nvq_vector<Format,2>(r,bank,sign_mode);
}
} // namespace mfq::cuda::packed_nvq
