#pragma once
#include "kernels/nvq/device.cuh"

namespace mfq::cuda {
namespace {
template<int Segment,int Bits>
__device__ __forceinline__ int2 parity_read_e8(
        packed_nvq::DenseE8Group<Bits> group,const int8_t* bank,uint32_t parity_mask) {
    const auto index=group.template index<Segment>();
    const auto signs=group.template signs<Segment>();
    const auto parity=__popc(signs|(index&parity_mask))&1u;
    const auto value=__ldg(reinterpret_cast<const int2*>(bank)+index);
    return packed_nvq::apply_sign8(value,signs|(parity<<7));
}
template<int Segment,int Bits>
__device__ __forceinline__ int2 parity_read_d4(
        packed_nvq::DenseD4Group<Bits> group,const int8_t* bank) {
    const auto* codes=reinterpret_cast<const int*>(bank);
    const auto x=__ldg(codes+group.template index<Segment,0>());
    const auto y=__ldg(codes+group.template index<Segment,1>());
    const auto signs=group.template signs<Segment>();
    return packed_nvq::apply_sign8(make_int2(x,y),signs|((__popc(signs)&1u)<<7));
}
template<int Format>
__device__ __forceinline__ float parity_read_scale(float anchor,uint32_t state,const int8_t* codebook) {
    if constexpr(Format==5 || Format==10 || Format==12 || Format==13 || Format==14 || Format==15) {
        const auto* lut=reinterpret_cast<const __half*>(codebook+packed_nvq::kJscLutOffset);
        return anchor*__half2float(__ldg(lut+state));
    }else return packed_nvq::format_scale<Format>(anchor,state,codebook);
}
}
}
