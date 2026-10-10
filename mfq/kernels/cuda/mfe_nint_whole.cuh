#pragma once
#include "mfe_ffn.h"
#include "warp_multi_sum.cuh"
#include "nint_route_group.cuh"
#include <climits>
#include <type_traits>

namespace mfq::cuda {
// Preserve the original MFE group order, affine scaling and two-row warp
// reduction. Only fixed q4/q5/q6 pairs with a proven payload extent use this.
__device__ __forceinline__ bool mfe_nint_try_whole_pair(const MfePackedProjection& d,int local,
        int first,const int8_t* input,const float* scales,float* result,bool multi_sum) {
    if(d.group_size!=24 && d.group_size!=28)return false;
    int stride=d.q_expert_stride;
    if(stride==0 && local==0 && d.local_experts==1 && d.sizes[0]>0 && d.sizes[0]<=INT_MAX)
        stride=int(d.sizes[0]); // Dense shared matrices have no expert stride.
    if(stride<=0 || d.sizes[0]<=0 || uint64_t(local+1)*uint64_t(stride)>uint64_t(d.sizes[0]))return false;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0])+size_t(local)*stride;
    if((reinterpret_cast<uintptr_t>(stream)&3u) || (reinterpret_cast<uintptr_t>(input)&3u))return false;
    if(first>=d.output_rows)return false;
    const auto* bits=static_cast<const uint8_t*>(d.fields[1]);
    const int width=bits[local*d.output_rows+first];
    if(width<4 || width>6 || d.groups>=INT_MAX/(d.group_size*width))return false;
    if(first+1<d.output_rows && bits[local*d.output_rows+first+1]!=width)return false;
    const auto decode=[&](auto group,auto bit_width) {
        packed_nint::nint_fixed_group_rows<decltype(group)::value,decltype(bit_width)::value,1,2,false>(
            static_cast<const uint8_t*>(d.fields[0]),static_cast<const int64_t*>(d.fields[2]),
            static_cast<const uint8_t*>(d.fields[3]),static_cast<const uint8_t*>(d.fields[4]),
            static_cast<const float*>(d.fields[5]),static_cast<const float*>(d.fields[6]),
            input,scales,local,first,d.output_rows,d.groups,stride,result,multi_sum);
    };
    const auto select=[&](auto group) {
        if(width==4)decode(group,std::integral_constant<int,4>{});
        else if(width==5)decode(group,std::integral_constant<int,5>{});
        else decode(group,std::integral_constant<int,6>{});
    };
    if(d.group_size==24)select(std::integral_constant<int,24>{});
    else select(std::integral_constant<int,28>{});
    return true;
}
}
