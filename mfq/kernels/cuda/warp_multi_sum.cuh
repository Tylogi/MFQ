// Adapted from Strata's src/kernels/cuda/s26_tsum.cuh.
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// MIT license: cpp_runtime/third_party/strata.LICENSE.
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>

namespace mfq::cuda::warp_multi_sum {
inline bool enabled() {
    const auto* setting=std::getenv("MFQ_CUDA_WARP_MULTI_SUM");
    return setting && setting[0]=='1' && setting[1]=='\0';
}
template<int Offset>
__device__ __forceinline__ float plain(float value) {
    if constexpr(Offset>=1) {
        value+=__shfl_xor_sync(0xffffffffu,value,Offset);
        return plain<Offset/2>(value);
    }else return value;
}

// Each stage retains half of the independent sums per lane. The pairs of
// finite values and the order of reduction levels match separate full-warp
// XOR butterflies. No matrix data or quantization layout changes.
template<int Count,int Offset=16>
__device__ __forceinline__ float distributed(float* values,int lane) {
    static_assert(Count>=1 && Count<=32 && (Count&(Count-1))==0);
    if constexpr(Count>1) {
        constexpr int half=Count/2;
        const bool high=(lane&Offset)!=0;
#pragma unroll
        for(int row=0;row<half;++row) {
            const float send=high?values[row]:values[row+half];
            const float keep=high?values[row+half]:values[row];
            values[row]=keep+__shfl_xor_sync(0xffffffffu,send,Offset);
        }
        return distributed<half,Offset/2>(values,lane);
    }else return plain<Offset>(values[0]);
}

template<int Count>
__host__ __device__ constexpr int result_lane(int row) {
    int result=0,offset=16;
    for(int count=Count;count>1;count/=2,offset/=2)
        if(row&(count/2))result|=offset;
    return result;
}

// Preserve callers that consume all row results in a single lane. For two
// rows this uses seven shuffle instructions instead of ten; four uses ten
// instead of twenty. Count==1 retains the ordinary five-step butterfly.
template<int Count>
__device__ __forceinline__ void broadcast(float (&values)[Count],int lane) {
    const float sum=distributed<Count>(values,lane);
#pragma unroll
    for(int row=0;row<Count;++row) {
        if constexpr(Count==1)values[row]=sum;
        else values[row]=__shfl_sync(0xffffffffu,sum,result_lane<Count>(row));
    }
}
}
