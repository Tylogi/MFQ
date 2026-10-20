#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Adapted from Strata native_bf16.cu and its llama.cpp MMVF implementation.
// Strata reference: fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
// llama.cpp reference: 3cf03257f219afbe7334045ff7c6a06ac68c627d.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace mfq::cuda::bf16_mmvf_detail {
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        value+=__shfl_xor_sync(0xffffffffu,value,offset);
    return value;
}

template<int Threads,int Rows>
__global__ void project_rows(const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,__nv_bfloat16* __restrict__ y,
        int width,int outputs) {
    static_assert(Threads>=32 && Threads%32==0);
    constexpr int Warps=Threads/32;
    const int thread=threadIdx.x,first=blockIdx.x*Rows;
    x+=std::size_t(blockIdx.y)*width;
    y+=std::size_t(blockIdx.y)*outputs;
    __shared__ float partials[Rows][Warps];
    float values[Rows]{};
    for(int pair=thread;pair<width/2;pair+=Threads) {
        const auto in=__bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x)[pair]);
#pragma unroll
        for(int row=0;row<Rows;++row) {
            if(first+row>=outputs)break;
            const auto w=__bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                weight+std::size_t(first+row)*width)[pair]);
            values[row]=__fmaf_rn(w.x,in.x,values[row]);
            values[row]=__fmaf_rn(w.y,in.y,values[row]);
        }
    }
#pragma unroll
    for(int row=0;row<Rows;++row) {
        values[row]=warp_sum(values[row]);
        if constexpr(Warps>1) {
            if((thread&31)==0)partials[row][thread/32]=values[row];
        }
    }
    if constexpr(Warps>1) {
        __syncthreads();
        if(thread>=32)return;
#pragma unroll
        for(int row=0;row<Rows;++row)
            values[row]=warp_sum(thread<Warps?partials[row][thread]:0.f);
    }
    if(thread==0) {
#pragma unroll
        for(int row=0;row<Rows;++row)
            if(first+row<outputs)y[first+row]=__float2bfloat16_rn(values[row]);
    }
}
}
