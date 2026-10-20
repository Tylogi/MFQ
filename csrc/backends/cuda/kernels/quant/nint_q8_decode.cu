#include "native/env.h"
#include "nint_short_decode.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <type_traits>
#include "kernels/kernel_dispatch_trace.h"

namespace {
#include "nint_q8_grouped.cuh"

// Small projections need enough independent CTAs to fill the device. Four
// lanes read adjacent pieces of a GS48 group; integer reduction precedes the
// affine, and four warps cooperate on one output row.
template<int Rows>
__global__ void __launch_bounds__(128) nint_q8_small_kernel(NintWholeArgs a) {
    const int row=blockIdx.x,lane=threadIdx.x,warp=threadIdx.y;
    const int member=lane&3;
    const auto* weights=a.weights+(uint64_t(a.offsets[row])>>3);
    const float outer_scale=a.outer_scale[row],outer_minimum=a.outer_minimum[row];
    float accumulator[Rows]{};
    for(int base=warp*8;base<a.groups;base+=32) {
        const int group=base+(lane>>2);
        const bool active=group<a.groups;
        int dots[Rows]{},sums[Rows]{};
        #pragma unroll
        for(int chunk=0;chunk<3;++chunk) {
            const int column=group*48+(chunk*4+member)*4;
            const int w=active?*reinterpret_cast<const int*>(weights+column):0;
            #pragma unroll
            for(int m=0;m<Rows;++m) {
                const int x=active?*reinterpret_cast<const int*>(a.activation+size_t(m)*a.groups*48+column):0;
                dots[m]=nint_q8_dot<true>(w,x,dots[m]);
                sums[m]=__dp4a(0x01010101,x,sums[m]);
            }
        }
        const float scale=active?outer_scale*float(a.scale[size_t(row)*a.groups+group]):0.f;
        const float minimum=active?outer_minimum*float(a.minimum[size_t(row)*a.groups+group]):0.f;
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            dots[m]+=__shfl_xor_sync(0xffffffffu,dots[m],2,4);
            dots[m]+=__shfl_xor_sync(0xffffffffu,dots[m],1,4);
            sums[m]+=__shfl_xor_sync(0xffffffffu,sums[m],2,4);
            sums[m]+=__shfl_xor_sync(0xffffffffu,sums[m],1,4);
            if(active && member==0)accumulator[m]+=a.activation_scale[size_t(m)*a.groups+group]*
                (scale*float(dots[m])-minimum*float(sums[m]));
        }
    }
    __shared__ float partial[Rows][4];
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset;offset>>=1)
            accumulator[m]+=__shfl_xor_sync(0xffffffffu,accumulator[m],offset);
        if(lane==0)partial[m][warp]=accumulator[m];
    }
    __syncthreads();
    if(warp==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            float value=lane<4?partial[m][lane]:0.f;
            #pragma unroll
            for(int offset=16;offset;offset>>=1)value+=__shfl_xor_sync(0xffffffffu,value,offset);
            if(lane==0)a.output[size_t(m)*a.rows+row]=__float2half_rn(value);
        }
    }
}
}

void nint_launch_grouped_q8(NintWholeArgs a,cudaStream_t stream) {
    const auto* vector_setting=mfq::cuda::env_lookup("MFQ_NINT_Q8_VECTOR");
    const bool head_vector_default=a.rows>=65536 && a.groups==54 &&
        a.activation_rows>=4 && a.activation_rows<=6;
    const bool vectors=vector_setting?vector_setting[0]=='1':head_vector_default;
    const auto* exact_setting=mfq::cuda::env_lookup("MFQ_NINT_Q8_EXACT_ROWS");
    const bool projection_exact_default=a.groups==54 && a.rows>=512 && a.rows<=16384 &&
        a.activation_rows>=2 && a.activation_rows<=6 && !a.activation_sum;
    const bool exact_rows=(exact_setting?exact_setting[0]=='1':
        (head_vector_default || projection_exact_default)) && a.activation_rows<=6;
    const bool tracing=mfq::cuda::dispatch_trace::enabled();
    const auto* unsigned_setting=mfq::cuda::env_lookup("MFQ_NINT_Q8_U8S8");
    const bool unsigned_default=a.rows>=65536 && a.groups==54 &&
        (a.activation_rows==1 || a.activation_rows==4 || a.activation_rows==6);
    const bool unsigned_weights=unsigned_setting?unsigned_setting[0]=='1':unsigned_default;
    const auto* small_setting=mfq::cuda::env_lookup("MFQ_NINT_Q8_SMALL_WARPS");
    const bool small=(!small_setting || small_setting[0]!='0') && a.rows<=128 &&
        a.groups<=64 && a.activation_rows>=2 && a.activation_rows<=6;
    const auto launch=[&](auto rows) {
        if(small) {
            nint_q8_small_kernel<decltype(rows)::value><<<a.rows,dim3(32,4),0,stream>>>(a);
            if(tracing)mfq::cuda::dispatch_trace::record(stream,"nint_q8_small_groups",
                reinterpret_cast<const void*>(nint_q8_small_kernel<decltype(rows)::value>),
                "M="+std::to_string(a.activation_rows)+" N="+std::to_string(a.rows)+" groups="+std::to_string(a.groups));
            return;
        }
        const auto sum=[&](auto cached) {
            const auto vector=[&](auto packed) {
                const auto exact=[&](auto shape) {
                const auto sign=[&](auto native) {
                nint_q8_grouped_kernel<decltype(rows)::value,decltype(cached)::value,decltype(packed)::value,decltype(shape)::value,decltype(native)::value><<<
                    (a.rows+3)/4,dim3(32,4),0,stream>>>(a.weights,a.offsets,a.scale,a.minimum,
                    a.outer_scale,a.outer_minimum,a.activation,a.activation_scale,a.output,
                    a.activation_rows,a.rows,a.groups,a.activation_sum);
                if(tracing)mfq::cuda::dispatch_trace::record(stream,decltype(cached)::value?
                    (decltype(packed)::value?"nint_q8_grouped_sum_vector":"nint_q8_grouped_sum"):
                    (decltype(packed)::value?"nint_q8_grouped_vector":"nint_q8_grouped"),
                    reinterpret_cast<const void*>(nint_q8_grouped_kernel<decltype(rows)::value,decltype(cached)::value,decltype(packed)::value,decltype(shape)::value,decltype(native)::value>),
                    "M="+std::to_string(a.activation_rows)+" N="+std::to_string(a.rows)+" groups="+std::to_string(a.groups)+
                    " capacity="+std::to_string(decltype(rows)::value)+" exact="+std::to_string(decltype(shape)::value)+
                    " u8s8="+std::to_string(decltype(native)::value));
                };
                if(unsigned_weights)sign(std::true_type{});else sign(std::false_type{});
                };
                if(exact_rows)exact(std::true_type{});else exact(std::false_type{});
            };
            if(vectors)vector(std::true_type{});else vector(std::false_type{});
        };
        if(a.activation_sum)sum(std::true_type{});else sum(std::false_type{});
    };
    if(a.activation_rows==1)launch(std::integral_constant<int,1>{});
    else if(a.activation_rows==2)launch(std::integral_constant<int,2>{});
    else if(a.activation_rows==3)launch(std::integral_constant<int,3>{});
    else if(a.activation_rows==4)launch(std::integral_constant<int,4>{});
    else if(a.activation_rows==5)launch(std::integral_constant<int,5>{});
    else if(a.activation_rows==6)launch(std::integral_constant<int,6>{});
    else launch(std::integral_constant<int,8>{});
}
