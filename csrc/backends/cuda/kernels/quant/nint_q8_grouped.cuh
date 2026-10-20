#pragma once

// Aligned GS48 storage remains packed. One lane consumes one complete group,
// keeping its affine outside twelve integer dot products. Compared with the
// chunk kernel this changes floating-point reduction order, but not activation
// quantization, encoded weights, or integer products.
template<bool UnsignedWeights>
__device__ __forceinline__ int nint_q8_dot(int w,int x,int sum) {
    if constexpr(UnsignedWeights) {
        asm("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(sum) : "r"(w),"r"(x));
        return sum;
    }else return __dp4a(w,x,sum);
}

template<int Rows,bool CachedSum=false,bool VectorLoads=false,bool ExactRows=false,bool UnsignedWeights=false>
__global__ void __launch_bounds__(128) nint_q8_grouped_kernel(
        const uint8_t* __restrict__ weights,const int64_t* __restrict__ offsets,
        const uint8_t* __restrict__ scales,const uint8_t* __restrict__ minima,
        const float* __restrict__ outer_scales,const float* __restrict__ outer_minima,
        const int8_t* __restrict__ activation,const float* __restrict__ activation_scales,
        __half* __restrict__ output,int activation_rows,int output_rows,int groups,
        const int* __restrict__ activation_sums=nullptr) {
    const int row=int(blockIdx.x)*4+int(threadIdx.y),lane=int(threadIdx.x);
    if(row>=output_rows)return;
    const auto* weight=weights+(uint64_t(offsets[row])>>3);
    const float outer_scale=outer_scales[row],outer_minimum=outer_minima[row];
    float accumulator[Rows]={};
    for(int group=lane;group<groups;group+=32) {
        const int column=group*48;
        int dot[Rows]={},sum[CachedSum?1:Rows]={};
        if constexpr(VectorLoads) {
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk) {
                int4 w;
                if((reinterpret_cast<uintptr_t>(weight)&15u)==0)
                    w=*reinterpret_cast<const int4*>(weight+column+chunk*16);
                else {
                    const auto* p=reinterpret_cast<const int*>(weight+column+chunk*16);
                    w=make_int4(p[0],p[1],p[2],p[3]);
                }
                if constexpr(!UnsignedWeights) {
                    w.x^=int(0x80808080u);w.y^=int(0x80808080u);
                    w.z^=int(0x80808080u);w.w^=int(0x80808080u);
                }
                #pragma unroll
                for(int m=0;m<Rows;++m)if(ExactRows || Rows==1 || m<activation_rows) {
                    const auto x=*reinterpret_cast<const int4*>(activation+size_t(m)*groups*48+column+chunk*16);
                    dot[m]=nint_q8_dot<UnsignedWeights>(w.x,x.x,dot[m]);dot[m]=nint_q8_dot<UnsignedWeights>(w.y,x.y,dot[m]);
                    dot[m]=nint_q8_dot<UnsignedWeights>(w.z,x.z,dot[m]);dot[m]=nint_q8_dot<UnsignedWeights>(w.w,x.w,dot[m]);
                    if constexpr(!CachedSum) {
                        sum[m]=__dp4a(0x01010101,x.x,sum[m]);sum[m]=__dp4a(0x01010101,x.y,sum[m]);
                        sum[m]=__dp4a(0x01010101,x.z,sum[m]);sum[m]=__dp4a(0x01010101,x.w,sum[m]);
                    }
                }
            }
        }else {
        #pragma unroll
        for(int chunk=0;chunk<12;++chunk) {
            const int qw=*reinterpret_cast<const int*>(weight+column+chunk*4)^(UnsignedWeights?0:int(0x80808080u));
            #pragma unroll
            for(int m=0;m<Rows;++m)if(ExactRows || Rows==1 || m<activation_rows) {
                const int qx=*reinterpret_cast<const int*>(activation+size_t(m)*groups*48+column+chunk*4);
                dot[m]=nint_q8_dot<UnsignedWeights>(qw,qx,dot[m]);
                if constexpr(!CachedSum)sum[m]=__dp4a(0x01010101,qx,sum[m]);
            }
        }
        }
        const float scale=outer_scale*float(scales[size_t(row)*groups+group]);
        const float minimum=outer_minimum*float(minima[size_t(row)*groups+group]);
        #pragma unroll
        for(int m=0;m<Rows;++m)if(ExactRows || Rows==1 || m<activation_rows) {
            const int total=CachedSum?activation_sums[size_t(m)*groups+group]:sum[m];
            accumulator[m]+=activation_scales[size_t(m)*groups+group]*
                (scale*float(dot[m]+(UnsignedWeights?0:128*total))-minimum*float(total));
        }
    }
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)
            accumulator[m]+=__shfl_xor_sync(0xffffffffu,accumulator[m],offset);
        if(lane==0 && (ExactRows || Rows==1 || m<activation_rows))
            output[size_t(m)*output_rows+row]=__float2half_rn(accumulator[m]);
    }
}
