#pragma once

// Same Uniform16 activation integers and original unsigned NINT bytes.
// Two independent int32 chains retain instruction-level parallelism; their
// exact integer sum replaces two float conversions and the plane merge.
__device__ __forceinline__ int gr_s16_u8_low(int values,unsigned weights,int sum) {
    asm("dp2a.lo.s32.u32 %0, %1, %2, %0;" : "+r"(sum) : "r"(values),"r"(weights));
    return sum;
}

__device__ __forceinline__ int gr_s16_u8_high(int values,unsigned weights,int sum) {
    asm("dp2a.hi.s32.u32 %0, %1, %2, %0;" : "+r"(sum) : "r"(values),"r"(weights));
    return sum;
}

// One lane owns both halves of a GS48 group. Keep their FP32 chains
// separate so the eight-lane tree followed by the half sum reproduces the
// original sixteen-lane tree exactly, while sharing row/group metadata.
template<int Capacity>
__device__ __forceinline__ void gr_up_direct_s16_four_outputs(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    constexpr int Width=320;
    const int group=lane&7;
    int4 values[Capacity][2][3];
    float2 statistics[Capacity];
    if(group<7) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
            statistics[k]=q.scales[k*7+group];
            #pragma unroll
            for(int half=0;half<2;++half)for(int chunk=0;chunk<3;++chunk) {
                const int at=group*48+half*24+chunk*8;
                if(at<Width)values[k][half][chunk]=*reinterpret_cast<const int4*>(q.high+2*(k*336+at));
            }
        }
    }
    #pragma unroll
    for(int s=0;s<4;++s) {
        float partial[Capacity][2]{};
        if(group<7) {
            const int row=s*hidden+column;
            const auto* source=w.q+(w.offsets[row]>>3)+group*48;
            const size_t meta=size_t(row)*7+group;
            const float scale=w.row_scale[row]*float(w.scales[meta]);
            const float minimum=w.row_minimum[row]*float(w.minima[meta]);
            int even[2][Capacity]{},odd[2][Capacity]{};
            // Interleave the two halves instead of serializing their integer
            // chains. Independent accumulators hide both load and DP2A latency.
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk) {
                #pragma unroll
                for(int half=0;half<2;++half)if(group*48+half*24+chunk*8<Width) {
                    const auto* address=source+half*24+chunk*8;
                    int2 weights;
                    if((reinterpret_cast<uintptr_t>(address)&7u)==0)
                        weights=*reinterpret_cast<const int2*>(address);
                    else {
                        const auto* words=reinterpret_cast<const int*>(address);
                        weights=make_int2(words[0],words[1]);
                    }
                    #pragma unroll
                    for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                        const auto v=values[k][half][chunk];
                        even[half][k]=gr_s16_u8_low(v.x,unsigned(weights.x),even[half][k]);
                        odd[half][k]=gr_s16_u8_high(v.y,unsigned(weights.x),odd[half][k]);
                        even[half][k]=gr_s16_u8_low(v.z,unsigned(weights.y),even[half][k]);
                        odd[half][k]=gr_s16_u8_high(v.w,unsigned(weights.y),odd[half][k]);
                    }
                }
            }
            #pragma unroll
            for(int half=0;half<2;++half)for(int k=0;k<Capacity;++k)if(sample_begin+k<samples)
                partial[k][half]=statistics[k].y*fmaf(scale,float(even[half][k]+odd[half][k]),
                    -minimum*(half==0?statistics[k].x:0.f));
        }
        #pragma unroll
        for(int k=0;k<Capacity;++k) {
            #pragma unroll
            for(int shift=4;shift;shift>>=1) {
                partial[k][0]+=__shfl_down_sync(0xffffffffu,partial[k][0],shift,8);
                partial[k][1]+=__shfl_down_sync(0xffffffffu,partial[k][1],shift,8);
            }
            sums[k][s]=__fadd_rn(partial[k][0],partial[k][1]);
        }
    }
}

template<int Capacity,int PreloadStreams=2,bool PairedStats=false,bool VectorWeights=false>
__device__ __forceinline__ void gr_up_direct_s16(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    static_assert(PreloadStreams==2 || PreloadStreams==4);
    constexpr int Width=320;
    const int group=(lane&15)>>1,half=lane&1,at=group*48+half*24;
    if(group<7) {
        int4 values[Capacity][3];
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<Width)
                values[k][chunk]=*reinterpret_cast<const int4*>(q.high+2*(k*336+at+chunk*8));
        }
        #pragma unroll
        for(int pair=0;pair<4;pair+=PreloadStreams) {
            int2 weights[PreloadStreams][3];float scales[PreloadStreams],minima[PreloadStreams];
            #pragma unroll
            for(int s=0;s<PreloadStreams;++s) {
                const int row=(pair+s)*hidden+column;
                const auto* source=w.q+(w.offsets[row]>>3)+at;
                const size_t meta=size_t(row)*7+group;
                scales[s]=w.row_scale[row]*float(w.scales[meta]);
                minima[s]=w.row_minimum[row]*float(w.minima[meta]);
                const uintptr_t alignment=reinterpret_cast<uintptr_t>(source)&15u;
                if constexpr(VectorWeights) {
                    if(at+24<=Width && alignment==0) {
                        const auto first=*reinterpret_cast<const int4*>(source);
                        weights[s][0]=make_int2(first.x,first.y);
                        weights[s][1]=make_int2(first.z,first.w);
                        weights[s][2]=*reinterpret_cast<const int2*>(source+16);
                    }else if(at+24<=Width && alignment==8) {
                        weights[s][0]=*reinterpret_cast<const int2*>(source);
                        const auto last=*reinterpret_cast<const int4*>(source+8);
                        weights[s][1]=make_int2(last.x,last.y);
                        weights[s][2]=make_int2(last.z,last.w);
                    }else {
                        #pragma unroll
                        for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<Width) {
                            if((alignment&7u)==0)
                                weights[s][chunk]=*reinterpret_cast<const int2*>(source+chunk*8);
                            else {
                                const auto* words=reinterpret_cast<const int*>(source+chunk*8);
                                weights[s][chunk]=make_int2(words[0],words[1]);
                            }
                        }
                    }
                }else {
                    #pragma unroll
                    for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<Width) {
                        if((alignment&7u)==0)
                            weights[s][chunk]=*reinterpret_cast<const int2*>(source+chunk*8);
                        else {
                            const auto* words=reinterpret_cast<const int*>(source+chunk*8);
                            weights[s][chunk]=make_int2(words[0],words[1]);
                        }
                    }
                }
            }
            #pragma unroll
            for(int s=0;s<PreloadStreams;++s)for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                int even=0,odd=0;
                #pragma unroll
                for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<Width) {
                    const auto v=values[k][chunk];const auto weight=weights[s][chunk];
                    even=gr_s16_u8_low(v.x,unsigned(weight.x),even);
                    odd=gr_s16_u8_high(v.y,unsigned(weight.x),odd);
                    even=gr_s16_u8_low(v.z,unsigned(weight.y),even);
                    odd=gr_s16_u8_high(v.w,unsigned(weight.y),odd);
                }
                float activation_scale,total;
                if constexpr(PairedStats) {
                    const auto statistics=q.scales[k*7+group];
                    activation_scale=statistics.y;total=half==0?statistics.x:0.f;
                }else {
                    activation_scale=q.scales[k*7+group].y;
                    total=float(half==0?q.sums[k*7+group].x:0);
                }
                sums[k][pair+s]=activation_scale*fmaf(scales[s],float(even+odd),-minima[s]*total);
            }
        }
    }
    #pragma unroll
    for(int s=0;s<4;++s)for(int k=0;k<Capacity;++k) {
        #pragma unroll
        for(int shift=8;shift;shift>>=1)sums[k][s]+=__shfl_down_sync(0xffffffffu,sums[k][s],shift,16);
    }
}
