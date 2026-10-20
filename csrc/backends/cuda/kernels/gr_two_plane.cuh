#pragma once

// Temporary high/residual activation planes. Original NINT weights stay
// packed; no persistent weight expansion or additional weight copy is used.
struct GrTwoPlane {
    int8_t *high,*middle,*low;
    float2* scales;
    int4* sums;
};

__device__ __forceinline__ int gr_unsigned_signed_dot(int weights,int values,int sum) {
    asm("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(sum) : "r"(weights),"r"(values));
    return sum;
}

template<bool UnsignedWeights>
__device__ __forceinline__ int gr_plane_dot(int weights,int values,int sum) {
    if constexpr(UnsignedWeights)return gr_unsigned_signed_dot(weights,values,sum);
    else return __dp4a(weights,values,sum);
}

template<int Kind>struct GrDotFlags {
    static constexpr bool triple=Kind==3 || Kind==5 || Kind==7 || Kind==9;
    static constexpr bool unsigned_low=Kind==4 || Kind==6 || Kind==8 || Kind==10;
    static constexpr bool combined=Kind==5 || Kind==6 || Kind==9 || Kind==10;
    static constexpr bool unsigned_weights=Kind>=7;
};

#include "gr_up_preload.cuh"
#include "gr_up_full_warp.cuh"
#include "gr_up_dp2a.cuh"

// Original unsigned weights, two activation bytes and two outputs per warp.
template<int Capacity>
__device__ __forceinline__ void gr_up_uniform16_half_groups(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    const int group=(lane&15)>>1,half=lane&1,at=group*48+half*24;
    if(group<7) {
        #pragma unroll
        for(int stream=0;stream<4;++stream) {
            const int row=stream*hidden+column;
            const auto* source=w.q+(w.offsets[row]>>3)+at;
            const size_t meta=size_t(row)*7+group;
            const float scale=w.row_scale[row]*float(w.scales[meta]);
            const float minimum=w.row_minimum[row]*float(w.minima[meta]);
            int high[Capacity]{},low[Capacity]{};
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                int2 weight;
                if((reinterpret_cast<uintptr_t>(source)&7u)==0)weight=*reinterpret_cast<const int2*>(source+chunk*8);
                else {
                    const auto* words=reinterpret_cast<const int*>(source+chunk*8);
                    weight=make_int2(words[0],words[1]);
                }
                #pragma unroll
                for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                    const int offset=k*336+at+chunk*8;
                    auto x=*reinterpret_cast<const int2*>(q.high+offset);
                    high[k]=gr_unsigned_signed_dot(weight.x,x.x,high[k]);
                    high[k]=gr_unsigned_signed_dot(weight.y,x.y,high[k]);
                    x=*reinterpret_cast<const int2*>(q.low+offset);
                    low[k]=int(__dp4a(unsigned(weight.x),unsigned(x.x),unsigned(low[k])));
                    low[k]=int(__dp4a(unsigned(weight.y),unsigned(x.y),unsigned(low[k])));
                }
            }
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                int total=0;
                if(half==0) {
                    const auto sum=q.sums[k*7+group];total=sum.x*256+sum.z;
                }
                const float dot=fmaf(256.f,float(high[k]),float(low[k]));
                sums[k][stream]=q.scales[k*7+group].y*fmaf(scale,dot,-minimum*float(total));
            }
        }
    }
    #pragma unroll
    for(int stream=0;stream<4;++stream) {
        #pragma unroll
        for(int k=0;k<Capacity;++k) {
            #pragma unroll
            for(int shift=8;shift;shift>>=1)sums[k][stream]+=__shfl_down_sync(0xffffffffu,sums[k][stream],shift,16);
        }
    }
}

// Four independent output columns share a warp. Seven lanes each consume an
// entire GS48 group, reusing that row's bytes across all activation planes.
// The eighth lane supplies zero for each eight-lane reduction.
template<int Capacity>
__device__ __forceinline__ void gr_up_integer_groups(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    const int group=lane&7,at=group*48;
    if(group<7) {
        #pragma unroll
        for(int s=0;s<4;++s) {
            const int row=s*hidden+column;
            const auto* source=w.q+(w.offsets[row]>>3)+at;
            const size_t meta=size_t(row)*7+group;
            const float scale=w.row_scale[row]*float(w.scales[meta]);
            const float minimum=w.row_minimum[row]*float(w.minima[meta]);
            const float correction=fmaf(128.f,scale,-minimum);
            int high[Capacity]{},middle[Capacity]{},low[Capacity]{};
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk) {
                if(at+chunk*16>=w.width)continue;
                int4 weights;
                if((reinterpret_cast<uintptr_t>(source)&15u)==0)
                    weights=*reinterpret_cast<const int4*>(source+chunk*16);
                else {
                    const auto* words=reinterpret_cast<const int*>(source+chunk*16);
                    weights=make_int4(words[0],words[1],words[2],words[3]);
                }
                weights.x^=int(0x80808080u);weights.y^=int(0x80808080u);
                weights.z^=int(0x80808080u);weights.w^=int(0x80808080u);
                #pragma unroll
                for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                    const int offset=k*336+at+chunk*16;
                    auto x=*reinterpret_cast<const int4*>(q.high+offset);
                    high[k]=__dp4a(weights.x,x.x,high[k]);high[k]=__dp4a(weights.y,x.y,high[k]);
                    high[k]=__dp4a(weights.z,x.z,high[k]);high[k]=__dp4a(weights.w,x.w,high[k]);
                    x=*reinterpret_cast<const int4*>(q.middle+offset);
                    middle[k]=__dp4a(weights.x,x.x,middle[k]);middle[k]=__dp4a(weights.y,x.y,middle[k]);
                    middle[k]=__dp4a(weights.z,x.z,middle[k]);middle[k]=__dp4a(weights.w,x.w,middle[k]);
                    x=*reinterpret_cast<const int4*>(q.low+offset);
                    low[k]=__dp4a(weights.x,x.x,low[k]);low[k]=__dp4a(weights.y,x.y,low[k]);
                    low[k]=__dp4a(weights.z,x.z,low[k]);low[k]=__dp4a(weights.w,x.w,low[k]);
                }
            }
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                const auto sum=q.sums[k*7+group];
                const int total=sum.x*65536+sum.y*256+sum.z;
                const float dot=fmaf(65536.f,float(high[k]),fmaf(256.f,float(middle[k]),float(low[k])));
                sums[k][s]=q.scales[k*7+group].y*fmaf(scale,dot,correction*float(total));
            }
        }
    }
    #pragma unroll
    for(int s=0;s<4;++s) {
        #pragma unroll
        for(int k=0;k<Capacity;++k) {
            #pragma unroll
            for(int shift=4;shift;shift>>=1)sums[k][s]+=__shfl_down_sync(0xffffffffu,sums[k][s],shift,8);
        }
    }
}

// Two outputs per warp, with two lanes sharing each GS48 group. The 24-byte
// halves use aligned int2 reads; fourteen of each sixteen lanes do useful work.
template<int Capacity>
__device__ __forceinline__ void gr_up_integer_half_groups(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    const int group=(lane&15)>>1,half=lane&1,at=group*48+half*24;
    if(group<7) {
        #pragma unroll
        for(int s=0;s<4;++s) {
            const int row=s*hidden+column;
            const auto* source=w.q+(w.offsets[row]>>3)+at;
            const size_t meta=size_t(row)*7+group;
            const float scale=w.row_scale[row]*float(w.scales[meta]);
            const float minimum=w.row_minimum[row]*float(w.minima[meta]);
            const float correction=fmaf(128.f,scale,-minimum);
            int high[Capacity]{},middle[Capacity]{},low[Capacity]{};
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                int2 weight;
                if((reinterpret_cast<uintptr_t>(source)&7u)==0)weight=*reinterpret_cast<const int2*>(source+chunk*8);
                else {
                    const auto* words=reinterpret_cast<const int*>(source+chunk*8);
                    weight=make_int2(words[0],words[1]);
                }
                weight.x^=int(0x80808080u);weight.y^=int(0x80808080u);
                #pragma unroll
                for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                    const int offset=k*336+at+chunk*8;
                    auto x=*reinterpret_cast<const int2*>(q.high+offset);
                    high[k]=__dp4a(weight.x,x.x,high[k]);high[k]=__dp4a(weight.y,x.y,high[k]);
                    x=*reinterpret_cast<const int2*>(q.middle+offset);
                    middle[k]=__dp4a(weight.x,x.x,middle[k]);middle[k]=__dp4a(weight.y,x.y,middle[k]);
                    x=*reinterpret_cast<const int2*>(q.low+offset);
                    low[k]=__dp4a(weight.x,x.x,low[k]);low[k]=__dp4a(weight.y,x.y,low[k]);
                }
            }
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                int total=0;
                if(half==0) {
                    const auto sum=q.sums[k*7+group];
                    total=sum.x*65536+sum.y*256+sum.z;
                }
                const float dot=fmaf(65536.f,float(high[k]),fmaf(256.f,float(middle[k]),float(low[k])));
                sums[k][s]=q.scales[k*7+group].y*fmaf(scale,dot,correction*float(total));
            }
        }
    }
    #pragma unroll
    for(int s=0;s<4;++s) {
        #pragma unroll
        for(int k=0;k<Capacity;++k) {
            #pragma unroll
            for(int shift=8;shift;shift>>=1)sums[k][s]+=__shfl_down_sync(0xffffffffu,sums[k][s],shift,16);
        }
    }
}
__device__ __forceinline__ int gr_q8_round(float x,float scale,float inverse) {
    if(scale==0.f)return 0;
    const float scaled=isfinite(inverse)?x*inverse:__fdiv_rn(x,scale);
    return max(-127,min(127,__float2int_rn(scaled)));
}

// Seven-group Up: one 16-byte chunk per lane. Read each stream's weight
// once for all token columns, and combine integer planes before scaling.
template<int Capacity>
__device__ __forceinline__ void gr_up_integer_chunks_mix(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,
        const void* normalized,void* mixed,bool half) {
    const int group=lane/3,chunk=lane%3,at=group*48+chunk*16;
    const auto accumulate=[&](int s,float* target) {
    if(at<w.width) {
            const int row=s*hidden+column;
            const auto* source=w.q+(w.offsets[row]>>3)+at;
            int4 weights;
            if((reinterpret_cast<uintptr_t>(source)&15u)==0)weights=*reinterpret_cast<const int4*>(source);
            else {
                int words[4];
                #pragma unroll
                for(int item=0;item<4;++item) {
                    unsigned word=0;
                    #pragma unroll
                    for(int b=0;b<4;++b)word|=unsigned(source[item*4+b])<<(8*b);
                    words[item]=int(word);
                }
                weights=make_int4(words[0],words[1],words[2],words[3]);
            }
            weights.x^=int(0x80808080u);weights.y^=int(0x80808080u);
            weights.z^=int(0x80808080u);weights.w^=int(0x80808080u);
            const size_t meta=size_t(row)*w.groups+group;
            const float scale=w.row_scale[row]*float(w.scales[meta]);
            const float minimum=w.row_minimum[row]*float(w.minima[meta]);
            const float correction=fmaf(128.f,scale,-minimum);
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                const int offset=k*w.groups*48+at;
                auto x=*reinterpret_cast<const int4*>(q.high+offset);
                int high=__dp4a(weights.x,x.x,0);high=__dp4a(weights.y,x.y,high);
                high=__dp4a(weights.z,x.z,high);high=__dp4a(weights.w,x.w,high);
                x=*reinterpret_cast<const int4*>(q.middle+offset);
                int middle=__dp4a(weights.x,x.x,0);middle=__dp4a(weights.y,x.y,middle);
                middle=__dp4a(weights.z,x.z,middle);middle=__dp4a(weights.w,x.w,middle);
                x=*reinterpret_cast<const int4*>(q.low+offset);
                int low=__dp4a(weights.x,x.x,0);low=__dp4a(weights.y,x.y,low);
                low=__dp4a(weights.z,x.z,low);low=__dp4a(weights.w,x.w,low);
                const float dot=fmaf(65536.f,float(high),fmaf(256.f,float(middle),float(low)));
                int sum=0;
                if(chunk==0) {
                    const auto g=q.sums[k*w.groups+group];
                    sum=g.x*65536+g.y*256+g.z;
                }
                target[k]=q.scales[k*w.groups+group].y*fmaf(scale,dot,correction*float(sum));
            }
    }
    };
    {
        float value[Capacity]{};
        #pragma unroll
        for(int s=0;s<4;++s) {
            float acc[Capacity]{};accumulate(s,acc);
            #pragma unroll
            for(int k=0;k<Capacity;++k) {
                #pragma unroll
                for(int shift=16;shift;shift>>=1)acc[k]+=__shfl_down_sync(0xffffffffu,acc[k],shift);
            }
            if(lane==0) {
                #pragma unroll
                for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                    const int sample=sample_begin+k;
                    const float projected=mhc_round(acc[k],half);
                    const float gate=mhc_round(1.f/(1.f+expf(-projected)),half);
                    const float product=mhc_round(__fmul_rn(gate,mhc_read(normalized,
                        (size_t(sample)*4+s)*hidden+column,half)),half);
                    value[k]=__fadd_rn(value[k],product);
                }
            }
        }
        if(lane==0) {
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples)
                mhc_write(mixed,size_t(sample_begin+k)*hidden+column,__fdiv_rn(value[k],4.f),half);
        }
    }
}

// Keep the default's stream loop inside one bounds branch. The experimental
// immediate-mix path above has different register and branch costs.
template<int Capacity>
__device__ __forceinline__ void gr_up_integer_chunks(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    const int group=lane/3,chunk=lane%3,at=group*48+chunk*16;
    if(at<w.width) {
        #pragma unroll
        for(int s=0;s<4;++s) {
            const int row=s*hidden+column;
            const auto* source=w.q+(w.offsets[row]>>3)+at;
            int4 weights;
            if((reinterpret_cast<uintptr_t>(source)&15u)==0)weights=*reinterpret_cast<const int4*>(source);
            else {
                int words[4];
                #pragma unroll
                for(int item=0;item<4;++item) {
                    unsigned word=0;
                    #pragma unroll
                    for(int b=0;b<4;++b)word|=unsigned(source[item*4+b])<<(8*b);
                    words[item]=int(word);
                }
                weights=make_int4(words[0],words[1],words[2],words[3]);
            }
            weights.x^=int(0x80808080u);weights.y^=int(0x80808080u);
            weights.z^=int(0x80808080u);weights.w^=int(0x80808080u);
            const size_t meta=size_t(row)*w.groups+group;
            const float scale=w.row_scale[row]*float(w.scales[meta]);
            const float minimum=w.row_minimum[row]*float(w.minima[meta]);
            const float correction=fmaf(128.f,scale,-minimum);
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                const int offset=k*w.groups*48+at;
                auto x=*reinterpret_cast<const int4*>(q.high+offset);
                int high=__dp4a(weights.x,x.x,0);high=__dp4a(weights.y,x.y,high);
                high=__dp4a(weights.z,x.z,high);high=__dp4a(weights.w,x.w,high);
                x=*reinterpret_cast<const int4*>(q.middle+offset);
                int middle=__dp4a(weights.x,x.x,0);middle=__dp4a(weights.y,x.y,middle);
                middle=__dp4a(weights.z,x.z,middle);middle=__dp4a(weights.w,x.w,middle);
                x=*reinterpret_cast<const int4*>(q.low+offset);
                int low=__dp4a(weights.x,x.x,0);low=__dp4a(weights.y,x.y,low);
                low=__dp4a(weights.z,x.z,low);low=__dp4a(weights.w,x.w,low);
                const float dot=fmaf(65536.f,float(high),fmaf(256.f,float(middle),float(low)));
                int sum=0;
                if(chunk==0) {
                    const auto g=q.sums[k*w.groups+group];
                    sum=g.x*65536+g.y*256+g.z;
                }
                sums[k][s]=q.scales[k*w.groups+group].y*fmaf(scale,dot,correction*float(sum));
            }
        }
    }
    #pragma unroll
    for(int s=0;s<4;++s) {
        #pragma unroll
        for(int k=0;k<Capacity;++k) {
            #pragma unroll
            for(int shift=16;shift;shift>>=1)sums[k][s]+=__shfl_down_sync(0xffffffffu,sums[k][s],shift);
        }
    }
}

template<int GS,bool ThreePlane=false,bool UnsignedLow=false,bool Packed16=false,bool PairedStats=false,bool MmaPlanes=false>
__device__ __forceinline__ void gr_quantize_activation_values(MhcMatrix w,float x0,float x1,GrTwoPlane q,
        int group,int sample,int lane) {
    static_assert(!Packed16 || (!ThreePlane && UnsignedLow),"direct int16 requires Uniform16 quantization");
    static_assert(!PairedStats || (Packed16 && GS*32767<(1<<24)),"group sum must be exactly representable in FP32");
    static_assert(!MmaPlanes || (GS==48 && Packed16 && PairedStats));
    float maximum=fmaxf(fabsf(x0),fabsf(x1));
    #pragma unroll
    for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    float high_scale,low_scale;
    int h0,h1,m0=0,m1=0,l0,l1;
    if constexpr(ThreePlane) {
        // A 24-bit integer is split into centered signed bytes. All three
        // planes share one max reduction; powers of two preserve their scale.
        constexpr int limit=127*65536+127*256+127;
        low_scale=maximum/float(limit);high_scale=low_scale*65536.f;
        const float inverse=low_scale?1.f/low_scale:0.f;
        const auto quantize=[&](float x) {
            if(low_scale==0.f)return 0;
            const float scaled=isfinite(inverse)?x*inverse:__fdiv_rn(x,low_scale);
            return max(-limit,min(limit,__float2int_rn(scaled)));
        };
        const int c0=quantize(x0),c1=quantize(x1);
        // The centered low byte may carry before the middle byte is split.
        // Include both 0x80 biases so middle also remains in [-128,127].
        h0=(c0+32896)>>16;h1=(c1+32896)>>16;
        m0=((c0+128)>>8)-h0*256;m1=((c1+128)>>8)-h1*256;
        l0=c0-h0*65536-m0*256;l1=c1-h1*65536-m1*256;
    }else if constexpr(UnsignedLow) {
        constexpr int limit=32767;
        low_scale=maximum/float(limit);high_scale=low_scale*256.f;
        const float inverse=low_scale?1.f/low_scale:0.f;
        const auto quantize=[&](float x) {
            if(low_scale==0.f)return 0;
            const float scaled=isfinite(inverse)?x*inverse:__fdiv_rn(x,low_scale);
            return max(-limit,min(limit,__float2int_rn(scaled)));
        };
        const int c0=quantize(x0),c1=quantize(x1);
        h0=c0>>8;h1=c1>>8;
        l0=c0-h0*256;l1=c1-h1*256;
    }else {
        high_scale=maximum/127.f;const float high_inverse=high_scale?1.f/high_scale:0.f;
        h0=gr_q8_round(x0,high_scale,high_inverse);h1=gr_q8_round(x1,high_scale,high_inverse);
        const float r0=fmaf(-high_scale,float(h0),x0),r1=fmaf(-high_scale,float(h1),x1);
        float residual_max=fmaxf(fabsf(r0),fabsf(r1));
        #pragma unroll
        for(int offset=16;offset;offset>>=1)residual_max=fmaxf(residual_max,__shfl_xor_sync(0xffffffffu,residual_max,offset));
        low_scale=residual_max/127.f;const float low_inverse=low_scale?1.f/low_scale:0.f;
        l0=gr_q8_round(r0,low_scale,low_inverse);l1=gr_q8_round(r1,low_scale,low_inverse);
    }
    const int at=(sample*w.groups+group)*GS+lane;
    if constexpr(Packed16) {
        const int first=h0*256+l0,second=h1*256+l1;
        if constexpr(MmaPlanes) {
            // Two 24-element halves, padded to K32, and three signed digits.
            // x = (x>>14)*16384 + ((x>>7)&127)*128 + (x&127), exactly.
            auto* digits=q.high+(sample*w.groups+group)*192;
            const int first_at=lane+(lane>=24?8:0),second_at=lane+40;
            #pragma unroll
            for(int plane=0;plane<3;++plane) {
                const int d0=plane==0?first>>14:plane==1?(first>>7)&127:first&127;
                const int d1=plane==0?second>>14:plane==1?(second>>7)&127:second&127;
                digits[plane*64+first_at]=int8_t(d0);
                if(lane<16)digits[plane*64+second_at]=int8_t(d1);
                if(lane>=24) {digits[plane*64+lane]=0;digits[plane*64+32+lane]=0;}
            }
        }else {
            auto* values=reinterpret_cast<int16_t*>(q.high);
            if(lane<GS)values[at]=int16_t(first);
            if(lane+32<GS)values[at+32]=int16_t(second);
        }
        int total=first+second;
        #pragma unroll
        for(int offset=16;offset;offset>>=1)total+=__shfl_xor_sync(0xffffffffu,total,offset);
        if(lane==0) {
            if constexpr(PairedStats)q.scales[sample*w.groups+group]=make_float2(float(total),low_scale);
            else {
                q.scales[sample*w.groups+group]=make_float2(high_scale,low_scale);
                q.sums[sample*w.groups+group]=make_int4(total,0,0,0);
            }
        }
        return;
    }
    if(lane<GS){q.high[at]=int8_t(h0);q.low[at]=int8_t(l0);}
    if constexpr(ThreePlane)if(lane<GS)q.middle[at]=int8_t(m0);
    if(lane+32<GS){q.high[at+32]=int8_t(h1);q.low[at+32]=int8_t(l1);}
    if constexpr(ThreePlane)if(lane+32<GS)q.middle[at+32]=int8_t(m1);
    int high_sum=h0+h1,middle_sum=m0+m1,low_sum=l0+l1;
    #pragma unroll
    for(int offset=16;offset;offset>>=1) {
        high_sum+=__shfl_xor_sync(0xffffffffu,high_sum,offset);
        if constexpr(ThreePlane)middle_sum+=__shfl_xor_sync(0xffffffffu,middle_sum,offset);
        low_sum+=__shfl_xor_sync(0xffffffffu,low_sum,offset);
    }
    if(lane==0) {
        q.scales[sample*w.groups+group]=make_float2(high_scale,low_scale);
        q.sums[sample*w.groups+group]=make_int4(high_sum,middle_sum,low_sum,0);
    }
}

template<int GS,bool ThreePlane=false,bool UnsignedLow=false,bool Packed16=false>
__device__ __forceinline__ void gr_quantize_activation_group(MhcMatrix w,const void* normalized,bool half,GrTwoPlane q,
        int group,int sample,int lane) {
    const int column=group*GS+lane;
    const size_t base=size_t(sample)*w.width;
    const float x0=lane<GS && column<w.width?mhc_read(normalized,base+column,half):0.f;
    const float x1=lane+32<GS && column+32<w.width?mhc_read(normalized,base+column+32,half):0.f;
    gr_quantize_activation_values<GS,ThreePlane,UnsignedLow,Packed16>(w,x0,x1,q,group,sample,lane);
}

template<int GS,bool ThreePlane=false,bool UnsignedLow=false>
__global__ void gr_global_two_plane_kernel(MhcMatrix w,const void* normalized,bool half,GrTwoPlane q) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int group=int(blockIdx.x)*8+warp,sample=blockIdx.y;
    if(group<w.groups)gr_quantize_activation_group<GS,ThreePlane,UnsignedLow>(w,normalized,half,q,group,sample,lane);
}

template<int Bits,int Planes=3>
__device__ __forceinline__ float gr_packed24_affine_bits(const MhcMatrix& w,GrTwoPlane q,
        int row,int first,int last,int lane) {
    const auto offset=uint64_t(w.offsets[row]);
    const float outer_scale=w.row_scale[row],outer_minimum=w.row_minimum[row];
    float partial=0.f;
    for(int group=first+lane;group<last;group+=32) {
        const NintWholeGroup<24,Bits> packed(w.q,offset+uint64_t(group)*24*Bits,w.bytes);
        int high=0,middle=0,low=0;
        const auto consume=[&](auto index) {
            constexpr int chunk=decltype(index)::value;
            const int codes=packed.template codes<chunk>()^(Bits==8?int(0x80808080u):0);
            const int at=(group-first)*24+chunk*4;
            high=__dp4a(codes,*reinterpret_cast<const int*>(q.high+at),high);
            if constexpr(Planes==3)middle=__dp4a(codes,*reinterpret_cast<const int*>(q.middle+at),middle);
            const int activation=*reinterpret_cast<const int*>(q.low+at);
            if constexpr(Planes==4) {
                const int weights=packed.template codes<chunk>();
                asm("dp4a.u32.u32 %0, %1, %2, %0;" : "+r"(low) : "r"(weights),"r"(activation));
            }else low=__dp4a(codes,activation,low);
        };
        consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
        consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
        consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
        const auto sum=q.sums[group-first];
        if constexpr(Bits==8) {
            high+=128*sum.x;
            if constexpr(Planes==3){middle+=128*sum.y;low+=128*sum.z;}
        }
        const float scale=outer_scale*float(w.scales[size_t(row)*w.groups+group]);
        const float minimum=outer_minimum*float(w.minima[size_t(row)*w.groups+group]);
        const float dot=Planes==3?fmaf(65536.f,float(high),fmaf(256.f,float(middle),float(low))):
            fmaf(256.f,float(high),float(low));
        const int total=Planes==3?sum.x*65536+sum.y*256+sum.z:sum.x*256+sum.z;
        partial=fmaf(q.scales[group-first].y,fmaf(scale,dot,-minimum*float(total)),partial);
    }
    #pragma unroll
    for(int shift=16;shift;shift>>=1)partial+=__shfl_down_sync(0xffffffffu,partial,shift);
    return partial;
}
template<int Planes=3>
__device__ __forceinline__ float gr_packed24_affine(const MhcMatrix& w,GrTwoPlane q,
        int row,int first,int last,int lane) {
    switch(w.bits[row]) {
    case 1:return gr_packed24_affine_bits<1,Planes>(w,q,row,first,last,lane);
    case 2:return gr_packed24_affine_bits<2,Planes>(w,q,row,first,last,lane);
    case 3:return gr_packed24_affine_bits<3,Planes>(w,q,row,first,last,lane);
    case 4:return gr_packed24_affine_bits<4,Planes>(w,q,row,first,last,lane);
    case 5:return gr_packed24_affine_bits<5,Planes>(w,q,row,first,last,lane);
    case 6:return gr_packed24_affine_bits<6,Planes>(w,q,row,first,last,lane);
    case 7:return gr_packed24_affine_bits<7,Planes>(w,q,row,first,last,lane);
    case 8:return gr_packed24_affine_bits<8,Planes>(w,q,row,first,last,lane);
    default:return 0.f;
    }
}

template<int Bits,int Capacity,int Planes>
__device__ __forceinline__ void gr_packed24_affine_batch_bits(const MhcMatrix& w,GrTwoPlane q,
        int row,int first,int last,int sample,int samples,int lane,float (&partial)[Capacity]) {
    const auto offset=uint64_t(w.offsets[row]);
    const float outer_scale=w.row_scale[row],outer_minimum=w.row_minimum[row];
    for(int group=first+lane;group<last;group+=32) {
        const NintWholeGroup<24,Bits> packed(w.q,offset+uint64_t(group)*24*Bits,w.bytes);
        int high[Capacity]{},middle[Capacity]{},low[Capacity]{};
        const auto consume=[&](auto index) {
            constexpr int chunk=decltype(index)::value;
            const int weights=packed.template codes<chunk>();
            const int centered=weights^(Bits==8?int(0x80808080u):0);
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample+k<samples) {
                const size_t at=(size_t(sample+k)*w.groups+group)*24+chunk*4;
                high[k]=__dp4a(centered,*reinterpret_cast<const int*>(q.high+at),high[k]);
                if constexpr(Planes==3)middle[k]=__dp4a(centered,*reinterpret_cast<const int*>(q.middle+at),middle[k]);
                const int activation=*reinterpret_cast<const int*>(q.low+at);
                if constexpr(Planes==4) {
                    asm("dp4a.u32.u32 %0, %1, %2, %0;" : "+r"(low[k]) : "r"(weights),"r"(activation));
                }else low[k]=__dp4a(centered,activation,low[k]);
            }
        };
        consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
        consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
        consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
        const float scale=outer_scale*float(w.scales[size_t(row)*w.groups+group]);
        const float minimum=outer_minimum*float(w.minima[size_t(row)*w.groups+group]);
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample+k<samples) {
            const size_t at=size_t(sample+k)*w.groups+group;
            const auto sum=q.sums[at];
            if constexpr(Bits==8) {
                high[k]+=128*sum.x;
                if constexpr(Planes==3){middle[k]+=128*sum.y;low[k]+=128*sum.z;}
            }
            const float dot=Planes==3?fmaf(65536.f,float(high[k]),fmaf(256.f,float(middle[k]),float(low[k]))):
                fmaf(256.f,float(high[k]),float(low[k]));
            const int total=Planes==3?sum.x*65536+sum.y*256+sum.z:sum.x*256+sum.z;
            partial[k]=fmaf(q.scales[at].y,fmaf(scale,dot,-minimum*float(total)),partial[k]);
        }
    }
    #pragma unroll
    for(int shift=16;shift;shift>>=1) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)partial[k]+=__shfl_down_sync(0xffffffffu,partial[k],shift);
    }
}
template<int Capacity,int Planes>
__device__ __forceinline__ void gr_packed24_affine_batch(const MhcMatrix& w,GrTwoPlane q,
        int row,int first,int last,int sample,int samples,int lane,float (&partial)[Capacity]) {
    switch(w.bits[row]) {
    case 1:gr_packed24_affine_batch_bits<1,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    case 2:gr_packed24_affine_batch_bits<2,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    case 3:gr_packed24_affine_batch_bits<3,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    case 4:gr_packed24_affine_batch_bits<4,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    case 5:gr_packed24_affine_batch_bits<5,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    case 6:gr_packed24_affine_batch_bits<6,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    case 7:gr_packed24_affine_batch_bits<7,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    case 8:gr_packed24_affine_batch_bits<8,Capacity,Planes>(w,q,row,first,last,sample,samples,lane,partial);break;
    }
}

template<int GS,bool ThreePlane=false,bool UnsignedLow=false,bool ParallelChunks=false,bool Combined=false,bool UnsignedWeights=false>
__device__ __forceinline__ float gr_dp4a2_affine(const MhcMatrix& w,GrTwoPlane q,
        int row,int first,int last,int lane) {
    const auto* packed=w.q+(w.offsets[row]>>3);
    const float outer_scale=w.row_scale[row],outer_minimum=w.row_minimum[row];
    float partial=0.f;
    static_assert(!ParallelChunks || GS%16==0);
    const int chunks=GS/16;
    // The 320-wide Up has seven groups. Assign its three 16-byte chunks
    // to separate lanes; larger geometries use the original group mapping.
    for(int group=first+(ParallelChunks?lane/chunks:lane);group<last;group+=ParallelChunks?32/chunks:32) {
        int high_dot=0,middle_dot=0,low_dot=0;
        const int at=(group-first)*GS;
        #pragma unroll
        for(int chunk=0;chunk<GS/16;++chunk) {
            if constexpr(ParallelChunks)if(chunk!=lane%chunks)continue;
            const int column=group*GS+chunk*16;
            if(column<w.width) {
                int4 weights;
                if((reinterpret_cast<uintptr_t>(packed+column)&15u)==0)
                    weights=*reinterpret_cast<const int4*>(packed+column);
                else {
                    int words[4];
                    #pragma unroll
                    for(int item=0;item<4;++item) {
                        uint32_t word=0;
                        #pragma unroll
                        for(int b=0;b<4;++b)word|=uint32_t(packed[column+item*4+b])<<(8*b);
                        words[item]=int(word);
                    }
                    weights=make_int4(words[0],words[1],words[2],words[3]);
                }
                const auto h=*reinterpret_cast<const int4*>(q.high+at+chunk*16);
                const auto l=*reinterpret_cast<const int4*>(q.low+at+chunk*16);
                const int mask=UnsignedWeights?0:int(0x80808080u);
                high_dot=gr_plane_dot<UnsignedWeights>(weights.x^mask,h.x,high_dot);
                high_dot=gr_plane_dot<UnsignedWeights>(weights.y^mask,h.y,high_dot);
                high_dot=gr_plane_dot<UnsignedWeights>(weights.z^mask,h.z,high_dot);
                high_dot=gr_plane_dot<UnsignedWeights>(weights.w^mask,h.w,high_dot);
                if constexpr(UnsignedLow) {
                    low_dot=int(__dp4a(unsigned(weights.x),unsigned(l.x),unsigned(low_dot)));
                    low_dot=int(__dp4a(unsigned(weights.y),unsigned(l.y),unsigned(low_dot)));
                    low_dot=int(__dp4a(unsigned(weights.z),unsigned(l.z),unsigned(low_dot)));
                    low_dot=int(__dp4a(unsigned(weights.w),unsigned(l.w),unsigned(low_dot)));
                }else {
                    low_dot=gr_plane_dot<UnsignedWeights>(weights.x^mask,l.x,low_dot);
                    low_dot=gr_plane_dot<UnsignedWeights>(weights.y^mask,l.y,low_dot);
                    low_dot=gr_plane_dot<UnsignedWeights>(weights.z^mask,l.z,low_dot);
                    low_dot=gr_plane_dot<UnsignedWeights>(weights.w^mask,l.w,low_dot);
                }
                if constexpr(ThreePlane) {
                    const auto m=*reinterpret_cast<const int4*>(q.middle+at+chunk*16);
                    middle_dot=gr_plane_dot<UnsignedWeights>(weights.x^mask,m.x,middle_dot);
                    middle_dot=gr_plane_dot<UnsignedWeights>(weights.y^mask,m.y,middle_dot);
                    middle_dot=gr_plane_dot<UnsignedWeights>(weights.z^mask,m.z,middle_dot);
                    middle_dot=gr_plane_dot<UnsignedWeights>(weights.w^mask,m.w,middle_dot);
                }
            }
        }
        const auto sums=(!ParallelChunks || lane%chunks==0)?q.sums[group-first]:make_int4(0,0,0,0);
        if constexpr(!UnsignedWeights) {
            high_dot+=128*sums.x;
            if constexpr(!UnsignedLow)low_dot+=128*sums.z;
            if constexpr(ThreePlane)middle_dot+=128*sums.y;
        }
        const size_t meta=size_t(row)*w.groups+group;
        const float scale=outer_scale*float(w.scales[meta]);
        const float minimum=outer_minimum*float(w.minima[meta]);
        const auto scales=q.scales[group-first];
        if constexpr(Combined) {
            static_assert(ThreePlane || UnsignedLow);
            const float dot=ThreePlane?fmaf(65536.f,float(high_dot),fmaf(256.f,float(middle_dot),float(low_dot))):
                fmaf(256.f,float(high_dot),float(low_dot));
            const int total=ThreePlane?sums.x*65536+sums.y*256+sums.z:sums.x*256+sums.z;
            partial=fmaf(scales.y,fmaf(scale,dot,-minimum*float(total)),partial);
        }else {
        const float high=fmaf(scale,float(high_dot),-minimum*float(sums.x));
        const float low=fmaf(scale,float(low_dot),-minimum*float(sums.z));
        partial=fmaf(scales.x,high,partial);
        if constexpr(ThreePlane) {
            const float middle=fmaf(scale,float(middle_dot),-minimum*float(sums.y));
            partial=fmaf(scales.x*(1.f/256.f),middle,partial);
        }
        partial=fmaf(scales.y,low,partial);
        }
    }
    #pragma unroll
    for(int offset=16;offset;offset>>=1)partial+=__shfl_down_sync(0xffffffffu,partial,offset);
    return partial;
}

template<int GS,int Capacity,bool ThreePlane,bool UnsignedLow,bool Combined=false,bool UnsignedWeights=false>
__device__ __forceinline__ void gr_dp4a_affine_batch(const MhcMatrix& w,GrTwoPlane q,
        int row,int first,int last,int sample,int samples,int lane,float (&partial)[Capacity]) {
    const auto* packed=w.q+(w.offsets[row]>>3);
    const float outer_scale=w.row_scale[row],outer_minimum=w.row_minimum[row];
    for(int group=first+lane;group<last;group+=32) {
        int high[Capacity]{},middle[Capacity]{},low[Capacity]{};
        #pragma unroll
        for(int chunk=0;chunk<GS/16;++chunk) {
            const int column=group*GS+chunk*16;
            if(column<w.width) {
                int4 weights;
                if((reinterpret_cast<uintptr_t>(packed+column)&15u)==0)
                    weights=*reinterpret_cast<const int4*>(packed+column);
                else {
                    int words[4];
                    #pragma unroll
                    for(int item=0;item<4;++item) {
                        unsigned word=0;
                        #pragma unroll
                        for(int b=0;b<4;++b)word|=unsigned(packed[column+item*4+b])<<(8*b);
                        words[item]=int(word);
                    }
                    weights=make_int4(words[0],words[1],words[2],words[3]);
                }
                const int mask=UnsignedWeights?0:int(0x80808080u);
                const int4 centered=make_int4(weights.x^mask,weights.y^mask,weights.z^mask,weights.w^mask);
                #pragma unroll
                for(int k=0;k<Capacity;++k)if(sample+k<samples) {
                    const size_t at=(size_t(sample+k)*w.groups+group)*GS+chunk*16;
                    auto x=*reinterpret_cast<const int4*>(q.high+at);
                    high[k]=gr_plane_dot<UnsignedWeights>(centered.x,x.x,high[k]);high[k]=gr_plane_dot<UnsignedWeights>(centered.y,x.y,high[k]);
                    high[k]=gr_plane_dot<UnsignedWeights>(centered.z,x.z,high[k]);high[k]=gr_plane_dot<UnsignedWeights>(centered.w,x.w,high[k]);
                    x=*reinterpret_cast<const int4*>(q.low+at);
                    if constexpr(UnsignedLow) {
                        low[k]=int(__dp4a(unsigned(weights.x),unsigned(x.x),unsigned(low[k])));
                        low[k]=int(__dp4a(unsigned(weights.y),unsigned(x.y),unsigned(low[k])));
                        low[k]=int(__dp4a(unsigned(weights.z),unsigned(x.z),unsigned(low[k])));
                        low[k]=int(__dp4a(unsigned(weights.w),unsigned(x.w),unsigned(low[k])));
                    }else {
                        low[k]=gr_plane_dot<UnsignedWeights>(centered.x,x.x,low[k]);low[k]=gr_plane_dot<UnsignedWeights>(centered.y,x.y,low[k]);
                        low[k]=gr_plane_dot<UnsignedWeights>(centered.z,x.z,low[k]);low[k]=gr_plane_dot<UnsignedWeights>(centered.w,x.w,low[k]);
                    }
                    if constexpr(ThreePlane) {
                        x=*reinterpret_cast<const int4*>(q.middle+at);
                        middle[k]=gr_plane_dot<UnsignedWeights>(centered.x,x.x,middle[k]);middle[k]=gr_plane_dot<UnsignedWeights>(centered.y,x.y,middle[k]);
                        middle[k]=gr_plane_dot<UnsignedWeights>(centered.z,x.z,middle[k]);middle[k]=gr_plane_dot<UnsignedWeights>(centered.w,x.w,middle[k]);
                    }
                }
            }
        }
        const size_t meta=size_t(row)*w.groups+group;
        const float scale=outer_scale*float(w.scales[meta]),minimum=outer_minimum*float(w.minima[meta]);
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample+k<samples) {
            const size_t at=size_t(sample+k)*w.groups+group;
            const auto sum=q.sums[at];const auto scales=q.scales[at];
            if constexpr(!UnsignedWeights) {
                high[k]+=128*sum.x;
                if constexpr(!UnsignedLow)low[k]+=128*sum.z;
                if constexpr(ThreePlane)middle[k]+=128*sum.y;
            }
            if constexpr(Combined) {
                static_assert(ThreePlane || UnsignedLow);
                const float dot=ThreePlane?fmaf(65536.f,float(high[k]),fmaf(256.f,float(middle[k]),float(low[k]))):
                    fmaf(256.f,float(high[k]),float(low[k]));
                const int total=ThreePlane?sum.x*65536+sum.y*256+sum.z:sum.x*256+sum.z;
                partial[k]=fmaf(scales.y,fmaf(scale,dot,-minimum*float(total)),partial[k]);
            }else {
            const float h=fmaf(scale,float(high[k]),-minimum*float(sum.x));
            const float l=fmaf(scale,float(low[k]),-minimum*float(sum.z));
            partial[k]=fmaf(scales.x,h,partial[k]);
            if constexpr(ThreePlane) {
                const float m=fmaf(scale,float(middle[k]),-minimum*float(sum.y));
                partial[k]=fmaf(scales.x*(1.f/256.f),m,partial[k]);
            }
            partial[k]=fmaf(scales.y,l,partial[k]);
            }
        }
    }
    #pragma unroll
    for(int shift=16;shift;shift>>=1) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)partial[k]+=__shfl_down_sync(0xffffffffu,partial[k],shift);
    }
}
