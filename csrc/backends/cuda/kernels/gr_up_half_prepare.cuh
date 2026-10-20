#pragma once

struct GrUpHalf {
    __half *high,*low;
    float2* statistics;
};

template<bool OnePlane>
__device__ __forceinline__ void gr_prepare_up_half_values(float x0,float x1,GrUpHalf q,
        int sample,int group,int lane) {
    constexpr int GS=48,Groups=7,Elements=336;
    float scale=1.f;
    if constexpr(!OnePlane) {
        float maximum=fmaxf(fabsf(x0),fabsf(x1));
        #pragma unroll
        for(int offset=16;offset;offset>>=1)
            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
        if(isfinite(maximum) && maximum>32768.f)
            scale=__int_as_float((ilogbf(maximum)-13+127)<<23);
    }
    const float v0=x0/scale,v1=x1/scale;
    const __half h0=__float2half_rn(v0),h1=__float2half_rn(v1);
    const __half l0=OnePlane?__float2half_rn(0.f):__float2half_rn(v0-__half2float(h0));
    const __half l1=OnePlane?__float2half_rn(0.f):__float2half_rn(v1-__half2float(h1));
    const int at=sample*Elements+group*GS+lane;
    q.high[at]=h0;
    if(lane+32<GS)q.high[at+32]=h1;
    if constexpr(!OnePlane) {
        q.low[at]=l0;
        if(lane+32<GS)q.low[at+32]=l1;
    }
    float sum=__fadd_rn(__fadd_rn(__half2float(h0),__half2float(l0)),
        __fadd_rn(__half2float(h1),__half2float(l1)));
    #pragma unroll
    for(int offset=16;offset;offset>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
    if(lane==0)q.statistics[sample*Groups+group]=make_float2(scale,sum);
}
