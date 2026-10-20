#pragma once

// RMS statistics stay on four CTAs with the original reduction order. Move
// the normalization stores to the already parallel activation-quantization
// grid, and quantize the rounded values directly from the warp's shared tile.
template<int GS,bool ThreePlane,bool UnsignedLow,int FixedHalf=-1,bool RegisterValues=false,int Threads=256>
__global__ void gated_residual_finish_norm_quant_kernel(MhcInput input,MhcMatrix down,
        const float* inverses,void* updated,void* normalized,float* promoted_normalized,
        GrTwoPlane quantized) {
    if constexpr(FixedHalf>=0) {
        input.output_half=FixedHalf!=0;
        down.width=10240;down.groups=(10240+GS-1)/GS;
    }
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sample=blockIdx.y;
    static_assert(Threads==128 || Threads==256);
    const int group=int(blockIdx.x)*(Threads/32)+warp;
    __shared__ float values[RegisterValues?1:(Threads/32)*GS];
    float x0=0.f,x1=0.f;
    if(group>=down.groups)return;
    #pragma unroll
    for(int part=0;part<2;++part) {
        const int local=lane+part*32,column=group*GS+local;
        if(local<GS) {
            float norm_value=0.f;
            if(column<down.width) {
                const int stream=column/2560;
                const size_t at=size_t(sample)*10240+column;
                const float value=mhc_read(updated,at,input.output_half);
                norm_value=mhc_round(__fmul_rn(__fmul_rn(value,inverses[sample*4+stream]),
                    __fadd_rn(input.norm[column],1.f)),input.output_half);
                mhc_write(normalized,at,norm_value,input.output_half);
                if(promoted_normalized)promoted_normalized[at]=norm_value;
            }
            if constexpr(RegisterValues) {
                if(part==0)x0=norm_value;else x1=norm_value;
            }else values[warp*GS+local]=norm_value;
        }
    }
    if constexpr(!RegisterValues)__syncwarp();
    const size_t at=(size_t(sample)*down.groups+group)*GS;
    GrTwoPlane local{quantized.high+at,
        ThreePlane?quantized.middle+at:nullptr,quantized.low+at,
        quantized.scales+size_t(sample)*down.groups+group,
        quantized.sums+size_t(sample)*down.groups+group};
    MhcMatrix tile{};tile.width=GS;tile.groups=1;
    if constexpr(RegisterValues)
        gr_quantize_activation_values<GS,ThreePlane,UnsignedLow>(tile,x0,x1,local,0,0,lane);
    else gr_quantize_activation_group<GS,ThreePlane,UnsignedLow>(tile,values+warp*GS,
        false,local,0,0,lane);
}

template<bool Half>
__global__ void gated_residual_finish_norm_quant_s16_kernel(MhcInput input,
        const float* inverses,const void* updated,void* normalized,GrTwoPlane quantized) {
    constexpr int GS=48,Groups=214;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,sample=blockIdx.y;
    const int group=int(blockIdx.x)*4+warp;
    if(group>=Groups)return;
    float x0=0.f,x1=0.f;
    #pragma unroll
    for(int part=0;part<2;++part) {
        const int local=lane+part*32,column=group*GS+local;
        if(local<GS && column<10240) {
            const size_t at=size_t(sample)*10240+column;
            const float value=mhc_read(updated,at,Half);
            const float rounded=mhc_round(__fmul_rn(__fmul_rn(value,inverses[sample*4+column/2560]),
                __fadd_rn(input.norm[column],1.f)),Half);
            mhc_write(normalized,at,rounded,Half);
            if(part==0)x0=rounded;else x1=rounded;
        }
    }
    const size_t slot=size_t(sample)*Groups+group;
    GrTwoPlane local{quantized.high+2*slot*GS,nullptr,nullptr,quantized.scales+slot,nullptr};
    MhcMatrix tile{};tile.width=GS;tile.groups=1;
    gr_quantize_activation_values<48,false,true,true,true>(tile,x0,x1,local,0,0,lane);
}
