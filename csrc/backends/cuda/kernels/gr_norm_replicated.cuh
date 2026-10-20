#pragma once

// Decode candidate: each CTA owns a few complete GS24/48 groups, recomputes
// their stream's RMS, and immediately quantizes them. Repeated cached reads
// buy more independent CTAs and remove the statistics->quantization launch.
// Cross-stream groups compute both inverses locally: no counters or global
// producer synchronization, and every output element has a single writer.
template<int GS,int Groups48,bool ThreePlane,bool After,bool BranchHalf>
__global__ void __launch_bounds__(256) gated_residual_norm_replicated_kernel(
        MhcInput input,void* updated,void* normalized,GrTwoPlane quantized,float eps) {
    constexpr int Groups=Groups48*48/GS,TotalGroups=(10240+GS-1)/GS;
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sample=blockIdx.y;
    const size_t row=size_t(sample)*10240;
    const int first_group=int(blockIdx.x)*Groups;
    const int begin=first_group*GS,end=min(10240,(first_group+Groups)*GS);
    const int first_stream=begin/2560,last_stream=(end-1)/2560;
    __shared__ float partial[256],inverse[2];
    for(int stream=first_stream;stream<=last_stream;++stream) {
        const float gate=After?mhc_read(input.injection,size_t(sample)*4+stream,input.injection_half):0.f;
        float sum=0.f;
        #pragma unroll
        for(int item=0;item<10;++item) {
            const int column=tid+item*256;
            float value=mhc_read(input.residual,row+stream*2560+column,true);
            if constexpr(After) {
                const float product=mhc_round(__fmul_rn(mhc_read(input.branch,size_t(sample)*2560+column,BranchHalf),gate),
                    BranchHalf && input.injection_half);
                value=mhc_round(__fadd_rn(value,product),true);
            }
            sum=__fadd_rn(sum,__fmul_rn(value,value));
        }
        partial[tid]=sum;__syncthreads();
        if(tid<32) {
            const float a=__fadd_rn(partial[tid],partial[tid+128]);
            const float b=__fadd_rn(partial[tid+64],partial[tid+192]);
            const float c=__fadd_rn(partial[tid+32],partial[tid+160]);
            const float d=__fadd_rn(partial[tid+96],partial[tid+224]);
            float value=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
            #pragma unroll
            for(int shift=16;shift;shift>>=1) {
                const float other=__shfl_down_sync(0xffffffffu,value,shift);
                if(tid<shift)value=__fadd_rn(value,other);
            }
            if(!tid)inverse[stream-first_stream]=mhc_norm_inverse(value,2560,eps,input.fast_norm);
        }
        __syncthreads();
    }
    for(int group=first_group+warp;group<first_group+Groups && group<TotalGroups;group+=8) {
        float x0=0.f,x1=0.f;
        #pragma unroll
        for(int part=0;part<2;++part) {
            const int local=lane+part*32,column=group*GS+local;
            if(local<GS && column<10240) {
                const int stream=column/2560;
                float value=mhc_read(input.residual,row+column,true);
                if constexpr(After) {
                    const float product=mhc_round(__fmul_rn(mhc_read(input.branch,size_t(sample)*2560+column%2560,BranchHalf),
                        mhc_read(input.injection,size_t(sample)*4+stream,input.injection_half)),BranchHalf && input.injection_half);
                    value=mhc_round(__fadd_rn(value,product),true);
                }
                mhc_write(updated,row+column,value,true);
                const float rounded=mhc_round(__fmul_rn(__fmul_rn(value,inverse[stream-first_stream]),
                    __fadd_rn(input.norm[column],1.f)),true);
                mhc_write(normalized,row+column,rounded,true);
                if(part==0)x0=rounded;else x1=rounded;
            }
        }
        const size_t slot=size_t(sample)*TotalGroups+group;
        GrTwoPlane local{quantized.high+slot*GS,ThreePlane?quantized.middle+slot*GS:nullptr,quantized.low+slot*GS,
            quantized.scales+slot,quantized.sums+slot};
        MhcMatrix tile{};tile.width=GS;tile.groups=1;
        gr_quantize_activation_values<GS,ThreePlane,!ThreePlane>(tile,x0,x1,local,0,0,lane);
    }
}
