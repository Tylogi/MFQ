#pragma once

constexpr int GrNormCounterSlots=1024;
__device__ unsigned gr_norm_quant_counters[GrNormCounterSlots*6*2]{};

// Independent graphs may be captured on the same reused stream. Give each
// capture its own counter slot; eager operations retain a per-stream slot.
// A graph's mutable buffers and stream dependencies serialize its replays.
int gr_norm_counter_slot() {
    static std::atomic<unsigned> next{0};
    thread_local std::unordered_map<cudaStream_t,int> slots;
    thread_local std::unordered_map<unsigned long long,int> capture_slots;
    const auto stream=mfq_current_cuda_stream();
    cudaStreamCaptureStatus status;unsigned long long capture_id;
    MFQ_CUDA_CHECK(cudaStreamGetCaptureInfo(stream,&status,&capture_id));
    if(status==cudaStreamCaptureStatusActive) {
        const auto found=capture_slots.find(capture_id);
        if(found!=capture_slots.end())return found->second;
        const unsigned index=next.fetch_add(1,std::memory_order_relaxed);
        const int slot=index<GrNormCounterSlots?int(index):-1;
        capture_slots.emplace(capture_id,slot);return slot;
    }
    const auto found=slots.find(stream);
    if(found!=slots.end())return found->second;
    const unsigned index=next.fetch_add(1,std::memory_order_relaxed);
    const int slot=index<GrNormCounterSlots?int(index):-1;
    slots.emplace(stream,slot);return slot;
}

// More warps load/store the 2560 values; the first 256 threads retain the
// original ten-value sums and the exact original 256-thread reduction tree.
template<int GS=0,int Planes=0,bool InlineBoundary=false,int NormThreads=1024,bool StatsOnly=false,int Slices=1,int FixedKind=0>
__global__ void __launch_bounds__(NormThreads,1) gated_residual_norm_wide_kernel(
        MhcInput input,void* updated,void* normalized,float* promoted_normalized,
        float eps,const float* prepared_right,MhcMatrix down={},GrTwoPlane quantized={},int counter_slot=-1,float* saved_inverse=nullptr) {
    const int tid=threadIdx.x,sample=blockIdx.y,stream=blockIdx.x;
    const int slice=blockIdx.z;
    static_assert(Slices==1 || Slices==2 || Slices==4 || Slices==8 || Slices==16 || Slices==32);
    static_assert(Slices==1 || (GS>0 && Planes>0 && InlineBoundary && !StatsOnly));
    int first_column=0,last_column=2560;
    if constexpr(Slices>1) {
        if(slice>0)first_column=((stream*2560+slice*(2560/Slices)+GS-1)/GS)*GS-stream*2560;
        if(slice+1<Slices)last_column=((stream*2560+(slice+1)*(2560/Slices)+GS-1)/GS)*GS-stream*2560;
    }
    input.hidden=2560;input.streams=4;
    if constexpr(FixedKind>0) {
        static_assert(FixedKind<=6);
        input.residual_half=FixedKind<=3;
        input.output_half=FixedKind<=3;
        input.after=FixedKind!=1 && FixedKind!=4;
        input.branch_half=FixedKind==2 || FixedKind==5;
    }
    __shared__ float values[NormThreads==256?2560/Slices+GS:2560],partial[256],inverse,prior_gates[4];
    if(tid<4)prior_gates[tid]=input.after?
        mhc_read(input.injection,size_t(sample)*4+tid,input.injection_half):0.f;
    input.cached_gates=prior_gates;
    __syncthreads();
    static_assert(NormThreads==256 || NormThreads==512 || NormThreads==640 || NormThreads==1024);
    static_assert(Planes==0 || NormThreads==1024 || Slices>1);
    static_assert(!StatsOnly || Planes==0);
    constexpr int Items=(2560+NormThreads-1)/NormThreads;
    float cached[Items];
    #pragma unroll
    for(int item=0;item<Items;++item) {
        const int column=tid+NormThreads*item;
        if(column<2560) {
            const float value=mhc_updated(input,sample,stream,column);
            cached[item]=value;
            if constexpr(NormThreads!=256)values[column]=value;
            if(column>=first_column && column<last_column)
                mhc_write(updated,(size_t(sample)*4+stream)*2560+column,value,input.output_half);
        }
    }
    if constexpr(NormThreads!=256)__syncthreads();
    if(tid<256) {
        float sum=0.f;
        #pragma unroll
        for(int item=0;item<10;++item) {
            const float value=NormThreads==256?cached[item]:values[tid+256*item];
            sum=__fadd_rn(sum,__fmul_rn(value,value));
        }
        partial[tid]=sum;
    }
    __syncthreads();
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
        if(tid==0)inverse=mhc_norm_inverse(value,2560,eps,input.fast_norm);
    }
    __syncthreads();
    if constexpr(StatsOnly) {
        if(tid==0)saved_inverse[size_t(sample)*4+stream]=inverse;
        return;
    }
    bool boundary_writer=false;
    #pragma unroll
    for(int item=0;item<Items;++item) {
        const int column=tid+NormThreads*item;
        if(column>=first_column && column<last_column) {
            const size_t at=(size_t(sample)*4+stream)*2560+column;
            const float value=mhc_round(__fmul_rn(__fmul_rn(cached[item],inverse),
                __fadd_rn(input.norm[stream*2560+column],1.f)),input.output_half);
            mhc_write(normalized,at,value,input.output_half);
            if(promoted_normalized)promoted_normalized[at]=value;
            if constexpr(Planes>0)values[column-(NormThreads==256?first_column:0)]=value;
            if constexpr(InlineBoundary && Slices>1) {
                const int absolute=stream*2560+column;
                const int first_edge=(2560/GS)*GS,second_edge=(5120/GS)*GS;
                boundary_writer|=(absolute>=first_edge && absolute<first_edge+GS) ||
                    (absolute>=second_edge && absolute<second_edge+GS);
            }
        }
    }
    if constexpr(Planes>0) {
        static_assert(GS==24 || GS==48);
        __syncthreads();
        const int first=(stream*2560+first_column+GS-1)/GS;
        const int last=stream==3 && slice==Slices-1?(10240+GS-1)/GS:(stream*2560+last_column)/GS;
        const int lane=tid&31,warp=tid>>5;
        constexpr int Groups=(10240+GS-1)/GS;
        for(int group=first+warp;group<last;group+=NormThreads/32) {
            const size_t slot=size_t(sample)*Groups+group,at=slot*GS;
            GrTwoPlane q{quantized.high+at,Planes==3?quantized.middle+at:nullptr,
                quantized.low+at,quantized.scales+slot,quantized.sums+slot};
            MhcMatrix tile{};tile.width=min(GS,10240-group*GS);tile.groups=1;
            gr_quantize_activation_group<GS,Planes==3,Planes==4>(tile,
                values+group*GS-stream*2560-(NormThreads==256?first_column:0),false,q,0,0,lane);
        }
    }
    if constexpr(InlineBoundary)if((stream==0 && slice==Slices-1) ||
        (stream==1 && (slice==0 || slice==Slices-1)) || (stream==2 && slice==0)) {
        // Strata's second-arrival protocol: publish every writer, then let
        // only the later neighbor quantize the boundary through volatile reads.
        __shared__ int last[2];
        if constexpr(Slices>1) {if(boundary_writer)__threadfence();}
        else __threadfence();
        __syncthreads();
        if(tid<2) {
            last[tid]=0;
            if((stream==tid && slice==Slices-1) || (stream==tid+1 && slice==0)) {
                auto* counter=gr_norm_quant_counters+counter_slot*12+sample*2+tid;
                last[tid]=atomicAdd(counter,1u)==1u;
                if(last[tid])atomicExch(counter,0u);
            }
        }
        __syncthreads();
        const int warp=tid>>5,lane=tid&31;
        if(warp<2 && last[warp]) {
            __threadfence();
            const int group=warp==0?2560/GS:5120/GS;
            #pragma unroll
            for(int chunk=0;chunk<2;++chunk) {
                const int j=lane+chunk*32;
                if(j<GS) {
                    const size_t at=size_t(sample)*10240+group*GS+j;
                    const float value=input.output_half?
                        __half2float(__ushort_as_half(reinterpret_cast<const volatile unsigned short*>(normalized)[at])):
                        reinterpret_cast<const volatile float*>(normalized)[at];
                    partial[warp*GS+j]=value;
                }
            }
            __syncwarp();
            constexpr int Groups=(10240+GS-1)/GS;
            const size_t slot=size_t(sample)*Groups+group,at=slot*GS;
            GrTwoPlane q{quantized.high+at,Planes==3?quantized.middle+at:nullptr,
                quantized.low+at,quantized.scales+slot,quantized.sums+slot};
            MhcMatrix tile{};tile.width=GS;tile.groups=1;
            gr_quantize_activation_group<GS,Planes==3,Planes==4>(tile,
                partial+warp*GS,false,q,0,0,lane);
        }
    }
    if(prepared_right && blockIdx.x==0 && blockIdx.y==0 && slice==0) {
        const auto* address=reinterpret_cast<const char*>(prepared_right);
        constexpr size_t bytes=2560*4*4*sizeof(float);
        for(size_t at=size_t(tid)*128;at<bytes;at+=NormThreads*128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(address+at));
    }
}

// Only the groups crossing the first and second stream boundaries remain.
template<int GS,bool ThreePlane,bool UnsignedLow>
__global__ void gated_residual_quantize_cross_stream_kernel(MhcMatrix down,
        const void* normalized,bool half,GrTwoPlane q) {
    static_assert(2560%GS!=0 && 5120%GS!=0 && 7680%GS==0);
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int group=warp==0?2560/GS:5120/GS;
    gr_quantize_activation_group<GS,ThreePlane,UnsignedLow>(down,normalized,half,
        q,group,int(blockIdx.y),lane);
}
