#pragma once

// Caller proves regular NINT group geometry and the original token plan.
// Fixed token and row indices keep result arrays out of thread-local memory.
namespace mfq::cuda { namespace {
template<int Capacity>
__device__ __forceinline__ int short_shared_count(const MfeFfnBatch& b,int group) {
    if constexpr(Capacity==1)if(b.tokens==1)return 1;
    return b.token_group_counts[b.tokens*b.routes+group];
}
template<int Capacity>
__device__ __forceinline__ int short_shared_slot(const MfeFfnBatch& b,int group,int k) {
    if constexpr(Capacity==1)if(b.tokens==1)return b.routes;
    return b.token_group_slots[(b.tokens*b.routes+group)*8+k];
}
template<int Capacity>
__device__ __forceinline__ void short_shared_project(const MfePackedProjection& d,int first,
        const int8_t* const* inputs,const float* const* scales,int count,
        __half values[Capacity][4],bool multi_sum) {
    if constexpr(Capacity>1) {
        if(d.group_size==24)nint_token_pair<24,Capacity>(d,0,first,inputs,scales,count,values,0);
        else if(d.group_size==28)nint_token_pair<28,Capacity>(d,0,first,inputs,scales,count,values,0);
        else nint_token_pair<32,Capacity>(d,0,first,inputs,scales,count,values,0);
    }else {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count)project_pair<true,0,true>(&d,0,first,inputs[k],scales[k],values[k],multi_sum);
    }
}
template<int Capacity,bool LowParallelism>
__global__ void __launch_bounds__(256,Capacity==2 && LowParallelism?3:Capacity<=2?4:Capacity<=4?3:2) short_shared_gate_up(MfeFfnBatch b,int group_start) {
    const int group=group_start+int(blockIdx.y),count=short_shared_count<Capacity>(b,group);
    if(count<1 || count>Capacity)return;
    const int lane=threadIdx.x,warp=threadIdx.y;
    const int base=int(blockIdx.x)*8,first=base+(warp%4)*2;
    if(base>=b.shared_intermediate)return;
    const auto& d=b.shared[warp>=4];
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
    #pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
        const int token=short_shared_slot<Capacity>(b,group,k)/(b.routes+1);
        inputs[k]=static_cast<const int8_t*>(d.fields[7])+int64_t(token)*d.groups*d.group_size;
        scales[k]=static_cast<const float*>(d.fields[8])+int64_t(token)*d.groups;
    }
    __half values[Capacity][4]{};
    if(first<b.shared_intermediate)short_shared_project<Capacity>(d,first,inputs,scales,count,values,b.warp_multi_sum);
    __shared__ __half projected[8][Capacity][2];
    if(!lane) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count) {
            projected[warp][k][0]=values[k][0];projected[warp][k][1]=values[k][1];
        }
    }
    __syncthreads();
    if(warp)return;
    if(lane<8 && base+lane<b.shared_intermediate) {
      #pragma unroll
      for(int k=0;k<Capacity;++k)if(k<count) {
        const int pair=lane/2,row=lane&1;
        const auto gate=projected[pair][k][row],up=projected[pair+4][k][row];
        const int slot=short_shared_slot<Capacity>(b,group,k);
        b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=
            mfe_ffn_swiglu(b,gate,up,b.sigmoid_table);
      }
    }
}
__global__ void __launch_bounds__(128) short_shared_quantize(MfeFfnBatch b) {
    const int group=int(blockIdx.x)*4+int(threadIdx.y);
    const int gs=b.shared[2].group_size;
    if(group*gs>=b.shared_intermediate)return;
    const int slot=int(blockIdx.y)*(b.routes+1)+b.routes;
    quantize_completed_group(b,slot,b.shared_intermediate,gs,group);
}
template<int Capacity,bool LowParallelism>
__global__ void __launch_bounds__(128,Capacity==2 && LowParallelism?4:Capacity<=2?8:Capacity<=4?6:4) short_shared_down(MfeFfnBatch b,int group_start) {
    const int group=group_start+int(blockIdx.y),count=short_shared_count<Capacity>(b,group);
    if(count<1 || count>Capacity)return;
    const int first=(int(blockIdx.x)*4+int(threadIdx.y))*2;
    if(first>=b.output_width)return;
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
    #pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
        const int slot=short_shared_slot<Capacity>(b,group,k);
        inputs[k]=b.hidden_quantized+int64_t(slot)*b.quantized_stride;
        scales[k]=b.hidden_scales+int64_t(slot)*b.scale_stride;
    }
    __half values[Capacity][4]{};
    short_shared_project<Capacity>(b.shared[2],first,inputs,scales,count,values,b.warp_multi_sum);
    #pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
        #pragma unroll
        for(int row=0;row<2;++row)if(threadIdx.x==row && first+row<b.output_width) {
            const int slot=short_shared_slot<Capacity>(b,group,k);
            b.resident_pairs[int64_t(slot)*b.output_width+first+row]=values[k][row];
        }
    }
}
}
}
