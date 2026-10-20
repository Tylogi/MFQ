#include "ffn_device.cuh"
#include "ffn_shared.h"
#include "ffn_prepare.h"
#include "ffn_token_groups.h"
#include "nint_tokens.cuh"
#include "ffn_shared_short.cuh"
#include "ffn_gu_group_quant.cuh"

namespace mfq::cuda {
namespace {
__global__ void __launch_bounds__(64) shared_prepare_input(const __half* input,int width,
        const MfeInputQuantization* inputs,int count) {
    const int lane=threadIdx.x&31;
    int entry=int(blockIdx.x)*2+(int(threadIdx.x)>>5);
    const MfeInputQuantization* d=nullptr;
    for(int i=0;i<count;++i) {
        if(entry<inputs[i].groups){d=inputs+i;break;}entry-=inputs[i].groups;
    }
    if(!d)return;
    float values[2]{};
#pragma unroll
    for(int part=0;part<2;++part) {
        const int local=lane+part*32,col=entry*d->group_size+local;
        if(local<d->group_size && col<width)values[part]=__half2float(input[col]);
    }
    float maximum=fmaxf(fabsf(values[0]),fabsf(values[1]));
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    const float scale=maximum>0.f?maximum/127.f:1.f;
    if(!lane)d->scales[entry]=scale;
#pragma unroll
    for(int part=0;part<2;++part)if(lane+part*32<d->group_size) {
        const int col=entry*d->group_size+lane+part*32;
        d->values[col]=col<width?int8_t(max(-127,min(127,int(roundf(values[part]/scale))))):int8_t(0);
    }
}
template<int Capacity>
__device__ __forceinline__ int shared_count(const MfeFfnBatch& b,int group) {
    if constexpr(Capacity==1)if(b.tokens==1)return 1;
    return b.token_group_counts[b.tokens*b.routes+group];
}
template<int Capacity>
__device__ __forceinline__ int shared_slot(const MfeFfnBatch& b,int group,int k) {
    if constexpr(Capacity==1)if(b.tokens==1)return b.routes;
    return b.token_group_slots[(b.tokens*b.routes+group)*8+k];
}
template<int Capacity>
__device__ __forceinline__ void shared_project(const MfePackedProjection& d,int first,
        const int8_t* const* inputs,const float* const* scales,int count,
        __half values[Capacity][4],bool multi_sum) {
    if constexpr(Capacity>1)if(!multi_sum) {
        if(d.group_size==24){nint_token_pair<24,Capacity>(d,0,first,inputs,scales,count,values,0);return;}
        if(d.group_size==28){nint_token_pair<28,Capacity>(d,0,first,inputs,scales,count,values,0);return;}
        if(d.group_size==32){nint_token_pair<32,Capacity>(d,0,first,inputs,scales,count,values,0);return;}
    }
    for(int k=0;k<count;++k)project_pair<true,0,true>(&d,0,first,inputs[k],scales[k],values[k],multi_sum);
}
template<int Capacity>
__global__ void __launch_bounds__(256,Capacity<=2?4:Capacity<=4?3:2) shared_gate_up(MfeFfnBatch b,int group_start) {
    const int group=group_start+int(blockIdx.y),count=shared_count<Capacity>(b,group);
    const int lane=threadIdx.x,warp=threadIdx.y;
    const int base=int(blockIdx.x)*8,first=base+(warp%4)*2;
    if(base>=b.shared_intermediate)return;
    const auto& d=b.shared[warp>=4];
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
    for(int k=0;k<count;++k) {
        const int token=shared_slot<Capacity>(b,group,k)/(b.routes+1);
        inputs[k]=static_cast<const int8_t*>(d.fields[7])+int64_t(token)*d.groups*d.group_size;
        scales[k]=static_cast<const float*>(d.fields[8])+int64_t(token)*d.groups;
    }
    __half values[Capacity][4]{};
    if(first<b.shared_intermediate)shared_project<Capacity>(d,first,inputs,scales,count,values,b.warp_multi_sum);
    __shared__ __half projected[8][Capacity][2];
    if(!lane)for(int k=0;k<count;++k) {
        projected[warp][k][0]=values[k][0];projected[warp][k][1]=values[k][1];
    }
    __syncthreads();
    if(warp)return;
    if(lane<8 && base+lane<b.shared_intermediate)for(int k=0;k<count;++k) {
        const int pair=lane/2,row=lane&1;
        const auto gate=projected[pair][k][row],up=projected[pair+4][k][row];
        const int slot=shared_slot<Capacity>(b,group,k);
        b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=
            mfe_ffn_swiglu(b,gate,up,b.sigmoid_table);
    }
}
__global__ void __launch_bounds__(128) shared_quantize(MfeFfnBatch b) {
    const int group=int(blockIdx.x)*4+int(threadIdx.y);
    const int gs=b.shared[2].group_size;
    if(group*gs>=b.shared_intermediate)return;
    const int slot=int(blockIdx.y)*(b.routes+1)+b.routes;
    quantize_completed_group(b,slot,b.shared_intermediate,gs,group);
}
template<int Capacity>
__device__ __forceinline__ void shared_down_rows(const MfeFfnBatch& b,int group_start) {
    const int group=group_start+int(blockIdx.y),count=shared_count<Capacity>(b,group);
    const int first=(int(blockIdx.x)*4+int(threadIdx.y))*2;
    if(first>=b.output_width)return;
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
    for(int k=0;k<count;++k) {
        const int slot=shared_slot<Capacity>(b,group,k);
        inputs[k]=b.hidden_quantized+int64_t(slot)*b.quantized_stride;
        scales[k]=b.hidden_scales+int64_t(slot)*b.scale_stride;
    }
    __half values[Capacity][4]{};
    shared_project<Capacity>(b.shared[2],first,inputs,scales,count,values,b.warp_multi_sum);
    if(threadIdx.x<2 && first+int(threadIdx.x)<b.output_width)for(int k=0;k<count;++k) {
        const int slot=shared_slot<Capacity>(b,group,k);
        b.resident_pairs[int64_t(slot)*b.output_width+first+int(threadIdx.x)]=values[k][threadIdx.x];
    }
}
template<int Capacity>
__global__ void __launch_bounds__(128,Capacity<=2?8:Capacity<=4?6:4) shared_down(MfeFfnBatch b,int group_start) {
    shared_down_rows<Capacity>(b,group_start);
}
__device__ __forceinline__ void shared_scalar_gate(const MfePrepareSharedGate& gate) {
    const int lane=threadIdx.x,warp=threadIdx.y;
    __shared__ float terms[64*12];
    __shared__ float activation_scales[64];
    for(int group=warp;group<gate.groups;group+=4) {
        if(lane<12) {
            const int q=reinterpret_cast<const int*>(gate.activation+group*48)[lane];
            const auto* weights=gate.packed+(uint64_t(gate.offset[0])>>3)+group*48;
            const int code=reinterpret_cast<const int*>(weights)[lane];
            const int sum=__dp4a(0x01010101,q,0);
            const int dot=__dp4a(code^int(0x80808080u),q,0)+128*sum;
            terms[group*12+lane]=gate.outer_scale[0]*float(gate.scales[group])*float(dot)-
                gate.outer_minimum[0]*float(gate.minima[group])*float(sum);
        }
        if(!lane)activation_scales[group]=gate.activation_scales[group];
    }
    __syncthreads();
    if(warp)return;
    // Match the original ordered readout, including the Half rounding before
    // sigmoid. The other three warps can start their shared Down rows now.
    const int relative=lane/12,chunk=lane%12;
    float result=0.f;
    if(relative<2)for(int group=relative;group<gate.groups;group+=2)
        result+=activation_scales[group]*terms[group*12+chunk];
#pragma unroll
    for(int step=16;step;step>>=1)result+=__shfl_xor_sync(0xffffffffu,result,step);
    if(!lane)gate.output[0]=__float2half_rn(1.f/(1.f+expf(-__half2float(__float2half_rn(result)))));
}
__global__ void __launch_bounds__(128,8) shared_down_with_gate(MfeFfnBatch b,MfePrepareSharedGate gate) {
    if(!blockIdx.x)shared_scalar_gate(gate);
    shared_down_rows<1>(b,0);
}
template<int Capacity>
void launch_shared(const MfeFfnBatch& b,int first,int groups,bool down,cudaStream_t stream) {
    if(down)shared_down<Capacity><<<dim3((b.output_width+7)/8,groups),dim3(32,4),0,stream>>>(b,first);
    else shared_gate_up<Capacity><<<dim3((b.shared_intermediate+7)/8,groups),dim3(32,8),0,stream>>>(b,first);
}
void launch_groups(const MfeFfnBatch& b,int count,int first,int groups,bool down,cudaStream_t stream) {
    if(count<=1)launch_shared<1>(b,first,groups,down,stream);
    else if(count<=2)launch_shared<2>(b,first,groups,down,stream);
    else if(count<=4)launch_shared<4>(b,first,groups,down,stream);
    else launch_shared<8>(b,first,groups,down,stream);
}
void launch_single_shared_gate_up(const MfeFfnBatch& b,cudaStream_t stream) {
    // Reuse the existing whole-quantization-group kernel. A one-route view
    // keeps the shared expert in its original output slot through rebasing.
    auto shared=b;
    shared.routes=0;
    shared.hidden+=int64_t(b.routes)*b.hidden_stride;
    shared.hidden_quantized+=int64_t(b.routes)*b.quantized_stride;
    shared.hidden_scales+=int64_t(b.routes)*b.scale_stride;
    mfe_ffn_gu_group_quant_kernel<0,8><<<dim3(b.scale_stride,1),dim3(32,8),0,stream>>>(shared,0);
}
}
void mfe_ffn_shared_prepare_input(const __half* input,int width,const MfeInputQuantization* views,
        int count,int groups,cudaStream_t stream) {
    shared_prepare_input<<<(groups+1)/2,64,0,stream>>>(input,width,views,count);
}
void mfe_ffn_shared_resident(const MfeFfnBatch& b,cudaStream_t stream) {
    if(b.tokens==1) {
        launch_single_shared_gate_up(b,stream);
        launch_shared<1>(b,0,1,true,stream);
        return;
    }
    const int capacity=b.token_group_capacity>kMfeSharedGroupEncoding?
        b.token_group_capacity-kMfeSharedGroupEncoding:b.token_group_capacity;
    const int full=b.tokens/capacity,tail=b.tokens%capacity;
    for(bool down:{false,true}) {
        if(full)launch_groups(b,capacity,0,full,down,stream);
        if(tail)launch_groups(b,tail,full,1,down,stream);
        if(!down)shared_quantize<<<dim3((b.scale_stride+3)/4,b.tokens),dim3(32,4),0,stream>>>(b);
    }
}
void mfe_ffn_shared_resident_with_gate(const MfeFfnBatch& b,const MfePrepareSharedGate& gate,cudaStream_t stream) {
    launch_single_shared_gate_up(b,stream);
    shared_down_with_gate<<<(b.output_width+7)/8,dim3(32,4),0,stream>>>(b,gate);
}
}

namespace mfq::cuda {
namespace {
template<int Capacity,bool LowParallelism=false>
void launch_short_shared(const MfeFfnBatch& b,int first,int groups,bool down,cudaStream_t stream) {
    if(down)short_shared_down<Capacity,LowParallelism><<<dim3((b.output_width+7)/8,groups),dim3(32,4),0,stream>>>(b,first);
    else short_shared_gate_up<Capacity,LowParallelism><<<dim3((b.shared_intermediate+7)/8,groups),dim3(32,8),0,stream>>>(b,first);
}
void launch_short_groups(const MfeFfnBatch& b,int count,int first,int groups,bool down,bool low,cudaStream_t stream) {
    if(count<=1)launch_short_shared<1>(b,first,groups,down,stream);
    else if(count<=2) {
        if(low)launch_short_shared<2,true>(b,first,groups,down,stream);
        else launch_short_shared<2,false>(b,first,groups,down,stream);
    }else launch_short_shared<4>(b,first,groups,down,stream);
}
}
bool mfe_ffn_shared_short_resident(const MfeFfnBatch& b,cudaStream_t stream) {
    const int capacity=b.token_group_capacity>kMfeSharedGroupEncoding?
        b.token_group_capacity-kMfeSharedGroupEncoding:b.token_group_capacity;
    if(b.tokens<=1 || !b.token_group_counts || !b.token_group_slots ||
            b.shared_dense_math || b.warp_multi_sum || capacity<1 || capacity>4)return false;
    const int full=b.tokens/capacity,tail=b.tokens%capacity;
    // Large token groups keep the tighter register budget so resident routed
    // CTAs can run alongside them. Small groups can trade spare occupancy
    // for fewer spills without changing the expert grouping or arithmetic.
    const bool low=capacity==2 && full<=2;
    for(bool down:{false,true}) {
        if(full)launch_short_groups(b,capacity,0,full,down,low,stream);
        if(tail)launch_short_groups(b,tail,full,1,down,false,stream);
        if(!down)short_shared_quantize<<<dim3((b.scale_stride+3)/4,b.tokens),dim3(32,4),0,stream>>>(b);
    }
    return true;
}
}
