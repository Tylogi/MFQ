#pragma once

namespace mfq::cuda {
namespace {
// Only cold GU routes enter this grid. Four workers cover the common small
// miss count; a worker loops over further routes without changing their slots.
template<uint32_t Formats>
__global__ void __launch_bounds__(512,2) mfe_ffn_gu_transfer_routes_kernel(
        MfeFfnBatch b,const int32_t* routes) {
    const int count=routes[0],worker=int(blockIdx.y);
    if(worker>=count)return;
    const int lane=threadIdx.x,warp=threadIdx.y;
    const int width=b.intermediate,base=int(blockIdx.x)*8;
    if(base>=width)return;
    __shared__ MfePackedProjection descriptors[3];
    __shared__ int locals[3],present[3];
    __shared__ __half projected[8][2];
    for(int item=worker;item<count;item+=int(gridDim.y)) {
        const int route=routes[1+item],expert=b.ids[route],slot=route;
        if(!warp && lane<3) {
            int local;
            const auto* source=projection(b,lane,expert,false,local,true);
            locals[lane]=local;present[lane]=source!=nullptr;
            if(source)descriptors[lane]=*source;
        }
        __syncthreads();
        const int role=warp>=4,first=base+(warp%4)*2;
        const auto* d=present[role]?descriptors+role:nullptr;
        __half values[2]={__float2half(0.f),__float2half(0.f)};
        if(first<width)project_pair<true,Formats,true,false>(d,locals[role],first,
            d?static_cast<const int8_t*>(d->fields[7]):nullptr,
            d?static_cast<const float*>(d->fields[8]):nullptr,values,b.warp_multi_sum,b.input);
        if(!lane){projected[warp][0]=values[0];projected[warp][1]=values[1];}
        __syncthreads();
        if(!warp) {
            const int end=min(base+8,width);
            const auto* table=b.sigmoid_table+(b.routes>1?65536:0);
            if(base+lane<end) {
                const int pair=lane/2,row=lane&1;
                const auto gate=projected[pair][row],up=projected[pair+4][row];
                b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=
                    mfe_ffn_swiglu(b,gate,up,table);
                __threadfence();
            }
            __syncwarp();
            const auto* down=present[2]?descriptors+2:nullptr;
            if(down && locals[2]>=0 && locals[2]<down->local_experts) {
                const int gs=down->group_size;
                for(int group=base/gs;group<=(end-1)/gs;++group) {
                    const int written=min(end,(group+1)*gs)-max(base,group*gs);
                    int complete=0;
                    if(!lane)complete=atomicAdd(b.completion+int64_t(slot)*b.scale_stride+group,written)+written
                        ==min(gs,width-group*gs);
                    complete=__shfl_sync(0xffffffffu,complete,0);
                    if(complete)quantize_completed_group(b,slot,width,gs,group);
                }
            }
        }
        if(item+int(gridDim.y)>=count)return;
        __syncthreads();
    }
}
}
}
