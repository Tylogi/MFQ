#pragma once

namespace mfq::cuda {
namespace {
// The following hidden-quantization node owns completion and Down geometry.
// One warp reads the immutable route classification and descriptors before
// their common CTA barrier. Other warps do not duplicate those reads.
template<uint32_t Formats,bool SimpleM1,bool TokenSingles=false>
__global__ void __launch_bounds__(512,2) mfe_ffn_gu_prequantized_kernel(MfeFfnBatch b,int partition) {
    const bool shared_first=(partition&4)!=0,early_gu=(partition&8)!=0;
    partition&=3;
    const int all_routes=b.routes+int(b.shared!=nullptr);
    int token=SimpleM1?0:int(blockIdx.y);
    int tile=int(blockIdx.x),route;
    if constexpr(TokenSingles) {
        const int group=token_group_index<1>(b,blockIdx.y);
        if(group<0 || b.token_group_counts[group]!=1)return;
        const int original=b.token_group_slots[group*8];
        token=original/all_routes;route=original%all_routes;
    }else if constexpr(SimpleM1) {
        const int position=int(blockIdx.y);
        route=position+(shared_first?(position?-1:b.routes):0);
    }else {
        const int routed_tiles=(b.intermediate+7)/8;
        const int shared_tiles=b.shared?(b.shared_intermediate+7)/8:0;
        if(shared_first && b.shared) {
            if(tile<shared_tiles)route=b.routes;
            else {tile-=shared_tiles;route=tile/routed_tiles;tile%=routed_tiles;}
        }else {
            const int routed=b.routes*routed_tiles;
            if(tile>=routed){tile-=routed;route=b.routes;}
            else {route=tile/routed_tiles;tile%=routed_tiles;}
        }
    }
    const int slot=token*all_routes+route;
    const bool shared=route==b.routes;
    const int width=shared?b.shared_intermediate:b.intermediate;
    const int lane=threadIdx.x,warp=threadIdx.y,team=int(blockDim.y)/2;
    const int base=tile*8;
    if(base>=width)return;
    const int expert=shared?0:b.ids[token*b.routes+route];
    if(!shared && unsigned(expert)>=unsigned(b.experts))return;
    __shared__ int block_kind;
    __shared__ MfePackedProjection descriptors[2];
    __shared__ int locals[2],present[2];
    if(!warp) {
        int kind=0;
        if(!lane) {
            const int tagged=shared?1:aborted(b)?-1:b.kinds?b.kinds[expert]:1;
            kind=tagged>0?(tagged&3):tagged;
            if(early_gu && !shared && kind==2 && (tagged&4)) {
                if(partition==1)kind=1;
                else if(partition==2)kind=-1;
            }
        }
        kind=__shfl_sync(0xffffffffu,kind,0);
        if(!lane)block_kind=kind;
        if(kind>0 && (!partition || kind==partition) && lane<2) {
            int local;
            const auto* source=projection(b,lane,expert,shared,local,b.plan_ready && kind==2);
            locals[lane]=local;present[lane]=source!=nullptr;
            if(source)descriptors[lane]=*source;
        }
    }
    __syncthreads();
    if(block_kind<=0 || (partition && block_kind!=partition))return;
    const int role=warp>=team,first=base+(warp%team)*2;
    const auto* d=present[role]?descriptors+role:nullptr;
    __half values[2]={__float2half(0.f),__float2half(0.f)};
    if(first<width)project_pair<true,Formats,true,false>(d,locals[role],first,
        d?static_cast<const int8_t*>(d->fields[7])+int64_t(token)*d->groups*d->group_size:nullptr,
        d?static_cast<const float*>(d->fields[8])+int64_t(token)*d->groups:nullptr,values,b.warp_multi_sum,
        b.input+int64_t(token)*b.input_width);
    __shared__ __half projected[8][2];
    if(!lane){projected[warp][0]=values[0];projected[warp][1]=values[1];}
    __syncthreads();
    if(warp)return;
    const int end=min(base+int(blockDim.y),width);
    const auto* table=b.sigmoid_table+(!shared && b.tokens*b.routes>1?65536:0);
    if(base+lane<end) {
        const int pair=lane/2,row=lane&1;
        const auto gate=projected[pair][row],up=projected[pair+team][row];
        b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=
            mfe_ffn_swiglu(b,gate,up,table);
    }
}
}
}
