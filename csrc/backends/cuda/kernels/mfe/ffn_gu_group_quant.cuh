#pragma once

namespace mfq::cuda { namespace {
template<uint32_t Formats,int Warps,bool Worklist=false>
__global__ void __launch_bounds__(32*Warps,Worklist?3:Warps<=16?32/Warps:1) mfe_ffn_gu_group_quant_kernel(
        MfeFfnBatch b,int partition,const int32_t* worklist=nullptr) {
    const bool early_gu=(partition&8)!=0;partition&=3;
    // The transfer packet lists only cold GU routes and keeps their original
    // slots. Each CTA owns one item, avoiding a loop's extra live registers.
    int slot=int(blockIdx.y);
    if constexpr(Worklist) {
        if(slot>=worklist[0])return;
        slot=worklist[1+slot];
    }
    const int all_routes=b.routes+int(b.shared!=nullptr);
    const int token=slot/all_routes,route=slot%all_routes;
    const bool shared=route==b.routes;
    const int width=shared?b.shared_intermediate:b.intermediate;
    if(width<=0)return;
    const int expert=shared?0:b.ids[token*b.routes+route];
    if(!shared && unsigned(expert)>=unsigned(b.experts))return;
    const int lane=threadIdx.x,warp=threadIdx.y;
    __shared__ int block_kind,locals[3],present[3];
    __shared__ MfePackedProjection descriptors[3];
    if(!warp) {
        int kind=0;
        if(!lane) {
            const int tagged=shared?1:aborted(b)?-1:b.kinds?b.kinds[expert]:1;
            kind=tagged>0?(tagged&3):tagged;
            if(!shared && early_gu && kind==2 && (tagged&4)) {
                if(partition==1)kind=1;else if(partition==2)kind=-1;
            }
        }
        kind=__shfl_sync(0xffffffffu,kind,0);
        if(!lane)block_kind=kind;
        if(kind>0 && (!partition || kind==partition)) {
            uint64_t address=0;
            if(lane<3) {
                int local;const auto* d=projection(b,lane,expert,shared,local,b.plan_ready && kind==2);
                locals[lane]=local;present[lane]=d!=nullptr;address=reinterpret_cast<uint64_t>(d);
            }
            // Broadcast the three sources through registers. Consecutive lanes
            // copy consecutive descriptor words; the existing CTA barrier below
            // publishes their fields and the original per-projection metadata.
            static_assert(sizeof(MfePackedProjection)%sizeof(uint64_t)==0);
            constexpr int words=sizeof(MfePackedProjection)/sizeof(uint64_t);
#pragma unroll
            for(int role=0;role<3;++role) {
                const auto source=__shfl_sync(0xffffffffu,address,role);
#pragma unroll
                for(int chunk=0;chunk<(words+31)/32;++chunk) {
                    const int word=chunk*32+lane;
                    if(source && word<words)
                        reinterpret_cast<uint64_t*>(descriptors+role)[word]=
                            __ldg(reinterpret_cast<const uint64_t*>(source)+word);
                }
            }
        }
    }
    __syncthreads();
    if(block_kind<=0 || (partition && block_kind!=partition) || !present[2])return;
    const int gs=descriptors[2].group_size,group=blockIdx.x,base=group*gs;
    if(base>=width)return;
    __shared__ __half projected[2][64];
    // Visit Gate row pairs before Up while keeping every warp occupied. This
    // reduces concurrent codebook traffic without another CTA barrier.
    const int pairs=(gs+1)/2;
    for(int item=warp;item<pairs*2;item+=Warps) {
        const int role=item>=pairs,offset=(item-role*pairs)*2;
        const auto* d=present[role]?descriptors+role:nullptr;
        const int first=base+offset;
        __half values[2]{};
        if(first<width)project_pair<true,Formats,true>(d,locals[role],first,
            d?static_cast<const int8_t*>(d->fields[7])+int64_t(token)*d->groups*d->group_size:nullptr,
            d?static_cast<const float*>(d->fields[8])+int64_t(token)*d->groups:nullptr,values,b.warp_multi_sum,
            b.input+int64_t(token)*b.input_width);
#pragma unroll
        for(int r=0;r<2;++r)if(lane==r && offset+r<gs)projected[role][offset+r]=values[r];
    }
    __syncthreads();
    const auto* table=b.sigmoid_table+(!shared && b.tokens*b.routes>1?65536:0);
    if(warp)return;
    float values[2]{};
#pragma unroll
    for(int half=0;half<2;++half) {
        const int offset=lane+half*32;
        if(offset<gs && base+offset<width) {
            const auto gate=projected[0][offset],up=projected[1][offset];
            const auto hidden=mfe_ffn_swiglu(b,gate,up,table);
            b.hidden[int64_t(slot)*b.hidden_stride+base+offset]=hidden;
            values[half]=__half2float(hidden);
        }
    }
    // The same warp retains both rounded Half values for the original max
    // reduction and quantization. No global hidden reload or CTA join is
    // needed after the final projected-value barrier above.
    float maximum=fmaxf(fabsf(values[0]),fabsf(values[1]));
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    const float scale=maximum>0.0f?maximum/127.0f:1.0f;
    if(lane==0)b.hidden_scales[int64_t(slot)*b.scale_stride+group]=scale;
#pragma unroll
    for(int half=0;half<2;++half)if(lane+half*32<gs) {
        const int column=base+lane+half*32;
        int code=0;
        if(column<width)code=max(-127,min(127,int(roundf(values[half]/scale))));
        b.hidden_quantized[int64_t(slot)*b.quantized_stride+column]=int8_t(code);
    }
}
}
}
