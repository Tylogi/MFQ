// The single-entry fallback shares the wide decoder; grouped weight chunks
// retain their specialized, inlined dot loops below.
#define MFQ_MFE_PROJECT_INLINE __noinline__
#include "mfe_ffn_device.cuh"
#include "mfe_nint_tokens.cuh"
#include "mfe_ffn_token_groups.h"
#include "mfe_ffn_compact.h"
#include "nvq_dense_window32.cuh"
#include "nvq_jsc_bank_map.cuh"
#include <atomic>
#include <cstdio>

namespace mfq::cuda {
namespace {
// Expert grouping follows Strata's s2_expert_grouped.cu (MIT; see
// cpp_runtime/third_party/strata.LICENSE). MFQ keeps its own packed formats,
// lane reductions, rounded SwiGLU, transfer partitions and route sum order.
// Decoded chunks live only in registers for this group of routed tokens.
template<int Format,int Lanes,int Capacity>
__device__ __forceinline__ void nvq1_token_dot(const MfePackedProjection& d,int local,int first,
        const int8_t* const* inputs,const float* const* scales,int count,__half result[Capacity][4]) {
    constexpr int bits=Format==8?9:11;
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* indices=static_cast<const uint8_t*>(d.fields[0]);
    const auto* signs=static_cast<const uint8_t*>(d.fields[1]);
    const auto* states=static_cast<const uint8_t*>(d.fields[2]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    const bool integer=d.fields[5]!=nullptr;
    const int neuron=local*d.output_rows+first+row;
    float partial[Capacity]{};
    for(int group=lane;group<d.groups;group+=Lanes) {
        int2 weight[3]{};float weight_scale=0.f;int delta=1;
        if(first+row<d.output_rows) {
            uint64_t record=0;uint32_t sub=0;int negative=0;
            if(d.fields[6] || !d.sizes[1]) {
                record=d.fields[6]?nvq1_complete_record<Format>(d,neuron,group):
                    packed_nvq::load_nvq1_record64<Format>(indices,d.sizes[0],neuron,group,d.groups);
                sub=uint32_t(record>>(3*bits))&(Format==8?15u:7u);
                negative=int((record>>(Format==8?31:36))&1u);
            }else {
                const auto linear=int64_t(neuron)*d.groups+group;
                sub=packed_nvq::load_nvq_state<Format>(indices,d.sizes[0],states,d.sizes[2],linear,d.sub_bits,d.groups);
                negative=packed_nvq::load_packed_bits(signs,linear,1,d.sizes[1]);
                record=packed_nvq::load_packed_group_window<bits*3>(indices,(int64_t(neuron)*d.nvec+group*3)*bits,d.sizes[0]);
            }
            delta=negative?-1:1;weight_scale=anchors[neuron]*float(sub);
            const auto* bank=integer?static_cast<const int8_t*>(d.fields[5])+negative*(Format==8?packed_nvq::kNvq1SBankBytes:2048*8):
                codebook+(Format==8?negative*packed_nvq::kNvq1SBankBytes:0);
#pragma unroll
            for(int segment=0;segment<3;++segment)
                weight[segment]=reinterpret_cast<const int2*>(bank)[uint32_t(record>>(segment*bits))&((1u<<bits)-1u)];
        }
#pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count && first+row<d.output_rows) {
            int dot=0,sum=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                const auto activation=*reinterpret_cast<const int2*>(inputs[k]+group*24+segment*8);
                dot=__dp4a(weight[segment].y,activation.y,__dp4a(weight[segment].x,activation.x,dot));
                if(!integer)sum=__dp4a(0x01010101,activation.y,__dp4a(0x01010101,activation.x,sum));
            }
            float value=float(dot);
            if(integer)value*=Format==8?0.03125f:0.125f;
            else value+=(Format==8?0.15625f:0.125f)*float(delta*sum);
            partial[k]=fmaf(weight_scale*scales[k][group],value,partial[k]);
        }
    }
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
#pragma unroll
        for(int offset=Lanes/2;offset>0;offset>>=1)partial[k]+=__shfl_xor_sync(0xffffffffu,partial[k],offset,Lanes);
#pragma unroll
        for(int r=0;r<32/Lanes;++r)result[k][r]=__float2half(__shfl_sync(0xffffffffu,partial[k],r*Lanes));
    }
}

template<int Format,int Lanes,int Capacity,bool Compact=false>
__device__ __forceinline__ void nvq_token_dot(const MfePackedProjection& d,int local,int first,
        const int8_t* const* inputs,const float* const* scales,int count,__half result[Capacity][4],bool narrow) {
    constexpr bool d4=Format==10 || Format==11 || Format==12 || Format==15;
    constexpr int bits=packed_nvq::format_index_bits(Format);
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0]);
    const auto* auxiliary=static_cast<const uint8_t*>(d.fields[1]);
    const auto* states=static_cast<const uint8_t*>(d.fields[2]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    const int neuron=local*d.output_rows+first+row;
    const int row_bits=(d4?d.nsign*(2*bits+7):d.nvec*(bits+7))+d.groups*4;
    const int expert_bytes=(Compact || narrow)?d.output_rows*row_bits/8:0;
    const auto* local_stream=stream+int64_t(local)*expert_bytes;
    const uint32_t bank_map=d.nvq_bank_map;
    const bool analytic_scale=Capacity<=2 && packed_nvq::jsc_d4_analytic_scales<Format>(codebook);
    const float anchor=first+row<d.output_rows?anchors[neuron]:0.f;
    float partial[Capacity]{};
    for(int group=lane;group<d.groups;group+=Lanes) {
        int2 weight[3]{};float weight_scale=0.f;
        if(first+row<d.output_rows) {
            if(!Compact && !(d4?d.d4_dense_groups:d.e8_dense_groups)) {
                // The source layout keeps indices, parity and joint states in
                // separate packed streams. Decode the same register group as
                // the scalar path, then reuse it for every routed token.
                const auto state=packed_nvq::load_nvq_state<Format>(stream,d.sizes[0],states,d.sizes[2],
                    int64_t(neuron)*d.groups+group,d.sub_bits,d.groups);
                const auto* bank=packed_nvq::active_codebook<Format>(codebook,state);
                const auto signs=packed_nvq::load_packed_group_window<21>(auxiliary,
                    (int64_t(neuron)*d.nsign+group*3)*7,d.sizes[1]);
                constexpr int vectors=d4?6:3;
                const auto indices=packed_nvq::load_packed_group_window<bits*vectors>(stream,
                    (int64_t(neuron)*d.nvec+group*vectors)*bits,d.sizes[0]);
                weight_scale=packed_nvq::format_scale<Format>(anchor,state,codebook);
#pragma unroll
                for(int segment=0;segment<3;++segment)
                    weight[segment]=packed_nvq::load_nvq_group_vec8<Format,true>(stream,d.sizes[0],
                        auxiliary,d.sizes[1],codebook,bank,neuron,group,segment,d.groups,d.nvec,
                        d.nsign,d.sign_mode,state,uint32_t(signs),indices,nullptr,0).values;
            }else if constexpr(d4) {
                packed_nvq::DenseD4Group<bits> packed;
                if(Compact || narrow) {
                    const auto words=packed_nvq::load_dense_window32<4+3*(2*bits+7)>(local_stream,expert_bytes,
                        (first+row)*row_bits+group*(4+3*(2*bits+7)));
                    packed={words.x,words.y,words.z};
                }else packed=packed_nvq::load_dense_d4_group<bits>(stream,d.sizes[0],neuron,group,d.nsign,d.groups);
                const auto state=packed.state();const int8_t* bank;
                if constexpr(Format==packed_nvq::kNvq3Jsc || Format==packed_nvq::kNvq3Jsc512 ||
                             Format==packed_nvq::kNvq3JscL) {
                    bank=bank_map?packed_nvq::packed_jsc_codebook<Format>(codebook,bank_map,state)
                                 :packed_nvq::active_codebook<Format>(codebook,state);
                }else bank=packed_nvq::active_codebook<Format>(codebook,state);
                weight_scale=analytic_scale?anchor*float((state>>1)+1):packed_nvq::format_scale<Format>(anchor,state,codebook);
                weight[0]=packed_nvq::decode_dense_d4_vector<0>(packed,bank);
                weight[1]=packed_nvq::decode_dense_d4_vector<1>(packed,bank);
                weight[2]=packed_nvq::decode_dense_d4_vector<2>(packed,bank);
            }else {
                packed_nvq::DenseE8Group<bits> packed;
                if(Compact || narrow) {
                    const auto words=packed_nvq::load_dense_window32<4+3*(bits+7)>(local_stream,expert_bytes,
                        (first+row)*row_bits+group*(4+3*(bits+7)));
                    packed={uint64_t(words.x)|(uint64_t(words.y)<<32)};
                }else packed=packed_nvq::load_dense_e8_group<bits>(stream,d.sizes[0],neuron,group,d.nvec,d.groups);
                const auto state=packed.state();const int8_t* bank;
                if constexpr(Format==packed_nvq::kNvq2Jsc || Format==packed_nvq::kNvq2JscL ||
                             Format==packed_nvq::kNvq2JscXL) {
                    bank=bank_map?packed_nvq::packed_jsc_codebook<Format>(codebook,bank_map,state)
                                 :packed_nvq::active_codebook<Format>(codebook,state);
                }else bank=packed_nvq::active_codebook<Format>(codebook,state);
                weight_scale=packed_nvq::format_scale<Format>(anchor,state,codebook);
                weight[0]=packed_nvq::decode_dense_e8_vector<0>(packed,bank,d.sign_mode);
                weight[1]=packed_nvq::decode_dense_e8_vector<1>(packed,bank,d.sign_mode);
                weight[2]=packed_nvq::decode_dense_e8_vector<2>(packed,bank,d.sign_mode);
            }
        }
#pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count && first+row<d.output_rows) {
            int dot=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                const auto activation=*reinterpret_cast<const int2*>(inputs[k]+group*24+segment*8);
                dot=__dp4a(weight[segment].y,activation.y,__dp4a(weight[segment].x,activation.x,dot));
            }
            partial[k]=fmaf(weight_scale*scales[k][group],float(dot),partial[k]);
        }
    }
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
#pragma unroll
        for(int offset=Lanes/2;offset>0;offset>>=1)
            partial[k]+=__shfl_xor_sync(0xffffffffu,partial[k],offset,Lanes);
#pragma unroll
        for(int r=0;r<32/Lanes;++r)result[k][r]=__float2half(__shfl_sync(0xffffffffu,partial[k],r*Lanes));
    }
}


template<int Capacity>
__device__ __forceinline__ void shared_nint_project(const MfePackedProjection& d,int first,
        const int8_t* const* inputs,const float* const* scales,int count,
        __half result[Capacity][4],bool multi_sum) {
    if(!d.dense_reduction_warps && !d.nint_late_bits && !multi_sum) {
        if(d.group_size==24){nint_token_pair<24,Capacity,1>(d,0,first,inputs,scales,count,result,0);return;}
        if(d.group_size==28){nint_token_pair<28,Capacity,1>(d,0,first,inputs,scales,count,result,0);return;}
        if(d.group_size==32){nint_token_pair<32,Capacity,1>(d,0,first,inputs,scales,count,result,0);return;}
    }
    for(int k=0;k<count;++k)
        project_pair<true,0>(&d,0,first,inputs[k],scales[k],result[k],multi_sum);
}

template<int Capacity,int Rows,bool Compact=false>
__device__ __forceinline__ void token_project(const MfePackedProjection* d,int local,int first,
        const int8_t* const* inputs,const float* const* scales,int count,
        __half result[Capacity][4],bool multi_sum,bool canonical,bool narrow,
        const __half* original,int original_stride,const int* slots,int slot_divisor) {
    if(d && local>=0 && local<d->local_experts) {
        if(d->family==3) {
            if constexpr(Capacity==1) {
                const auto* source=original?original+int64_t(slots[0]/slot_divisor)*original_stride:nullptr;
#pragma unroll
                for(int pair=0;pair<Rows/2;++pair)
                    mfe_mxfp4_pair(*d,local,first+pair*2,source,result[0]+pair*2);
                return;
            }
            const __half* sources[Capacity]{};
#pragma unroll
            for(int k=0;k<Capacity;++k)if(k<count) {
                sources[k]=original?original+int64_t(slots[k]/slot_divisor)*original_stride:nullptr;
            }
            mfe_mxfp4_tokens<Capacity,Rows>(*d,local,first,sources,count,result);
            return;
        }
        if(d->family==2) {
            constexpr int lanes=Rows==4?8:16;
            if(Compact || !canonical || Rows==2) {
                if(d->format==1){nvq1_token_dot<1,lanes,Capacity>(*d,local,first,inputs,scales,count,result);return;}
                if(d->format==8){nvq1_token_dot<8,lanes,Capacity>(*d,local,first,inputs,scales,count,result);return;}
#define MFQ_TOKEN_NVQ(F,DENSE) case F: if(Capacity>1 || Compact || d->DENSE) {nvq_token_dot<F,lanes,Capacity,Compact>(*d,local,first,inputs,scales,count,result,narrow);return;}break
                switch(d->format) {
                    MFQ_TOKEN_NVQ(5,e8_dense_groups);MFQ_TOKEN_NVQ(13,e8_dense_groups);MFQ_TOKEN_NVQ(14,e8_dense_groups);
                    MFQ_TOKEN_NVQ(10,d4_dense_groups);MFQ_TOKEN_NVQ(11,d4_dense_groups);
                    MFQ_TOKEN_NVQ(12,d4_dense_groups);MFQ_TOKEN_NVQ(15,d4_dense_groups);
                }
#undef MFQ_TOKEN_NVQ
            }
        }else if(d->family==1 && (Compact || (!d->dense_reduction_warps && !d->nint_late_bits && !multi_sum))) {
#pragma unroll
            for(int pair=0;pair<Rows/2;++pair) {
                if(d->group_size==24)nint_token_pair<24,Capacity>(*d,local,first+pair*2,inputs,scales,count,result,pair*2);
                else if(d->group_size==28)nint_token_pair<28,Capacity>(*d,local,first+pair*2,inputs,scales,count,result,pair*2);
                else if(d->group_size==32)nint_token_pair<32,Capacity>(*d,local,first+pair*2,inputs,scales,count,result,pair*2);
                else break;
            }
            if(d->group_size==24 || d->group_size==28 || d->group_size==32)return;
        }
    }
    if constexpr(!Compact) {
    // Unusual arithmetic contracts retain their old decoder. Keep its address
    // argument separate from the register-resident multi-token results.
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
        __half fallback[4];
        if constexpr(Rows==4)project_quad<true>(d,local,first,inputs[k],scales[k],fallback,multi_sum,canonical);
        else project_pair<true>(d,local,first,inputs[k],scales[k],fallback,multi_sum);
#pragma unroll
        for(int r=0;r<Rows;++r)result[k][r]=fallback[r];
    }
    }
}

template<int Capacity>
__device__ __forceinline__ bool accepts_count(int count) {
    if constexpr(Capacity==1)return count==1;
    return count>Capacity/2 && count<=Capacity;
}

__global__ void __launch_bounds__(64) token_shared_plan(MfeFfnBatch b,int capacity) {
    const int lane=threadIdx.x,entries=b.tokens*b.routes;
    if(b.token_group_buckets && lane<4)b.token_group_buckets[lane*(b.token_groups+1)]=0;
    for(int group=entries+lane;group<b.token_groups;group+=64) {
        const int first=(group-entries)*capacity;
        const int count=max(0,min(capacity,b.tokens-first));
        b.token_group_counts[group]=count;
        for(int k=0;k<count;++k)b.token_group_slots[group*8+k]=
            (first+k)*(b.routes+1)+b.routes;
    }
    __syncthreads();
    if(!b.token_group_buckets)return;
    for(int group=lane;group<b.token_groups;group+=64) {
        const int count=b.token_group_counts[group];
        if(!count)continue;
        const int bucket=count==1?0:count<=2?1:count<=4?2:3;
        auto* plan=b.token_group_buckets+bucket*(b.token_groups+1);
        const int index=atomicAdd(plan,1);plan[index+1]=group;
    }
}

template<int Capacity>
int token_launch_bound(const MfeFfnBatch& b) {
    if(b.token_group_capacity<=kMfeSharedGroupEncoding || !b.token_group_buckets)
        return mfe_token_group_bound(b,Capacity);
    // Larger routed groups cannot exist in this plan. Bound their grids by
    // the shared groups instead of the total number of routed token slots.
    int groups=Capacity<=2?b.tokens*b.routes/Capacity:0;
    const int shared_capacity=b.token_group_capacity-kMfeSharedGroupEncoding;
    for(int first=0;first<b.tokens;first+=shared_capacity) {
        const int count=std::min(shared_capacity,b.tokens-first);
        groups+=Capacity==1?count==1:count>Capacity/2 && count<=Capacity;
    }
    return groups;
}

__device__ __forceinline__ int group_kind(const MfeFfnBatch& b,int expert,bool shared,int partition,bool& early) {
    early=false;
    const int tagged=shared?1:aborted(b)?-1:b.kinds?b.kinds[expert]:1;
    int kind=tagged>0?(tagged&3):tagged;
    if((partition&8) && !shared && kind==2 && (tagged&4)) {
        if((partition&3)==1){kind=1;early=true;}
        else if((partition&3)==2)kind=-1;
    }
    return kind;
}

// Gate/Up and Down execute in one stream. A completed producer kernel is
// sufficient publication for the hidden tensor; no per-neuron fence or
// cross-CTA completion counter is needed by this separate quantizer.
template<int Warps>
__global__ void __launch_bounds__(32*Warps) hidden_quantize(MfeFfnBatch b,int partition) {
    const int slot=int(blockIdx.y),all_routes=b.routes+int(b.shared!=nullptr);
    const int token=slot/all_routes,route=slot%all_routes;
    const bool shared=route==b.routes;
    if(shared && b.shared_intermediate==0)return;
    const int expert=shared?0:b.ids[token*b.routes+route];
    if(!shared && unsigned(expert)>=unsigned(b.experts))return;
    bool early;const int kind=group_kind(b,expert,shared,partition,early);
    if(kind<=0 || ((partition&3) && kind!=(partition&3)))return;
    int local;const auto* d=projection(b,2,expert,shared,local,b.plan_ready && kind==2);
    if(!d || (!early && (local<0 || local>=d->local_experts)))return;
    const int width=shared?b.shared_intermediate:b.intermediate;
    const int group=int(blockIdx.x)*Warps+int(threadIdx.y);
    if(group*d->group_size>=width)return;
    quantize_completed_group(b,slot,width,d->group_size,group);
}

template<int Capacity,bool Compact=false>
__global__ void __launch_bounds__(256,Capacity<=2?4:Capacity<=4?3:2) token_gate_up(MfeFfnBatch b,int partition) {
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slots[0]%all_routes;
    const bool shared=route==b.routes;
    if(shared && b.shared_intermediate==0)return;
    const int expert=shared?0:b.ids[(slots[0]/all_routes)*b.routes+route];
    bool early;const int kind=group_kind(b,expert,shared,partition,early);
    if(kind<=0 || ((partition&3) && kind!=(partition&3)))return;
    const int lane=threadIdx.x,warp=threadIdx.y,team=blockDim.y/2;
    const int width=shared?b.shared_intermediate:b.intermediate;
    const int base=blockIdx.x*blockDim.y,first=base+(warp%team)*2;
    if(base>=width)return;
    __shared__ MfePackedProjection descriptors[3];
    __shared__ int locals[3],present[3];
    if(warp==0 && lane<3) {
        int local;const auto* d=projection(b,lane,expert,shared,local,b.plan_ready && kind==2);
        locals[lane]=local;present[lane]=d!=nullptr;if(d)descriptors[lane]=*d;
    }
    __syncthreads();
    const int p=warp>=team?1:0;const auto* d=present[p]?descriptors+p:nullptr;
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count && d) {
        const int token=slots[k]/all_routes;
        inputs[k]=static_cast<const int8_t*>(d->fields[7])+int64_t(token)*d->groups*d->group_size;
        scales[k]=static_cast<const float*>(d->fields[8])+int64_t(token)*d->groups;
    }
    __half values[Capacity][4]{};
    token_project<Capacity,2,Compact>(d,locals[p],first,inputs,scales,count,values,b.warp_multi_sum,false,b.compact_nvq,
        b.input,b.input_width,slots,all_routes);
    __shared__ __half projected[8][Capacity][2];
    if(lane==0) {
#pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count){projected[warp][k][0]=values[k][0];projected[warp][k][1]=values[k][1];}
    }
    __syncthreads();
    if(warp!=0)return;
    const int end=min(base+int(blockDim.y),width);
    const auto* table=b.sigmoid_table+(!shared && b.tokens*b.routes>1?65536:0);
    for(int k=0;k<count;++k) {
        const int slot=slots[k];
        if(base+lane<end) {
            const auto gate=projected[lane/2][k][lane&1],up=projected[lane/2+team][k][lane&1];
            b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=mfe_ffn_swiglu(b,gate,up,table);
            if(b.completion)__threadfence();
        }
        if(!b.completion)continue;
        __syncwarp();
        const auto* down=present[2]?descriptors+2:nullptr;
        if(!down || (!early && (locals[2]<0 || locals[2]>=down->local_experts)))continue;
        const int gs=down->group_size;
        for(int g=base/gs;g<=(end-1)/gs;++g) {
            const int written=min(end,(g+1)*gs)-max(base,g*gs);int complete=0;
            if(lane==0)complete=atomicAdd(b.completion+int64_t(slot)*b.scale_stride+g,written)+written==min(gs,width-g*gs);
            complete=__shfl_sync(0xffffffffu,complete,0);
            if(complete)quantize_completed_group(b,slot,width,gs,g);
        }
    }
}

// One CTA owns an entire Down input-quantization group. Gate and Up retain
// their original pair decoders and rounded SwiGLU, while the CTA barrier
// replaces cross-CTA completion atomics and per-neuron global fences.
template<int Capacity,bool Compact>
__device__ __forceinline__ void token_aligned_pair(const MfeFfnBatch& b,
        const MfePackedProjection* descriptors,const int* locals,const int* present,
        const int* slots,int count,int all_routes,bool shared,int first,int width) {
    const int lane=threadIdx.x;
    __half gate[Capacity][2]{};
    const auto* table=b.sigmoid_table+(!shared && b.tokens*b.routes>1?65536:0);
    // Reuse one decoder instruction body for Gate and Up. Unrolling the
    // format switch twice adds instruction-cache pressure without sharing
    // their different weights, and keeps both result arrays live.
    #pragma unroll 1
    for(int p=0;p<2;++p) {
        const auto* d=present[p]?descriptors+p:nullptr;
        const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count && d) {
            const int token=slots[k]/all_routes;
            inputs[k]=static_cast<const int8_t*>(d->fields[7])+int64_t(token)*d->groups*d->group_size;
            scales[k]=static_cast<const float*>(d->fields[8])+int64_t(token)*d->groups;
        }
        __half values[Capacity][4]{};
        token_project<Capacity,2,Compact>(d,locals[p],first,inputs,scales,count,values,b.warp_multi_sum,false,b.compact_nvq,
            b.input,b.input_width,slots,all_routes);
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count) {
            if(p==0) {
                gate[k][0]=values[k][0];gate[k][1]=values[k][1];
            }else if(lane<2 && first+lane<width) {
                const auto g=gate[k][lane],u=values[k][lane];
                b.hidden[int64_t(slots[k])*b.hidden_stride+first+lane]=mfe_ffn_swiglu(b,g,u,table);
            }
        }
    }
}

template<int Capacity,bool Compact=false,int Warps=16>
__global__ void __launch_bounds__(32*Warps,Warps==12?(Capacity<=2?3:2):(Capacity<=2?2:1)) token_gate_up_aligned(MfeFfnBatch b,int partition) {
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slots[0]%all_routes;
    const bool shared=route==b.routes;
    if(shared && b.shared_intermediate==0)return;
    const int expert=shared?0:b.ids[(slots[0]/all_routes)*b.routes+route];
    bool early;const int kind=group_kind(b,expert,shared,partition,early);
    if(kind<=0 || ((partition&3) && kind!=(partition&3)))return;
    const int lane=threadIdx.x,warp=threadIdx.y;
    const int width=shared?b.shared_intermediate:b.intermediate;
    __shared__ MfePackedProjection descriptors[3];
    __shared__ int locals[3],present[3];
    if(warp==0 && lane<3) {
        int local;const auto* d=projection(b,lane,expert,shared,local,b.plan_ready && kind==2);
        locals[lane]=local;present[lane]=d!=nullptr;if(d)descriptors[lane]=*d;
    }
    __syncthreads();
    const auto* down=present[2]?descriptors+2:nullptr;
    if(!down || (!early && (locals[2]<0 || locals[2]>=down->local_experts)))return;
    const int gs=down->group_size,base=int(blockIdx.x)*gs,first=base+warp*2;
    if(base>=width)return;
    if constexpr(Warps==16) {
        if(warp<gs/2)token_aligned_pair<Capacity,Compact>(b,descriptors,locals,present,
            slots,count,all_routes,shared,first,width);
    }else {
        // GS28/32 need a second column pair in some warps. All active warps
        // retain the original reduction order for each output.
        for(int column=first;column<base+gs;column+=2*Warps)
            token_aligned_pair<Capacity,Compact>(b,descriptors,locals,present,
                slots,count,all_routes,shared,column,width);
    }
    __syncthreads();
    if(warp==0) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count)quantize_completed_group(b,slots[k],width,gs,blockIdx.x);
    }
}

// Shared experts always use NINT. One output per warp halves live result
// state and supplies enough CTAs for a single expert with four/eight tokens.
template<int Capacity>
__global__ void __launch_bounds__(256,4) shared_token_gate_up(MfeFfnBatch b,int partition) {
    if((partition&3)==2)return;
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int lane=threadIdx.x,warp=threadIdx.y,team=4;
    const int base=int(blockIdx.x)*team,first=base+warp%team,p=warp/team;
    if(base>=b.shared_intermediate)return;
    // An ungated shared expert has no gate descriptor; both teams read Up.
    const auto& d=b.shared[b.shared_activation==1?1:p];
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
    #pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
        const int token=slots[k]/(b.routes+1);
        inputs[k]=static_cast<const int8_t*>(d.fields[7])+int64_t(token)*d.groups*d.group_size;
        scales[k]=static_cast<const float*>(d.fields[8])+int64_t(token)*d.groups;
    }
    __half values[Capacity][4]{};
    shared_nint_project<Capacity>(d,first,inputs,scales,count,values,b.warp_multi_sum);
    __shared__ __half projected[8][Capacity];
    if(lane==0)for(int k=0;k<count;++k)projected[warp][k]=values[k][0];
    __syncthreads();
    if(warp!=0)return;
    const int end=min(base+team,b.shared_intermediate),gs=b.shared[2].group_size;
    for(int k=0;k<count;++k) {
        const int slot=slots[k];
        if(base+lane<end) {
            const auto gate=projected[lane][k],up=projected[lane+team][k];
            b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=mfe_ffn_swiglu(b,gate,up,b.sigmoid_table,true);
            if(b.completion)__threadfence();
        }
        if(!b.completion)continue;
        __syncwarp();
        for(int g=base/gs;g<=(end-1)/gs;++g) {
            const int written=min(end,(g+1)*gs)-max(base,g*gs);int complete=0;
            if(lane==0)complete=atomicAdd(b.completion+int64_t(slot)*b.scale_stride+g,written)+written==
                min(gs,b.shared_intermediate-g*gs);
            complete=__shfl_sync(0xffffffffu,complete,0);
            if(complete)quantize_completed_group(b,slot,b.shared_intermediate,gs,g);
        }
    }
}

template<int Capacity>
__global__ void __launch_bounds__(128,8) shared_token_down(MfeFfnBatch b,int partition) {
    if(partition && partition!=1)return;
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int first=int(blockIdx.x)*4+threadIdx.y;
    if(first>=b.output_width)return;
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
    #pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
        inputs[k]=b.hidden_quantized+int64_t(slots[k])*b.quantized_stride;
        scales[k]=b.hidden_scales+int64_t(slots[k])*b.scale_stride;
    }
    __half values[Capacity][4]{};
    shared_nint_project<Capacity>(b.shared[2],first,inputs,scales,count,values,b.warp_multi_sum);
    if(threadIdx.x==0)for(int k=0;k<count;++k)
        b.resident_pairs[int64_t(slots[k])*b.output_width+first]=values[k][0];
}

template<int Capacity,int Rows,bool Compact=false>
__global__ void __launch_bounds__(128,Capacity<=2?8:Capacity<=4?6:4) token_down(MfeFfnBatch b,int partition) {
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slots[0]%all_routes;
    const bool shared=route==b.routes;
    if(shared && b.shared_intermediate==0)return;
    const int expert=shared?0:b.ids[(slots[0]/all_routes)*b.routes+route];
    const int kind=shared?1:route_kind(b,expert);
    if(kind<=0 || (partition && kind!=partition))return;
    const int first=(blockIdx.x*blockDim.y+threadIdx.y)*Rows;
    if(first>=b.output_width)return;
    int local;const auto* d=projection(b,2,expert,shared,local,b.plan_ready && kind==2);
    const int8_t* inputs[Capacity]{};const float* scales[Capacity]{};
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
        inputs[k]=b.hidden_quantized+int64_t(slots[k])*b.quantized_stride;
        scales[k]=b.hidden_scales+int64_t(slots[k])*b.scale_stride;
    }
    __half values[Capacity][4]{};
    token_project<Capacity,Rows,Compact>(d,local,first,inputs,scales,count,values,b.warp_multi_sum,b.shared_dense_math,b.compact_nvq,
        b.hidden,b.hidden_stride,slots,1);
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
#pragma unroll
        for(int r=0;r<Rows;++r)if(threadIdx.x==r && first+r<b.output_width)
            b.resident_pairs[int64_t(slots[k])*b.output_width+first+r]=values[k][r];
    }
}

__global__ void token_reduce(MfeFfnBatch b) {
    const int column=blockIdx.x*blockDim.x+threadIdx.x,token=blockIdx.y;
    if(column>=b.output_width)return;
    const int all_routes=b.routes+int(b.shared!=nullptr);
    float sum=0.f;
    for(int route=0;route<b.routes;++route) {
        const int pair=token*b.routes+route,expert=b.ids[pair];
        __half value=__float2half(0.f);
        if(unsigned(expert)<unsigned(b.experts) && !aborted(b)) {
            const int kind=route_kind(b,expert);
            if(kind==0)value=b.cpu_pairs[int64_t(pair)*b.output_width+column];
            else if(kind>0)value=b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+column];
        }
        if(b.output_pairs)static_cast<__half*>(b.output)[int64_t(pair)*b.output_width+column]=value;
        else sum+=b.route_weights[pair]*__half2float(value);
    }
    if(b.output_pairs)return;
    const float routed=__half2float(__float2half(sum));
    const float shared_gate=b.shared?(b.shared_product_half?__half2float(static_cast<const __half*>(b.shared_gate)[token]):
        b.shared_gate_bfloat?__bfloat162float(static_cast<const __nv_bfloat16*>(b.shared_gate)[token]):
        static_cast<const float*>(b.shared_gate)[token]):0.f;
    float contribution=b.shared?__fmul_rn(__half2float(b.resident_pairs[(int64_t(token)*all_routes+b.routes)*b.output_width+column]),shared_gate):0.f;
    if(b.shared_product_half)contribution=__half2float(__float2half(contribution));
    const float output=b.shared?__fadd_rn(routed,contribution):routed;
    const auto index=int64_t(token)*b.output_width+column;
    if(b.output_float)static_cast<float*>(b.output)[index]=output;
    else static_cast<__half*>(b.output)[index]=__float2half(output);
}

bool enabled(const MfeFfnBatch& b) {
    const auto* setting=std::getenv("MFQ_MFE_TOKEN_GROUPS");
    const auto* grouped=std::getenv("MFQ_MFE_GROUP_DOT");
    return (!setting || setting[0]!='0') && (!grouped || grouped[0]!='0') &&
        b.tokens>=2 && b.tokens<=32 && b.token_group_counts && b.token_group_slots && b.resident_pairs;
}
template<int Capacity,bool Compact=false>
void launch_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(b.plan_ready && b.resident_plan_overlap) {
        const auto* early=std::getenv("MFQ_MFE_EARLY_GU");
        if(!early || early[0]!='0')partition|=8;
    }
    const int groups=token_launch_bound<Capacity>(b);
    if(groups<=0)return;
    if constexpr(Capacity>=4)if(b.token_group_capacity>kMfeSharedGroupEncoding) {
        if((partition&3)!=2 && b.shared_intermediate>0)shared_token_gate_up<Capacity><<<
            dim3((b.shared_intermediate+3)/4,groups),dim3(32,8),0,stream>>>(b,partition);
        return;
    }
    if(b.token_group_aligned_gu) {
        const auto* option=std::getenv("MFQ_MFE_ALIGNED_GU_WARPS");
        if(option && std::atoi(option)==12)
            token_gate_up_aligned<Capacity,Compact,12><<<dim3(b.scale_stride,groups),dim3(32,12),0,stream>>>(b,partition);
        else token_gate_up_aligned<Capacity,Compact,16><<<dim3(b.scale_stride,groups),dim3(32,16),0,stream>>>(b,partition);
    }else token_gate_up<Capacity,Compact><<<dim3((width+7)/8,groups),dim3(32,8),0,stream>>>(b,partition);
}
template<int Capacity,bool Compact=false>
void launch_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    const int groups=token_launch_bound<Capacity>(b);
    if(groups<=0)return;
    if constexpr(Capacity>=4)if(b.token_group_capacity>kMfeSharedGroupEncoding) {
        if(b.shared_intermediate>0 && (!partition || partition==1))shared_token_down<Capacity><<<
            dim3((b.output_width+3)/4,groups),dim3(32,4),0,stream>>>(b,partition);
        return;
    }
    if(b.shared && !b.shared_dense_math)
        token_down<Capacity,4,Compact><<<dim3((b.output_width+15)/16,groups),dim3(32,4),0,stream>>>(b,partition);
    else token_down<Capacity,2,Compact><<<dim3((b.output_width+7)/8,groups),dim3(32,4),0,stream>>>(b,partition);
}
}
void mfe_ffn_token_group_shared_plan(const MfeFfnBatch& b,cudaStream_t stream) {
    token_shared_plan<<<1,64,0,stream>>>(b,b.token_group_capacity-kMfeSharedGroupEncoding);
}
void mfe_ffn_quantize_hidden(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    const auto* option=std::getenv("MFQ_MFE_HIDDEN_QUANT_WARPS");
    const int warps=option?std::atoi(option):1;
    const auto launch=[&](auto count) {
        constexpr int value=decltype(count)::value;
        hidden_quantize<value><<<dim3((b.scale_stride+value-1)/value,b.tokens*(b.routes+int(b.shared!=nullptr))),
            dim3(32,value),0,stream>>>(b,partition);
    };
    if(warps==4)launch(std::integral_constant<int,4>{});
    else if(warps==8)launch(std::integral_constant<int,8>{});
    else launch(std::integral_constant<int,1>{});
}
bool mfe_ffn_token_group_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!enabled(b))return false;
    const auto* compact=std::getenv("MFQ_MFE_TOKEN_COMPACT");
    if(b.token_group_aligned_gu) {
        const auto* aligned=std::getenv("MFQ_MFE_ALIGNED_COMPACT");
        if(b.tokens<=8 && b.token_group_compact && (!compact || compact[0]!='0') && (!aligned || aligned[0]!='0')) {
            launch_gate_up<1,true>(b,partition,width,stream);
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "mfe_aligned_compact_dispatch scope=eligible_single_group enabled=1 tokens=%d capacity=%d mask=%u\n",
                b.tokens,b.token_group_capacity,b.nvq_format_mask);
        }else launch_gate_up<1>(b,partition,width,stream);
    }else if(!mfe_ffn_compact_token_single_gate_up(b,partition,width,stream))
        launch_gate_up<1>(b,partition,width,stream);
    if(b.tokens<=8 && b.token_group_compact && (!compact || compact[0]!='0')) {
        launch_gate_up<2,true>(b,partition,width,stream);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_token_compact_dispatch scope=first_pair_gu enabled=1 tokens=%d partition=%d mask=%u\n",
            b.tokens,partition,b.nvq_format_mask);
    }else launch_gate_up<2>(b,partition,width,stream);
    bool wide_compact=false;
    if(b.token_group_capacity>=4 && b.token_group_capacity<=8 && b.tokens<=8 && b.token_group_compact && (!compact || compact[0]!='0')) {
        const auto* wide=std::getenv("MFQ_MFE_TOKEN_WIDE_COMPACT");
        wide_compact=!wide || wide[0]!='0';
    }
    if(wide_compact) {
        static std::atomic<unsigned> reported{0};const unsigned bit=b.token_group_capacity==8?2:1;
        if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "mfe_token_wide_compact_dispatch scope=eligible_routed_gu enabled=1 tokens=%d capacity=%d mask=%u\n",
            b.tokens,b.token_group_capacity,b.nvq_format_mask);
    }
    if(b.token_group_capacity>=4) {
        if(wide_compact)
            launch_gate_up<4,true>(b,partition,width,stream);
        else launch_gate_up<4>(b,partition,width,stream);
    }
    if(b.token_group_capacity>=8) {
        if(wide_compact)
            launch_gate_up<8,true>(b,partition,width,stream);
        else launch_gate_up<8>(b,partition,width,stream);
    }
    return true;
}
bool mfe_ffn_token_group_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!enabled(b))return false;
    if(!mfe_ffn_compact_token_single_down(b,partition,stream))launch_down<1>(b,partition,stream);
    const auto* compact=std::getenv("MFQ_MFE_TOKEN_COMPACT");
    if(b.tokens<=8 && b.token_group_compact && (!compact || compact[0]!='0')) {
        launch_down<2,true>(b,partition,stream);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_token_compact_dispatch scope=first_pair_down enabled=1 tokens=%d partition=%d mask=%u\n",
            b.tokens,partition,b.nvq_format_mask);
    }else launch_down<2>(b,partition,stream);
    bool wide_compact=false;
    if(b.token_group_capacity>=4 && b.token_group_capacity<=8 && b.tokens<=8 && b.token_group_compact && (!compact || compact[0]!='0')) {
        const auto* wide=std::getenv("MFQ_MFE_TOKEN_WIDE_COMPACT");
        wide_compact=!wide || wide[0]!='0';
    }
    if(wide_compact) {
        static std::atomic<unsigned> reported{0};const unsigned bit=b.token_group_capacity==8?2:1;
        if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "mfe_token_wide_compact_dispatch scope=eligible_routed_down enabled=1 tokens=%d capacity=%d mask=%u\n",
            b.tokens,b.token_group_capacity,b.nvq_format_mask);
    }
    if(b.token_group_capacity>=4) {
        if(wide_compact)
            launch_down<4,true>(b,partition,stream);
        else launch_down<4>(b,partition,stream);
    }
    if(b.token_group_capacity>=8) {
        if(wide_compact)
            launch_down<8,true>(b,partition,stream);
        else launch_down<8>(b,partition,stream);
    }
    if(partition!=1)token_reduce<<<dim3((b.output_width+255)/256,b.tokens),256,0,stream>>>(b);
    return true;
}
}
