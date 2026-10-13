// The single-entry fallback shares the wide decoder; grouped weight chunks
// retain their specialized, inlined dot loops below.
#define MFQ_MFE_PROJECT_INLINE __noinline__
#include "native/env.h"
#include "mfe_ffn_device.cuh"
#include "mfe_ffn_token_groups.h"
#include "mfe_ffn_compact.h"
#include "nvq_dense_window32.cuh"

namespace mfq::cuda {
namespace {
// Expert grouping follows Strata's s2_expert_grouped.cu (MIT; see
// csrc/third_party/strata.LICENSE). MFQ keeps its own packed formats,
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

template<int Format,int Lanes,int Capacity>
__device__ __forceinline__ void nvq_token_dot(const MfePackedProjection& d,int local,int first,
        const int8_t* const* inputs,const float* const* scales,int count,__half result[Capacity][4],bool narrow) {
    constexpr bool d4=Format==10 || Format==11 || Format==12 || Format==15;
    constexpr int bits=packed_nvq::format_index_bits(Format);
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    const int neuron=local*d.output_rows+first+row;
    const int row_bits=(d4?d.nsign*(2*bits+7):d.nvec*(bits+7))+d.groups*4;
    const int expert_bytes=narrow?d.output_rows*row_bits/8:0;
    const auto* local_stream=stream+int64_t(local)*expert_bytes;
    const float anchor=first+row<d.output_rows?anchors[neuron]:0.f;
    float partial[Capacity]{};
    for(int group=lane;group<d.groups;group+=Lanes) {
        int2 weight[3]{};float weight_scale=0.f;
        if(first+row<d.output_rows) {
            if constexpr(d4) {
                packed_nvq::DenseD4Group<bits> packed;
                if(narrow) {
                    const auto words=packed_nvq::load_dense_window32<4+3*(2*bits+7)>(local_stream,expert_bytes,
                        (first+row)*row_bits+group*(4+3*(2*bits+7)));
                    packed={words.x,words.y,words.z};
                }else packed=packed_nvq::load_dense_d4_group<bits>(stream,d.sizes[0],neuron,group,d.nsign,d.groups);
                const auto state=packed.state();const auto* bank=packed_nvq::active_codebook<Format>(codebook,state);
                weight_scale=packed_nvq::format_scale<Format>(anchor,state,codebook);
                weight[0]=packed_nvq::decode_dense_d4_vector<0>(packed,bank);
                weight[1]=packed_nvq::decode_dense_d4_vector<1>(packed,bank);
                weight[2]=packed_nvq::decode_dense_d4_vector<2>(packed,bank);
            }else {
                packed_nvq::DenseE8Group<bits> packed;
                if(narrow) {
                    const auto words=packed_nvq::load_dense_window32<4+3*(bits+7)>(local_stream,expert_bytes,
                        (first+row)*row_bits+group*(4+3*(bits+7)));
                    packed={uint64_t(words.x)|(uint64_t(words.y)<<32)};
                }else packed=packed_nvq::load_dense_e8_group<bits>(stream,d.sizes[0],neuron,group,d.nvec,d.groups);
                const auto state=packed.state();const auto* bank=packed_nvq::active_codebook<Format>(codebook,state);
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

template<int GroupSize,int Capacity>
__device__ __forceinline__ void nint_token_pair(const MfePackedProjection& d,int local,int first,
        const int8_t* const* inputs,const float* const* scales,int count,__half result[Capacity][4],int destination) {
    constexpr int chunks=GroupSize/4;
    const int lane=threadIdx.x;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0])+size_t(local)*d.q_expert_stride;
    const auto* bits=static_cast<const uint8_t*>(d.fields[1]);
    const auto* offsets=static_cast<const int64_t*>(d.fields[2]);
    const auto* subscale=static_cast<const uint8_t*>(d.fields[3]);
    const auto* submin=static_cast<const uint8_t*>(d.fields[4]);
    const auto* outer_scale=static_cast<const float*>(d.fields[5]);
    const auto* outer_min=static_cast<const float*>(d.fields[6]);
    float partial[Capacity][2]{};
    int widths[2]{};uint64_t starts[2]{};float factors[2]{},minima[2]{};
#pragma unroll
    for(int r=0;r<2;++r)if(first+r<d.output_rows) {
        const int neuron=local*d.output_rows+first+r;
        widths[r]=bits[neuron];starts[r]=uint64_t(offsets[neuron]);
        factors[r]=outer_scale[neuron];minima[r]=outer_min[neuron];
    }
    for(int group=lane;group<d.groups;group+=32) {
#pragma unroll
        for(int r=0;r<2;++r)if(first+r<d.output_rows) {
            int weight[chunks];
#pragma unroll
            for(int chunk=0;chunk<chunks;chunk+=2) {
                const auto offset=starts[r]+uint64_t(group*GroupSize+chunk*4)*widths[r];
                if(chunk+1<chunks) {
                    const auto packed=packed_nint::unpack_nint_codes8_packed(stream,offset,widths[r]);
                    const uint32_t low=uint32_t(packed),second=uint32_t(packed>>(4*widths[r]));
                    const uint32_t a=__byte_perm(low,low>>(2*widths[r]),0x5410);
                    const uint32_t b=__byte_perm(second,second>>(2*widths[r]),0x5410);
                    const uint32_t mask=((1u<<widths[r])-1u)*0x01010101u;
                    weight[chunk]=int(__byte_perm(a,a>>widths[r],0x6240)&mask);
                    weight[chunk+1]=int(__byte_perm(b,b>>widths[r],0x6240)&mask);
                }else weight[chunk]=packed_nint::unpack_nint_codes4(stream,offset,widths[r]);
            }
            const auto meta=int64_t(local*d.output_rows+first+r)*d.groups+group;
            const float scale=float(subscale[meta]),minimum=float(submin[meta]);
#pragma unroll
            for(int k=0;k<Capacity;++k)if(k<count) {
                int dot=0,sum=0;
#pragma unroll
                for(int chunk=0;chunk<chunks;++chunk) {
                    const int activation=packed_nint::load_i8x4(inputs[k]+group*GroupSize+chunk*4);
                    const int s=__dp4a(0x01010101,activation,0);sum+=s;
                    dot+=widths[r]==8?__dp4a(weight[chunk]^int(0x80808080u),activation,0)+128*s:
                        __dp4a(weight[chunk],activation,0);
                }
                partial[k][r]+=scales[k][group]*(factors[r]*scale*float(dot)-minima[r]*minimum*float(sum));
            }
        }
    }
#pragma unroll
    for(int k=0;k<Capacity;++k)if(k<count) {
#pragma unroll
        for(int r=0;r<2;++r) {
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)partial[k][r]+=__shfl_xor_sync(0xffffffffu,partial[k][r],offset);
            result[k][destination+r]=__float2half(partial[k][r]);
        }
    }
}

template<int Capacity,int Rows>
__device__ __forceinline__ void token_project(const MfePackedProjection* d,int local,int first,
        const int8_t* const* inputs,const float* const* scales,int count,
        __half result[Capacity][4],bool multi_sum,bool canonical,bool narrow) {
    if(d && local>=0 && local<d->local_experts) {
        if(d->family==2) {
            constexpr int lanes=Rows==4?8:16;
            if(!canonical || Rows==2) {
                if(d->format==1){nvq1_token_dot<1,lanes,Capacity>(*d,local,first,inputs,scales,count,result);return;}
                if(d->format==8){nvq1_token_dot<8,lanes,Capacity>(*d,local,first,inputs,scales,count,result);return;}
#define MFQ_TOKEN_NVQ(F,DENSE) case F: if(d->DENSE) {nvq_token_dot<F,lanes,Capacity>(*d,local,first,inputs,scales,count,result,narrow);return;}break
                switch(d->format) {
                    MFQ_TOKEN_NVQ(5,e8_dense_groups);MFQ_TOKEN_NVQ(13,e8_dense_groups);MFQ_TOKEN_NVQ(14,e8_dense_groups);
                    MFQ_TOKEN_NVQ(10,d4_dense_groups);MFQ_TOKEN_NVQ(11,d4_dense_groups);
                    MFQ_TOKEN_NVQ(12,d4_dense_groups);MFQ_TOKEN_NVQ(15,d4_dense_groups);
                }
#undef MFQ_TOKEN_NVQ
            }
        }else if(d->family==1 && !d->dense_reduction_warps && !d->nint_late_bits && !multi_sum) {
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

template<int Capacity>
__device__ __forceinline__ bool accepts_count(int count) {
    if constexpr(Capacity==1)return count==1;
    return count>Capacity/2 && count<=Capacity;
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

template<int Capacity>
__global__ void __launch_bounds__(256,Capacity<=2?4:Capacity<=4?3:2) token_gate_up(MfeFfnBatch b,int partition) {
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slots[0]%all_routes;
    const bool shared=route==b.routes;
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
    token_project<Capacity,2>(d,locals[p],first,inputs,scales,count,values,b.warp_multi_sum,false,b.compact_nvq);
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
            const auto activated=__float2half_rn(__half2float(gate)*__half2float(table[__half_as_ushort(gate)]));
            b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=__float2half_rn(__half2float(activated)*__half2float(up));
            __threadfence();
        }
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
template<int Capacity>
__global__ void __launch_bounds__(512,Capacity<=2?2:1) token_gate_up_aligned(MfeFfnBatch b,int partition) {
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slots[0]%all_routes;
    const bool shared=route==b.routes;
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
    if(warp<gs/2) {
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
            token_project<Capacity,2>(d,locals[p],first,inputs,scales,count,values,b.warp_multi_sum,false,b.compact_nvq);
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(k<count) {
                if(p==0) {
                    gate[k][0]=values[k][0];gate[k][1]=values[k][1];
                }else if(lane<2 && first+lane<width) {
                    const auto g=gate[k][lane],u=values[k][lane];
                    const auto activated=__float2half_rn(__half2float(g)*__half2float(table[__half_as_ushort(g)]));
                    b.hidden[int64_t(slots[k])*b.hidden_stride+first+lane]=__float2half_rn(__half2float(activated)*__half2float(u));
                }
            }
        }
    }
    __syncthreads();
    if(warp==0) {
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(k<count)quantize_completed_group(b,slots[k],width,gs,blockIdx.x);
    }
}

template<int Capacity,int Rows>
__global__ void __launch_bounds__(128,Capacity<=2?8:Capacity<=4?6:4) token_down(MfeFfnBatch b,int partition) {
    const int group=token_group_index<Capacity>(b,blockIdx.y);
    if(group<0)return;
    const int count=b.token_group_counts[group];
    if(!accepts_count<Capacity>(count))return;
    const int* slots=b.token_group_slots+group*8;
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slots[0]%all_routes;
    const bool shared=route==b.routes;
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
    token_project<Capacity,Rows>(d,local,first,inputs,scales,count,values,b.warp_multi_sum,b.shared_dense_math,b.compact_nvq);
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
    const auto* setting=mfq::cuda::env_lookup("MFQ_MFE_TOKEN_GROUPS");
    const auto* grouped=mfq::cuda::env_lookup("MFQ_MFE_GROUP_DOT");
    return (!setting || setting[0]!='0') && (!grouped || grouped[0]!='0') &&
        b.tokens>=2 && b.tokens<=32 && b.token_group_counts && b.token_group_slots && b.resident_pairs;
}
template<int Capacity>
void launch_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(b.plan_ready && b.resident_plan_overlap) {
        const auto* early=mfq::cuda::env_lookup("MFQ_MFE_EARLY_GU");
        if(!early || early[0]!='0')partition|=8;
    }
    const int groups=mfe_token_group_bound(b,Capacity);
    if(groups<=0)return;
    if(b.token_group_aligned_gu)
        token_gate_up_aligned<Capacity><<<dim3(b.scale_stride,groups),dim3(32,16),0,stream>>>(b,partition);
    else token_gate_up<Capacity><<<dim3((width+7)/8,groups),dim3(32,8),0,stream>>>(b,partition);
}
template<int Capacity>
void launch_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    const int groups=mfe_token_group_bound(b,Capacity);
    if(groups<=0)return;
    if(b.shared && !b.shared_dense_math)
        token_down<Capacity,4><<<dim3((b.output_width+15)/16,groups),dim3(32,4),0,stream>>>(b,partition);
    else token_down<Capacity,2><<<dim3((b.output_width+7)/8,groups),dim3(32,4),0,stream>>>(b,partition);
}
}
bool mfe_ffn_token_group_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!enabled(b))return false;
    if(b.token_group_aligned_gu || !mfe_ffn_compact_token_single_gate_up(b,partition,width,stream))launch_gate_up<1>(b,partition,width,stream);
    launch_gate_up<2>(b,partition,width,stream);
    if(b.token_group_capacity>=4)launch_gate_up<4>(b,partition,width,stream);
    if(b.token_group_capacity>=8)launch_gate_up<8>(b,partition,width,stream);
    return true;
}
bool mfe_ffn_token_group_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!enabled(b))return false;
    if(!mfe_ffn_compact_token_single_down(b,partition,stream))launch_down<1>(b,partition,stream);
    launch_down<2>(b,partition,stream);
    if(b.token_group_capacity>=4)launch_down<4>(b,partition,stream);
    if(b.token_group_capacity>=8)launch_down<8>(b,partition,stream);
    if(partition!=1)token_reduce<<<dim3((b.output_width+255)/256,b.tokens),256,0,stream>>>(b);
    return true;
}
}
