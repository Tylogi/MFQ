#pragma once
#ifndef MFQ_MFE_COMPACT_NVQ
#define MFQ_MFE_COMPACT_NVQ 0
#endif
#if MFQ_MFE_COMPACT_NVQ
#include "mfe_nvq_compact_dot.cuh"
#endif
#include "mfe_ffn.h"
#include <cuda_bf16.h>
#include "packed_nint.cuh"
#include "packed_nvq.cuh"
#include "mfe_e8_narrow.cuh"
#include "nvq_e8_dense_group.cuh"
#include "nvq_d4_dense_group.cuh"
#include "mfe_ffn_launch.h"
#include "warp_multi_sum.cuh"
#include "nint_route_group.cuh"
#include "mfe_nint_whole.cuh"
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <type_traits>

namespace mfq::cuda {
namespace {
template<int Format>
__device__ __forceinline__ uint64_t nvq1_complete_record(const MfePackedProjection& d,int neuron,int group) {
    const auto* records=static_cast<const uint8_t*>(d.fields[6]);
    const int64_t record=int64_t(neuron)*d.groups+group;
    if constexpr(Format==packed_nvq::kNvq1S)return reinterpret_cast<const uint32_t*>(records)[record];
    else {
        const int64_t byte=record*5,aligned=byte&~int64_t(3);
        const int shift=int(byte&3)*8;
        if(aligned+8<=d.decode_record_bytes) {
            const auto* words=reinterpret_cast<const uint32_t*>(records+aligned);
            return uint64_t(__funnelshift_r(words[0],words[1],shift)) |
                (uint64_t(shift?words[1]>>shift:words[1])<<32);
        }
        uint64_t packed=0;
#pragma unroll
        for(int i=0;i<5;++i)packed|=uint64_t(records[byte+i])<<(8*i);
        return packed;
    }
}
template<int Format,int Lanes>
__device__ __forceinline__ void nvq_dense_e8_dot(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
#if MFQ_MFE_COMPACT_NVQ
    nvq_compact32_e8_dot<Format,Lanes>(d,expert,first,input,scales,result);
#else
    constexpr int bits=packed_nvq::format_index_bits(Format);
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    float partial=0.f;
    for(int group=lane;group<d.groups;group+=Lanes) {
        const int2 activation[3]={*reinterpret_cast<const int2*>(input+group*24),
            *reinterpret_cast<const int2*>(input+group*24+8),*reinterpret_cast<const int2*>(input+group*24+16)};
        const float input_scale=scales[group];
        if(first+row<d.output_rows) {
            const int neuron=expert*d.output_rows+first+row;
            const auto packed=packed_nvq::load_dense_e8_group<bits>(stream,d.sizes[0],neuron,group,d.nvec,d.groups);
            const auto state=packed.state();
            const auto* bank=packed_nvq::active_codebook<Format>(codebook,state);
            const float weight_scale=packed_nvq::format_scale<Format>(anchors[neuron],state,codebook);
            int dot=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                int2 value{};
                if(segment==0)value=packed_nvq::decode_dense_e8_vector<0>(packed,bank,d.sign_mode);
                if(segment==1)value=packed_nvq::decode_dense_e8_vector<1>(packed,bank,d.sign_mode);
                if(segment==2)value=packed_nvq::decode_dense_e8_vector<2>(packed,bank,d.sign_mode);
                dot=__dp4a(value.y,activation[segment].y,__dp4a(value.x,activation[segment].x,dot));
            }
            partial=fmaf(weight_scale*input_scale,float(dot),partial);
        }
    }
#pragma unroll
    for(int offset=Lanes/2;offset>0;offset>>=1)partial+=__shfl_xor_sync(0xffffffffu,partial,offset,Lanes);
#pragma unroll
    for(int row=0;row<32/Lanes;++row)result[row]=__shfl_sync(0xffffffffu,partial,row*Lanes);
#endif
}

template<int Format,int Lanes>
__device__ __forceinline__ void nvq_dense_d4_dot(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
#if MFQ_MFE_COMPACT_NVQ
    nvq_compact32_d4_dot<Format,Lanes>(d,expert,first,input,scales,result);
#else
    constexpr int bits=packed_nvq::format_index_bits(Format);
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    float partial=0.f;
    for(int group=lane;group<d.groups;group+=Lanes) {
        const int2 activation[3]={*reinterpret_cast<const int2*>(input+group*24),
            *reinterpret_cast<const int2*>(input+group*24+8),*reinterpret_cast<const int2*>(input+group*24+16)};
        const float input_scale=scales[group];
        if(first+row<d.output_rows) {
            const int neuron=expert*d.output_rows+first+row;
            const auto packed=packed_nvq::load_dense_d4_group<bits>(stream,d.sizes[0],neuron,group,d.nsign,d.groups);
            const auto state=packed.state();
            const auto* bank=packed_nvq::active_codebook<Format>(codebook,state);
            const float weight_scale=packed_nvq::format_scale<Format>(anchors[neuron],state,codebook);
            int dot=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                int2 value{};
                if(segment==0)value=packed_nvq::decode_dense_d4_vector<0>(packed,bank);
                if(segment==1)value=packed_nvq::decode_dense_d4_vector<1>(packed,bank);
                if(segment==2)value=packed_nvq::decode_dense_d4_vector<2>(packed,bank);
                dot=__dp4a(value.y,activation[segment].y,__dp4a(value.x,activation[segment].x,dot));
            }
            partial=fmaf(weight_scale*input_scale,float(dot),partial);
        }
    }
#pragma unroll
    for(int offset=Lanes/2;offset>0;offset>>=1)partial+=__shfl_xor_sync(0xffffffffu,partial,offset,Lanes);
#pragma unroll
    for(int row=0;row<32/Lanes;++row)result[row]=__shfl_sync(0xffffffffu,partial,row*Lanes);
#endif
}

template<int Format,bool IntegerDelta=false,bool Records=false,bool E8Narrow=false>
__device__ __forceinline__ void nvq_group_dot(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
    if constexpr(Format==5 || Format==13 || Format==14)if(MFQ_MFE_COMPACT_NVQ || d.e8_dense_groups) {
        nvq_dense_e8_dot<Format,16>(d,expert,first,input,scales,result);return;
    }
    if constexpr(Format==10 || Format==11 || Format==12 || Format==15)if(MFQ_MFE_COMPACT_NVQ || d.d4_dense_groups) {
        nvq_dense_d4_dot<Format,16>(d,expert,first,input,scales,result);return;
    }
    const int lane=threadIdx.x&15,row=threadIdx.x>>4;
    float partial=0.0f;
    const auto* indices=static_cast<const uint8_t*>(d.fields[0]);
    const auto* auxiliary=static_cast<const uint8_t*>(d.fields[1]);
    const auto* states=static_cast<const uint8_t*>(d.fields[2]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    int narrow_index_bytes=0,narrow_sign_bytes=0;
    if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14)) {
        constexpr int bits=packed_nvq::format_index_bits(Format);
        narrow_index_bytes=int(int64_t(d.output_rows)*d.nvec*bits/8);
        narrow_sign_bytes=int(int64_t(d.output_rows)*d.nsign*7/8);
        indices+=int64_t(expert)*narrow_index_bytes;
        auxiliary+=int64_t(expert)*narrow_sign_bytes;
        states+=int64_t(expert)*d.output_rows*d.groups/2;
        anchors+=int64_t(expert)*d.output_rows;
    }
    for(int group=lane;group<d.groups;group+=16) {
        const int2 activation[3]={*reinterpret_cast<const int2*>(input+group*24),
            *reinterpret_cast<const int2*>(input+group*24+8),*reinterpret_cast<const int2*>(input+group*24+16)};
        const float input_scale=scales[group];
        if(first+row<d.output_rows) {
            const int neuron=(E8Narrow && (Format==5 || Format==13 || Format==14))?
                first+row:expert*d.output_rows+first+row;
            const int64_t state_linear=int64_t(neuron)*d.groups+group;
            uint32_t signs=0,exec96[3]{};uint64_t exec64=0;int delta=0;
            uint32_t sub;
            if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14)) {
                sub=load_e8_state32(states,neuron*d.groups+group);
            }else if constexpr(Records) {
                exec64=nvq1_complete_record<Format>(d,neuron,group);
                constexpr int bits=Format==packed_nvq::kNvq1S?9:11;
                sub=uint32_t(exec64>>(3*bits))&(Format==packed_nvq::kNvq1S?15u:7u);
            }else if constexpr(Format==packed_nvq::kNvq2JscXLGroupExec) {
                exec64=packed_nvq::load_group_exec64(indices,neuron,group,d.groups);
                sub=uint32_t(exec64>>60);
            }else if constexpr(Format==packed_nvq::kNvq3JscLGroupExec) {
                packed_nvq::load_group_exec96_words(indices,neuron,group,d.groups,exec96);
                sub=(exec96[2]>>20)&15u;
            }else {
                if constexpr(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S) {
                    if(d.sizes[1]==0) {
                        exec64=packed_nvq::load_nvq1_record64<Format>(indices,d.sizes[0],neuron,group,d.groups);
                        sub=uint32_t(exec64>>(Format==packed_nvq::kNvq1S?27:33))&(Format==packed_nvq::kNvq1S?15u:7u);
                    }else sub=packed_nvq::load_nvq_state<Format>(indices,d.sizes[0],states,d.sizes[2],state_linear,d.sub_bits,d.groups);
                }else sub=packed_nvq::load_nvq_state<Format>(indices,d.sizes[0],states,d.sizes[2],state_linear,d.sub_bits,d.groups);
            }
            const float weight_scale=(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S)
                ? anchors[neuron]*float(sub):packed_nvq::format_scale<Format>(anchors[neuron],sub,codebook);
            const int8_t* bank=packed_nvq::active_codebook<Format>(codebook,sub);
            if constexpr(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S) {
                const int negative=(Records || d.sizes[1]==0)?int((exec64>>(Format==packed_nvq::kNvq1S?31:36))&1u):
                    packed_nvq::load_packed_bits(auxiliary,state_linear,1,d.sizes[1]);
                delta=negative?-1:1;
                if constexpr(Format==packed_nvq::kNvq1S)bank=codebook+negative*packed_nvq::kNvq1SBankBytes;
                if constexpr(IntegerDelta)bank=static_cast<const int8_t*>(d.fields[5])+
                    negative*(Format==packed_nvq::kNvq1S?packed_nvq::kNvq1SBankBytes:2048*8);
            }else if constexpr(Format!=packed_nvq::kNvq2Exec && Format!=packed_nvq::kNvq2JscExec &&
                              Format!=packed_nvq::kNpq0L && Format!=packed_nvq::kNpq0S &&
                              Format!=packed_nvq::kNvq2JscXLGroupExec && Format!=packed_nvq::kNvq3JscLGroupExec) {
                if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14)) {
                    const int sign_bit=(neuron*d.nsign+group*3)*7;
                    signs=d.nvq_word_signs?load_mfe_e8_signs_words32(auxiliary,sign_bit,narrow_sign_bytes):
                        load_e8_signs32(auxiliary,sign_bit,narrow_sign_bytes);
                }else {
                const int64_t sign_bit=(int64_t(neuron)*d.nsign+group*3)*7;
                signs=d.nvq_word_signs?packed_nvq::load_packed_sign_group3_words(auxiliary,sign_bit,d.sizes[1]):
                    packed_nvq::load_packed_sign_group3(auxiliary,sign_bit,d.sizes[1]);
                }
            }
            if constexpr(!Records && Format!=packed_nvq::kNvq2Exec && Format!=packed_nvq::kNvq2JscExec &&
                Format!=packed_nvq::kNvq2JscXLGroupExec && Format!=packed_nvq::kNvq3JscLGroupExec &&
                Format!=packed_nvq::kNpq0L && Format!=packed_nvq::kNpq0S) {
                constexpr int bits=packed_nvq::format_index_bits(Format);
                constexpr int vectors=packed_nvq::is_d4_format(Format)?6:3;
                if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14))
                    exec64=load_e8_indices32<bits,false>(indices,neuron*d.nvec+group*3,narrow_index_bytes);
                else if(!((Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S) && d.sizes[1]==0))
                    exec64=packed_nvq::load_packed_group_window<bits*vectors>(indices,
                        (int64_t(neuron)*d.nvec+group*vectors)*bits,d.sizes[0]);
            }
            int dot=0,activation_sum=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                // GS24 quantizers zero-pad activation tails. The checked packed
                // index window can supply the remaining codebook vectors without
                // per-vector tail branches; their dot contributions are zero.
                packed_nvq::NvqVec8Values<Format> w{};
                if constexpr(Format!=4 && Format!=6 && Format!=7 && Format!=9 && Format<16) {
                    constexpr int index_bits=packed_nvq::format_index_bits(Format);
                    constexpr bool delta_format=Format==1 || Format==8;
                    const uint64_t indices=exec64;int2 weight{};
                    constexpr uint32_t mask=(1u<<index_bits)-1u;
                    if constexpr(packed_nvq::is_d4_format(Format)) {
                        const uint32_t i0=uint32_t(indices>>(segment*2*index_bits))&mask;
                        const uint32_t i1=uint32_t(indices>>((segment*2+1)*index_bits))&mask;
                        weight=make_int2(reinterpret_cast<const int*>(bank)[i0],reinterpret_cast<const int*>(bank)[i1]);
                    }else {
                        const uint32_t index=uint32_t(indices>>(segment*index_bits))&mask;
                        weight=reinterpret_cast<const int2*>(bank)[index];
                    }
                    if constexpr(!delta_format) {
                        const uint32_t mask7=(signs>>(segment*7))&127u;
                        int last=__popc(mask7)&1;
                        if constexpr(!packed_nvq::is_d4_format(Format))
                            if(d.sign_mode&1)last^=int((indices>>(segment*index_bits+7))&1u);
                        weight=packed_nvq::apply_sign8(weight,mask7|(uint32_t(last)<<7));
                    }
                    w={weight,delta,true};
                }else w=packed_nvq::load_nvq_group_vec8<Format, true>(indices,d.sizes[0],auxiliary,d.sizes[1],
                    codebook,bank,neuron,group,segment,d.groups,d.nvec,d.nsign,d.sign_mode,sub,signs,exec64,exec96,delta);
                dot=__dp4a(w.values.y,activation[segment].y,__dp4a(w.values.x,activation[segment].x,dot));
                if constexpr(!IntegerDelta && (Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S))
                    if(w.valid)activation_sum=__dp4a(0x01010101,activation[segment].y,__dp4a(0x01010101,activation[segment].x,activation_sum));
            }
            float value=float(dot);
            if constexpr(IntegerDelta)value*=Format==packed_nvq::kNvq1L?0.125f:0.03125f;
            else {
                if constexpr(Format==packed_nvq::kNvq1L)value+=0.125f*float(delta*activation_sum);
                if constexpr(Format==packed_nvq::kNvq1S)value+=0.15625f*float(delta*activation_sum);
            }
            partial=fmaf(weight_scale*input_scale,value,partial);
        }
    }
#pragma unroll
    for(int offset=8;offset>0;offset>>=1)partial+=__shfl_xor_sync(0xffffffffu,partial,offset,16);
    result[0]=__shfl_sync(0xffffffffu,partial,0);
    result[1]=__shfl_sync(0xffffffffu,partial,16);
}
template<int Format,bool IntegerDelta=false,bool Records=false,bool E8Narrow=false>
__device__ __forceinline__ void nvq_quad_dot(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
    if constexpr(Format==5 || Format==13 || Format==14)if(MFQ_MFE_COMPACT_NVQ || d.e8_dense_groups) {
        nvq_dense_e8_dot<Format,8>(d,expert,first,input,scales,result);return;
    }
    if constexpr(Format==10 || Format==11 || Format==12 || Format==15)if(MFQ_MFE_COMPACT_NVQ || d.d4_dense_groups) {
        nvq_dense_d4_dot<Format,8>(d,expert,first,input,scales,result);return;
    }
    const int lane=threadIdx.x&7,row=threadIdx.x>>3;
    float partial=0.0f;
    const auto* indices=static_cast<const uint8_t*>(d.fields[0]);
    const auto* auxiliary=static_cast<const uint8_t*>(d.fields[1]);
    const auto* states=static_cast<const uint8_t*>(d.fields[2]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    int narrow_index_bytes=0,narrow_sign_bytes=0;
    if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14)) {
        constexpr int bits=packed_nvq::format_index_bits(Format);
        narrow_index_bytes=int(int64_t(d.output_rows)*d.nvec*bits/8);
        narrow_sign_bytes=int(int64_t(d.output_rows)*d.nsign*7/8);
        indices+=int64_t(expert)*narrow_index_bytes;
        auxiliary+=int64_t(expert)*narrow_sign_bytes;
        states+=int64_t(expert)*d.output_rows*d.groups/2;
        anchors+=int64_t(expert)*d.output_rows;
    }
    for(int group=lane;group<d.groups;group+=8) {
        const int2 activation[3]={*reinterpret_cast<const int2*>(input+group*24),
            *reinterpret_cast<const int2*>(input+group*24+8),*reinterpret_cast<const int2*>(input+group*24+16)};
        const float input_scale=scales[group];
        if(first+row<d.output_rows) {
            const int neuron=(E8Narrow && (Format==5 || Format==13 || Format==14))?
                first+row:expert*d.output_rows+first+row;
            const int64_t state_linear=int64_t(neuron)*d.groups+group;
            uint32_t signs=0,exec96[3]{};uint64_t exec64=0;int delta=0;
            uint32_t sub;
            if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14)) {
                sub=load_e8_state32(states,neuron*d.groups+group);
            }else if constexpr(Records) {
                exec64=nvq1_complete_record<Format>(d,neuron,group);
                constexpr int bits=Format==packed_nvq::kNvq1S?9:11;
                sub=uint32_t(exec64>>(3*bits))&(Format==packed_nvq::kNvq1S?15u:7u);
            }else if constexpr(Format==packed_nvq::kNvq2JscXLGroupExec) {
                exec64=packed_nvq::load_group_exec64(indices,neuron,group,d.groups);
                sub=uint32_t(exec64>>60);
            }else if constexpr(Format==packed_nvq::kNvq3JscLGroupExec) {
                packed_nvq::load_group_exec96_words(indices,neuron,group,d.groups,exec96);
                sub=(exec96[2]>>20)&15u;
            }else {
                if constexpr(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S) {
                    if(d.sizes[1]==0) {
                        exec64=packed_nvq::load_nvq1_record64<Format>(indices,d.sizes[0],neuron,group,d.groups);
                        sub=uint32_t(exec64>>(Format==packed_nvq::kNvq1S?27:33))&(Format==packed_nvq::kNvq1S?15u:7u);
                    }else sub=packed_nvq::load_nvq_state<Format>(indices,d.sizes[0],states,d.sizes[2],state_linear,d.sub_bits,d.groups);
                }else sub=packed_nvq::load_nvq_state<Format>(indices,d.sizes[0],states,d.sizes[2],state_linear,d.sub_bits,d.groups);
            }
            const float weight_scale=(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S)
                ? anchors[neuron]*float(sub):packed_nvq::format_scale<Format>(anchors[neuron],sub,codebook);
            const int8_t* bank=packed_nvq::active_codebook<Format>(codebook,sub);
            if constexpr(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S) {
                const int negative=(Records || d.sizes[1]==0)?int((exec64>>(Format==packed_nvq::kNvq1S?31:36))&1u):
                    packed_nvq::load_packed_bits(auxiliary,state_linear,1,d.sizes[1]);
                delta=negative?-1:1;
                if constexpr(Format==packed_nvq::kNvq1S)bank=codebook+negative*packed_nvq::kNvq1SBankBytes;
                if constexpr(IntegerDelta)bank=static_cast<const int8_t*>(d.fields[5])+
                    negative*(Format==packed_nvq::kNvq1S?packed_nvq::kNvq1SBankBytes:2048*8);
            }else if constexpr(Format!=packed_nvq::kNvq2Exec && Format!=packed_nvq::kNvq2JscExec &&
                              Format!=packed_nvq::kNpq0L && Format!=packed_nvq::kNpq0S &&
                              Format!=packed_nvq::kNvq2JscXLGroupExec && Format!=packed_nvq::kNvq3JscLGroupExec) {
                if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14)) {
                    const int sign_bit=(neuron*d.nsign+group*3)*7;
                    signs=d.nvq_word_signs?load_mfe_e8_signs_words32(auxiliary,sign_bit,narrow_sign_bytes):
                        load_e8_signs32(auxiliary,sign_bit,narrow_sign_bytes);
                }else {
                const int64_t sign_bit=(int64_t(neuron)*d.nsign+group*3)*7;
                signs=d.nvq_word_signs?packed_nvq::load_packed_sign_group3_words(auxiliary,sign_bit,d.sizes[1]):
                    packed_nvq::load_packed_sign_group3(auxiliary,sign_bit,d.sizes[1]);
                }
            }
            if constexpr(!Records && Format!=packed_nvq::kNvq2Exec && Format!=packed_nvq::kNvq2JscExec &&
                Format!=packed_nvq::kNvq2JscXLGroupExec && Format!=packed_nvq::kNvq3JscLGroupExec &&
                Format!=packed_nvq::kNpq0L && Format!=packed_nvq::kNpq0S) {
                constexpr int bits=packed_nvq::format_index_bits(Format);
                constexpr int vectors=packed_nvq::is_d4_format(Format)?6:3;
                if constexpr(E8Narrow && (Format==5 || Format==13 || Format==14))
                    exec64=load_e8_indices32<bits,false>(indices,neuron*d.nvec+group*3,narrow_index_bytes);
                else if(!((Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S) && d.sizes[1]==0))
                    exec64=packed_nvq::load_packed_group_window<bits*vectors>(indices,
                        (int64_t(neuron)*d.nvec+group*vectors)*bits,d.sizes[0]);
            }
            int dot=0,activation_sum=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                // GS24 quantizers zero-pad activation tails. The checked packed
                // index window can supply the remaining codebook vectors without
                // per-vector tail branches; their dot contributions are zero.
                packed_nvq::NvqVec8Values<Format> w{};
                if constexpr(Format!=4 && Format!=6 && Format!=7 && Format!=9 && Format<16) {
                    constexpr int index_bits=packed_nvq::format_index_bits(Format);
                    constexpr bool delta_format=Format==1 || Format==8;
                    const uint64_t indices=exec64;int2 weight{};
                    constexpr uint32_t mask=(1u<<index_bits)-1u;
                    if constexpr(packed_nvq::is_d4_format(Format)) {
                        const uint32_t i0=uint32_t(indices>>(segment*2*index_bits))&mask;
                        const uint32_t i1=uint32_t(indices>>((segment*2+1)*index_bits))&mask;
                        weight=make_int2(reinterpret_cast<const int*>(bank)[i0],reinterpret_cast<const int*>(bank)[i1]);
                    }else {
                        const uint32_t index=uint32_t(indices>>(segment*index_bits))&mask;
                        weight=reinterpret_cast<const int2*>(bank)[index];
                    }
                    if constexpr(!delta_format) {
                        const uint32_t mask7=(signs>>(segment*7))&127u;
                        int last=__popc(mask7)&1;
                        if constexpr(!packed_nvq::is_d4_format(Format))
                            if(d.sign_mode&1)last^=int((indices>>(segment*index_bits+7))&1u);
                        weight=packed_nvq::apply_sign8(weight,mask7|(uint32_t(last)<<7));
                    }
                    w={weight,delta,true};
                }else w=packed_nvq::load_nvq_group_vec8<Format, true>(indices,d.sizes[0],auxiliary,d.sizes[1],
                    codebook,bank,neuron,group,segment,d.groups,d.nvec,d.nsign,d.sign_mode,sub,signs,exec64,exec96,delta);
                dot=__dp4a(w.values.y,activation[segment].y,__dp4a(w.values.x,activation[segment].x,dot));
                if constexpr(!IntegerDelta && (Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S))
                    if(w.valid)activation_sum=__dp4a(0x01010101,activation[segment].y,__dp4a(0x01010101,activation[segment].x,activation_sum));
            }
            float value=float(dot);
            if constexpr(IntegerDelta)value*=Format==packed_nvq::kNvq1L?0.125f:0.03125f;
            else {
                if constexpr(Format==packed_nvq::kNvq1L)value+=0.125f*float(delta*activation_sum);
                if constexpr(Format==packed_nvq::kNvq1S)value+=0.15625f*float(delta*activation_sum);
            }
            partial=fmaf(weight_scale*input_scale,value,partial);
        }
    }
#pragma unroll
    for(int offset=4;offset>0;offset>>=1)partial+=__shfl_xor_sync(0xffffffffu,partial,offset,8);
    result[0]=__shfl_sync(0xffffffffu,partial,0);
    result[1]=__shfl_sync(0xffffffffu,partial,8);
    result[2]=__shfl_sync(0xffffffffu,partial,16);
    result[3]=__shfl_sync(0xffffffffu,partial,24);
}
template<int Format,bool Quad>
__device__ __forceinline__ void nvq1_decode_dot(const MfePackedProjection& d,int local,
        int first,const int8_t* input,const float* scales,float* values) {
    if constexpr(Quad) {
        if(d.fields[6]) {
            if(d.fields[5])nvq_quad_dot<Format,true,true>(d,local,first,input,scales,values);
            else nvq_quad_dot<Format,false,true>(d,local,first,input,scales,values);
        }else if(d.fields[5])nvq_quad_dot<Format,true>(d,local,first,input,scales,values);
        else nvq_quad_dot<Format>(d,local,first,input,scales,values);
    }else {
        if(d.fields[6]) {
            if(d.fields[5])nvq_group_dot<Format,true,true>(d,local,first,input,scales,values);
            else nvq_group_dot<Format,false,true>(d,local,first,input,scales,values);
        }else if(d.fields[5])nvq_group_dot<Format,true>(d,local,first,input,scales,values);
        else nvq_group_dot<Format>(d,local,first,input,scales,values);
    }
}
template<int Format>
__device__ __forceinline__ void nvq_pair(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,__half* result) {
    const int lane=threadIdx.x;
    const int parts=d.input_width>=4096?8:4;
    float values[2]{};
    const auto* indices=static_cast<const uint8_t*>(d.fields[0]);
    const auto* auxiliary=static_cast<const uint8_t*>(d.fields[1]);
    const auto* states=static_cast<const uint8_t*>(d.fields[2]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    // Each virtual warp keeps exactly the old segment assignment and FMA
    // order. Independent output warps remove the cross-warp block barrier.
#pragma unroll 1
    for(int part=0;part<parts;++part) {
        float partial[2]{};
        for(int segment=part*32+lane;segment<d.nsign;segment+=parts*32) {
            const int group=segment/3;
            const auto* activation=input+group*24+(segment-group*3)*8;
            const int2 activation_codes=*reinterpret_cast<const int2*>(activation);
            const float input_scale=scales[group];
            int activation_sum=0;
            if constexpr(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S)
                activation_sum=__dp4a(0x01010101,activation_codes.y,__dp4a(0x01010101,activation_codes.x,0));
#pragma unroll
            for(int row=0;row<2;++row)if(first+row<d.output_rows) {
                const int neuron=expert*d.output_rows+first+row;
                const int64_t state_linear=int64_t(neuron)*d.groups+group;
                const uint32_t sub=packed_nvq::load_nvq_state<Format>(indices,d.sizes[0],
                    states,d.sizes[2],state_linear,d.sub_bits,d.groups);
                const float weight_scale=(Format==packed_nvq::kNvq1L || Format==packed_nvq::kNvq1S)
                    ? anchors[neuron]*float(sub)
                    : packed_nvq::format_scale<Format>(anchors[neuron],sub,codebook);
                const auto weight=packed_nvq::load_nvq_vec8<Format>(indices,d.sizes[0],auxiliary,d.sizes[1],
                    codebook,neuron,segment,group,d.groups,d.nvec,d.nsign,d.sign_mode,sub);
                float dot=0.0f;
                if(weight.valid) {
                    const int integer_dot=__dp4a(weight.values.y,activation_codes.y,
                        __dp4a(weight.values.x,activation_codes.x,0));
                    dot=static_cast<float>(integer_dot);
                    if constexpr(Format==packed_nvq::kNvq1L)
                        dot=dot+0.125f*static_cast<float>(weight.delta*activation_sum);
                    if constexpr(Format==packed_nvq::kNvq1S)
                        dot=dot+0.15625f*static_cast<float>(weight.delta*activation_sum);
                }
                partial[row]=fmaf(weight_scale*input_scale,dot,partial[row]);
            }
        }
#pragma unroll
        for(int row=0;row<2;++row) {
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)
                partial[row]+=__shfl_xor_sync(0xffffffffu,partial[row],offset);
            if(lane==part)values[row]=partial[row];
        }
    }
#pragma unroll
    for(int row=0;row<2;++row) {
        // Only lanes 0 and 1 consume the projection results. The omitted
        // upper XOR steps add zero; normalize signed zero as those steps did.
        float value=values[row];
        if(value==0.0f)value=0.0f;
#pragma unroll 1
        for(int offset=parts/2;offset>0;offset>>=1)value+=__shfl_xor_sync(0xffffffffu,value,offset);
        result[row]=__float2half(value);
    }
}
__device__ __forceinline__ void nint_group_pair_legacy(const MfePackedProjection& d,int local,
        int first,const int8_t* input,const float* scales,float* result,bool multi_sum) {
    const int lane=threadIdx.x;
    const uint8_t* stream=static_cast<const uint8_t*>(d.fields[0])+size_t(local)*d.q_expert_stride;
    const auto* qbits=static_cast<const uint8_t*>(d.fields[1]);
    const auto* offsets=static_cast<const int64_t*>(d.fields[2]);
    const auto* subscale=static_cast<const uint8_t*>(d.fields[3]);
    const auto* submin=static_cast<const uint8_t*>(d.fields[4]);
    const auto* scales_outer=static_cast<const float*>(d.fields[5]);
    const auto* minima_outer=static_cast<const float*>(d.fields[6]);
    float partial[2]{},outer_scale[2]{},outer_min[2]{};
    int bits[2]{};uint64_t row_offset[2]{};
#pragma unroll
    for(int row=0;row<2;++row)if(first+row<d.output_rows) {
        const int neuron=local*d.output_rows+first+row;
        bits[row]=qbits[neuron];row_offset[row]=uint64_t(offsets[neuron]);
        outer_scale[row]=scales_outer[neuron];outer_min[row]=minima_outer[neuron];
    }
    for(int group=lane;group<d.groups;group+=32) {
        int dot[2]{},activation_sum=0;
#pragma unroll 1
        for(int element=0;element<d.group_size;element+=4) {
            const int column=group*d.group_size+element,width=min(4,d.group_size-element);
            uint32_t codes=0;
            if(width==4)codes=uint32_t(packed_nint::load_i8x4(input+column));
            else {
#pragma unroll
                for(int c=0;c<4;++c)if(c<width)codes|=(uint32_t(input[column+c])&255u)<<(8*c);
            }
            const int sum=__dp4a(0x01010101,int(codes),0);activation_sum+=sum;
#pragma unroll
            for(int row=0;row<2;++row)if(first+row<d.output_rows) {
                uint32_t weight=0;
                if(width==4)weight=uint32_t(packed_nint::unpack_nint_codes4(stream,
                    row_offset[row]+uint64_t(column)*bits[row],bits[row]));
                else {
#pragma unroll
                    for(int c=0;c<4;++c)if(c<width)weight|=uint32_t(packed_nint::unpack_nint_code(
                        stream,row_offset[row],column+c,bits[row]))<<(8*c);
                }
                dot[row]+=bits[row]==8?__dp4a(int(weight^0x80808080u),int(codes),0)+128*sum:
                    __dp4a(int(weight),int(codes),0);
            }
        }
        const float input_scale=scales[group];
#pragma unroll
        for(int row=0;row<2;++row)if(first+row<d.output_rows) {
            const size_t meta=size_t(local*d.output_rows+first+row)*d.groups+group;
            partial[row]+=input_scale*(outer_scale[row]*float(subscale[meta])*float(dot[row])-
                outer_min[row]*float(submin[meta])*float(activation_sum));
        }
    }
    if(multi_sum)warp_multi_sum::broadcast(partial,lane);
    else {
#pragma unroll
        for(int row=0;row<2;++row) {
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)partial[row]+=__shfl_xor_sync(0xffffffffu,partial[row],offset);
        }
    }
    result[0]=partial[0];result[1]=partial[1];
}
__device__ __forceinline__ void nint_group_pair(const MfePackedProjection& d,int local,
        int first,const int8_t* input,const float* scales,float* result,bool multi_sum) {
    if(d.nint_whole && mfe_nint_try_whole_pair(d,local,first,input,scales,result,multi_sum))return;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0])+size_t(local)*d.q_expert_stride;
    if((d.group_size&7) || (reinterpret_cast<uintptr_t>(input)&7u) || (reinterpret_cast<uintptr_t>(stream)&3u)) {
        nint_group_pair_legacy(d,local,first,input,scales,result,multi_sum);return;
    }
    const int lane=threadIdx.x;
    const auto* qbits=static_cast<const uint8_t*>(d.fields[1]);
    const auto* offsets=static_cast<const int64_t*>(d.fields[2]);
    const auto* subscale=static_cast<const uint8_t*>(d.fields[3]);
    const auto* submin=static_cast<const uint8_t*>(d.fields[4]);
    const auto* scales_outer=static_cast<const float*>(d.fields[5]);
    const auto* minima_outer=static_cast<const float*>(d.fields[6]);
    float partial[2]{},outer_scale[2]{},outer_min[2]{};
    int bits[2]{};uint64_t row_offset[2]{};
#pragma unroll
    for(int row=0;row<2;++row)if(first+row<d.output_rows) {
        const int neuron=local*d.output_rows+first+row;
        bits[row]=qbits[neuron];row_offset[row]=uint64_t(offsets[neuron]);
        outer_scale[row]=scales_outer[neuron];outer_min[row]=minima_outer[neuron];
    }
    for(int group=lane;group<d.groups;group+=32) {
        int dot[2]{},activation_sum=0;
#pragma unroll 1
        for(int element=0;element<d.group_size;element+=8) {
            const int column=group*d.group_size+element;
            const int2 codes=*reinterpret_cast<const int2*>(input+column);
            activation_sum=__dp4a(0x01010101,codes.y,__dp4a(0x01010101,codes.x,activation_sum));
#pragma unroll
            for(int row=0;row<2;++row)if(first+row<d.output_rows) {
                int2 weight;
                const uint64_t packed=packed_nint::unpack_nint_codes8_packed(stream,
                    row_offset[row]+uint64_t(column)*bits[row],bits[row]);
                const uint32_t low=uint32_t(packed),second=uint32_t(packed>>(4*bits[row]));
                const uint32_t pairs0=__byte_perm(low,low>>(2*bits[row]),0x5410);
                const uint32_t pairs1=__byte_perm(second,second>>(2*bits[row]),0x5410);
                const uint32_t mask=((1u<<bits[row])-1u)*0x01010101u;
                weight=make_int2(int(__byte_perm(pairs0,pairs0>>bits[row],0x6240)&mask),
                    int(__byte_perm(pairs1,pairs1>>bits[row],0x6240)&mask));
                if(bits[row]==8){weight.x^=int(0x80808080u);weight.y^=int(0x80808080u);}
                dot[row]=__dp4a(weight.y,codes.y,__dp4a(weight.x,codes.x,dot[row]));
            }
        }
        const float input_scale=scales[group];
#pragma unroll
        for(int row=0;row<2;++row)if(first+row<d.output_rows) {
            if(bits[row]==8)dot[row]+=128*activation_sum;
            const size_t meta=size_t(local*d.output_rows+first+row)*d.groups+group;
            partial[row]+=input_scale*(outer_scale[row]*float(subscale[meta])*float(dot[row])-
                outer_min[row]*float(submin[meta])*float(activation_sum));
        }
    }
    if(multi_sum)warp_multi_sum::broadcast(partial,lane);
    else {
#pragma unroll
        for(int row=0;row<2;++row) {
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)partial[row]+=__shfl_xor_sync(0xffffffffu,partial[row],offset);
        }
    }
    result[0]=partial[0];result[1]=partial[1];
}


// Shared projections use the dense NINT kernel in the unfused FFN. Its long
// rows distribute groups across four warps, so a single-warp group sum can
// round differently at an FP16 boundary. Reproduce each virtual warp's sum
// and the final dense XOR tree without changing compressed weight storage.
template<int Warps>
__device__ __forceinline__ void nint_dense_group_pair(const MfePackedProjection& d,int local,
        int first,const int8_t* input,const float* input_scales,float* result) {
    const int lane=threadIdx.x;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0])+size_t(local)*d.q_expert_stride;
    const auto* qbits=static_cast<const uint8_t*>(d.fields[1]);
    const auto* offsets=static_cast<const int64_t*>(d.fields[2]);
    const auto* subscale=static_cast<const uint8_t*>(d.fields[3]);
    const auto* submin=static_cast<const uint8_t*>(d.fields[4]);
    const auto* outer_scale=static_cast<const float*>(d.fields[5]);
    const auto* outer_min=static_cast<const float*>(d.fields[6]);
    float sums[Warps][2]{};
#pragma unroll
    for(int part=0;part<Warps;++part) {
        for(int group=part*32+lane;group<d.groups;group+=32*Warps) {
            int dots[2]{},activation_sum=0;
#pragma unroll 1
            for(int chunk=0;chunk<d.group_size/4;++chunk) {
                const int column=group*d.group_size+chunk*4;
                const int x=packed_nint::load_i8x4(input+column);
                activation_sum=__dp4a(0x01010101,x,activation_sum);
#pragma unroll
                for(int row=0;row<2;++row)if(first+row<d.output_rows) {
                    const int neuron=local*d.output_rows+first+row,bits=qbits[neuron];
                    int q=packed_nint::unpack_nint_codes4_aligned(stream,
                        uint64_t(offsets[neuron])+uint64_t(column)*bits,bits,d.q_expert_stride);
                    if(bits==8)q^=int(0x80808080u);
                    dots[row]=__dp4a(q,x,dots[row]);
                }
            }
#pragma unroll
            for(int row=0;row<2;++row)if(first+row<d.output_rows) {
                const int neuron=local*d.output_rows+first+row;
                const int dot=dots[row]+(qbits[neuron]==8?128*activation_sum:0);
                const auto meta=size_t(neuron)*d.groups+group;
                const float scale=outer_scale[neuron]*float(subscale[meta]);
                const float minimum=outer_min[neuron]*float(submin[meta]);
                sums[part][row]+=input_scales[group]*(scale*float(dot)-minimum*float(activation_sum));
            }
        }
#pragma unroll
        for(int row=0;row<2;++row) {
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)
                sums[part][row]+=__shfl_xor_sync(0xffffffffu,sums[part][row],offset);
        }
    }
#pragma unroll
    for(int row=0;row<2;++row) {
        if constexpr(Warps==1)result[row]=sums[0][row];
        else {
            float value=0.0f;
#pragma unroll
            for(int part=0;part<Warps;++part)if(lane==part)value=sums[part][row];
#pragma unroll
            for(int offset=16;offset>0;offset>>=1)value+=__shfl_xor_sync(0xffffffffu,value,offset);
            result[row]=__shfl_sync(0xffffffffu,value,0);
        }
    }
}
__device__ __forceinline__ bool nint_late_group_pair(const MfePackedProjection& d,int local,
        int first,const int8_t* input,const float* scales,float* result,bool multi_sum) {
    const auto* bits=static_cast<const uint8_t*>(d.fields[1]);
    bool matching=d.groups<INT_MAX/(d.group_size*d.nint_late_bits);
    #pragma unroll
    for(int row=0;row<2;++row)if(first+row<d.output_rows)
        matching=matching && bits[local*d.output_rows+first+row]==d.nint_late_bits;
    if(!matching)return false;
    const auto launch=[&](auto group,auto width) {
        packed_nint::nint_fixed_group_rows<decltype(group)::value,decltype(width)::value,1,2,true>(
            static_cast<const uint8_t*>(d.fields[0]),static_cast<const int64_t*>(d.fields[2]),
            static_cast<const uint8_t*>(d.fields[3]),static_cast<const uint8_t*>(d.fields[4]),
            static_cast<const float*>(d.fields[5]),static_cast<const float*>(d.fields[6]),
            input,scales,local,first,d.output_rows,d.groups,d.q_expert_stride,result,multi_sum);
    };
    const auto choose_bits=[&](auto group) {
        if(d.nint_late_bits==4)launch(group,std::integral_constant<int,4>{});
        else if(d.nint_late_bits==5)launch(group,std::integral_constant<int,5>{});
        else launch(group,std::integral_constant<int,6>{});
    };
    if(d.group_size==24)choose_bits(std::integral_constant<int,24>{});
    else choose_bits(std::integral_constant<int,28>{});
    return true;
}
template<bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
__device__ __forceinline__ void project_pair(const MfePackedProjection* d,int local,
        int first,const int8_t* input,const float* scales,__half* result,bool multi_sum=false) {
    result[0]=result[1]=__float2half(0.0f);
    if(!d || local<0 || local>=d->local_experts)return;
    if(d->family==1) {
        if constexpr(!DefaultMath) {
        if(d->dense_reduction_warps) {
            float values[2];
            if(d->dense_reduction_warps<0)
                packed_nint::nint_matmul_routed_pair<true>(static_cast<const uint8_t*>(d->fields[0]),
                    static_cast<const uint8_t*>(d->fields[1]),static_cast<const int64_t*>(d->fields[2]),
                    static_cast<const uint8_t*>(d->fields[3]),static_cast<const uint8_t*>(d->fields[4]),
                    static_cast<const float*>(d->fields[5]),static_cast<const float*>(d->fields[6]),
                    input,scales,nullptr,0,0,local,first,d->output_rows,d->groups,d->groups*d->group_size,
                    d->group_size,d->q_expert_stride,0,values);
            else if(d->dense_reduction_warps==4)nint_dense_group_pair<4>(*d,local,first,input,scales,values);
            else nint_dense_group_pair<1>(*d,local,first,input,scales,values);
            result[0]=__float2half(values[0]);result[1]=__float2half(values[1]);return;
        }
        if(d->nint_late_bits) {
            float values[2];
            if(d->nint_late_bits<0 || !nint_late_group_pair(*d,local,first,input,scales,values,multi_sum))
                packed_nint::nint_matmul_routed_pair<true>(static_cast<const uint8_t*>(d->fields[0]),
                    static_cast<const uint8_t*>(d->fields[1]),static_cast<const int64_t*>(d->fields[2]),
                    static_cast<const uint8_t*>(d->fields[3]),static_cast<const uint8_t*>(d->fields[4]),
                    static_cast<const float*>(d->fields[5]),static_cast<const float*>(d->fields[6]),
                    input,scales,nullptr,0,0,local,first,d->output_rows,d->groups,d->groups*d->group_size,
                    d->group_size,d->q_expert_stride,0,values);
            result[0]=__float2half(values[0]);result[1]=__float2half(values[1]);return;
        }
        }
        if constexpr(Grouped) {
            float values[2];nint_group_pair(*d,local,first,input,scales,values,multi_sum);
            result[0]=__float2half(values[0]);result[1]=__float2half(values[1]);return;
        }else {
        float values[2];
        packed_nint::nint_matmul_routed_pair<true>(static_cast<const uint8_t*>(d->fields[0]),
            static_cast<const uint8_t*>(d->fields[1]),static_cast<const int64_t*>(d->fields[2]),
            static_cast<const uint8_t*>(d->fields[3]),static_cast<const uint8_t*>(d->fields[4]),
            static_cast<const float*>(d->fields[5]),static_cast<const float*>(d->fields[6]),
            input,scales,nullptr,0,0,local,first,d->output_rows,d->groups,d->groups*d->group_size,
            d->group_size,d->q_expert_stride,0,values);
        result[0]=__float2half(values[0]);result[1]=__float2half(values[1]);return;
        }
    }
    if(d->family!=2)return;
    if constexpr(Grouped) {
    float floating[2]{};
#define MFQ_MFE_NVQ_CASE(F) case F: if constexpr(Formats&(1u<<F))nvq_group_dot<F,false,false,E8Narrow>(*d,local,first,input,scales,floating);break
    switch(d->format) {
        case 1: if constexpr(Formats&(1u<<1))nvq1_decode_dot<1,false>(*d,local,first,input,scales,floating);break;
        MFQ_MFE_NVQ_CASE(2);MFQ_MFE_NVQ_CASE(3);MFQ_MFE_NVQ_CASE(4);
        MFQ_MFE_NVQ_CASE(5);MFQ_MFE_NVQ_CASE(6);MFQ_MFE_NVQ_CASE(7);case 8: if constexpr(Formats&(1u<<8))nvq1_decode_dot<8,false>(*d,local,first,input,scales,floating);break;
        MFQ_MFE_NVQ_CASE(9);MFQ_MFE_NVQ_CASE(10);MFQ_MFE_NVQ_CASE(11);MFQ_MFE_NVQ_CASE(12);
        MFQ_MFE_NVQ_CASE(13);MFQ_MFE_NVQ_CASE(14);MFQ_MFE_NVQ_CASE(15);MFQ_MFE_NVQ_CASE(16);
        MFQ_MFE_NVQ_CASE(17);
    }
#undef MFQ_MFE_NVQ_CASE
    result[0]=__float2half(floating[0]);result[1]=__float2half(floating[1]);return;
    }else {
#define MFQ_MFE_NVQ_CASE(F) case F: if constexpr(Formats&(1u<<F))nvq_pair<F>(*d,local,first,input,scales,result);break
    switch(d->format) {
        MFQ_MFE_NVQ_CASE(1);MFQ_MFE_NVQ_CASE(2);MFQ_MFE_NVQ_CASE(3);MFQ_MFE_NVQ_CASE(4);
        MFQ_MFE_NVQ_CASE(5);MFQ_MFE_NVQ_CASE(6);MFQ_MFE_NVQ_CASE(7);MFQ_MFE_NVQ_CASE(8);
        MFQ_MFE_NVQ_CASE(9);MFQ_MFE_NVQ_CASE(10);MFQ_MFE_NVQ_CASE(11);MFQ_MFE_NVQ_CASE(12);
        MFQ_MFE_NVQ_CASE(13);MFQ_MFE_NVQ_CASE(14);MFQ_MFE_NVQ_CASE(15);MFQ_MFE_NVQ_CASE(16);
        MFQ_MFE_NVQ_CASE(17);
    }
#undef MFQ_MFE_NVQ_CASE
    }
}
template<bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
__device__ __forceinline__ void project_quad(const MfePackedProjection* d,int local,
        int first,const int8_t* input,const float* scales,__half* result,bool multi_sum=false,bool canonical=false) {
    if constexpr(!Grouped) {
        project_pair<false,Formats,DefaultMath,E8Narrow>(d,local,first,input,scales,result,multi_sum);
        project_pair<false,Formats,DefaultMath,E8Narrow>(d,local,first+2,input,scales,result+2,multi_sum);
        return;
    }
    if(canonical) {
        // Preserve the two-row NVQ reduction order while processing four
        // output rows per block, avoiding the doubled Down grid.
        project_pair<Grouped,Formats,DefaultMath,E8Narrow>(d,local,first,input,scales,result,multi_sum);
        project_pair<Grouped,Formats,DefaultMath,E8Narrow>(d,local,first+2,input,scales,result+2,multi_sum);
        return;
    }
#pragma unroll
    for(int row=0;row<4;++row)result[row]=__float2half(0.0f);
    if(!d || local<0 || local>=d->local_experts)return;
    if(d->family==1){
        if constexpr(!DefaultMath) if(d->dense_reduction_warps || d->nint_late_bits) {
            project_pair<Grouped,Formats,DefaultMath,E8Narrow>(d,local,first,input,scales,result,multi_sum);
            project_pair<Grouped,Formats,DefaultMath,E8Narrow>(d,local,first+2,input,scales,result+2,multi_sum);
            return;
        }
#pragma unroll
        for(int pair=0;pair<2;++pair){float values[2];
            nint_group_pair(*d,local,first+pair*2,input,scales,values,multi_sum);
            result[pair*2]=__float2half(values[0]);result[pair*2+1]=__float2half(values[1]);}
        return;
    }
    if(d->family!=2)return;
    float values[4]{};
    switch(d->format){
        case 1:if constexpr(Formats&(1u<<1))nvq1_decode_dot<1,true>(*d,local,first,input,scales,values);break;
        case 2:if constexpr(Formats&(1u<<2))nvq_quad_dot<2>(*d,local,first,input,scales,values);break;
        case 3:if constexpr(Formats&(1u<<3))nvq_quad_dot<3>(*d,local,first,input,scales,values);break;
        case 4:if constexpr(Formats&(1u<<4))nvq_quad_dot<4>(*d,local,first,input,scales,values);break;
        case 5:if constexpr(Formats&(1u<<5))nvq_quad_dot<5,false,false,E8Narrow>(*d,local,first,input,scales,values);break;
        case 6:if constexpr(Formats&(1u<<6))nvq_quad_dot<6>(*d,local,first,input,scales,values);break;
        case 7:if constexpr(Formats&(1u<<7))nvq_quad_dot<7>(*d,local,first,input,scales,values);break;
        case 8:if constexpr(Formats&(1u<<8))nvq1_decode_dot<8,true>(*d,local,first,input,scales,values);break;
        case 9:if constexpr(Formats&(1u<<9))nvq_quad_dot<9>(*d,local,first,input,scales,values);break;
        case 10:if constexpr(Formats&(1u<<10))nvq_quad_dot<10>(*d,local,first,input,scales,values);break;
        case 11:if constexpr(Formats&(1u<<11))nvq_quad_dot<11>(*d,local,first,input,scales,values);break;
        case 12:if constexpr(Formats&(1u<<12))nvq_quad_dot<12>(*d,local,first,input,scales,values);break;
        case 13:if constexpr(Formats&(1u<<13))nvq_quad_dot<13,false,false,E8Narrow>(*d,local,first,input,scales,values);break;
        case 14:if constexpr(Formats&(1u<<14))nvq_quad_dot<14,false,false,E8Narrow>(*d,local,first,input,scales,values);break;
        case 15:if constexpr(Formats&(1u<<15))nvq_quad_dot<15>(*d,local,first,input,scales,values);break;
        case 16:if constexpr(Formats&(1u<<16))nvq_quad_dot<16>(*d,local,first,input,scales,values);break;
        case 17:if constexpr(Formats&(1u<<17))nvq_quad_dot<17>(*d,local,first,input,scales,values);break;
    }
#pragma unroll
    for(int row=0;row<4;++row)result[row]=__float2half(values[row]);
}
__device__ __forceinline__ const MfePackedProjection* projection(const MfeFfnBatch& b,
        int which,int expert,bool shared,int& local,bool transferred=false) {
    local=-1;
    if(shared) {local=0;return b.shared+which;}
    if(unsigned(expert)>=unsigned(b.experts))return nullptr;
    if(transferred) {
        const int index=reinterpret_cast<const volatile int32_t*>(b.transfer_index)[expert];
        if(index<0)return nullptr;
        local=0;return b.transferred+index*3+which;
    }
    const int pool=b.cohort_by_expert[which*b.experts+expert];
    if(unsigned(pool)>=unsigned(b.projection_counts[which]))return nullptr;
    const auto* d=b.projections+b.projection_offsets[which]+pool;
    local=d->expert_local?d->expert_local[expert]:0;
    if(local<0 && d->fallback>=0 && d->fallback<b.projection_counts[which]) {
        d=b.projections+b.projection_offsets[which]+d->fallback;
        local=d->expert_local?d->expert_local[expert]:0;
    }
    return d;
}
__device__ __forceinline__ bool aborted(const MfeFfnBatch& b) {
    return b.aborted && *reinterpret_cast<const volatile uint32_t*>(b.aborted);
}
__device__ __forceinline__ int primary_residency_kind(const MfeFfnBatch& b,int expert) {
    for(int which=0;which<3;++which) {
        const int pool=b.cohort_by_expert[which*b.experts+expert];
        if(unsigned(pool)>=unsigned(b.projection_counts[which]))return which==2?6:2;
        const auto& d=b.projections[b.projection_offsets[which]+pool];
        const int local=d.expert_local?d.expert_local[expert]:0;
        if(local<0 || local>=d.local_experts)return which==2?6:2;
    }
    return 1;
}
__device__ __forceinline__ int route_kind(const MfeFfnBatch& b,int expert) {
    if(aborted(b))return -1;
    const int tagged=b.kinds?b.kinds[expert]:1;
    return tagged>0?(tagged&3):tagged;
}
__device__ __forceinline__ void quantize_completed_group(const MfeFfnBatch& b,int slot,
        int width,int gs,int group) {
    const int lane=threadIdx.x;
    float values[2]{};
#pragma unroll
    for(int half=0;half<2;++half) {
        const int column=group*gs+lane+half*32;
        if(lane+half*32<gs && column<width)
            values[half]=__half2float(__ushort_as_half(*reinterpret_cast<const volatile unsigned short*>(
                b.hidden+int64_t(slot)*b.hidden_stride+column)));
    }
    float maximum=fmaxf(fabsf(values[0]),fabsf(values[1]));
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    const float scale=maximum>0.0f?maximum/127.0f:1.0f;
    if(lane==0)b.hidden_scales[int64_t(slot)*b.scale_stride+group]=scale;
#pragma unroll
    for(int half=0;half<2;++half)if(lane+half*32<gs) {
        const int column=group*gs+lane+half*32;
        int code=0;
        if(column<width)code=max(-127,min(127,int(roundf(values[half]/scale))));
        b.hidden_quantized[int64_t(slot)*b.quantized_stride+column]=int8_t(code);
    }
}
__global__ void __launch_bounds__(64) mfe_ffn_prepare_kernel(MfeFfnBatch b,
        const MfeInputQuantization* inputs,int count) {
    const int lane=threadIdx.x;
    const auto flat=int64_t(blockIdx.x)*64+lane;
    const int slots=b.tokens*(b.routes+(b.shared?1:0));
    if(flat<int64_t(slots)*b.scale_stride)b.completion[flat]=0;
    // One control block reads mapped host state. Compute warps consume VRAM
    // snapshots after this kernel completes, avoiding PCIe polling per warp.
    if(b.plan_ready && blockIdx.x==0) {
        __shared__ uint32_t cancelled;
        if(lane==0) {
            const auto* host_abort=reinterpret_cast<const volatile uint32_t*>(b.host_aborted);
            if(!b.resident_plan_overlap)
                while(!*reinterpret_cast<const volatile uint32_t*>(b.plan_ready) && !*host_abort)__nanosleep(64);
            cancelled=*host_abort?1:0;*b.aborted=cancelled;
        }
        __syncthreads();
        if(b.resident_plan_overlap) {
            for(int pair=lane;pair<b.tokens*b.routes;pair+=64) {
                const int expert=b.ids[pair];if(unsigned(expert)>=unsigned(b.experts))continue;
                // Repeated routes publish identical values through atomic
                // stores. Entries absent from this call are never consumed.
                atomicExch(b.kinds+expert,cancelled?-1:primary_residency_kind(b,expert));
            }
        }else for(int expert=lane;expert<b.experts;expert+=64) {
            b.kinds[expert]=cancelled?-1:reinterpret_cast<const volatile int32_t*>(b.host_kinds)[expert];
            b.transfer_index[expert]=cancelled?-1:reinterpret_cast<const volatile int32_t*>(b.host_transfer_index)[expert];
        }
    }
    int entry=blockIdx.x;const MfeInputQuantization* d=nullptr;
    for(int i=0;i<count;++i) {
        const int length=b.tokens*inputs[i].groups;
        if(entry<length){d=inputs+i;break;}entry-=length;
    }
    if(!d)return;
    const int token=entry/d->groups,group=entry%d->groups,column=group*d->group_size+lane;
    const float value=lane<d->group_size && column<b.input_width
        ? __half2float(b.input[int64_t(token)*b.input_width+column]):0.0f;
    float maximum=fabsf(value);
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
    __shared__ float maxima[2];
    if((lane&31)==0)maxima[lane>>5]=maximum;
    __syncthreads();
    if(lane<32) {
        maximum=lane<2?maxima[lane]:0.0f;
#pragma unroll
        for(int offset=16;offset>0;offset>>=1)
            maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
    }
    __shared__ float scale;
    if(lane==0){scale=maximum>0.0f?maximum/127.0f:1.0f;d->scales[entry]=scale;}
    __syncthreads();
    if(lane<d->group_size) {
        int code=column<b.input_width?max(-127,min(127,int(roundf(value/scale)))):0;
        d->values[int64_t(token)*d->groups*d->group_size+column]=int8_t(code);
    }
}
template<bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
__global__ void __launch_bounds__(128) mfe_ffn_gate_up_kernel(MfeFfnBatch b,int partition) {
    const bool grouped=(partition&16)!=0;partition&=3;
    const int all_routes=b.routes+(b.shared?1:0),slot=blockIdx.y;
    const int token=slot/all_routes,route=slot%all_routes;
    const bool shared=route==b.routes;
    const int width=shared?b.shared_intermediate:b.intermediate;
    const int first=int(blockIdx.x)*8+int(threadIdx.y)*2,lane=threadIdx.x;
    if(first>=width)return;
    const int expert=shared?0:b.ids[token*b.routes+route];
    if(!shared && unsigned(expert)>=unsigned(b.experts))return;
    const int kind=shared?1:route_kind(b,expert);
    if(kind<=0)return;
    if(partition && kind!=partition)return;
    const bool transferred=b.plan_ready && kind==2;
    if(!shared && aborted(b))return;
    int g_local,u_local,d_local;
    const auto* g=projection(b,0,expert,shared,g_local,transferred);
    const auto* u=projection(b,1,expert,shared,u_local,transferred);
    const auto* down=projection(b,2,expert,shared,d_local,transferred);
    __half gate[2],up[2];
    project_pair<Grouped,Formats,DefaultMath,E8Narrow>(g,g_local,first,g?static_cast<const int8_t*>(g->fields[7])+int64_t(token)*g->groups*g->group_size:nullptr,
        g?static_cast<const float*>(g->fields[8])+int64_t(token)*g->groups:nullptr,gate,b.warp_multi_sum);
    project_pair<Grouped,Formats,DefaultMath,E8Narrow>(u,u_local,first,u?static_cast<const int8_t*>(u->fields[7])+int64_t(token)*u->groups*u->group_size:nullptr,
        u?static_cast<const float*>(u->fields[8])+int64_t(token)*u->groups:nullptr,up,b.warp_multi_sum);
    const auto* table=b.sigmoid_table+(!shared && b.tokens*b.routes>1?65536:0);
    if(lane==0) {
#pragma unroll
        for(int row=0;row<2;++row)if(first+row<width) {
            const auto sigmoid=table[__half_as_ushort(gate[row])];
            const auto activated=__float2half_rn(__half2float(gate[row])*__half2float(sigmoid));
            b.hidden[int64_t(slot)*b.hidden_stride+first+row]=
                __float2half_rn(__half2float(activated)*__half2float(up[row]));
        }
        __threadfence();
    }
    __syncwarp();
    if(!down || d_local<0 || d_local>=down->local_experts)return;
    const int gs=down->group_size;
    for(int row=0;row<2;++row)if(first+row<width) {
        const int group=(first+row)/gs;
        if(row && first/gs==group)continue;
        const int written=(row==0 && first+1<width && (first+1)/gs==group)?2:1;
        int complete=0;
        if(lane==0)complete=atomicAdd(b.completion+int64_t(slot)*b.scale_stride+group,written)+written
            ==min(gs,width-group*gs);
        complete=__shfl_sync(0xffffffffu,complete,0);
        if(complete)quantize_completed_group(b,slot,width,gs,group);
    }
}
template<int MinimumBlocks,bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
__global__ void __launch_bounds__(512,MinimumBlocks) mfe_ffn_parallel_gate_up_kernel(MfeFfnBatch b,int partition) {
    const bool grouped=(partition&16)!=0;
    const bool shared_first=(partition&4)!=0;
    const bool early_gu=(partition&8)!=0;
    partition&=3;
    const int all_routes=b.routes+(b.shared?1:0);
    const int position=int(blockIdx.y)%all_routes,token=int(blockIdx.y)/all_routes;
    const int shift=shared_first?(position?-1:b.routes):0;
    const int route=position+shift,slot=int(blockIdx.y)+shift;
    const bool shared=route==b.routes;
    const int width=shared?b.shared_intermediate:b.intermediate;
    const int lane=threadIdx.x,warp=threadIdx.y;
    const int team=int(blockDim.y)/2;
    const int base=int(blockIdx.x)*int(blockDim.y);
    if(base>=width)return;
    const int first=base+(warp%team)*2;
    const bool is_up=warp>=team;
    const int expert=shared?0:b.ids[token*b.routes+route];
    if(!shared && unsigned(expert)>=unsigned(b.experts))return;
    __shared__ int block_kind,block_early;
    if(lane==0 && warp==0) {
        const int tagged=shared?1:aborted(b)?-1:b.kinds?b.kinds[expert]:1;
        block_kind=tagged>0?(tagged&3):tagged;block_early=0;
        if(early_gu && !shared && block_kind==2 && (tagged&4)) {
            if(partition==1) {block_kind=1;block_early=1;}
            else if(partition==2)block_kind=-1;
        }
    }
    __syncthreads();
    const int kind=block_kind;
    if(kind<=0 || (partition && kind!=partition))return;
    const bool transferred=b.plan_ready && kind==2;
    __shared__ MfePackedProjection block_descriptors[3];
    __shared__ int block_locals[3],block_present[3];
    if(warp==0 && lane<3) {
        int source_local;
        const auto* source=projection(b,lane,expert,shared,source_local,transferred);
        block_locals[lane]=source_local;block_present[lane]=source!=nullptr;
        if(source)block_descriptors[lane]=*source;
    }
    __syncthreads();
    const int local=block_locals[is_up?1:0];
    const auto* d=block_present[is_up?1:0]?block_descriptors+(is_up?1:0):nullptr;
    __half values[2]={__float2half(0.0f),__float2half(0.0f)};
    if(first<width)project_pair<Grouped,Formats,DefaultMath,E8Narrow>(d,local,first,
        d?static_cast<const int8_t*>(d->fields[7])+int64_t(token)*d->groups*d->group_size:nullptr,
        d?static_cast<const float*>(d->fields[8])+int64_t(token)*d->groups:nullptr,values,b.warp_multi_sum);
    __shared__ __half projected[32][2];
    if(lane==0) {
        projected[warp][0]=values[0];projected[warp][1]=values[1];
    }
    __syncthreads();
    if(warp!=0)return;
    const int end=min(base+int(blockDim.y),width);
    const auto* table=b.sigmoid_table+(!shared && b.tokens*b.routes>1?65536:0);
    if(base+lane<end) {
        const int pair=lane/2,row=lane&1;
        const auto gate=projected[pair][row],up=projected[pair+team][row];
        const auto sigmoid=table[__half_as_ushort(gate)];
        const auto activated=__float2half_rn(__half2float(gate)*__half2float(sigmoid));
        b.hidden[int64_t(slot)*b.hidden_stride+base+lane]=
            __float2half_rn(__half2float(activated)*__half2float(up));
        __threadfence();
    }
    __syncwarp();
    const int d_local=block_locals[2];
    const auto* down=block_present[2]?block_descriptors+2:nullptr;
    if(!down || (!block_early && (d_local<0 || d_local>=down->local_experts)))return;
    const int gs=down->group_size;
    for(int group=base/gs;group<=(end-1)/gs;++group) {
        const int written=min(end,(group+1)*gs)-max(base,group*gs);
        int complete=0;
        if(lane==0)complete=atomicAdd(b.completion+int64_t(slot)*b.scale_stride+group,written)+written
            ==min(gs,width-group*gs);
        complete=__shfl_sync(0xffffffffu,complete,0);
        if(complete)quantize_completed_group(b,slot,width,gs,group);
    }
}

__global__ void mfe_ffn_cpu_ready_kernel(MfeFfnBatch b) {
    if(threadIdx.x)return;
    const auto* host_abort=reinterpret_cast<const volatile uint32_t*>(b.host_aborted);
    while(!*reinterpret_cast<const volatile uint32_t*>(b.cpu_ready) &&
          !(host_abort && *host_abort))__nanosleep(64);
    if(b.aborted && host_abort)*b.aborted=*host_abort?1:0;
}
__global__ void mfe_ffn_transfer_ready_kernel(MfeFfnBatch b) {
    __shared__ uint32_t cancelled;
    if(threadIdx.x==0) {
        const auto* host_abort=reinterpret_cast<const volatile uint32_t*>(b.host_aborted);
        while((!*reinterpret_cast<const volatile uint32_t*>(b.transfer_ready) ||
                (b.resident_plan_overlap && !*reinterpret_cast<const volatile uint32_t*>(b.plan_ready))) &&
              !(host_abort && *host_abort))__nanosleep(64);
        cancelled=host_abort && *host_abort?1:0;
        if(b.aborted)*b.aborted=cancelled;
        __threadfence_system();
    }
    if(!b.resident_plan_overlap)return;
    __syncthreads();
    for(int pair=threadIdx.x;pair<b.tokens*b.routes;pair+=32) {
        const int expert=b.ids[pair];if(unsigned(expert)>=unsigned(b.experts))continue;
        const int previous=b.kinds[expert];
        const int kind=cancelled?-1:reinterpret_cast<const volatile int32_t*>(b.host_kinds)[expert];
        atomicExch(b.kinds+expert,kind==2 && previous==6?6:kind);
        atomicExch(b.transfer_index+expert,cancelled?-1:
            reinterpret_cast<const volatile int32_t*>(b.host_transfer_index)[expert]);
    }
}
template<bool UnifiedProjection,bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
__global__ void __launch_bounds__(1024) mfe_ffn_down_reduce_kernel(MfeFfnBatch b,int partition) {
    const bool grouped=(partition&16)!=0;partition&=3;
    const int token=blockIdx.y,first=int(blockIdx.x)*2,lane=threadIdx.x;
    if(first>=b.output_width)return;
    const int all_routes=b.routes+(b.shared?1:0);
    __shared__ __half results[32][2];
    for(int route=threadIdx.y;route<all_routes;route+=blockDim.y) {
        __half values[2]={__float2half(0.0f),__float2half(0.0f)};
        const MfePackedProjection* descriptor=nullptr;int descriptor_local=-1;
        if(route<b.routes) {
            const int pair=token*b.routes+route,expert=b.ids[pair];
            if(unsigned(expert)<unsigned(b.experts) && !aborted(b)) {
                const int kind=route_kind(b,expert);
                if(partition==2 && kind==1) {
    #pragma unroll
                    for(int row=0;row<2;++row)if(first+row<b.output_width)
                        values[row]=b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+first+row];
                } else if(kind==0 && partition!=1) {
    #pragma unroll
                    for(int row=0;row<2;++row)if(first+row<b.output_width)
                        values[row]=aborted(b)?__float2half(0.0f):b.cpu_pairs[int64_t(pair)*b.output_width+first+row];
                } else if(kind>0 && (!partition || kind==partition)) {
                    if constexpr(UnifiedProjection) {
                        descriptor=projection(b,2,expert,false,descriptor_local,b.plan_ready && kind==2);
                    }else {
                        int local;const auto* d=projection(b,2,expert,false,local,b.plan_ready && kind==2);
                        const int slot=token*all_routes+route;
                        project_pair<Grouped,Formats,DefaultMath,E8Narrow>(d,local,first,b.hidden_quantized+int64_t(slot)*b.quantized_stride,
                            b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum);
                    }
                }
            }
        }else if(partition==2) {
    #pragma unroll
            for(int row=0;row<2;++row)if(first+row<b.output_width)
                values[row]=b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+first+row];
        }else {
            if constexpr(UnifiedProjection) {
                descriptor=b.shared+2;descriptor_local=0;
            }else {
                const int slot=token*all_routes+b.routes;
                project_pair<Grouped,Formats,DefaultMath,E8Narrow>(b.shared+2,0,first,b.hidden_quantized+int64_t(slot)*b.quantized_stride,
                    b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum);
            }
        }
        if constexpr(UnifiedProjection)if(descriptor) {
            const int slot=token*all_routes+route;
            project_pair<Grouped,Formats,DefaultMath,E8Narrow>(descriptor,descriptor_local,first,b.hidden_quantized+int64_t(slot)*b.quantized_stride,
                b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum);
        }
        if(partition==1) {
            if(lane<2 && first+lane<b.output_width)
                b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+first+lane]=values[lane];
            continue;
        }
        if(b.output_pairs) {
            if(lane<2 && first+lane<b.output_width)
                static_cast<__half*>(b.output)[(int64_t(token)*b.routes+route)*b.output_width+first+lane]=values[lane];
            continue;
        }
        if(lane==0) {
    #pragma unroll
            for(int row=0;row<2;++row)results[route][row]=values[row];
        }
    }
    if(partition==1 || b.output_pairs)return;
    __syncthreads();
    // Warps compute independent routes. The final two output lanes
    // retain the original sequential route FMA order and Half boundary.
    if(threadIdx.y==0 && lane<2 && first+lane<b.output_width) {
        float sum=0.0f;
        for(int r=0;r<b.routes;++r)sum+=b.route_weights[token*b.routes+r]*__half2float(results[r][lane]);
        const float routed=__half2float(__float2half(sum));
        const float shared_gate=b.shared?(b.shared_product_half?
            __half2float(static_cast<const __half*>(b.shared_gate)[token]):b.shared_gate_bfloat?
            __bfloat162float(static_cast<const __nv_bfloat16*>(b.shared_gate)[token]):
            static_cast<const float*>(b.shared_gate)[token]):0.0f;
        float contribution=b.shared?__fmul_rn(__half2float(results[b.routes][lane]),shared_gate):0.0f;
        if(b.shared_product_half)contribution=__half2float(__float2half(contribution));
        const float output=b.shared?__fadd_rn(routed,contribution):routed;
        const auto index=int64_t(token)*b.output_width+first+lane;
        if(b.output_float)static_cast<float*>(b.output)[index]=output;
        else static_cast<__half*>(b.output)[index]=__float2half(output);
    }
}
template<bool UnifiedProjection,bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
__global__ void __launch_bounds__(1024) mfe_ffn_down_reduce_kernel_quad(MfeFfnBatch b,int partition) {
    const bool grouped=(partition&16)!=0;partition&=3;
    const int token=blockIdx.y,first=int(blockIdx.x)*4,lane=threadIdx.x;
    if(first>=b.output_width)return;
    const int all_routes=b.routes+(b.shared?1:0);
    __shared__ __half results[32][4];
    for(int route=threadIdx.y;route<all_routes;route+=blockDim.y) {
        __half values[4]={__float2half(0.0f),__float2half(0.0f)};
        const MfePackedProjection* descriptor=nullptr;int descriptor_local=-1;
        if(route<b.routes) {
            const int pair=token*b.routes+route,expert=b.ids[pair];
            if(unsigned(expert)<unsigned(b.experts) && !aborted(b)) {
                const int kind=route_kind(b,expert);
                if(partition==2 && kind==1) {
    #pragma unroll
                    for(int row=0;row<4;++row)if(first+row<b.output_width)
                        values[row]=b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+first+row];
                } else if(kind==0 && partition!=1) {
    #pragma unroll
                    for(int row=0;row<4;++row)if(first+row<b.output_width)
                        values[row]=aborted(b)?__float2half(0.0f):b.cpu_pairs[int64_t(pair)*b.output_width+first+row];
                } else if(kind>0 && (!partition || kind==partition)) {
                    if constexpr(UnifiedProjection) {
                        descriptor=projection(b,2,expert,false,descriptor_local,b.plan_ready && kind==2);
                    }else {
                        int local;const auto* d=projection(b,2,expert,false,local,b.plan_ready && kind==2);
                        const int slot=token*all_routes+route;
                        project_quad<Grouped,Formats,DefaultMath,E8Narrow>(d,local,first,b.hidden_quantized+int64_t(slot)*b.quantized_stride,
                            b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum,b.shared_dense_math);
                    }
                }
            }
        }else if(partition==2) {
    #pragma unroll
            for(int row=0;row<4;++row)if(first+row<b.output_width)
                values[row]=b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+first+row];
        }else {
            if constexpr(UnifiedProjection) {
                descriptor=b.shared+2;descriptor_local=0;
            }else {
                const int slot=token*all_routes+b.routes;
                project_quad<Grouped,Formats,DefaultMath,E8Narrow>(b.shared+2,0,first,b.hidden_quantized+int64_t(slot)*b.quantized_stride,
                    b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum,b.shared_dense_math);
            }
        }
        if constexpr(UnifiedProjection)if(descriptor) {
            const int slot=token*all_routes+route;
            project_quad<Grouped,Formats,DefaultMath,E8Narrow>(descriptor,descriptor_local,first,b.hidden_quantized+int64_t(slot)*b.quantized_stride,
                b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum,b.shared_dense_math);
        }
        if(partition==1) {
            if(lane<4 && first+lane<b.output_width)
                b.resident_pairs[(int64_t(token)*all_routes+route)*b.output_width+first+lane]=values[lane];
            continue;
        }
        if(b.output_pairs) {
            if(lane<4 && first+lane<b.output_width)
                static_cast<__half*>(b.output)[(int64_t(token)*b.routes+route)*b.output_width+first+lane]=values[lane];
            continue;
        }
        if(lane==0) {
    #pragma unroll
            for(int row=0;row<4;++row)results[route][row]=values[row];
        }
    }
    if(partition==1 || b.output_pairs)return;
    __syncthreads();
    // Warps compute independent routes. The final two output lanes
    // retain the original sequential route FMA order and Half boundary.
    if(threadIdx.y==0 && lane<4 && first+lane<b.output_width) {
        float sum=0.0f;
        for(int r=0;r<b.routes;++r)sum+=b.route_weights[token*b.routes+r]*__half2float(results[r][lane]);
        const float routed=__half2float(__float2half(sum));
        const float shared_gate=b.shared?(b.shared_product_half?
            __half2float(static_cast<const __half*>(b.shared_gate)[token]):b.shared_gate_bfloat?
            __bfloat162float(static_cast<const __nv_bfloat16*>(b.shared_gate)[token]):
            static_cast<const float*>(b.shared_gate)[token]):0.0f;
        float contribution=b.shared?__fmul_rn(__half2float(results[b.routes][lane]),shared_gate):0.0f;
        if(b.shared_product_half)contribution=__half2float(__float2half(contribution));
        const float output=b.shared?__fadd_rn(routed,contribution):routed;
        const auto index=int64_t(token)*b.output_width+first+lane;
        if(b.output_float)static_cast<float*>(b.output)[index]=output;
        else static_cast<__half*>(b.output)[index]=__float2half(output);
    }
}

} // anonymous namespace
} // namespace mfq::cuda
