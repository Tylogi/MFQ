#pragma once
#include "mfe_ffn.h"
#include "nvq_e8_dense_group.cuh"
#include "nvq_d4_dense_group.cuh"
#include "nvq_dense_window32.cuh"
// Host descriptors prove expert-local bit offsets and exact compact payload extents.
namespace mfq::cuda {
namespace {
template<int Format,int Lanes>
__device__ __forceinline__ void nvq_compact32_e8_dot(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
    constexpr int bits=packed_nvq::format_index_bits(Format);
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    const int row_bits=d.nvec*(bits+7)+d.groups*4;
    const int expert_bytes=d.output_rows*row_bits/8;
    stream+=int64_t(expert)*expert_bytes;
    const int neuron=expert*d.output_rows+first+row;
    const float anchor=first+row<d.output_rows?anchors[neuron]:0.f;
    float partial=0.f;
    for(int group=lane;group<d.groups;group+=Lanes) {
        const int2 activation[3]={*reinterpret_cast<const int2*>(input+group*24),
            *reinterpret_cast<const int2*>(input+group*24+8),*reinterpret_cast<const int2*>(input+group*24+16)};
        const float input_scale=scales[group];
        if(first+row<d.output_rows) {
            const int bit=(first+row)*row_bits+group*(4+3*(bits+7));
            const auto words=packed_nvq::load_dense_window32<4+3*(bits+7)>(stream,expert_bytes,bit);
            const packed_nvq::DenseE8Group<bits> packed{uint64_t(words.x)|(uint64_t(words.y)<<32)};
            const auto state=packed.state();
            const auto* bank=packed_nvq::active_codebook<Format>(codebook,state);
            const float weight_scale=packed_nvq::format_scale<Format>(anchor,state,codebook);
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
}

template<int Format,int Lanes>
__device__ __forceinline__ void nvq_compact32_d4_dot(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
    constexpr int bits=packed_nvq::format_index_bits(Format);
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    const int row_bits=d.nsign*(2*bits+7)+d.groups*4;
    const int expert_bytes=d.output_rows*row_bits/8;
    stream+=int64_t(expert)*expert_bytes;
    const int neuron=expert*d.output_rows+first+row;
    const float anchor=first+row<d.output_rows?anchors[neuron]:0.f;
    float partial=0.f;
    for(int group=lane;group<d.groups;group+=Lanes) {
        const int2 activation[3]={*reinterpret_cast<const int2*>(input+group*24),
            *reinterpret_cast<const int2*>(input+group*24+8),*reinterpret_cast<const int2*>(input+group*24+16)};
        const float input_scale=scales[group];
        if(first+row<d.output_rows) {
            const int bit=(first+row)*row_bits+group*(4+3*(2*bits+7));
            const auto words=packed_nvq::load_dense_window32<4+3*(2*bits+7)>(stream,expert_bytes,bit);
            const packed_nvq::DenseD4Group<bits> packed{words.x,words.y,words.z};
            const auto state=packed.state();
            const auto* bank=packed_nvq::active_codebook<Format>(codebook,state);
            const float weight_scale=packed_nvq::format_scale<Format>(anchor,state,codebook);
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
}

}
}
