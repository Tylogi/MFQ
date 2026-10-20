#pragma once
#include "kernels/mfe/ffn.h"
#include "kernels/nvq/device.cuh"
#include "kernels/nvq/dense_window32.cuh"
#include "kernels/nvq/jsc_bank_map.cuh"
#if defined(MFQ_MFE_PARITY_READ) && MFQ_MFE_PARITY_READ
#include "nvq_parity_read.cuh"
#endif
// Host descriptors prove expert-local bit offsets and exact compact payload extents.
namespace mfq::cuda {
namespace {
template<int Format,int Lanes>
__device__ __forceinline__ void nvq_compact32_e8_dot_original(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
    constexpr int bits=packed_nvq::format_index_bits(Format);
    const int lane=threadIdx.x&(Lanes-1),row=threadIdx.x/Lanes;
    const auto* stream=static_cast<const uint8_t*>(d.fields[0]);
    const auto* anchors=static_cast<const float*>(d.fields[3]);
    const auto* codebook=static_cast<const int8_t*>(d.fields[4]);
    const uint32_t bank_map=d.nvq_bank_map;
#if defined(MFQ_MFE_PARITY_READ) && MFQ_MFE_PARITY_READ
    const uint32_t parity_mask=(uint32_t(d.sign_mode)&1u)<<7;
#endif
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
            const int8_t* bank;
            if constexpr(Format==packed_nvq::kNvq2Jsc || Format==packed_nvq::kNvq2JscL ||
                         Format==packed_nvq::kNvq2JscXL) {
                bank=bank_map?packed_nvq::packed_jsc_codebook<Format>(codebook,bank_map,state)
                             :packed_nvq::active_codebook<Format>(codebook,state);
            }else bank=packed_nvq::active_codebook<Format>(codebook,state);
#if defined(MFQ_MFE_PARITY_READ) && MFQ_MFE_PARITY_READ
            const float weight_scale=parity_read_scale<Format>(anchor,state,codebook);
#else
            const float weight_scale=packed_nvq::format_scale<Format>(anchor,state,codebook);
#endif
            // 24 int8 products fit in int32; independent chains retain the exact sum.
            int dot_low=0,dot_high=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                int2 value{};
#if defined(MFQ_MFE_PARITY_READ) && MFQ_MFE_PARITY_READ
                if(segment==0)value=parity_read_e8<0>(packed,bank,parity_mask);
                if(segment==1)value=parity_read_e8<1>(packed,bank,parity_mask);
                if(segment==2)value=parity_read_e8<2>(packed,bank,parity_mask);
#else
                if(segment==0)value=packed_nvq::decode_dense_e8_vector<0>(packed,bank,d.sign_mode);
                if(segment==1)value=packed_nvq::decode_dense_e8_vector<1>(packed,bank,d.sign_mode);
                if(segment==2)value=packed_nvq::decode_dense_e8_vector<2>(packed,bank,d.sign_mode);
#endif
                dot_low=__dp4a(value.x,activation[segment].x,dot_low);
                dot_high=__dp4a(value.y,activation[segment].y,dot_high);
            }
            partial=fmaf(weight_scale*input_scale,float(dot_low+dot_high),partial);
        }
    }
#pragma unroll
    for(int offset=Lanes/2;offset>0;offset>>=1)partial+=__shfl_xor_sync(0xffffffffu,partial,offset,Lanes);
#pragma unroll
    for(int row=0;row<32/Lanes;++row)result[row]=__shfl_sync(0xffffffffu,partial,row*Lanes);
}

// The per-group LUT read keeps the packed decoder's control flow compact.
// Real-weight resident GU measurements favor it over a warp LUT shuffle.
template<int Format,int Lanes>
__device__ __forceinline__ void nvq_compact32_e8_dot(const MfePackedProjection& d,int expert,
        int first,const int8_t* input,const float* scales,float* result) {
    nvq_compact32_e8_dot_original<Format,Lanes>(d,expert,first,input,scales,result);
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
    const uint32_t bank_map=d.nvq_bank_map;
    const bool analytic_scale=packed_nvq::jsc_d4_analytic_scales<Format>(codebook);
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
            const int8_t* bank;
            if constexpr(Format==packed_nvq::kNvq3Jsc || Format==packed_nvq::kNvq3Jsc512 ||
                         Format==packed_nvq::kNvq3JscL) {
                bank=bank_map?packed_nvq::packed_jsc_codebook<Format>(codebook,bank_map,state)
                             :packed_nvq::active_codebook<Format>(codebook,state);
            }else bank=packed_nvq::active_codebook<Format>(codebook,state);
#if defined(MFQ_MFE_PARITY_READ) && MFQ_MFE_PARITY_READ
            const float weight_scale=analytic_scale?anchor*float((state>>1)+1):parity_read_scale<Format>(anchor,state,codebook);
#else
            const float weight_scale=analytic_scale?anchor*float((state>>1)+1):packed_nvq::format_scale<Format>(anchor,state,codebook);
#endif
            // 24 int8 products fit in int32; independent chains retain the exact sum.
            int dot_low=0,dot_high=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                int2 value{};
#if defined(MFQ_MFE_PARITY_READ) && MFQ_MFE_PARITY_READ
                if(segment==0)value=parity_read_d4<0>(packed,bank);
                if(segment==1)value=parity_read_d4<1>(packed,bank);
                if(segment==2)value=parity_read_d4<2>(packed,bank);
#else
                if(segment==0)value=packed_nvq::decode_dense_d4_vector<0>(packed,bank);
                if(segment==1)value=packed_nvq::decode_dense_d4_vector<1>(packed,bank);
                if(segment==2)value=packed_nvq::decode_dense_d4_vector<2>(packed,bank);
#endif
                dot_low=__dp4a(value.x,activation[segment].x,dot_low);
                dot_high=__dp4a(value.y,activation[segment].y,dot_high);
            }
            partial=fmaf(weight_scale*input_scale,float(dot_low+dot_high),partial);
        }
    }
#pragma unroll
    for(int offset=Lanes/2;offset>0;offset>>=1)partial+=__shfl_xor_sync(0xffffffffu,partial,offset,Lanes);
#pragma unroll
    for(int row=0;row<32/Lanes;++row)result[row]=__shfl_sync(0xffffffffu,partial,row*Lanes);
}

}
}
