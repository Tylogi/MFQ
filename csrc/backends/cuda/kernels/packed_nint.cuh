#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "glu.cuh"
#include "warp_multi_sum.cuh"

namespace mfq::cuda::packed_nint {
__device__ __forceinline__ uint8_t unpack_nint_code(
        const uint8_t * stream,
        uint64_t row_bit_offset,
        int element,
        int bits) {
    const uint64_t bit = row_bit_offset +
        static_cast<uint64_t>(element) * static_cast<uint64_t>(bits);
    const uint64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7u);
    uint32_t word = static_cast<uint32_t>(stream[byte]);
    if (shift + bits > 8) {
        word |= static_cast<uint32_t>(stream[byte + 1]) << 8;
    }
    return static_cast<uint8_t>(
        (word >> shift) & ((1u << bits) - 1u));
}


__device__ __forceinline__ int unpack_nint_codes4(
        const uint8_t * stream,
        uint64_t bit_offset,
        int bits) {
    const uint64_t byte = bit_offset >> 3;
    const int shift = static_cast<int>(bit_offset & 7u);
    const int required_bits = shift + 4 * bits;
    uint32_t packed = static_cast<uint32_t>(stream[byte]);
    if (required_bits > 8) {
        packed |= static_cast<uint32_t>(stream[byte + 1]) << 8;
    }
    if (required_bits > 16) {
        packed |= static_cast<uint32_t>(stream[byte + 2]) << 16;
    }
    if (required_bits > 24) {
        packed |= static_cast<uint32_t>(stream[byte + 3]) << 24;
    }
    packed >>= shift;
    if (required_bits > 32) {
        // A crossing q8 window needs only the low bits of one more byte.
        // All four resulting codes fit in a 32-bit register.
        packed |= static_cast<uint32_t>(stream[byte + 4]) << (32-shift);
    }
    const uint32_t mask = (1u << bits) - 1u;
    const uint32_t codes =
        static_cast<uint32_t>(packed & mask) |
        (static_cast<uint32_t>((packed >> bits) & mask) << 8) |
        (static_cast<uint32_t>((packed >> (2 * bits)) & mask) << 16) |
        (static_cast<uint32_t>((packed >> (3 * bits)) & mask) << 24);
    return static_cast<int>(codes);
}


// Decode one eight-value micro-tile for every runtime q width.  The packed
// stream carries eight padding bytes, so the final row can safely furnish the
// one look-ahead byte needed by an unaligned q8 window.  Keeping q dynamic is
// what lets uniform presets and heterogeneous NINTv2 rows share this kernel.
__device__ __forceinline__ uint64_t unpack_nint_codes8_packed(
        const uint8_t * stream,
        uint64_t bit_offset,
        int bits) {
    const uint64_t byte = bit_offset >> 3;
    const int shift = static_cast<int>(bit_offset & 7u);
    const int required_bytes = (shift + 8 * bits + 7) >> 3;
    uint64_t packed = 0;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
        if (index < required_bytes) {
            packed |= static_cast<uint64_t>(stream[byte + index]) <<
                (8 * index);
        }
    }
    if (shift == 0) {
        return packed;
    }
    const uint64_t look_ahead = required_bytes > 8
        ? static_cast<uint64_t>(stream[byte + 8])
        : 0u;
    return (packed >> shift) | (look_ahead << (64 - shift));
}


__device__ __forceinline__ int load_i8x4(const int8_t * source) {
    if((reinterpret_cast<uintptr_t>(source)&3u)==0)
        return *reinterpret_cast<const int*>(source);
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(source);
    const uint32_t packed = static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
    return static_cast<int>(packed);
}


__device__ __forceinline__ int unpack_nint_codes4_aligned(
        const uint8_t* stream,uint64_t bit,int bits,uint64_t bytes) {
    const uint64_t byte=(bit>>3)&~uint64_t(3);
    if((reinterpret_cast<uintptr_t>(stream)&3u) || byte+8>bytes)
        return unpack_nint_codes4(stream,bit,bits);
    const auto* words=reinterpret_cast<const uint32_t*>(stream+byte);
    const uint32_t packed=__funnelshift_r(words[0],words[1],int(bit&31u));
    const uint32_t pairs=__byte_perm(packed,packed>>(2*bits),0x5410);
    return int(__byte_perm(pairs,pairs>>bits,0x6240)&(((1u<<bits)-1u)*0x01010101u));
}

__device__ __forceinline__ int2 unpack_nint_codes8_aligned(
        const uint8_t* stream,uint64_t bit,int bits,uint64_t bytes) {
    const uint64_t byte=(bit>>3)&~uint64_t(3);
    if((reinterpret_cast<uintptr_t>(stream)&3u) || byte+12>bytes)
        return make_int2(unpack_nint_codes4_aligned(stream,bit,bits,bytes),
            unpack_nint_codes4_aligned(stream,bit+4*bits,bits,bytes));
    const auto* words=reinterpret_cast<const uint32_t*>(stream+byte);
    const int shift=int(bit&31u);
    const uint32_t low=__funnelshift_r(words[0],words[1],shift);
    const uint32_t high=__funnelshift_r(words[1],words[2],shift);
    const uint32_t second=__funnelshift_rc(low,high,4*bits);
    const uint32_t a=__byte_perm(low,low>>(2*bits),0x5410);
    const uint32_t b=__byte_perm(second,second>>(2*bits),0x5410);
    const uint32_t mask=((1u<<bits)-1u)*0x01010101u;
    return make_int2(int(__byte_perm(a,a>>bits,0x6240)&mask),
        int(__byte_perm(b,b>>bits,0x6240)&mask));
}

__device__ __forceinline__ int2 unpack_nint4_codes8(
        const uint8_t* stream,uint64_t bit,uint64_t bytes) {
    const uint64_t byte=(bit>>3)&~uint64_t(3);
    if((reinterpret_cast<uintptr_t>(stream)&3u) || byte+8>bytes)
        return unpack_nint_codes8_aligned(stream,bit,4,bytes);
    const auto* words=reinterpret_cast<const uint32_t*>(stream+byte);
    const uint32_t low=__funnelshift_r(words[0],words[1],int(bit&31u)),high=low>>16;
    return make_int2(int(__byte_perm(low,low>>4,0x5140)&0x0f0f0f0fu),
        int(__byte_perm(high,high>>4,0x5140)&0x0f0f0f0fu));
}

// Whole-group integer dots for two routed output rows. Lanes=16 gives each
// half warp its own row; Lanes=32 shares activation loads across both rows.
// Packed row bit widths remain dynamic, including unsigned eight-bit codes.
template<int GroupSize,int Lanes,int Rows=2,int FixedBits=0>
__device__ __forceinline__ void nint_grouped_routed_pair(
        const uint8_t* stream,const uint8_t* bits,const int64_t* offsets,
        const uint8_t* subscale,const uint8_t* submin,
        const float* outer_scale,const float* outer_min,
        const int8_t* input,const float* input_scale,
        int local,int first,int output_rows,int groups,int stride,float* result,bool multi_sum=false) {
    static_assert(Lanes==32 || Rows==2);
    constexpr int accumulators=Lanes==16?1:Rows;
    const int lane=threadIdx.x&(Lanes-1);
    const int first_row=Lanes==16 ? int(threadIdx.x)>>4 : 0;
    const auto* packed=stream+size_t(local)*stride;
    float partial[accumulators]{},scale[accumulators]{},minimum[accumulators]{};
    int qbits[accumulators]{},neuron[accumulators]{};
    uint64_t row_bit[accumulators]{};
#pragma unroll
    for(int a=0;a<accumulators;++a)if(first+first_row+a<output_rows) {
        neuron[a]=local*output_rows+first+first_row+a;
        qbits[a]=FixedBits?FixedBits:bits[neuron[a]];row_bit[a]=uint64_t(offsets[neuron[a]]);
        scale[a]=outer_scale[neuron[a]];minimum[a]=outer_min[neuron[a]];
    }
    for(int group=lane;group<groups;group+=Lanes) {
        int dot[accumulators]{},sum=0;
#pragma unroll
        for(int chunk=0;chunk<GroupSize/8;++chunk) {
            const int column=group*GroupSize+chunk*8;
            const int2 x=make_int2(load_i8x4(input+column),load_i8x4(input+column+4));
            sum=__dp4a(0x01010101,x.y,__dp4a(0x01010101,x.x,sum));
#pragma unroll
            for(int a=0;a<accumulators;++a)if(first+first_row+a<output_rows) {
                int2 w;
                if constexpr(FixedBits==4)w=unpack_nint4_codes8(packed,row_bit[a]+uint64_t(column)*4,stride);
                else w=unpack_nint_codes8_aligned(packed,row_bit[a]+uint64_t(column)*qbits[a],qbits[a],stride);
                if(qbits[a]==8){w.x^=int(0x80808080u);w.y^=int(0x80808080u);}
                dot[a]=__dp4a(w.y,x.y,__dp4a(w.x,x.x,dot[a]));
            }
        }
        if constexpr(GroupSize%8==4) {
            const int column=group*GroupSize+GroupSize-4,x=load_i8x4(input+column);
            sum=__dp4a(0x01010101,x,sum);
#pragma unroll
            for(int a=0;a<accumulators;++a)if(first+first_row+a<output_rows) {
                int w=unpack_nint_codes4_aligned(packed,row_bit[a]+uint64_t(column)*qbits[a],qbits[a],stride);
                if(qbits[a]==8)w^=int(0x80808080u);
                dot[a]=__dp4a(w,x,dot[a]);
            }
        }
        const float xs=input_scale[group];
#pragma unroll
        for(int a=0;a<accumulators;++a)if(first+first_row+a<output_rows) {
            const size_t meta=size_t(neuron[a])*groups+group;
            if(qbits[a]==8)dot[a]+=128*sum;
            partial[a]+=xs*(scale[a]*float(subscale[meta])*float(dot[a])-
                minimum[a]*float(submin[meta])*float(sum));
        }
    }
    if(multi_sum && Lanes==32)warp_multi_sum::broadcast(partial,lane);
    else {
#pragma unroll
        for(int a=0;a<accumulators;++a) {
#pragma unroll
            for(int offset=Lanes/2;offset>0;offset>>=1)
                partial[a]+=__shfl_xor_sync(0xffffffffu,partial[a],offset,Lanes);
        }
    }
    if constexpr(Lanes==16) {
        result[0]=__shfl_sync(0xffffffffu,partial[0],0);
        result[1]=__shfl_sync(0xffffffffu,partial[0],16);
    } else {
#pragma unroll
        for(int a=0;a<accumulators;++a)result[a]=partial[a];
    }
}

template <bool ReturnValues = false>
__device__ __forceinline__ void nint_matmul_routed_pair(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int pair,
        int source_row,
        int local_expert,
        int output_row0,
        int output_rows,
        int groups,
        int padded_width,
        int group_size,
        int q_expert_stride,
        int epilogue_mode, float* row_values=nullptr) {
    constexpr int routed_rows_per_warp = 2;
    const int result_rows = epilogue_mode == 0
        ? output_rows
        : output_rows / 2;
    const int chunks = (group_size + 3) / 4;
    const int groups_per_warp = 32 / chunks;
    const int lane = static_cast<int>(threadIdx.x);
    const uint8_t * expert_stream = bitstream +
        static_cast<size_t>(local_expert) *
            static_cast<size_t>(q_expert_stride);

    float accumulators[routed_rows_per_warp] = {0.0f, 0.0f};
    float outer_scales[routed_rows_per_warp] = {};
    float outer_minima[routed_rows_per_warp] = {};
    int row_bits[routed_rows_per_warp] = {};
    uint64_t row_bit_offsets[routed_rows_per_warp] = {};
#pragma unroll
    for (int row = 0; row < routed_rows_per_warp; ++row) {
        const int output_row = epilogue_mode == 0
            ? output_row0 + row
            : output_row0 + row * result_rows;
        if (output_row0 < result_rows && output_row < output_rows) {
            const int weight_row =
                local_expert * output_rows + output_row;
            outer_scales[row] = neuron_scale[weight_row];
            outer_minima[row] = neuron_minimum[weight_row];
            row_bits[row] = static_cast<int>(row_q_bits[weight_row]);
            row_bit_offsets[row] = static_cast<uint64_t>(
                row_q_bit_offsets[weight_row]);
        }
    }

    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int element = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const int width = min(4, group_size - element);
    for (int group = active_lane ? relative_group : groups,
             column = relative_group * group_size + element;
         group < groups;
         group += groups_per_warp, column += groups_per_warp * group_size) {
        const int8_t * activation_ptr = activation +
            static_cast<size_t>(source_row) * padded_width + column;
        uint32_t activation_code_bits = 0;
        int activation_sum = 0;
        if (width == 4) {
            activation_code_bits = static_cast<uint32_t>(
                load_i8x4(activation_ptr));
            const int activation_codes = static_cast<int>(
                activation_code_bits);
            activation_sum = __dp4a(0x01010101, activation_codes, 0);
        } else {
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                if (component < width) {
                    const int code = static_cast<int>(
                        activation_ptr[component]);
                    activation_code_bits |=
                        (static_cast<uint32_t>(code) & 255u) <<
                            (8 * component);
                    activation_sum += code;
                }
            }
        }
        const int activation_codes = static_cast<int>(activation_code_bits);
        const float input_scale = activation_scale[
            static_cast<size_t>(source_row) * groups + group];
#pragma unroll
        for (int row = 0; row < routed_rows_per_warp; ++row) {
            const int output_row = epilogue_mode == 0
                ? output_row0 + row
                : output_row0 + row * result_rows;
            if (output_row0 >= result_rows || output_row >= output_rows) {
                continue;
            }
            const int weight_row =
                local_expert * output_rows + output_row;
            const size_t metadata_index =
                static_cast<size_t>(weight_row) * groups + group;
            const int bits = row_bits[row];
            uint32_t weight_code_bits = 0;
            if (width == 4) {
                const uint64_t bit_offset = row_bit_offsets[row] +
                    static_cast<uint64_t>(column) *
                        static_cast<uint64_t>(bits);
                weight_code_bits = static_cast<uint32_t>(
                    unpack_nint_codes4(
                        expert_stream, bit_offset, bits));
            } else {
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    if (component < width) {
                        weight_code_bits |= static_cast<uint32_t>(
                            unpack_nint_code(
                                expert_stream,
                                row_bit_offsets[row],
                                column + component,
                                bits)) << (8 * component);
                    }
                }
            }
            const int weight_codes = static_cast<int>(weight_code_bits);
            const int dot = bits == 8
                ? __dp4a(
                      weight_codes ^ static_cast<int>(0x80808080u),
                      activation_codes,
                      0) + 128 * activation_sum
                : __dp4a(weight_codes, activation_codes, 0);
            accumulators[row] += input_scale * (
                outer_scales[row] *
                    static_cast<float>(subgroup_scale[metadata_index]) *
                    static_cast<float>(dot) -
                outer_minima[row] *
                    static_cast<float>(subgroup_minimum[metadata_index]) *
                    static_cast<float>(activation_sum));
        }
    }

#pragma unroll
    for (int row = 0; row < routed_rows_per_warp; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if constexpr (ReturnValues) {
        row_values[0]=accumulators[0];row_values[1]=accumulators[1];return;
    }
    if (lane == 0) {
        if (epilogue_mode == 0) {
#pragma unroll
            for (int row = 0; row < routed_rows_per_warp; ++row) {
                const int output_row = output_row0 + row;
                if (output_row < output_rows) {
                    output[
                        static_cast<size_t>(pair) * output_rows +
                        output_row] = __float2half(accumulators[row]);
                }
            }
        } else if (output_row0 < result_rows) {
            const int activation = epilogue_mode == 2 ? 1 : 0;
            const float gate = __half2float(
                __float2half_rn(accumulators[0]));
            const float up = __half2float(
                __float2half_rn(accumulators[1]));
            output[
                static_cast<size_t>(pair) * result_rows + output_row0] =
                __float2half_rn(mfq_glu_runtime(
                    gate, up, activation));
        }
    }
}



}
