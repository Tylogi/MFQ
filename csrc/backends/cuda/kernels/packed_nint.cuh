#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "glu.cuh"

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
