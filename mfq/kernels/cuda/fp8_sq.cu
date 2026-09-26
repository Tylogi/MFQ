#include "fp8_sq.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "packed_backward.cuh"

namespace {

constexpr int kDirectPackedMaxRows = 48;

struct Layout {
    int outputs = 0;
    int width = 0;
    int block_rows = 0;
    int block_columns = 0;
    int scale_rows = 0;
    int scale_columns = 0;
    int scale_kind = 0;
    std::size_t palettes = 0;
    std::size_t symbols = 0;
    std::size_t scales = 0;
};

__device__ __forceinline__ unsigned read_bits(
        const std::uint8_t* data, std::size_t index, int bits) {
    const auto bit = index * static_cast<std::size_t>(bits);
    const unsigned shift = static_cast<unsigned>(bit & 7);
    unsigned value = data[bit >> 3];
    if (shift + static_cast<unsigned>(bits) > 8) {
        value |= static_cast<unsigned>(data[(bit >> 3) + 1]) << 8;
    }
    return (value >> shift) & ((1u << bits) - 1u);
}

__device__ __forceinline__ float decode_e4m3fn(std::uint8_t raw) {
    const unsigned magnitude = static_cast<unsigned>(raw & 0x7fu);
    const unsigned exponent = magnitude >> 3u;
    const unsigned mantissa = magnitude & 7u;
    const float value = exponent == 0u
        ? static_cast<float>(mantissa) * 0.001953125f
        : __uint_as_float(((exponent + 120u) << 23u) | (mantissa << 20u));
    return (raw & 0x80u) == 0u ? value : -value;
}


__device__ __forceinline__ float4 decode_e4m3fn4(std::uint32_t raw) {
#if __CUDA_ARCH__ >= 890
    const __half2_raw low_raw = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(raw), __NV_E4M3);
    const __half2_raw high_raw = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(raw >> 16), __NV_E4M3);
    const float2 low = __half22float2(__halves2half2(
        __ushort_as_half(low_raw.x), __ushort_as_half(low_raw.y)));
    const float2 high = __half22float2(__halves2half2(
        __ushort_as_half(high_raw.x), __ushort_as_half(high_raw.y)));
    return make_float4(low.x, low.y, high.x, high.y);
#else
    return make_float4(
        decode_e4m3fn(static_cast<std::uint8_t>(raw)),
        decode_e4m3fn(static_cast<std::uint8_t>(raw >> 8)),
        decode_e4m3fn(static_cast<std::uint8_t>(raw >> 16)),
        decode_e4m3fn(static_cast<std::uint8_t>(raw >> 24)));
#endif
}

__device__ __forceinline__ float decode_e8m0(std::uint8_t raw) {
    return raw == 0u
        ? __uint_as_float(0x00400000u)
        : __uint_as_float(static_cast<unsigned>(raw) << 23u);
}

__device__ __forceinline__ std::uint16_t load_u16(
        const std::uint8_t* source) {
    return static_cast<std::uint16_t>(source[0]) |
        static_cast<std::uint16_t>(source[1]) << 8;
}

__device__ __forceinline__ std::uint32_t load_u32(
        const std::uint8_t* source) {
    return static_cast<std::uint32_t>(source[0]) |
        static_cast<std::uint32_t>(source[1]) << 8 |
        static_cast<std::uint32_t>(source[2]) << 16 |
        static_cast<std::uint32_t>(source[3]) << 24;
}

__device__ __forceinline__ float decode_fp8_128_scale(
        const std::uint8_t* source, int scale_kind) {
    if (scale_kind == 2) {
        return __uint_as_float(static_cast<unsigned>(load_u16(source)) << 16);
    }
    if (scale_kind == 3) {
        const auto raw = load_u16(source);
        const unsigned exponent = (raw >> 10u) & 31u;
        const unsigned mantissa = raw & 1023u;
        if (exponent == 0u) {
            return static_cast<float>(mantissa) * 0x1p-24f;
        }
        return (1.0f + static_cast<float>(mantissa) / 1024.0f) *
            __uint_as_float((exponent + 112u) << 23u);
    }
    return __uint_as_float(load_u32(source));
}

__device__ __forceinline__ std::uint8_t decode_code(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const Layout& layout,
        int output,
        int column) {
    const int bits = static_cast<int>(row_q[output]);
    const auto* row_symbols = blob + layout.symbols +
        static_cast<std::size_t>(row_symbol_byte_offsets[output]);
    if (bits == 8) {
        return row_symbols[column];
    }
    const auto symbol = read_bits(
        row_symbols, static_cast<std::size_t>(column), bits);
    return blob[layout.palettes + (std::size_t{1} << bits) - 2 + symbol];
}

template <bool MXFP8>
__device__ __forceinline__ float decode_scale(
        const std::uint8_t* blob,
        const Layout& layout,
        int output,
        int column) {
    const auto scale_index =
        static_cast<std::size_t>(output / layout.block_rows) *
            layout.scale_columns +
        column / layout.block_columns;
    if constexpr (MXFP8) {
        return decode_e8m0(blob[layout.scales + scale_index]);
    } else {
        const int itemsize = layout.scale_kind == 4 ? 4 : 2;
        return decode_fp8_128_scale(
            blob + layout.scales + scale_index * itemsize,
            layout.scale_kind);
    }
}

__device__ __forceinline__ float decode_bf16_block_scale(
        const std::uint8_t* blob,
        const Layout& layout,
        int output,
        int column) {
    const auto scale_index =
        static_cast<std::size_t>(output / layout.block_rows) *
            layout.scale_columns +
        column / layout.block_columns;
    return __uint_as_float(
        static_cast<unsigned>(load_u16(
            blob + layout.scales + scale_index * 2)) << 16);
}

template <bool MXFP8>
__device__ __forceinline__ float decode_weight(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const Layout& layout,
        int output,
        int column) {
    return decode_e4m3fn(decode_code(
        blob, row_q, row_symbol_byte_offsets, layout, output, column)) *
        decode_scale<MXFP8>(blob, layout, output, column);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float as_float(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
}

template <bool MXFP8, typename T>
__device__ __forceinline__ void dequant_body(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    const auto count = static_cast<std::size_t>(layout.outputs) * layout.width;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / layout.width);
        const int column = static_cast<int>(index % layout.width);
        output[index] = from_float<T>(decode_weight<MXFP8>(
            blob, row_q, row_symbol_byte_offsets, layout, row, column));
    }
}

template <typename T>
__global__ void mxfp8_sq_dequant_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    dequant_body<true>(blob, row_q, row_symbol_byte_offsets, output, layout);
}

template <typename T>
__global__ void fp8_128_sq_dequant_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    dequant_body<false>(blob, row_q, row_symbol_byte_offsets, output, layout);
}

template <bool MXFP8, int TILE_M, typename T, bool ROUTED>
__device__ __forceinline__ void mmq_body(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int logical_outputs = ROUTED ? out_per_expert : layout.outputs;
    const auto output_tiles = (static_cast<std::int64_t>(logical_outputs) + 3) / 4;
    const auto row_tiles = (static_cast<std::int64_t>(rows) + TILE_M - 1) / TILE_M;
    for (std::int64_t task = blockIdx.x;
         task < output_tiles * row_tiles;
         task += gridDim.x) {
        const int logical_output = static_cast<int>(task % output_tiles) * 4 + warp;
        if (logical_output >= logical_outputs) continue;
        const int first_row = static_cast<int>(task / output_tiles) * TILE_M;
        int local_expert = 0;
        if constexpr (ROUTED) {
            const int expert = expert_ids[first_row];
            if (expert < 0 || expert >= global_experts) continue;
            local_expert = expert_local[expert];
            if (local_expert < 0 || local_expert >= local_experts) continue;
        }
        const int weight_row = ROUTED
            ? local_expert * out_per_expert + logical_output
            : logical_output;
        float accumulator[TILE_M] = {};
        for (int column = lane; column < layout.width; column += 32) {
            const float weight = decode_weight<MXFP8>(
                blob,
                row_q,
                row_symbol_byte_offsets,
                layout,
                weight_row,
                column);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                if (first_row + item >= rows) continue;
                const int source_row = ROUTED && shared_input
                    ? (first_row + item) / routes
                    : first_row + item;
                accumulator[item] = fmaf(
                    weight,
                    as_float(input[
                        static_cast<std::size_t>(source_row) * layout.width + column]),
                    accumulator[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const float value = warp_sum(accumulator[item]);
            if (lane == 0 && first_row + item < rows) {
                output[
                    static_cast<std::size_t>(first_row + item) * logical_outputs +
                    logical_output] = from_float<T>(value);
            }
        }
    }
}

template <int TILE_M, typename T, bool ROUTED = false>
__global__ void mxfp8_sq_mmq_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    mmq_body<true, TILE_M, T, ROUTED>(
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
        expert_ids, expert_local, global_experts, local_experts,
        out_per_expert, routes, shared_input);
}

template <int TILE_M, typename T, bool ROUTED = false>
__global__ void fp8_128_sq_mmq_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    mmq_body<false, TILE_M, T, ROUTED>(
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
        expert_ids, expert_local, global_experts, local_experts,
        out_per_expert, routes, shared_input);
}

template <typename T>
__device__ __forceinline__ float4 load_activation4(const T* values) {
    if constexpr (std::is_same_v<T, __half>) {
        if ((reinterpret_cast<std::uintptr_t>(values) & 3u) == 0) {
            const auto* pairs = reinterpret_cast<const __half2*>(values);
            const float2 low = __half22float2(pairs[0]);
            const float2 high = __half22float2(pairs[1]);
            return make_float4(low.x, low.y, high.x, high.y);
        }
    } else if ((reinterpret_cast<std::uintptr_t>(values) & 15u) == 0) {
        return *reinterpret_cast<const float4*>(values);
    }
    return make_float4(
        as_float(values[0]), as_float(values[1]),
        as_float(values[2]), as_float(values[3]));
}

__device__ __forceinline__ void accumulate_m2(
        const float4& weights,
        const float4& activation0,
        const float4& activation1,
        float& accumulator0,
        float& accumulator1) {
    accumulator0 = fmaf(weights.x, activation0.x, accumulator0);
    accumulator1 = fmaf(weights.x, activation1.x, accumulator1);
    accumulator0 = fmaf(weights.y, activation0.y, accumulator0);
    accumulator1 = fmaf(weights.y, activation1.y, accumulator1);
    accumulator0 = fmaf(weights.z, activation0.z, accumulator0);
    accumulator1 = fmaf(weights.z, activation1.z, accumulator1);
    accumulator0 = fmaf(weights.w, activation0.w, accumulator0);
    accumulator1 = fmaf(weights.w, activation1.w, accumulator1);
}

template <int TILE_M, bool BF16_SCALE, typename T>
__global__ void __launch_bounds__(128) fp8_128_sq_q8_kernel(
        const std::uint8_t* __restrict__ blob,
        const std::uint8_t* __restrict__ row_q,
        const std::int32_t* __restrict__ row_symbol_byte_offsets,
        const T* __restrict__ input,
        T* __restrict__ output,
        Layout layout,
        int rows) {
    constexpr int outputs_per_block = 4;
    constexpr int values_per_lane = 4;
    constexpr int columns_per_warp = 128;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int output_tiles = (layout.outputs + outputs_per_block - 1) /
        outputs_per_block;
    const int task = static_cast<int>(blockIdx.x);
    const int output_row = (task % output_tiles) * outputs_per_block + warp;
    const int first_row = (task / output_tiles) * TILE_M;
    if (output_row >= layout.outputs || first_row >= rows) {
        return;
    }

    float accumulators[TILE_M] = {};
    const int bits = static_cast<int>(row_q[output_row]);
    if (bits == 8) {
        const auto* row_symbols = blob + layout.symbols +
            static_cast<std::size_t>(row_symbol_byte_offsets[output_row]);
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(row_symbols) & 3u) == 0;
#pragma unroll 4
        for (int block_column = 0;
             block_column < layout.width;
             block_column += columns_per_warp) {
            float scale = lane == 0
                ? (BF16_SCALE
                    ? decode_bf16_block_scale(
                          blob, layout, output_row, block_column)
                    : decode_scale<false>(
                          blob, layout, output_row, block_column))
                : 0.0f;
            scale = __shfl_sync(0xffffffffu, scale, 0);
            const int column = block_column + lane * values_per_lane;
            if (column + values_per_lane <= layout.width) {
                const std::uint32_t codes = aligned
                    ? *reinterpret_cast<const std::uint32_t*>(
                          row_symbols + column)
                    : load_u32(row_symbols + column);
                const float4 weights = decode_e4m3fn4(codes);
#pragma unroll
                for (int item = 0; item < TILE_M; ++item) {
                    if (first_row + item >= rows) continue;
                    const T* input_row = input +
                        static_cast<std::size_t>(first_row + item) *
                            layout.width;
                    float4 activations;
                    if constexpr (std::is_same_v<T, __half>) {
                        const T* values = input_row + column;
                        if ((reinterpret_cast<std::uintptr_t>(values) & 3u) ==
                                0) {
                            const auto* pairs =
                                reinterpret_cast<const __half2*>(values);
                            const float2 low = __half22float2(pairs[0]);
                            const float2 high = __half22float2(pairs[1]);
                            activations = make_float4(
                                low.x, low.y, high.x, high.y);
                        } else {
                            activations = make_float4(
                                as_float(values[0]), as_float(values[1]),
                                as_float(values[2]), as_float(values[3]));
                        }
                    } else {
                        const T* values = input_row + column;
                        activations =
                            (reinterpret_cast<std::uintptr_t>(values) & 15u) ==
                                0
                            ? *reinterpret_cast<const float4*>(values)
                            : make_float4(
                                  as_float(values[0]), as_float(values[1]),
                                  as_float(values[2]), as_float(values[3]));
                    }
                    accumulators[item] = fmaf(
                        weights.x * scale,
                        activations.x,
                        accumulators[item]);
                    accumulators[item] = fmaf(
                        weights.y * scale,
                        activations.y,
                        accumulators[item]);
                    accumulators[item] = fmaf(
                        weights.z * scale,
                        activations.z,
                        accumulators[item]);
                    accumulators[item] = fmaf(
                        weights.w * scale,
                        activations.w,
                        accumulators[item]);
                }
            } else {
#pragma unroll
                for (int component = 0;
                     component < values_per_lane;
                     ++component) {
                    if (column + component >= layout.width) continue;
                    const float weight =
                        decode_e4m3fn(row_symbols[column + component]) *
                        scale;
#pragma unroll
                    for (int item = 0; item < TILE_M; ++item) {
                        if (first_row + item < rows) {
                            accumulators[item] = fmaf(
                                weight,
                                as_float(input[
                                    static_cast<std::size_t>(
                                        first_row + item) * layout.width +
                                    column + component]),
                                accumulators[item]);
                        }
                    }
                }
            }
        }
    } else {
        for (int column = lane; column < layout.width; column += 32) {
            const float weight = decode_weight<false>(
                blob, row_q, row_symbol_byte_offsets,
                layout, output_row, column);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                if (first_row + item < rows) {
                    accumulators[item] = fmaf(
                        weight,
                        as_float(input[
                            static_cast<std::size_t>(first_row + item) *
                                layout.width + column]),
                        accumulators[item]);
                }
            }
        }
    }

#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
        const float value = warp_sum(accumulators[item]);
        if (lane == 0 && first_row + item < rows) {
            output[
                static_cast<std::size_t>(first_row + item) * layout.outputs +
                output_row] = from_float<T>(value);
        }
    }
}

template <int GROUP_SIZE, bool BF16_SCALE, typename T>
__global__ void __launch_bounds__(128) fp8_128_sq_q8_m2_kernel(
        const std::uint8_t* __restrict__ blob,
        const std::uint8_t* __restrict__ row_q,
        const std::int32_t* __restrict__ row_symbol_byte_offsets,
        const T* __restrict__ input,
        T* __restrict__ output,
        Layout layout) {
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 32);
    constexpr int values_per_lane = 128 / GROUP_SIZE;
    constexpr int outputs_per_block = 128 / GROUP_SIZE;
    const int lane = static_cast<int>(threadIdx.x) & (GROUP_SIZE - 1);
    const int output_row = static_cast<int>(blockIdx.x) * outputs_per_block +
        static_cast<int>(threadIdx.x) / GROUP_SIZE;
    if (output_row >= layout.outputs) return;
    const int warp_lane = static_cast<int>(threadIdx.x) & 31;
    const unsigned group_mask = GROUP_SIZE == 32
        ? 0xffffffffu
        : 0xffffu << (warp_lane / GROUP_SIZE) * GROUP_SIZE;

    const T* input0 = input;
    const T* input1 = input + layout.width;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    if (row_q[output_row] == 8) {
        const auto* row_symbols = blob + layout.symbols +
            static_cast<std::size_t>(row_symbol_byte_offsets[output_row]);
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(row_symbols) & 3u) == 0;
#pragma unroll 4
        for (int block_column = 0;
             block_column < layout.width;
             block_column += 128) {
            float scale = lane == 0
                ? (BF16_SCALE
                    ? decode_bf16_block_scale(
                          blob, layout, output_row, block_column)
                    : decode_scale<false>(
                          blob, layout, output_row, block_column))
                : 0.0f;
            scale = __shfl_sync(group_mask, scale, 0, GROUP_SIZE);
            const int column = block_column + lane * values_per_lane;
            if (column + values_per_lane <= layout.width) {
                const std::uint32_t codes0 = aligned
                    ? *reinterpret_cast<const std::uint32_t*>(
                          row_symbols + column)
                    : load_u32(row_symbols + column);
                const float4 decoded0 = decode_e4m3fn4(codes0);
                const float4 weights0 = make_float4(
                    decoded0.x * scale, decoded0.y * scale,
                    decoded0.z * scale, decoded0.w * scale);
                const float4 activation00 = load_activation4(input0 + column);
                const float4 activation10 = load_activation4(input1 + column);
                accumulate_m2(
                    weights0, activation00, activation10,
                    accumulator0, accumulator1);
                if constexpr (values_per_lane == 8) {
                    const std::uint32_t codes1 = aligned
                        ? *reinterpret_cast<const std::uint32_t*>(
                              row_symbols + column + 4)
                        : load_u32(row_symbols + column + 4);
                    const float4 decoded1 = decode_e4m3fn4(codes1);
                    const float4 weights1 = make_float4(
                        decoded1.x * scale, decoded1.y * scale,
                        decoded1.z * scale, decoded1.w * scale);
                    const float4 activation01 =
                        load_activation4(input0 + column + 4);
                    const float4 activation11 =
                        load_activation4(input1 + column + 4);
                    accumulate_m2(
                        weights1, activation01, activation11,
                        accumulator0, accumulator1);
                }
            } else {
#pragma unroll
                for (int component = 0;
                     component < values_per_lane;
                     ++component) {
                    if (column + component >= layout.width) continue;
                    const float weight =
                        decode_e4m3fn(row_symbols[column + component]) *
                        scale;
                    accumulator0 = fmaf(
                        weight, as_float(input0[column + component]),
                        accumulator0);
                    accumulator1 = fmaf(
                        weight, as_float(input1[column + component]),
                        accumulator1);
                }
            }
        }
    } else {
        for (int column = lane;
             column < layout.width;
             column += GROUP_SIZE) {
            const float weight = decode_weight<false>(
                blob, row_q, row_symbol_byte_offsets,
                layout, output_row, column);
            accumulator0 = fmaf(
                weight, as_float(input0[column]), accumulator0);
            accumulator1 = fmaf(
                weight, as_float(input1[column]), accumulator1);
        }
    }

#pragma unroll
    for (int delta = GROUP_SIZE / 2; delta > 0; delta >>= 1) {
        accumulator0 += __shfl_down_sync(
            group_mask, accumulator0, delta, GROUP_SIZE);
        accumulator1 += __shfl_down_sync(
            group_mask, accumulator1, delta, GROUP_SIZE);
    }
    if (lane == 0) {
        output[output_row] = from_float<T>(accumulator0);
        output[layout.outputs + output_row] = from_float<T>(accumulator1);
    }
}

template <bool BF16_SCALE, typename T>
__global__ void __launch_bounds__(256) fp8_128_sq_q8_m4_kernel(
        const std::uint8_t* __restrict__ blob,
        const std::uint8_t* __restrict__ row_q,
        const std::int32_t* __restrict__ row_symbol_byte_offsets,
        const T* __restrict__ input,
        T* __restrict__ output,
        Layout layout) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int output_row = static_cast<int>(blockIdx.x) * 8 +
        (static_cast<int>(threadIdx.x) >> 5);
    if (output_row >= layout.outputs) return;

    float accumulators[4] = {};
    if (row_q[output_row] == 8) {
        const auto* row_symbols = blob + layout.symbols +
            static_cast<std::size_t>(row_symbol_byte_offsets[output_row]);
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(row_symbols) & 3u) == 0;
#pragma unroll 4
        for (int block_column = 0;
             block_column < layout.width;
             block_column += 128) {
            float scale = lane == 0
                ? (BF16_SCALE
                    ? decode_bf16_block_scale(
                          blob, layout, output_row, block_column)
                    : decode_scale<false>(
                          blob, layout, output_row, block_column))
                : 0.0f;
            scale = __shfl_sync(0xffffffffu, scale, 0);
            const int column = block_column + lane * 4;
            if (column + 4 <= layout.width) {
                const std::uint32_t codes = aligned
                    ? *reinterpret_cast<const std::uint32_t*>(
                          row_symbols + column)
                    : load_u32(row_symbols + column);
                const float4 decoded = decode_e4m3fn4(codes);
                const float4 weights = make_float4(
                    decoded.x * scale, decoded.y * scale,
                    decoded.z * scale, decoded.w * scale);
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    const float4 activation = load_activation4(
                        input + static_cast<std::size_t>(item) *
                            layout.width + column);
                    accumulators[item] = fmaf(
                        weights.x, activation.x, accumulators[item]);
                    accumulators[item] = fmaf(
                        weights.y, activation.y, accumulators[item]);
                    accumulators[item] = fmaf(
                        weights.z, activation.z, accumulators[item]);
                    accumulators[item] = fmaf(
                        weights.w, activation.w, accumulators[item]);
                }
            } else {
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    if (column + component >= layout.width) continue;
                    const float weight =
                        decode_e4m3fn(row_symbols[column + component]) *
                        scale;
#pragma unroll
                    for (int item = 0; item < 4; ++item) {
                        accumulators[item] = fmaf(
                            weight,
                            as_float(input[
                                static_cast<std::size_t>(item) *
                                    layout.width + column + component]),
                            accumulators[item]);
                    }
                }
            }
        }
    } else {
        for (int column = lane; column < layout.width; column += 32) {
            const float weight = decode_weight<false>(
                blob, row_q, row_symbol_byte_offsets,
                layout, output_row, column);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                accumulators[item] = fmaf(
                    weight,
                    as_float(input[
                        static_cast<std::size_t>(item) * layout.width +
                        column]),
                    accumulators[item]);
            }
        }
    }

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const float value = warp_sum(accumulators[item]);
        if (lane == 0) {
            output[
                static_cast<std::size_t>(item) * layout.outputs +
                output_row] = from_float<T>(value);
        }
    }
}

template <int GROUP_SIZE, bool BF16_SCALE, typename T>
__global__ void __launch_bounds__(256) fp8_128_sq_q8_m5_kernel(
        const std::uint8_t* __restrict__ blob,
        const std::uint8_t* __restrict__ row_q,
        const std::int32_t* __restrict__ row_symbol_byte_offsets,
        const T* __restrict__ input,
        T* __restrict__ output,
        Layout layout) {
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 32);
    constexpr int values_per_lane = 128 / GROUP_SIZE;
    const int outputs_per_block = static_cast<int>(blockDim.x) / GROUP_SIZE;
    const int lane = static_cast<int>(threadIdx.x) & (GROUP_SIZE - 1);
    const int output_row = static_cast<int>(blockIdx.x) * outputs_per_block +
        static_cast<int>(threadIdx.x) / GROUP_SIZE;
    if (output_row >= layout.outputs) return;
    const int warp_lane = static_cast<int>(threadIdx.x) & 31;
    const unsigned group_mask = GROUP_SIZE == 32
        ? 0xffffffffu
        : 0xffffu << (warp_lane / GROUP_SIZE) * GROUP_SIZE;

    float accumulators[5] = {};
    if (row_q[output_row] == 8) {
        const auto* row_symbols = blob + layout.symbols +
            static_cast<std::size_t>(row_symbol_byte_offsets[output_row]);
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(row_symbols) & 3u) == 0;
#pragma unroll 4
        for (int block_column = 0;
             block_column < layout.width;
             block_column += 128) {
            float scale = lane == 0
                ? (BF16_SCALE
                    ? decode_bf16_block_scale(
                          blob, layout, output_row, block_column)
                    : decode_scale<false>(
                          blob, layout, output_row, block_column))
                : 0.0f;
            scale = __shfl_sync(group_mask, scale, 0, GROUP_SIZE);
            const int column = block_column + lane * values_per_lane;
            if (column + values_per_lane <= layout.width) {
                const std::uint32_t codes0 = aligned
                    ? *reinterpret_cast<const std::uint32_t*>(
                          row_symbols + column)
                    : load_u32(row_symbols + column);
                const float4 decoded0 = decode_e4m3fn4(codes0);
                const float4 weights0 = make_float4(
                    decoded0.x * scale, decoded0.y * scale,
                    decoded0.z * scale, decoded0.w * scale);
#pragma unroll
                for (int item = 0; item < 5; ++item) {
                    const float4 activation = load_activation4(
                        input + static_cast<std::size_t>(item) *
                            layout.width + column);
                    accumulators[item] = fmaf(
                        weights0.x, activation.x, accumulators[item]);
                    accumulators[item] = fmaf(
                        weights0.y, activation.y, accumulators[item]);
                    accumulators[item] = fmaf(
                        weights0.z, activation.z, accumulators[item]);
                    accumulators[item] = fmaf(
                        weights0.w, activation.w, accumulators[item]);
                }
                if constexpr (values_per_lane == 8) {
                    const std::uint32_t codes1 = aligned
                        ? *reinterpret_cast<const std::uint32_t*>(
                              row_symbols + column + 4)
                        : load_u32(row_symbols + column + 4);
                    const float4 decoded1 = decode_e4m3fn4(codes1);
                    const float4 weights1 = make_float4(
                        decoded1.x * scale, decoded1.y * scale,
                        decoded1.z * scale, decoded1.w * scale);
#pragma unroll
                    for (int item = 0; item < 5; ++item) {
                        const float4 activation = load_activation4(
                            input + static_cast<std::size_t>(item) *
                                layout.width + column + 4);
                        accumulators[item] = fmaf(
                            weights1.x, activation.x, accumulators[item]);
                        accumulators[item] = fmaf(
                            weights1.y, activation.y, accumulators[item]);
                        accumulators[item] = fmaf(
                            weights1.z, activation.z, accumulators[item]);
                        accumulators[item] = fmaf(
                            weights1.w, activation.w, accumulators[item]);
                    }
                }
            } else {
#pragma unroll
                for (int component = 0;
                     component < values_per_lane;
                     ++component) {
                    if (column + component >= layout.width) continue;
                    const float weight =
                        decode_e4m3fn(row_symbols[column + component]) *
                        scale;
#pragma unroll
                    for (int item = 0; item < 5; ++item) {
                        accumulators[item] = fmaf(
                            weight,
                            as_float(input[
                                static_cast<std::size_t>(item) *
                                    layout.width + column + component]),
                            accumulators[item]);
                    }
                }
            }
        }
    } else {
        for (int column = lane;
             column < layout.width;
             column += GROUP_SIZE) {
            const float weight = decode_weight<false>(
                blob, row_q, row_symbol_byte_offsets,
                layout, output_row, column);
#pragma unroll
            for (int item = 0; item < 5; ++item) {
                accumulators[item] = fmaf(
                    weight,
                    as_float(input[
                        static_cast<std::size_t>(item) * layout.width +
                        column]),
                    accumulators[item]);
            }
        }
    }

#pragma unroll
    for (int item = 0; item < 5; ++item) {
#pragma unroll
        for (int delta = GROUP_SIZE / 2; delta > 0; delta >>= 1) {
            accumulators[item] += __shfl_down_sync(
                group_mask, accumulators[item], delta, GROUP_SIZE);
        }
        if (lane == 0) {
            output[
                static_cast<std::size_t>(item) * layout.outputs +
                output_row] = from_float<T>(accumulators[item]);
        }
    }
}

void validate_metadata(
        const mfq_tensor_backend::Tensor& blob,
        const mfq_tensor_backend::Tensor& row_q,
        const mfq_tensor_backend::Tensor& row_symbol_byte_offsets,
        const Layout& layout,
        bool mxfp8) {
    MFQ_RUNTIME_CHECK(
        blob.is_cuda() && blob.scalar_type() == mfq_tensor_backend::kUInt8 &&
        blob.dim() == 1 && blob.is_contiguous(),
        "FP8-SQ blob must be contiguous rank-1 CUDA uint8");
    MFQ_RUNTIME_CHECK(
        row_q.is_cuda() && row_q.get_device() == blob.get_device() &&
        row_q.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q.dim() == 1 && row_q.is_contiguous() &&
        row_q.numel() == layout.outputs &&
        row_symbol_byte_offsets.is_cuda() &&
        row_symbol_byte_offsets.get_device() == blob.get_device() &&
        row_symbol_byte_offsets.scalar_type() == mfq_tensor_backend::kInt32 &&
        row_symbol_byte_offsets.dim() == 1 &&
        row_symbol_byte_offsets.is_contiguous() &&
        row_symbol_byte_offsets.numel() == layout.outputs,
        "FP8-SQ row metadata must be matching CUDA arrays");
    MFQ_RUNTIME_CHECK(
        layout.outputs > 0 && layout.width > 0 &&
        layout.block_rows > 0 && layout.block_columns > 0 &&
        layout.scale_rows == (layout.outputs + layout.block_rows - 1) /
            layout.block_rows &&
        layout.scale_columns == (layout.width + layout.block_columns - 1) /
            layout.block_columns &&
        layout.palettes < layout.symbols && layout.symbols <= layout.scales &&
        layout.scales < static_cast<std::size_t>(blob.numel()),
        "FP8-SQ layout metadata is inconsistent");
    const std::size_t scale_count =
        static_cast<std::size_t>(layout.scale_rows) * layout.scale_columns;
    const std::size_t scale_itemsize = mxfp8 || layout.scale_kind == 1
        ? 1
        : layout.scale_kind == 4 ? 4 : 2;
    MFQ_RUNTIME_CHECK(
        scale_count <=
            (static_cast<std::size_t>(blob.numel()) - layout.scales) /
                scale_itemsize,
        "FP8-SQ scale payload exceeds its blob");
    if (mxfp8) {
        MFQ_RUNTIME_CHECK(
            layout.scale_kind == 1 &&
            ((layout.block_rows == 1 && layout.block_columns == 32) ||
             (layout.block_rows == 32 && layout.block_columns == 32) ||
             (layout.block_rows == 128 && layout.block_columns == 128)),
            "MXFP8-SQ scale contract is invalid");
    } else {
        MFQ_RUNTIME_CHECK(
            layout.block_rows == 128 && layout.block_columns == 128 &&
            (layout.scale_kind == 2 || layout.scale_kind == 3 ||
             layout.scale_kind == 4),
            "FP8-128SQ scale contract is invalid");
    }
}

Layout make_layout(
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t block_rows,
        std::int64_t block_columns,
        std::int64_t scale_rows,
        std::int64_t scale_columns,
        std::int64_t scale_kind,
        std::int64_t palettes,
        std::int64_t symbols,
        std::int64_t scales) {
    MFQ_RUNTIME_CHECK(
        outputs > 0 && outputs <= std::numeric_limits<int>::max() &&
        width > 0 && width <= std::numeric_limits<int>::max() &&
        block_rows > 0 && block_rows <= std::numeric_limits<int>::max() &&
        block_columns > 0 && block_columns <= std::numeric_limits<int>::max() &&
        scale_rows > 0 && scale_rows <= std::numeric_limits<int>::max() &&
        scale_columns > 0 && scale_columns <= std::numeric_limits<int>::max() &&
        palettes >= 0 && symbols >= 0 && scales >= 0,
        "FP8-SQ host layout exceeds CUDA limits");
    return {
        static_cast<int>(outputs),
        static_cast<int>(width),
        static_cast<int>(block_rows),
        static_cast<int>(block_columns),
        static_cast<int>(scale_rows),
        static_cast<int>(scale_columns),
        static_cast<int>(scale_kind),
        static_cast<std::size_t>(palettes),
        static_cast<std::size_t>(symbols),
        static_cast<std::size_t>(scales),
    };
}

template <bool MXFP8, typename T>
void launch_dequant(
        const mfq_tensor_backend::Tensor& blob,
        const mfq_tensor_backend::Tensor& row_q,
        const mfq_tensor_backend::Tensor& row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor& output,
        Layout layout,
        cudaStream_t stream) {
    const auto count = static_cast<std::int64_t>(layout.outputs) * layout.width;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        (count + 255) / 256, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_dequant_kernel<<<blocks, 256, 0, stream>>>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            reinterpret_cast<T*>(output.data_ptr()),
            layout);
    } else {
        fp8_128_sq_dequant_kernel<<<blocks, 256, 0, stream>>>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            reinterpret_cast<T*>(output.data_ptr()),
            layout);
    }
}

template <bool MXFP8, int TILE_M, typename T>
void launch_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        cudaStream_t stream) {
    const auto tasks =
        (static_cast<std::int64_t>(layout.outputs) + 3) / 4 *
        ((static_cast<std::int64_t>(rows) + TILE_M - 1) / TILE_M);
    const int blocks = static_cast<int>(std::min<std::int64_t>(tasks, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_mmq_kernel<TILE_M><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
            nullptr, nullptr, 0, 0, layout.outputs, 1, false);
    } else {
        fp8_128_sq_mmq_kernel<TILE_M><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
            nullptr, nullptr, 0, 0, layout.outputs, 1, false);
    }
}

template <int TILE_M, typename T>
void launch_fp8_128_sq_q8(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        cudaStream_t stream) {
    const auto output_tiles =
        (static_cast<std::int64_t>(layout.outputs) + 3) / 4;
    const auto row_tiles =
        (static_cast<std::int64_t>(rows) + TILE_M - 1) / TILE_M;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        output_tiles * row_tiles, 65535));
    if (layout.scale_kind == 2) {
        fp8_128_sq_q8_kernel<TILE_M, true><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout, rows);
    } else {
        fp8_128_sq_q8_kernel<TILE_M, false><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout, rows);
    }
}

template <typename T>
void launch_fp8_128_sq_q8_m2(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        cudaStream_t stream) {
    // M=2 half-warps win for balanced/tall matrices; very wide output
    // projections need full warps to hide HBM latency.
    const bool half_warp = static_cast<std::int64_t>(layout.outputs) <=
        2 * static_cast<std::int64_t>(layout.width);
    const int outputs_per_block = half_warp ? 8 : 4;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        (static_cast<std::int64_t>(layout.outputs) + outputs_per_block - 1) /
            outputs_per_block,
        65535));
    if (layout.scale_kind == 2) {
        if (half_warp) {
            fp8_128_sq_q8_m2_kernel<16, true><<<blocks, 128, 0, stream>>>(
                blob, row_q, row_symbol_byte_offsets,
                input, output, layout);
        } else {
            fp8_128_sq_q8_m2_kernel<32, true><<<blocks, 128, 0, stream>>>(
                blob, row_q, row_symbol_byte_offsets,
                input, output, layout);
        }
    } else if (half_warp) {
        fp8_128_sq_q8_m2_kernel<16, false><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout);
    } else {
        fp8_128_sq_q8_m2_kernel<32, false><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout);
    }
}

template <typename T>
void launch_fp8_128_sq_q8_m4(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        cudaStream_t stream) {
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        (static_cast<std::int64_t>(layout.outputs) + 7) / 8, 65535));
    if (layout.scale_kind == 2) {
        fp8_128_sq_q8_m4_kernel<true><<<blocks, 256, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout);
    } else {
        fp8_128_sq_q8_m4_kernel<false><<<blocks, 256, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout);
    }
}

template <typename T>
void launch_fp8_128_sq_q8_m5(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        cudaStream_t stream) {
    const bool half_warp = layout.outputs >= layout.width;
    constexpr int outputs_per_block = 8;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        (static_cast<std::int64_t>(layout.outputs) + outputs_per_block - 1) /
            outputs_per_block,
        65535));
    if (layout.scale_kind == 2) {
        if (half_warp) {
            fp8_128_sq_q8_m5_kernel<16, true><<<blocks, 128, 0, stream>>>(
                blob, row_q, row_symbol_byte_offsets,
                input, output, layout);
        } else {
            fp8_128_sq_q8_m5_kernel<32, true><<<blocks, 256, 0, stream>>>(
                blob, row_q, row_symbol_byte_offsets,
                input, output, layout);
        }
    } else if (half_warp) {
        fp8_128_sq_q8_m5_kernel<16, false><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout);
    } else {
        fp8_128_sq_q8_m5_kernel<32, false><<<blocks, 256, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets,
            input, output, layout);
    }
}

template <bool MXFP8, typename T>
void dispatch_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        cudaStream_t stream) {
#define MFQ_FP8_SQ_M_CASE(M) \
    case M: \
        if constexpr (MXFP8) { \
            launch_mmq<true, M>( \
                blob, row_q, row_symbol_byte_offsets, input, output, \
                layout, rows, stream); \
        } else { \
            launch_fp8_128_sq_q8<M>( \
                blob, row_q, row_symbol_byte_offsets, input, output, \
                layout, rows, stream); \
        } \
        break
    switch (rows) {
        MFQ_FP8_SQ_M_CASE(1);
        case 2:
            if constexpr (MXFP8) {
                launch_mmq<true, 2>(
                    blob, row_q, row_symbol_byte_offsets, input, output,
                    layout, rows, stream);
            } else {
                launch_fp8_128_sq_q8_m2(
                    blob, row_q, row_symbol_byte_offsets, input, output,
                    layout, stream);
            }
            break;
        MFQ_FP8_SQ_M_CASE(3);
        case 4:
            if constexpr (MXFP8) {
                launch_mmq<true, 4>(
                    blob, row_q, row_symbol_byte_offsets, input, output,
                    layout, rows, stream);
            } else {
                launch_fp8_128_sq_q8_m4(
                    blob, row_q, row_symbol_byte_offsets, input, output,
                    layout, stream);
            }
            break;
        case 5:
            if constexpr (MXFP8) {
                launch_mmq<true, 5>(
                    blob, row_q, row_symbol_byte_offsets, input, output,
                    layout, rows, stream);
            } else {
                launch_fp8_128_sq_q8_m5(
                    blob, row_q, row_symbol_byte_offsets, input, output,
                    layout, stream);
            }
            break;
        MFQ_FP8_SQ_M_CASE(6);
        default:
            launch_mmq<MXFP8, 8>(
                blob, row_q, row_symbol_byte_offsets,
                input, output, layout, rows, stream);
            break;
    }
#undef MFQ_FP8_SQ_M_CASE
}

template <bool MXFP8>
void launch_routed(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const __half* input,
        __half* output,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        Layout layout,
        int route_count,
        int routes,
        int global_experts,
        int local_experts,
        int out_per_expert,
        bool shared_input,
        cudaStream_t stream) {
    const auto tasks = static_cast<std::int64_t>(route_count) *
        ((static_cast<std::int64_t>(out_per_expert) + 3) / 4);
    const int blocks = static_cast<int>(std::min<std::int64_t>(tasks, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_mmq_kernel<1, __half, true><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout,
            route_count, expert_ids, expert_local, global_experts,
            local_experts, out_per_expert, routes, shared_input);
    } else {
        fp8_128_sq_mmq_kernel<1, __half, true><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout,
            route_count, expert_ids, expert_local, global_experts,
            local_experts, out_per_expert, routes, shared_input);
    }
}

template <bool MXFP8>
mfq_tensor_backend::Tensor dequant(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        Layout layout,
        bool fp32) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    const MfqCudaGuard guard(blob.device());
    auto output = mfq_tensor_backend::empty(
        {layout.outputs, layout.width},
        blob.options().dtype(
            fp32 ? mfq_tensor_backend::kFloat32 : mfq_tensor_backend::kFloat16));
    const auto stream = mfq_current_cuda_stream();
    if (fp32) {
        launch_dequant<MXFP8, float>(
            blob, row_q, row_symbol_byte_offsets, output, layout, stream);
    } else {
        launch_dequant<MXFP8, __half>(
            blob, row_q, row_symbol_byte_offsets, output, layout, stream);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void launch_dense_gemm_nt(
        const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight,
        mfq_tensor_backend::Tensor& output,
        cudaStream_t stream) {
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_RUNTIME_CHECK(
        cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "FP8-SQ cublasSetStream failed");

    // Row-major Y[M,N] = X[M,K] * W[N,K]^T maps to column-major
    // Y^T[N,M] = W[N,K] * X^T[K,M].
    if (input.scalar_type() == mfq_tensor_backend::kFloat16) {
        const __half alpha = __float2half(1.0f);
        const __half beta = __float2half(0.0f);
        MFQ_RUNTIME_CHECK(
            cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                outputs, rows, width,
                &alpha,
                weight.data_ptr<mfq_half>(), CUDA_R_16F, width,
                input.data_ptr<mfq_half>(), CUDA_R_16F, width,
                &beta,
                output.data_ptr<mfq_half>(), CUDA_R_16F, outputs,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS,
            "FP8-SQ FP16 GEMM failed");
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_RUNTIME_CHECK(
        cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            outputs, rows, width,
            &alpha,
            weight.data_ptr<float>(), CUDA_R_32F, width,
            input.data_ptr<float>(), CUDA_R_32F, width,
            &beta,
            output.data_ptr<float>(), CUDA_R_32F, outputs,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
        "FP8-SQ FP32 GEMM failed");
}

template <bool MXFP8>
mfq_tensor_backend::Tensor matmul(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() &&
        input.size(1) == layout.width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == mfq_tensor_backend::kFloat16 ||
         input.scalar_type() == mfq_tensor_backend::kFloat32),
        "FP8-SQ activation must be contiguous rank-2 CUDA FP16/FP32");
    const MfqCudaGuard guard(blob.device());
    mfq_tensor_backend::Tensor output = mfq_tensor_backend::empty(
        {input.size(0), layout.outputs}, input.options());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) return output;
    const auto stream = mfq_current_cuda_stream();
    if (rows > kDirectPackedMaxRows) {
        // The decoded matrix is a transient workspace owned by this call.  It
        // is never cached or attached to the packed weight.
        auto weight = dequant<MXFP8>(
            blob, row_q, row_symbol_byte_offsets, layout,
            input.scalar_type() == mfq_tensor_backend::kFloat32);
        launch_dense_gemm_nt(input, weight, output, stream);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    if (input.scalar_type() == mfq_tensor_backend::kFloat32) {
        dispatch_mmq<MXFP8>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            layout, rows, stream);
    } else {
        dispatch_mmq<MXFP8>(
            blob.data_ptr<std::uint8_t>(),
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            reinterpret_cast<const __half*>(input.data_ptr<mfq_half>()),
            reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
            layout, rows, stream);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool MXFP8>
void routed_matmul(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        std::int64_t n_experts,
        std::int64_t local_experts,
        std::int64_t out_per_expert,
        Layout layout,
        mfq_tensor_backend::Tensor output) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    MFQ_RUNTIME_CHECK(
        layout.outputs == local_experts * out_per_expert &&
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.is_contiguous() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        (input.dim() == 2 || input.dim() == 3) && input.size(-1) == layout.width &&
        expert_ids.is_cuda() && expert_ids.get_device() == blob.get_device() &&
        expert_ids.is_contiguous() &&
        expert_ids.scalar_type() == mfq_tensor_backend::kInt32 &&
        expert_ids.dim() == 2 && expert_ids.size(0) == input.size(0) &&
        expert_local.is_cuda() && expert_local.get_device() == blob.get_device() &&
        expert_local.is_contiguous() &&
        expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
        expert_local.dim() == 1 && expert_local.numel() == n_experts &&
        output.is_cuda() && output.get_device() == blob.get_device() &&
        output.is_contiguous() && output.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output.sizes() == mfq_tensor_backend::IntArrayRef(
            {expert_ids.size(0), expert_ids.size(1), out_per_expert}) &&
        (input.dim() == 2 || input.size(1) == expert_ids.size(1)) &&
        n_experts > 0 && n_experts <= std::numeric_limits<int>::max() &&
        local_experts > 0 && local_experts <= std::numeric_limits<int>::max() &&
        out_per_expert > 0 && out_per_expert <= std::numeric_limits<int>::max(),
        "FP8-SQ routed matmul requires matching contiguous CUDA FP16 tensors");
    MFQ_RUNTIME_CHECK(
        expert_ids.numel() <= std::numeric_limits<int>::max(),
        "FP8-SQ route count exceeds CUDA limits");
    if (expert_ids.numel() == 0) return;
    const MfqCudaGuard guard(blob.device());
    launch_routed<MXFP8>(
        blob.data_ptr<std::uint8_t>(),
        row_q.data_ptr<std::uint8_t>(),
        row_symbol_byte_offsets.data_ptr<std::int32_t>(),
        reinterpret_cast<const __half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
        expert_ids.data_ptr<std::int32_t>(),
        expert_local.data_ptr<std::int32_t>(),
        layout,
        static_cast<int>(expert_ids.numel()),
        static_cast<int>(expert_ids.size(1)),
        static_cast<int>(n_experts),
        static_cast<int>(local_experts),
        static_cast<int>(out_per_expert),
        input.dim() == 2,
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool MXFP8>
mfq_tensor_backend::Tensor backward_input(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == layout.outputs &&
        output_gradient.size(0) <= std::numeric_limits<int>::max() &&
        (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kBFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kFloat32),
        "FP8-SQ backward requires contiguous CUDA FP16/BF16/FP32 [M,N]");
    const MfqCudaGuard guard(blob.device());
    auto result = mfq_tensor_backend::empty(
        {output_gradient.size(0), layout.width}, output_gradient.options());
    if (output_gradient.size(0) == 0) return result;
    auto weight = dequant<MXFP8>(
        blob, row_q, row_symbol_byte_offsets, layout, false);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        static_cast<int>(output_gradient.size(0)),
        layout.outputs, layout.width, mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

} // namespace

mfq_tensor_backend::Tensor mxfp8_sq_dequant_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset, bool fp32) {
    return dequant<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset),
        fp32);
}

mfq_tensor_backend::Tensor fp8_128_sq_dequant_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset, bool fp32) {
    return dequant<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset),
        fp32);
}

mfq_tensor_backend::Tensor mxfp8_sq_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return matmul<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset));
}

mfq_tensor_backend::Tensor fp8_128_sq_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return matmul<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset));
}

mfq_tensor_backend::Tensor mxfp8_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset));
}

mfq_tensor_backend::Tensor fp8_128_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset));
}

void mxfp8_sq_moe_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        mfq_tensor_backend::Tensor output) {
    routed_matmul<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        std::move(expert_ids), std::move(expert_local),
        n_experts, local_experts, out_per_expert,
        make_layout(local_experts * out_per_expert, width,
                    block_rows, block_columns, scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset),
        std::move(output));
}

void fp8_128_sq_moe_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        mfq_tensor_backend::Tensor output) {
    routed_matmul<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        std::move(expert_ids), std::move(expert_local),
        n_experts, local_experts, out_per_expert,
        make_layout(local_experts * out_per_expert, width, 128, 128,
                    (local_experts * out_per_expert + 127) / 128,
                    (width + 127) / 128, scale_kind,
                    palettes_offset, symbols_offset, scales_offset),
        std::move(output));
}
