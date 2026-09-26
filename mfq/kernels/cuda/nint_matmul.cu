// Canonical CUDA execution for NINT.
//
// Every ordinary NINT tensor is normalized at the loader boundary to one
// row-major bitstream plus per-neuron q metadata.  Uniform presets and
// adaptive q/k allocations therefore execute through exactly the same
// kernels.  Large-M execution uses the same row decoder followed by GEMM.

// NINT8-0 is a separate symmetric format.  It retains one small-M packed
// kernel; larger batches use its row decoder followed by GEMM.


#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "async_copy.cuh"
#include "glu.cuh"
#include "mfq_tensor_backend.h"
#include "packed_backward.cuh"


#define MFQ_CUBLAS_CHECK(expression) \
    MFQ_RUNTIME_CHECK( \
        (expression) == CUBLAS_STATUS_SUCCESS, \
        "cuBLAS call failed: ", #expression)


namespace {


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
    uint64_t packed = static_cast<uint64_t>(stream[byte]);
    if (required_bits > 8) {
        packed |= static_cast<uint64_t>(stream[byte + 1]) << 8;
    }
    if (required_bits > 16) {
        packed |= static_cast<uint64_t>(stream[byte + 2]) << 16;
    }
    if (required_bits > 24) {
        packed |= static_cast<uint64_t>(stream[byte + 3]) << 24;
    }
    if (required_bits > 32) {
        packed |= static_cast<uint64_t>(stream[byte + 4]) << 32;
    }
    packed >>= shift;
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


__device__ __forceinline__ int unpack_nint4_codes4(
        uint32_t packed) {
    return static_cast<int>(
        (packed & 0x000fu) |
        ((packed & 0x00f0u) << 4) |
        ((packed & 0x0f00u) << 8) |
        ((packed & 0xf000u) << 12));
}


__device__ __forceinline__ int load_i8x4(const int8_t * source) {
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(source);
    const uint32_t packed = static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
    return static_cast<int>(packed);
}


__global__ void nint8_one_quantize_reconstruct_kernel(
        const __half * __restrict__ input,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scale_output,
        __half * __restrict__ sum_output,
        __half * __restrict__ reconstructed,
        int rows,
        int width,
        int groups) {
    const int group_index = static_cast<int>(blockIdx.x);
    if (group_index >= rows * groups) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int row = group_index / groups;
    const int group = group_index - row * groups;
    const int column = group * 32 + lane;
    const float value = column < width
        ? __half2float(input[static_cast<size_t>(row) * width + column])
        : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_xor_sync(0xffffffffu, maximum, offset));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    int code = static_cast<int>(roundf(value * inverse));
    code = max(-127, min(127, code));
    quantized[static_cast<size_t>(group_index) * 32 + lane] =
        static_cast<int8_t>(code);

    int sum = code;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    const __half stored_scale = __float2half_rn(scale);
    if (lane == 0) {
        scale_output[group_index] = stored_scale;
        sum_output[group_index] = __float2half_rn(
            static_cast<float>(sum) * scale);
    }
    if (column < width) {
        reconstructed[static_cast<size_t>(row) * width + column] =
            __float2half_rn(
                static_cast<float>(code) * __half2float(stored_scale));
    }
}


__global__ void nint_decode_rows_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        __half * __restrict__ output,
        int rows,
        int groups,
        int group_size,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int row = static_cast<int>(index / width);
        const int group = column / group_size;
        const size_t metadata = static_cast<size_t>(row) * groups + group;
        const int bits = static_cast<int>(row_q_bits[row]);
        const uint8_t code = unpack_nint_code(
            bitstream,
            static_cast<uint64_t>(row_q_bit_offsets[row]),
            column,
            bits);
        const float scale = neuron_scale[row] *
            static_cast<float>(subgroup_scale[metadata]);
        const float minimum = neuron_minimum[row] *
            static_cast<float>(subgroup_minimum[metadata]);
        output[index] = __float2half_rn(
            scale * static_cast<float>(code) - minimum);
    }
}


// Runtime group size deliberately avoids a kernel family per quantizer preset.
__global__ void __launch_bounds__(64) nint_quantize_activation_kernel(
        const __half * __restrict__ input,
        const __half * __restrict__ gate,
        int8_t * __restrict__ quantized,
        float * __restrict__ scale_output,
        int rows,
        int real_width,
        int padded_width,
        int groups,
        int group_size,
        int activation_mode) {
    const int row = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int column = group * group_size + lane;
    const bool valid = lane < group_size && column < real_width;
    float value = valid
        ? __half2float(input[static_cast<size_t>(row) * real_width + column])
        : 0.0f;
    if (valid && gate != nullptr) {
        const float gate_value = __half2float(
            gate[static_cast<size_t>(row) * real_width + column]);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        value *= activation_mode == 1 ? sigmoid : gate_value * sigmoid;
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    __shared__ float warp_maxima[2];
    if ((lane & 31) == 0) {
        warp_maxima[lane >> 5] = maximum;
    }
    __syncthreads();
    if (lane < 32) {
        const int active_warps = (blockDim.x + 31) / 32;
        maximum = lane < active_warps ? warp_maxima[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, offset));
        }
    }
    __shared__ float group_scale;
    if (lane == 0) {
        group_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scale_output[static_cast<size_t>(row) * groups + group] = group_scale;
    }
    __syncthreads();
    if (lane < group_size) {
        int code = 0;
        if (valid) {
            code = static_cast<int>(roundf(value / group_scale));
            code = max(-127, min(127, code));
        }
        quantized[static_cast<size_t>(row) * padded_width + column] =
            static_cast<int8_t>(code);
    }
}


constexpr int kNintPrefillTileN = 64;
constexpr int kNintPrefillTileK = 64;
constexpr int kNintPrefillStrideK = 72;
constexpr int kNintPrefillWarps = 8;


template <bool AsyncCopy, int TileM>
__device__ __forceinline__ void nint_prefill_load_activation_tile(
        const __half * __restrict__ input,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ activation_tile,
        int first,
        int last,
        int routes,
        int input_width,
        int column_base,
        bool routed_input,
        int thread) {
    constexpr int vector_width = 8;
    constexpr int vectors_per_row = kNintPrefillTileK / vector_width;
    constexpr int vectors = TileM * vectors_per_row;
    for (int index = thread; index < vectors;
         index += kNintPrefillWarps * 32) {
        const int matrix_row = index / vectors_per_row;
        const int vector = index - matrix_row * vectors_per_row;
        const int column = column_base + vector * vector_width;
        const int compact = first + matrix_row;
        int source_row = -1;
        if (compact < last) {
            const int pair = ids_dst[compact];
            source_row = routed_input ? pair : pair / routes;
        }
        __half * destination = activation_tile +
            matrix_row * kNintPrefillStrideK + vector * vector_width;
        if (source_row < 0 || column >= input_width) {
            *reinterpret_cast<int4 *>(destination) = make_int4(0, 0, 0, 0);
        } else if ((input_width & (vector_width - 1)) == 0 &&
                column + vector_width <= input_width) {
            const __half * source = input +
                static_cast<size_t>(source_row) * input_width + column;
            if constexpr (AsyncCopy) {
                mfq::cuda_detail::copy_16_async(destination, source);
            } else {
                *reinterpret_cast<int4 *>(destination) =
                    *reinterpret_cast<const int4 *>(source);
            }
        } else {
#pragma unroll
            for (int element = 0; element < vector_width; ++element) {
                destination[element] = column + element < input_width
                    ? input[static_cast<size_t>(source_row) * input_width +
                        column + element]
                    : __float2half_rn(0.0f);
            }
        }
    }
}


__device__ __forceinline__ void nint_prefill_load_weight_tile(
        const uint8_t * __restrict__ expert_stream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        __half * __restrict__ weight_tile,
        int local_expert,
        int output_rows,
        int result_rows,
        int groups,
        int group_size,
        int input_width,
        int result_row_base,
        int column_base,
        int projection,
        int thread) {
    constexpr int values_per_vector = 8;
    constexpr int vectors_per_row =
        kNintPrefillTileK / values_per_vector;
    constexpr int vectors = kNintPrefillTileN * vectors_per_row;
    for (int index = thread; index < vectors;
         index += kNintPrefillWarps * 32) {
        const int tile_row = index / vectors_per_row;
        const int vector = index - tile_row * vectors_per_row;
        const int result_row = result_row_base + tile_row;
        const int output_row = result_row + projection * result_rows;
        const int column = column_base + vector * values_per_vector;
        __half * destination = weight_tile +
            tile_row * kNintPrefillStrideK + vector * values_per_vector;
        const bool valid_row = result_row < result_rows &&
            output_row < output_rows;
        const int weight_row = valid_row
            ? local_expert * output_rows + output_row
            : 0;
        const int row_lane = thread & 7;
        int bits = valid_row && row_lane == 0
            ? static_cast<int>(row_q_bits[weight_row])
            : 0;
        uint64_t row_bit_offset = valid_row && row_lane == 0
            ? static_cast<uint64_t>(row_q_bit_offsets[weight_row])
            : 0u;
        float outer_scale = valid_row && row_lane == 0
            ? neuron_scale[weight_row]
            : 0.0f;
        float outer_minimum = valid_row && row_lane == 0
            ? neuron_minimum[weight_row]
            : 0.0f;
        const uint32_t offset_low = __shfl_sync(
            0xffffffffu, static_cast<uint32_t>(row_bit_offset), 0, 8);
        const uint32_t offset_high = __shfl_sync(
            0xffffffffu, static_cast<uint32_t>(row_bit_offset >> 32), 0, 8);
        bits = __shfl_sync(0xffffffffu, bits, 0, 8);
        outer_scale = __shfl_sync(0xffffffffu, outer_scale, 0, 8);
        outer_minimum = __shfl_sync(0xffffffffu, outer_minimum, 0, 8);
        row_bit_offset = static_cast<uint64_t>(offset_low) |
            (static_cast<uint64_t>(offset_high) << 32);

        const int valid_values = valid_row && column < input_width
            ? min(values_per_vector, input_width - column)
            : 0;
        const uint64_t codes = valid_values > 0
            ? unpack_nint_codes8_packed(
                expert_stream,
                row_bit_offset + static_cast<uint64_t>(column) * bits,
                bits)
            : 0u;
        const uint32_t code_mask = bits > 0 ? (1u << bits) - 1u : 0u;
        const size_t metadata_base = static_cast<size_t>(weight_row) * groups;

        // The eight lanes serving one row cooperatively load every subgroup
        // touched by this 64-wide tile exactly once.  Standard NINT groups
        // are at least eight values, so at most eight groups are present.
        // Smaller experimental groups retain this kernel and load metadata
        // directly while decoding each value.
        const int physical_last_column = groups * group_size - 1;
        const int tile_first_group = column_base / group_size;
        const int tile_last_column = column_base + min(
            kNintPrefillTileK - 1,
            physical_last_column - column_base);
        const int tile_last_group = tile_last_column / group_size;
        const int first_group = min(column, physical_last_column) / group_size;
        const int last_group = valid_values > 0
            ? (column + valid_values - 1) / group_size
            : first_group;
        float first_scale = 0.0f;
        float first_minimum = 0.0f;
        float last_scale = 0.0f;
        float last_minimum = 0.0f;
        if (group_size >= values_per_vector) {
            const int source_group = tile_first_group + row_lane;
            float source_scale = 0.0f;
            float source_minimum = 0.0f;
            if (valid_row && source_group <= tile_last_group) {
                const size_t metadata = metadata_base + source_group;
                source_scale = outer_scale *
                    static_cast<float>(subgroup_scale[metadata]);
                source_minimum = outer_minimum *
                    static_cast<float>(subgroup_minimum[metadata]);
            }
            const int first_source_lane = first_group - tile_first_group;
            const int last_source_lane = last_group - tile_first_group;
            first_scale = __shfl_sync(
                0xffffffffu, source_scale, first_source_lane, 8);
            first_minimum = __shfl_sync(
                0xffffffffu, source_minimum, first_source_lane, 8);
            last_scale = __shfl_sync(
                0xffffffffu, source_scale, last_source_lane, 8);
            last_minimum = __shfl_sync(
                0xffffffffu, source_minimum, last_source_lane, 8);
        }

#pragma unroll
        for (int element = 0; element < values_per_vector; element += 2) {
            float decoded[2] = {0.0f, 0.0f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int local = element + half;
                if (local < valid_values) {
                    const int value_column = column + local;
                    const int group = value_column / group_size;
                    float scale = first_scale;
                    float minimum = first_minimum;
                    if (group_size < values_per_vector) {
                        const size_t metadata = metadata_base + group;
                        scale = outer_scale *
                            static_cast<float>(subgroup_scale[metadata]);
                        minimum = outer_minimum *
                            static_cast<float>(subgroup_minimum[metadata]);
                    } else if (group != first_group) {
                        scale = last_scale;
                        minimum = last_minimum;
                    }
                    const int code = static_cast<int>(
                        (codes >> (local * bits)) & code_mask);
                    decoded[half] =
                        scale * static_cast<float>(code) - minimum;
                }
            }
            reinterpret_cast<__half2 *>(destination)[element / 2] =
                __floats2half2_rn(decoded[0], decoded[1]);
        }
    }
}


// FusedGlu is an epilogue contract, not a q/k specialization.  Both variants
// consume the same heterogeneous row metadata and execute the same tiled
// decoder.  Keeping the plain projection separate prevents it from reserving
// a second accumulator set that it can never use.
template <int TileM, bool AsyncActivation, bool FusedGlu>
__global__ void __launch_bounds__(256, 1) nint_matmul_tiled_prefill_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        int routes,
        int experts,
        int output_rows,
        int groups,
        int group_size,
        int input_width,
        int q_expert_stride,
        int pool_phase,
        int epilogue_mode,
        bool routed_input) {
    static_assert(
        TileM == 16 || TileM == 32 || TileM == 64 || TileM == 128);
    constexpr int matrix_tiles_m = TileM / 16;
    constexpr int matrix_tiles_n = kNintPrefillTileN / 16;
    constexpr int accumulators_per_warp = (matrix_tiles_m + 1) / 2;
    constexpr int operand_halves =
        (kNintPrefillTileN + TileM) * kNintPrefillStrideK;
    constexpr int output_floats =
        2 * kNintPrefillWarps * 16 * 16;
    constexpr int operand_bytes = operand_halves * sizeof(__half);
    constexpr int output_bytes = output_floats * sizeof(float);
    constexpr int shared_bytes =
        operand_bytes > output_bytes ? operand_bytes : output_bytes;
    __shared__ __align__(16) uint8_t storage[shared_bytes];
    __half * weight_tile = reinterpret_cast<__half *>(storage);
    __half * activation_tile = weight_tile +
        kNintPrefillTileN * kNintPrefillStrideK;

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int thread = warp * 32 + lane;
    const int warp_m0 = warp / matrix_tiles_n;
    const int warp_n = warp - warp_m0 * matrix_tiles_n;
    const int result_rows = FusedGlu ? output_rows / 2 : output_rows;
    constexpr int projections = FusedGlu ? 2 : 1;
    const int output_tiles =
        (result_rows + kNintPrefillTileN - 1) / kNintPrefillTileN;
    const int total_tiles = tile_bounds[experts];
    const int64_t total_tasks =
        static_cast<int64_t>(total_tiles) * output_tiles;

    using FragmentA = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, 16, 16, 16,
        __half, nvcuda::wmma::row_major>;
    using FragmentB = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, 16, 16, 16,
        __half, nvcuda::wmma::col_major>;
    using FragmentC = nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, 16, 16, 16, float>;

    const int64_t task_shift = total_tasks == 0
        ? 0
        : (static_cast<int64_t>(pool_phase) * 17 * output_tiles) %
            total_tasks;
    for (int64_t logical_task = blockIdx.x;
         logical_task < total_tasks;
         logical_task += gridDim.x) {
        const int64_t task = (logical_task + task_shift) % total_tasks;
        const int tile = static_cast<int>(task / output_tiles);
        const int output_tile = static_cast<int>(
            task - static_cast<int64_t>(tile) * output_tiles);
        const int expert = tile_experts[tile];
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TileM;
        const int last = min(first + TileM, expert_bounds[expert + 1]);
        const int result_row_base = output_tile * kNintPrefillTileN;
        const uint8_t * expert_stream = bitstream +
            static_cast<size_t>(local_expert) * q_expert_stride;

        FragmentC accumulators[projections][accumulators_per_warp];
#pragma unroll
        for (int projection = 0; projection < projections; ++projection) {
#pragma unroll
            for (int accumulator = 0;
                 accumulator < accumulators_per_warp; ++accumulator) {
                nvcuda::wmma::fill_fragment(
                    accumulators[projection][accumulator], 0.0f);
            }
        }

        for (int column_base = 0; column_base < input_width;
             column_base += kNintPrefillTileK) {
            nint_prefill_load_activation_tile<AsyncActivation, TileM>(
                input, ids_dst, activation_tile, first, last, routes,
                input_width, column_base, routed_input, thread);
            if constexpr (AsyncActivation) {
                mfq::cuda_detail::copy_async_commit();
            }
            nint_prefill_load_weight_tile(
                expert_stream, row_q_bits, row_q_bit_offsets,
                subgroup_scale, subgroup_minimum, neuron_scale,
                neuron_minimum, weight_tile, local_expert, output_rows,
                result_rows, groups, group_size, input_width,
                result_row_base, column_base, 0, thread);
            if constexpr (AsyncActivation) {
                mfq::cuda_detail::copy_async_wait();
            }
            __syncthreads();

            const bool active_warp = warp_m0 < matrix_tiles_m;
#pragma unroll
            for (int k_local = 0; k_local < kNintPrefillTileK;
                 k_local += 16) {
                if (active_warp && column_base + k_local < input_width) {
                    FragmentB weight_fragment;
                    nvcuda::wmma::load_matrix_sync(
                        weight_fragment,
                        weight_tile + warp_n * 16 *
                            kNintPrefillStrideK + k_local,
                        kNintPrefillStrideK);
#pragma unroll
                    for (int accumulator = 0;
                         accumulator < accumulators_per_warp;
                         ++accumulator) {
                        const int matrix_tile_m =
                            warp_m0 + accumulator * 2;
                        if (matrix_tile_m < matrix_tiles_m) {
                            FragmentA activation_fragment;
                            nvcuda::wmma::load_matrix_sync(
                                activation_fragment,
                                activation_tile + matrix_tile_m * 16 *
                                    kNintPrefillStrideK + k_local,
                                kNintPrefillStrideK);
                            nvcuda::wmma::mma_sync(
                                accumulators[0][accumulator],
                                activation_fragment, weight_fragment,
                                accumulators[0][accumulator]);
                        }
                    }
                }
            }
            __syncthreads();

            if constexpr (FusedGlu) {
                nint_prefill_load_weight_tile(
                    expert_stream, row_q_bits, row_q_bit_offsets,
                    subgroup_scale, subgroup_minimum, neuron_scale,
                    neuron_minimum, weight_tile, local_expert, output_rows,
                    result_rows, groups, group_size, input_width,
                    result_row_base, column_base, 1, thread);
                __syncthreads();
#pragma unroll
                for (int k_local = 0; k_local < kNintPrefillTileK;
                     k_local += 16) {
                    if (active_warp && column_base + k_local < input_width) {
                        FragmentB weight_fragment;
                        nvcuda::wmma::load_matrix_sync(
                            weight_fragment,
                            weight_tile + warp_n * 16 *
                                kNintPrefillStrideK + k_local,
                            kNintPrefillStrideK);
#pragma unroll
                        for (int accumulator = 0;
                             accumulator < accumulators_per_warp;
                             ++accumulator) {
                            const int matrix_tile_m =
                                warp_m0 + accumulator * 2;
                            if (matrix_tile_m < matrix_tiles_m) {
                                FragmentA activation_fragment;
                                nvcuda::wmma::load_matrix_sync(
                                    activation_fragment,
                                    activation_tile + matrix_tile_m * 16 *
                                        kNintPrefillStrideK + k_local,
                                    kNintPrefillStrideK);
                                nvcuda::wmma::mma_sync(
                                    accumulators[1][accumulator],
                                    activation_fragment, weight_fragment,
                                    accumulators[1][accumulator]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }

        float * output_tile_storage = reinterpret_cast<float *>(storage);
#pragma unroll
        for (int accumulator = 0;
             accumulator < accumulators_per_warp; ++accumulator) {
            const int matrix_tile_m = warp_m0 + accumulator * 2;
            if (matrix_tile_m >= matrix_tiles_m) {
                continue;
            }
            float * first_projection = output_tile_storage + warp * 16 * 16;
            float * second_projection = first_projection +
                kNintPrefillWarps * 16 * 16;
            nvcuda::wmma::store_matrix_sync(
                first_projection, accumulators[0][accumulator], 16,
                nvcuda::wmma::mem_row_major);
            if constexpr (FusedGlu) {
                nvcuda::wmma::store_matrix_sync(
                    second_projection, accumulators[1][accumulator], 16,
                    nvcuda::wmma::mem_row_major);
            }
            __syncwarp();
#pragma unroll
            for (int element = lane; element < 16 * 16; element += 32) {
                const int row = element / 16;
                const int column = element - row * 16;
                const int compact = first + matrix_tile_m * 16 + row;
                const int result_row =
                    result_row_base + warp_n * 16 + column;
                if (compact < last && result_row < result_rows) {
                    const int pair = ids_dst[compact];
                    float value = first_projection[element];
                    if constexpr (FusedGlu) {
                        const float gate = __half2float(
                            __float2half_rn(value));
                        const float up = __half2float(
                            __float2half_rn(second_projection[element]));
                        value = mfq_glu_runtime(
                            gate, up, epilogue_mode == 2 ? 1 : 0);
                    }
                    output[static_cast<size_t>(pair) * result_rows +
                        result_row] = __float2half_rn(value);
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }
}


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
        int epilogue_mode) {
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
    for (int group_base = 0;
         group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups || element >= group_size) {
            continue;
        }
        const int width = min(4, group_size - element);
        const int column = group * group_size + element;
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


// Decode M=1 one GS24 group per lane.  NINTv2 stores q rows in bit-width
// cohorts, so row_q_bit_offsets remains the source of truth instead of a
// presumed row stride.  Q4 gets fixed-width vector loads; adaptive rows keep
// the canonical metadata-driven unpack.
__global__ void __launch_bounds__(128) nint_matmul_m1_gs24_kernel(
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
        int output_rows,
        int groups) {
    constexpr int group_size = 24;
    constexpr int chunks = group_size / 4;
    constexpr int warps_per_block = 4;
    const int output_row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }

    const int bits = static_cast<int>(row_q_bits[output_row]);
    const uint64_t row_bit_offset = static_cast<uint64_t>(
        row_q_bit_offsets[output_row]);
    const bool q4 = bits == 4 && (row_bit_offset & 7u) == 0;
    const bool vector_q4 = q4 && (row_bit_offset & 31u) == 0;
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    float dot_accumulator = 0.0f;
    float minimum_accumulator = 0.0f;

    for (int group = warp * 32 + lane;
         group < groups;
         group += warps_per_block * 32) {
        int dot = 0;
        int activation_sum = 0;
        uint32_t qwords[3] = {};
        const uint8_t * qgroup = nullptr;
        if (q4) {
            qgroup = bitstream + (row_bit_offset >> 3) +
                static_cast<size_t>(group) * (group_size / 2);
            if (vector_q4) {
                const uint32_t * words =
                    reinterpret_cast<const uint32_t *>(qgroup);
                qwords[0] = words[0];
                qwords[1] = words[1];
                qwords[2] = words[2];
            }
        }
#pragma unroll
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int column = group * group_size + chunk * 4;
            const int activation_codes = load_i8x4(activation + column);
            const int chunk_sum = __dp4a(
                0x01010101, activation_codes, 0);
            int weight_codes;
            if (q4) {
                uint32_t packed;
                if (vector_q4) {
                    packed = qwords[chunk >> 1] >> ((chunk & 1) * 16);
                } else {
                    const uint8_t * pair = qgroup + chunk * 2;
                    packed = static_cast<uint32_t>(pair[0]) |
                        (static_cast<uint32_t>(pair[1]) << 8);
                }
                weight_codes = unpack_nint4_codes4(packed);
            } else {
                const uint64_t bit_offset = row_bit_offset +
                    static_cast<uint64_t>(column) *
                        static_cast<uint64_t>(bits);
                weight_codes = unpack_nint_codes4(
                    bitstream, bit_offset, bits);
            }
            dot += bits == 8
                ? __dp4a(
                      weight_codes ^ static_cast<int>(0x80808080u),
                      activation_codes,
                      0) + 128 * chunk_sum
                : __dp4a(weight_codes, activation_codes, 0);
            activation_sum += chunk_sum;
        }
        const float input_scale = activation_scale[group];
        dot_accumulator = fmaf(
            input_scale * static_cast<float>(scale_row[group]),
            static_cast<float>(dot),
            dot_accumulator);
        minimum_accumulator = fmaf(
            input_scale * static_cast<float>(minimum_row[group]),
            static_cast<float>(activation_sum),
            minimum_accumulator);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot_accumulator += __shfl_xor_sync(
            0xffffffffu, dot_accumulator, offset);
        minimum_accumulator += __shfl_xor_sync(
            0xffffffffu, minimum_accumulator, offset);
    }
    __shared__ float partial_dot[warps_per_block];
    __shared__ float partial_minimum[warps_per_block];
    if (lane == 0) {
        partial_dot[warp] = dot_accumulator;
        partial_minimum[warp] = minimum_accumulator;
    }
    __syncthreads();
    if (warp == 0) {
        dot_accumulator = lane < warps_per_block
            ? partial_dot[lane]
            : 0.0f;
        minimum_accumulator = lane < warps_per_block
            ? partial_minimum[lane]
            : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_accumulator += __shfl_xor_sync(
                0xffffffffu, dot_accumulator, offset);
            minimum_accumulator += __shfl_xor_sync(
                0xffffffffu, minimum_accumulator, offset);
        }
        if (lane == 0) {
            output[output_row] = __float2half_rn(
                neuron_scale[output_row] * dot_accumulator -
                neuron_minimum[output_row] * minimum_accumulator);
        }
    }
}


// One generic NINT compute kernel covers every q, k, group size, M<=8, and
// routed MFE projection. q is row metadata; k has already been baked into the
// subgroup metadata values. Routed execution changes only the indexing
// contract, not the packed-weight compute kernel.
__global__ void __launch_bounds__(128) nint_matmul_kernel(
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
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width,
        int group_size,
        const int32_t * __restrict__ route_ids,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        int routes,
        int experts,
        int q_expert_stride,
        int pool_phase,
        int epilogue_mode,
        bool routed_input) {
    constexpr int warps_per_block = 4;
    constexpr int maximum_activation_rows = 8;
    constexpr int routed_rows_per_warp = 2;
    const int chunks = (group_size + 3) / 4;
    const int groups_per_warp = 32 / chunks;
    const int lane = static_cast<int>(threadIdx.x);

    if (ids_dst != nullptr) {
        constexpr int route_tile = 8;
        const int rows_per_task = epilogue_mode == 0
            ? routed_rows_per_warp
            : 1;
        const int result_rows = epilogue_mode == 0
            ? output_rows
            : output_rows / 2;
        const int row_blocks =
            (result_rows + rows_per_task - 1) / rows_per_task;
        const int total_tiles = tile_bounds[experts];
        const int64_t total_tasks =
            static_cast<int64_t>(total_tiles) * row_blocks;
        const int64_t task_shift = total_tasks == 0
            ? 0
            : (static_cast<int64_t>(pool_phase) * 17 * row_blocks) %
                total_tasks;
        for (int64_t logical_task = blockIdx.x;
             logical_task < total_tasks;
             logical_task += gridDim.x) {
            const int64_t task =
                (logical_task + task_shift) % total_tasks;
            const int output_row0 =
                static_cast<int>(task % row_blocks) *
                    rows_per_task;
            const int tile = static_cast<int>(task / row_blocks);
            const int expert = tile_experts[tile];
            const int local_expert = expert_local[expert];
            if (local_expert < 0) {
                continue;
            }
            const int local_tile = tile - tile_bounds[expert];
            const int first =
                expert_bounds[expert] + local_tile * route_tile;
            const int last = min(
                first + route_tile, expert_bounds[expert + 1]);
            for (int compact = first + static_cast<int>(threadIdx.y);
                 compact < last;
                 compact += warps_per_block) {
                const int pair = ids_dst[compact];
                const int source_row = routed_input
                    ? pair
                    : pair / routes;
                nint_matmul_routed_pair(
                    bitstream, row_q_bits, row_q_bit_offsets,
                    subgroup_scale, subgroup_minimum, neuron_scale,
                    neuron_minimum, activation, activation_scale, output,
                    pair, source_row, local_expert, output_row0,
                    output_rows, groups, padded_width, group_size,
                    q_expert_stride, epilogue_mode);
            }
        }
        return;
    }

    if (route_ids != nullptr) {
        const int token = static_cast<int>(blockIdx.z) * warps_per_block +
            static_cast<int>(threadIdx.y);
        const int route = static_cast<int>(blockIdx.y);
        const int rows_per_task = epilogue_mode == 0
            ? routed_rows_per_warp
            : 1;
        const int output_row0 =
            static_cast<int>(blockIdx.x) * rows_per_task;
        if (token >= activation_rows || route >= routes) {
            return;
        }
        const int pair = token * routes + route;
        const int expert = route_ids[pair];
        if (static_cast<unsigned int>(expert) >=
                static_cast<unsigned int>(experts)) {
            return;
        }
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            return;
        }
        const int source_row = routed_input ? pair : token;
        nint_matmul_routed_pair(
            bitstream, row_q_bits, row_q_bit_offsets, subgroup_scale,
            subgroup_minimum, neuron_scale, neuron_minimum, activation,
            activation_scale, output, pair, source_row, local_expert,
            output_row0, output_rows, groups, padded_width, group_size,
            q_expert_stride, epilogue_mode);
        return;
    }

    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }

    const int bits = static_cast<int>(row_q_bits[output_row]);
    const uint64_t row_bit_offset =
        static_cast<uint64_t>(row_q_bit_offsets[output_row]);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }

    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int element = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const bool full_chunk = active_lane && element + 3 < group_size;
    const bool tail_chunk =
        active_lane && element < group_size && !full_chunk;
    for (int group_base = 0;
         group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        if (full_chunk) {
            const uint64_t bit_offset = row_bit_offset +
                static_cast<uint64_t>(column) * static_cast<uint64_t>(bits);
            const int weight_codes = unpack_nint_codes4(
                bitstream, bit_offset, bits);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = load_i8x4(
                        activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = bits == 8
                        ? __dp4a(
                              weight_codes ^ static_cast<int>(0x80808080u),
                              activation_codes,
                              0) + 128 * activation_sum
                        : __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        } else if (tail_chunk) {
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    int dot = 0;
                    int activation_sum = 0;
#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        if (element + component < group_size) {
                            const int activation_code = static_cast<int>(
                                activation[
                                    static_cast<size_t>(row) * padded_width +
                                    column + component]);
                            const int weight_code = static_cast<int>(
                                unpack_nint_code(
                                    bitstream,
                                    row_bit_offset,
                                    column + component,
                                    bits));
                            dot += weight_code * activation_code;
                            activation_sum += activation_code;
                        }
                    }
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }

#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}


// One metadata-driven packed input-gradient kernel covers every NINT q/k
// allocation.  Each warp owns four adjacent input columns and reduces over
// output neurons; q is read from row metadata and k is already represented by
// the baked subgroup metadata.
__global__ void __launch_bounds__(128) nint_backward_input_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const __half * __restrict__ output_gradient,
        __half * __restrict__ input_gradient,
        int activation_rows,
        int output_rows,
        int groups,
        int width,
        int group_size) {
    constexpr int warps_per_block = 4;
    constexpr int columns_per_warp = 4;
    constexpr int rows_per_tile = 4;
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int chunks_per_group = (group_size + columns_per_warp - 1) /
        columns_per_warp;
    const int chunk_index = static_cast<int>(blockIdx.x) * warps_per_block +
        warp;
    const int group = chunk_index / chunks_per_group;
    const int group_element =
        (chunk_index - group * chunks_per_group) * columns_per_warp;
    const int column_base = group * group_size + group_element;
    const int first_activation_row =
        static_cast<int>(blockIdx.y) * rows_per_tile;
    if (group >= groups || column_base >= width ||
            first_activation_row >= activation_rows) {
        return;
    }

    float accumulators[rows_per_tile][columns_per_warp] = {};
    for (int output_row = lane; output_row < output_rows; output_row += 32) {
        const int bits = static_cast<int>(row_q_bits[output_row]);
        const uint64_t row_bit_offset = static_cast<uint64_t>(
            row_q_bit_offsets[output_row]);
        const size_t metadata_index =
            static_cast<size_t>(output_row) * groups + group;
        const float scale = neuron_scale[output_row] *
            static_cast<float>(subgroup_scale[metadata_index]);
        const float minimum = neuron_minimum[output_row] *
            static_cast<float>(subgroup_minimum[metadata_index]);
        const int valid_columns = min(
            columns_per_warp,
            min(group_size - group_element, width - column_base));
        uint32_t packed_codes = 0;
        if (valid_columns == columns_per_warp) {
            packed_codes = static_cast<uint32_t>(
                unpack_nint_codes4(
                    bitstream,
                    row_bit_offset +
                        static_cast<uint64_t>(column_base) * bits,
                    bits));
        } else {
#pragma unroll
            for (int component = 0; component < columns_per_warp; ++component) {
                if (component < valid_columns) {
                    packed_codes |= static_cast<uint32_t>(unpack_nint_code(
                        bitstream,
                        row_bit_offset,
                        column_base + component,
                        bits)) << (8 * component);
                }
            }
        }
        float weights[columns_per_warp];
#pragma unroll
        for (int component = 0; component < columns_per_warp; ++component) {
            const int code = (packed_codes >> (8 * component)) & 255;
            weights[component] = scale * static_cast<float>(code) - minimum;
        }
#pragma unroll
        for (int local_row = 0; local_row < rows_per_tile; ++local_row) {
            const int activation_row = first_activation_row + local_row;
            const float gradient = activation_row < activation_rows
                ? __half2float(output_gradient[
                      static_cast<size_t>(activation_row) * output_rows +
                      output_row])
                : 0.0f;
#pragma unroll
            for (int component = 0; component < columns_per_warp; ++component) {
                accumulators[local_row][component] = fmaf(
                    gradient,
                    weights[component],
                    accumulators[local_row][component]);
            }
        }
    }

#pragma unroll
    for (int local_row = 0; local_row < rows_per_tile; ++local_row) {
#pragma unroll
        for (int component = 0; component < columns_per_warp; ++component) {
            float total = accumulators[local_row][component];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffffu, total, offset);
            }
            const int activation_row = first_activation_row + local_row;
            const int column = column_base + component;
            if (lane == 0 && activation_row < activation_rows &&
                    component < group_size - group_element && column < width) {
                input_gradient[
                    static_cast<size_t>(activation_row) * width + column] =
                    __float2half_rn(total);
            }
        }
    }
}


__global__ void nint8_zero_decode_rows_kernel(
        const int8_t * __restrict__ quantized,
        const __half * __restrict__ scale,
        __half * __restrict__ output,
        int rows,
        int groups,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / width);
        const int column = static_cast<int>(index % width);
        const int group = column / 32;
        const int lane = column & 31;
        const size_t block = static_cast<size_t>(row) * groups + group;
        output[index] = __float2half_rn(
            __half2float(scale[block]) * static_cast<float>(
                quantized[block * 32 + lane]));
    }
}


// NINT8-0 has one packed small-M compute kernel; M is a grid dimension rather
// than a template parameter, so it does not create one binary per M.
__global__ void __launch_bounds__(128) nint8_zero_matmul_kernel(
        const int8_t * __restrict__ weight,
        const __half * __restrict__ weight_scale,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width) {
    constexpr int warps_per_block = 4;
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    const int activation_row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (output_row >= output_rows || activation_row >= activation_rows) {
        return;
    }
    float accumulator = 0.0f;
    for (int base = lane * 4; base < padded_width; base += 32 * 4) {
        const int group = base / 32;
        const int offset = base & 31;
        const size_t block = static_cast<size_t>(output_row) * groups + group;
        const int weight_codes = *reinterpret_cast<const int *>(
            weight + block * 32 + offset);
        const int activation_codes = *reinterpret_cast<const int *>(
            activation + static_cast<size_t>(activation_row) * padded_width +
            base);
        accumulator += __half2float(weight_scale[block]) *
            activation_scale[
                static_cast<size_t>(activation_row) * groups + group] *
            static_cast<float>(__dp4a(weight_codes, activation_codes, 0));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(
            0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
        output[static_cast<size_t>(activation_row) * output_rows +
            output_row] = __float2half_rn(accumulator);
    }
}


void validate_nint8_zero(
        const mfq_tensor_backend::Tensor & quantized,
        const mfq_tensor_backend::Tensor & scale) {
    MFQ_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == mfq_tensor_backend::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 scale must be contiguous CUDA fp16 [N,G]");
    MFQ_RUNTIME_CHECK(
        quantized.device() == scale.device(),
        "NINT8-0 tensors must share one CUDA device");
}


mfq_tensor_backend::Tensor cublas_gemm_nt_f32_output(
        const mfq_tensor_backend::Tensor & input,
        const mfq_tensor_backend::Tensor & weight) {
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        input.options().dtype(mfq_tensor_backend::kFloat32));
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<float>(),
        CUDA_R_32F,
        outputs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


}  // namespace


void launch_nint_matmul_routed_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor route_ids,
        mfq_tensor_backend::Tensor expert_local,
        bool route_map_ready,
        mfq_tensor_backend::Tensor ids_dst,
        mfq_tensor_backend::Tensor expert_bounds,
        mfq_tensor_backend::Tensor tile_bounds,
        mfq_tensor_backend::Tensor tile_experts,
        mfq_tensor_backend::Tensor output,
        int tokens,
        int routes,
        int experts,
        int output_rows,
        int groups,
        int padded_width,
        int input_width,
        int group_size,
        int q_expert_stride,
        int route_tile_m,
        int pool_phase,
        bool masked_experts,
        int epilogue_mode,
        bool routed_input,
        cudaStream_t stream) {
    constexpr int warps_per_block = 4;
    constexpr int rows_per_warp = 2;
    const int rows_per_task = epilogue_mode == 0 ? rows_per_warp : 1;
    const int result_rows = epilogue_mode == 0
        ? output_rows
        : output_rows / 2;
    const int row_blocks =
        (result_rows + rows_per_task - 1) / rows_per_task;
    const bool use_compact = tokens > 8 && route_map_ready;
    const bool use_tiled_prefill = use_compact &&
        (route_tile_m == 16 || route_tile_m == 32 ||
         route_tile_m == 64 || route_tile_m == 128);
    if (use_tiled_prefill) {
        constexpr int output_tile = kNintPrefillTileN;
        const int output_tiles =
            (result_rows + output_tile - 1) / output_tile;
        const int pairs = tokens * routes;
        const int64_t maximum_tiles =
            (pairs + route_tile_m - 1) / route_tile_m + experts;
        const int64_t maximum_tasks = maximum_tiles * output_tiles;
        int block_cap = pairs >= 32768 ? 8192 : 4096;
        const int scaled_block_cap = static_cast<int>(
            std::min<int64_t>(INT_MAX, (maximum_tasks + 3) / 4));
        block_cap = std::max(block_cap, scaled_block_cap);
        if (masked_experts && maximum_tasks > block_cap) {
            const int routed_rows_per_expert = std::max(
                1, (pairs + experts - 1) / experts);
            const int task_period =
                ((routed_rows_per_expert + route_tile_m - 1) /
                    route_tile_m) * output_tiles;
            if (task_period > 0 && block_cap % task_period == 0) {
                block_cap += task_period;
            }
        }
        const int blocks = static_cast<int>(std::max<int64_t>(
            1, std::min<int64_t>(maximum_tasks, block_cap)));
#define MFQ_LAUNCH_NINT_TILED_PREFILL(TILE_M, FUSED_GLU) \
        nint_matmul_tiled_prefill_kernel<TILE_M, true, FUSED_GLU><<< \
            blocks, dim3(32, kNintPrefillWarps), 0, stream>>>( \
                bitstream.data_ptr<uint8_t>(), \
                row_q_bits.data_ptr<uint8_t>(), \
                row_q_bit_offsets.data_ptr<int64_t>(), \
                subgroup_scale.data_ptr<uint8_t>(), \
                subgroup_minimum.data_ptr<uint8_t>(), \
                neuron_scale.data_ptr<float>(), \
                neuron_minimum.data_ptr<float>(), \
                reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()), \
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), \
                expert_local.data_ptr<int32_t>(), \
                ids_dst.data_ptr<int32_t>(), \
                expert_bounds.data_ptr<int32_t>(), \
                tile_bounds.data_ptr<int32_t>(), \
                tile_experts.data_ptr<int32_t>(), \
                routes, experts, output_rows, groups, group_size, \
                input_width, q_expert_stride, pool_phase, epilogue_mode, \
                routed_input)
        if (route_tile_m == 128 && epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(128, false);
        } else if (route_tile_m == 128) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(128, true);
        } else if (route_tile_m == 32 && epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(32, false);
        } else if (route_tile_m == 32) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(32, true);
        } else if (route_tile_m == 16 && epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(16, false);
        } else if (route_tile_m == 16) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(16, true);
        } else if (epilogue_mode == 0) {
            MFQ_LAUNCH_NINT_TILED_PREFILL(64, false);
        } else {
            MFQ_LAUNCH_NINT_TILED_PREFILL(64, true);
        }
#undef MFQ_LAUNCH_NINT_TILED_PREFILL
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    MFQ_RUNTIME_CHECK(
        route_tile_m == 8,
        "NINT routed tile must be 8, 16, 32, 64, or 128");
    const int token_blocks =
        (tokens + warps_per_block - 1) / warps_per_block;
    const int64_t maximum_tasks =
        static_cast<int64_t>(tokens) * routes * row_blocks;
    constexpr int persistent_blocks = 4096;
    const int compact_blocks = static_cast<int>(
        std::min<int64_t>(maximum_tasks, persistent_blocks));
    const dim3 grid = use_compact
        ? dim3(compact_blocks)
        : dim3(row_blocks, routes, token_blocks);
    nint_matmul_kernel<<<
        grid,
        dim3(32, warps_per_block), 0, stream>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens,
            output_rows,
            groups,
            padded_width,
            group_size,
            route_ids.data_ptr<int32_t>(),
            expert_local.data_ptr<int32_t>(),
            use_compact ? ids_dst.data_ptr<int32_t>() : nullptr,
            use_compact ? expert_bounds.data_ptr<int32_t>() : nullptr,
            use_compact ? tile_bounds.data_ptr<int32_t>() : nullptr,
            use_compact ? tile_experts.data_ptr<int32_t>() : nullptr,
            routes,
            experts,
            q_expert_stride,
            pool_phase,
            epilogue_mode,
            routed_input);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}


std::vector<mfq_tensor_backend::Tensor> nint8_one_quantize_reconstruct_cuda(
        mfq_tensor_backend::Tensor input) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "NINT8-1 input must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    MFQ_RUNTIME_CHECK(rows > 0 && width > 0, "NINT8-1 input must be non-empty");
    const int groups = (width + 31) / 32;
    auto quantized = mfq_tensor_backend::empty(
        {rows, groups, 32},
        input.options().dtype(mfq_tensor_backend::kInt8));
    auto scale = mfq_tensor_backend::empty({rows, groups}, input.options());
    auto sum = mfq_tensor_backend::empty({rows, groups}, input.options());
    auto reconstructed = mfq_tensor_backend::empty_like(input);
    nint8_one_quantize_reconstruct_kernel<<<
        rows * groups, 32, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            quantized.data_ptr<int8_t>(),
            reinterpret_cast<__half *>(scale.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(sum.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(reconstructed.data_ptr<mfq_half>()),
            rows,
            width,
            groups);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {quantized, scale, sum, reconstructed};
}


mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f16acc_cuda(
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor weight) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == mfq_tensor_backend::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    MFQ_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<mfq_half>(),
        CUDA_R_16F,
        outputs,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f32acc_cuda(
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor weight) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == mfq_tensor_backend::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    MFQ_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<mfq_half>(),
        CUDA_R_16F,
        outputs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


mfq_tensor_backend::Tensor nint_decode_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1,
        "NINT bitstream must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1,
        "NINT q metadata must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bit_offsets.is_cuda() && row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT row offsets must be contiguous CUDA int64 rank-1");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    MFQ_RUNTIME_CHECK(
        rows > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT decode dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == rows && row_q_bit_offsets.numel() == rows &&
        neuron_scale.numel() == rows && neuron_minimum.numel() == rows,
        "NINT row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device(),
        "NINT tensors must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {rows, width},
        neuron_scale.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    nint_decode_rows_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            rows,
            groups,
            static_cast<int>(group_size),
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


static mfq_tensor_backend::Tensor nint_matmul_ws_impl(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor * gate,
        int64_t activation_mode,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "NINT input must be contiguous CUDA fp16 rank-2");
    if (gate != nullptr) {
        MFQ_RUNTIME_CHECK(
            activation_mode == 1 || activation_mode == 2,
            "NINT input gate mode must be sigmoid or SiLU");
        MFQ_RUNTIME_CHECK(
            gate->is_cuda() && gate->is_contiguous() &&
            gate->scalar_type() == mfq_tensor_backend::kFloat16 &&
            gate->dim() == 2 && gate->sizes() == input.sizes(),
            "NINT gate must be contiguous CUDA fp16 and match input shape");
    } else {
        MFQ_RUNTIME_CHECK(
            activation_mode == 0,
            "NINT activation mode requires a gate tensor");
    }
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(input.size(0));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * static_cast<int>(group_size);
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8,
        "NINT packed matmul supports M in [1,8]");
    MFQ_RUNTIME_CHECK(
        group_size >= 4 && group_size <= 64 && real_width <= padded_width,
        "NINT packed matmul dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == mfq_tensor_backend::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT activation workspace is invalid");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == input.device() &&
        row_q_bits.device() == input.device() &&
        row_q_bit_offsets.device() == input.device() &&
        subgroup_scale.device() == input.device() &&
        subgroup_minimum.device() == input.device() &&
        neuron_scale.device() == input.device() &&
        neuron_minimum.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device() &&
        (gate == nullptr || gate->device() == input.device()),
        "NINT tensors and workspace must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {activation_rows, output_rows}, input.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
    const int quantize_threads = group_size <= 32 ? 32 : 64;
    nint_quantize_activation_kernel<<<
        dim3(activation_rows, groups), quantize_threads, 0, stream>>>(
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            gate == nullptr
                ? nullptr
                : reinterpret_cast<const __half *>(
                    gate->data_ptr<mfq_half>()),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            activation_rows,
            real_width,
            padded_width,
            groups,
            static_cast<int>(group_size),
            static_cast<int>(activation_mode));
    if (activation_rows == 1 && group_size == 24) {
        nint_matmul_m1_gs24_kernel<<<
            output_rows, dim3(32, 4), 0, stream>>>(
                bitstream.data_ptr<uint8_t>(),
                row_q_bits.data_ptr<uint8_t>(),
                row_q_bit_offsets.data_ptr<int64_t>(),
                subgroup_scale.data_ptr<uint8_t>(),
                subgroup_minimum.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(),
                neuron_minimum.data_ptr<float>(),
                quantized_input.data_ptr<int8_t>(),
                input_scale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                output_rows,
                groups);
    } else {
        nint_matmul_kernel<<<
            dim3((output_rows + 3) / 4), dim3(32, 4), 0, stream>>>(
                bitstream.data_ptr<uint8_t>(),
                row_q_bits.data_ptr<uint8_t>(),
                row_q_bit_offsets.data_ptr<int64_t>(),
                subgroup_scale.data_ptr<uint8_t>(),
                subgroup_minimum.data_ptr<uint8_t>(),
                neuron_scale.data_ptr<float>(),
                neuron_minimum.data_ptr<float>(),
                quantized_input.data_ptr<int8_t>(),
                input_scale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                activation_rows,
                output_rows,
                groups,
                padded_width,
                static_cast<int>(group_size),
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                0,
                0,
                0,
                0,
                false);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint_matmul_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, nullptr, 0, group_size, quantized_input, input_scale);
}


mfq_tensor_backend::Tensor nint_matmul_input_mul_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor gate,
        int64_t activation_mode,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, &gate, activation_mode, group_size,
        quantized_input, input_scale);
}


mfq_tensor_backend::Tensor nint_backward_input_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor output_gradient,
        int64_t width,
        int64_t group_size) {
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output_gradient.dim() == 2,
        "NINT output gradient must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(output_gradient.size(0));
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        output_gradient.size(1) == output_rows && group_size >= 4 &&
        group_size <= 64 && width > 0 &&
        width <= static_cast<int64_t>(groups) * group_size,
        "NINT packed backward dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT backward row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == output_gradient.device() &&
        row_q_bits.device() == output_gradient.device() &&
        row_q_bit_offsets.device() == output_gradient.device() &&
        subgroup_scale.device() == output_gradient.device() &&
        subgroup_minimum.device() == output_gradient.device() &&
        neuron_scale.device() == output_gradient.device() &&
        neuron_minimum.device() == output_gradient.device(),
        "NINT backward tensors must share one CUDA device");
    auto result = mfq_tensor_backend::empty(
        {activation_rows, width}, output_gradient.options());
    constexpr int warps_per_block = 4;
    constexpr int rows_per_tile = 4;
    const int chunks_per_group =
        (static_cast<int>(group_size) + 3) / 4;
    const int total_chunks = groups * chunks_per_group;
    nint_backward_input_kernel<<<
        dim3(
            (total_chunks + warps_per_block - 1) / warps_per_block,
            (activation_rows + rows_per_tile - 1) / rows_per_tile),
        dim3(32, warps_per_block),
        0,
        mfq_current_cuda_stream()>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            activation_rows,
            output_rows,
            groups,
            static_cast<int>(width),
            static_cast<int>(group_size));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}


mfq_tensor_backend::Tensor nint8_zero_dequant_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 width is invalid");
    const int rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    auto output = mfq_tensor_backend::empty(
        {rows, width},
        quantized.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    nint8_zero_decode_rows_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const int8_t *>(
                quantized.data_ptr<uint8_t>()),
            reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            rows,
            groups,
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_gemv_ws_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "NINT8-0 input must be contiguous CUDA fp16 rank-2");
    const int activation_rows = static_cast<int>(input.size(0));
    const int output_rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * 32;
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        real_width <= padded_width,
        "NINT8-0 packed matmul expects M in [1,8] and K <= packed K");
    MFQ_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == mfq_tensor_backend::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT8-0 activation workspace is invalid");
    MFQ_RUNTIME_CHECK(
        quantized.device() == input.device() && scale.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device(),
        "NINT8-0 tensors and workspace must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {activation_rows, output_rows}, input.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
    nint_quantize_activation_kernel<<<
        dim3(activation_rows, groups), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            nullptr,
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            activation_rows,
            real_width,
            padded_width,
            groups,
            32,
            0);
    nint8_zero_matmul_kernel<<<
        dim3((output_rows + 3) / 4, activation_rows),
        dim3(32, 4),
        0,
        stream>>>(
            reinterpret_cast<const int8_t *>(
                quantized.data_ptr<uint8_t>()),
            reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            activation_rows,
            output_rows,
            groups,
            padded_width);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_mmq_f16_packed_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return nint_cublas_gemm_nt_f16acc_cuda(input, weight);
}


mfq_tensor_backend::Tensor nint8_zero_mmq_f32_packed_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 FP32 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return cublas_gemm_nt_f32_output(input, weight);
}


mfq_tensor_backend::Tensor nint8_zero_backward_input_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor output_gradient,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == quantized.size(0),
        "NINT8-0 output-gradient geometry is invalid");
    MFQ_RUNTIME_CHECK(
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
        output_gradient.scalar_type() == mfq_tensor_backend::kBFloat16 ||
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT8-0 output gradient must be fp16, bf16, or fp32");
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32 &&
        quantized.device() == output_gradient.device(),
        "NINT8-0 backward dimensions or device are invalid");
    const int rows = static_cast<int>(output_gradient.size(0));
    const int outputs = static_cast<int>(quantized.size(0));
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    auto result = mfq_tensor_backend::empty(
        {rows, width}, output_gradient.options());
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient,
        weight,
        result,
        rows,
        outputs,
        static_cast<int>(width),
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
