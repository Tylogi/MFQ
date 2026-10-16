// Dense and canonical packed embedding lookup helpers.  This unit holds only
// the device bodies and the raw-pointer + cudaStream_t launchers; the
// allocating, validating wrappers live in ops/embedding.cpp, which keeps this
// include closure free of any Tensor backend header.

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/kernels.h"


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


template <typename Scalar>
__global__ void dense_embedding_flat_kernel(
        const Scalar * __restrict__ weight,
        const int64_t * __restrict__ token_ids,
        Scalar * __restrict__ output,
        int token_count,
        int width,
        int vocabulary) {
    const size_t total = static_cast<size_t>(token_count) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int token_index = static_cast<int>(index / width);
        const int64_t token = token_ids[token_index];
        output[index] = token >= 0 && token < vocabulary
            ? weight[static_cast<size_t>(token) * width + column]
            : Scalar(0);
    }
}


template <typename Scalar>
__global__ void dense_embedding_vector_kernel(
        const Scalar * __restrict__ weight,
        const int64_t * __restrict__ token_ids,
        Scalar * __restrict__ output,
        int token_count,
        int width,
        int vocabulary) {
    constexpr int elements_per_vector = sizeof(uint4) / sizeof(Scalar);
    const int vectors_per_token = width / elements_per_vector;
    const int tiles_per_token =
        (vectors_per_token + blockDim.x - 1) / blockDim.x;
    const size_t total_tiles =
        static_cast<size_t>(token_count) * tiles_per_token;
    for (size_t tile_index = blockIdx.x;
         tile_index < total_tiles;
         tile_index += gridDim.x) {
        const int token_index = static_cast<int>(
            tile_index / tiles_per_token);
        const int tile = static_cast<int>(tile_index % tiles_per_token);
        const int vector_index = tile * blockDim.x + threadIdx.x;
        if (vector_index >= vectors_per_token) {
            continue;
        }
        const int64_t token = token_ids[token_index];
        auto* output_vectors = reinterpret_cast<uint4 *>(output);
        const auto* weight_vectors =
            reinterpret_cast<const uint4 *>(weight);
        const size_t output_index = static_cast<size_t>(token_index) *
            vectors_per_token + vector_index;
        output_vectors[output_index] = token >= 0 && token < vocabulary
            ? weight_vectors[static_cast<size_t>(token) * vectors_per_token +
                vector_index]
            : make_uint4(0, 0, 0, 0);
    }
}


__global__ void nint_embedding_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int64_t * __restrict__ token_ids,
        __half * __restrict__ output,
        int token_count,
        int vocabulary,
        int groups,
        int group_size,
        int width) {
    const size_t total = static_cast<size_t>(token_count) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int token_index = static_cast<int>(index / width);
        const int64_t token = token_ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocabulary) {
            const int group = column / group_size;
            const size_t metadata = static_cast<size_t>(token) * groups + group;
            const int bits = static_cast<int>(row_q_bits[token]);
            const uint8_t code = unpack_nint_code(
                bitstream,
                static_cast<uint64_t>(row_q_bit_offsets[token]),
                column,
                bits);
            const float scale = neuron_scale[token] *
                static_cast<float>(subgroup_scale[metadata]);
            const float minimum = neuron_minimum[token] *
                static_cast<float>(subgroup_minimum[metadata]);
            value = scale * static_cast<float>(code) - minimum;
        }
        output[index] = __float2half_rn(value);
    }
}


__global__ void nint_selected_rows_kernel(
        const uint8_t* __restrict__ packed,
        const uint32_t* __restrict__ descriptors,
        const int64_t* __restrict__ inverse,
        __half* __restrict__ output, size_t total, int width) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const auto selected = inverse ? inverse[index / width] : index / width;
        const uint32_t* row = descriptors + selected * 6;
        const uint32_t layout = row[3];
        const int qbits = layout & 15u, kbits = (layout >> 4u) & 15u;
        const int group = column / row[4];
        const float q = unpack_nint_code(packed + row[0], (layout >> 8u) & 7u, column, qbits);
        const float s = unpack_nint_code(packed + row[1], (layout >> 12u) & 7u, group, kbits);
        const float m = unpack_nint_code(packed + row[2], (layout >> 16u) & 7u, group, kbits);
        const float ns = __half2float(__ushort_as_half(static_cast<unsigned short>(row[5])));
        const float nm = __half2float(__ushort_as_half(static_cast<unsigned short>(row[5] >> 16u)));
        // Match the shared/Metal reconstruction order, without contraction.
        output[index] = __float2half_rn(__fsub_rn(__fmul_rn(__fmul_rn(ns, s), q), __fmul_rn(nm, m)));
    }
}

__global__ void nint8_zero_embedding_kernel(
        const int8_t * __restrict__ quantized,
        const __half * __restrict__ scale,
        const int64_t * __restrict__ token_ids,
        __half * __restrict__ output,
        int token_count,
        int vocabulary,
        int groups,
        int width) {
    const size_t total = static_cast<size_t>(token_count) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int token_index = static_cast<int>(index / width);
        const int64_t token = token_ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocabulary) {
            const int group = column / 32;
            const int lane = column & 31;
            const size_t block = static_cast<size_t>(token) * groups + group;
            value = __half2float(scale[block]) * static_cast<float>(
                quantized[block * 32 + lane]);
        }
        output[index] = __float2half_rn(value);
    }
}


int launch_blocks(size_t elements) {
    constexpr int threads = 256;
    return static_cast<int>(std::min<size_t>(
        (elements + threads - 1) / threads, 4096));
}


}  // namespace


namespace mfq::cuda::kernels {

template <typename Scalar>
static void launch_dense_embedding(
        const Scalar * weight,
        const int64_t * token_ids,
        Scalar * output,
        int token_count,
        int width,
        int vocabulary,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const auto vector_addresses =
        reinterpret_cast<std::uintptr_t>(weight) |
        reinterpret_cast<std::uintptr_t>(output);
    const bool vector_aligned =
        (vector_addresses & (alignof(uint4) - 1)) == 0;
    if (token_count >= 64 && vector_aligned &&
        (static_cast<size_t>(width) * sizeof(Scalar)) % sizeof(uint4) == 0) {
        constexpr int elements_per_vector = sizeof(uint4) / sizeof(Scalar);
        const size_t vectors = static_cast<size_t>(width) / elements_per_vector;
        const size_t tiles = (vectors + threads - 1) / threads;
        const int vector_blocks = launch_blocks(
            static_cast<size_t>(token_count) * tiles * threads);
        dense_embedding_vector_kernel<Scalar><<<
            vector_blocks, threads, 0, stream>>>(
                weight, token_ids, output, token_count, width, vocabulary);
    } else {
        const int flat_blocks = launch_blocks(
            static_cast<size_t>(token_count) * width);
        dense_embedding_flat_kernel<Scalar><<<
            flat_blocks, threads, 0, stream>>>(
                weight, token_ids, output, token_count, width, vocabulary);
    }
}


void dense_embedding_lookup(
        const void * weight,
        ScalarType weight_type,
        const int64_t * token_ids,
        void * output,
        int token_count,
        int width,
        int vocabulary,
        cudaStream_t stream) {
    if (weight_type == ScalarType::float32) {
        launch_dense_embedding(
            static_cast<const float *>(weight), token_ids,
            static_cast<float *>(output),
            token_count, width, vocabulary, stream);
    } else if (weight_type == ScalarType::float16) {
        launch_dense_embedding(
            static_cast<const __half *>(weight), token_ids,
            static_cast<__half *>(output),
            token_count, width, vocabulary, stream);
    } else {
        throw std::invalid_argument(
            "native embedding weight must be fp16 or fp32");
    }
}


void nint_embedding_lookup(
        const uint8_t * bitstream,
        const uint8_t * row_q_bits,
        const int64_t * row_q_bit_offsets,
        const uint8_t * subgroup_scale,
        const uint8_t * subgroup_minimum,
        const float * neuron_scale,
        const float * neuron_minimum,
        const int64_t * token_ids,
        __half * output,
        int token_count,
        int vocabulary,
        int groups,
        int64_t group_size,
        int64_t width,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = launch_blocks(
        static_cast<size_t>(token_count) * width);
    nint_embedding_kernel<<<blocks, threads, 0, stream>>>(
        bitstream,
        row_q_bits,
        row_q_bit_offsets,
        subgroup_scale,
        subgroup_minimum,
        neuron_scale,
        neuron_minimum,
        token_ids,
        output,
        token_count,
        vocabulary,
        groups,
        static_cast<int>(group_size),
        static_cast<int>(width));
}


void nint_selected_rows(
        const uint8_t * packed,
        const uint32_t * descriptors,
        const int64_t * inverse,
        __half * output,
        size_t total,
        int width,
        cudaStream_t stream) {
    nint_selected_rows_kernel<<<launch_blocks(total), 256, 0, stream>>>(
        packed, descriptors, inverse, output, total, width);
}


void nint8_zero_embedding_lookup(
        const int8_t * quantized,
        const __half * scale,
        const int64_t * token_ids,
        __half * output,
        int token_count,
        int vocabulary,
        int groups,
        int64_t width,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = launch_blocks(
        static_cast<size_t>(token_count) * width);
    nint8_zero_embedding_kernel<<<blocks, threads, 0, stream>>>(
        quantized, scale, token_ids, output, token_count, vocabulary, groups,
        static_cast<int>(width));
}

}  // namespace mfq::cuda::kernels
