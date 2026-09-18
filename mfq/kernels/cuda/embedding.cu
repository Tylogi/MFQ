// Dense and canonical packed embedding lookup helpers.

#include <algorithm>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mfq_tensor_backend.h"


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


mfq_tensor_backend::Tensor embedding_lookup_cuda(
        mfq_tensor_backend::Tensor weight,
        mfq_tensor_backend::Tensor token_ids) {
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() && weight.dim() == 2,
        "embedding weight must be contiguous CUDA rank-2");
    MFQ_RUNTIME_CHECK(
        weight.scalar_type() == mfq_tensor_backend::kFloat32 ||
        weight.scalar_type() == mfq_tensor_backend::kFloat16,
        "embedding weight must be fp16 or fp32");
    MFQ_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
        token_ids.device() == weight.device(),
        "embedding token IDs must be contiguous CUDA int64 on the weight device");
    const int vocabulary = static_cast<int>(weight.size(0));
    const int width = static_cast<int>(weight.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(shape, weight.options());
    if (token_count == 0) {
        return output;
    }
    constexpr int threads = 256;
    MFQ_DISPATCH_FLOATING_TYPES_AND_HALF(
        weight.scalar_type(), "embedding_lookup_cuda", [&] {
            if (token_count >= 64 &&
                (static_cast<size_t>(width) * sizeof(scalar_t)) %
                    sizeof(uint4) == 0) {
                constexpr int elements_per_vector =
                    sizeof(uint4) / sizeof(scalar_t);
                const size_t vectors =
                    static_cast<size_t>(width) / elements_per_vector;
                const size_t tiles = (vectors + threads - 1) / threads;
                const int vector_blocks = launch_blocks(
                    static_cast<size_t>(token_count) * tiles * threads);
                dense_embedding_vector_kernel<scalar_t><<<
                    vector_blocks, threads, 0, mfq_current_cuda_stream()>>>(
                        weight.data_ptr<scalar_t>(),
                        token_ids.data_ptr<int64_t>(),
                        output.data_ptr<scalar_t>(),
                        token_count,
                        width,
                        vocabulary);
            } else {
                const int flat_blocks = launch_blocks(
                    static_cast<size_t>(token_count) * width);
                dense_embedding_flat_kernel<scalar_t><<<
                    flat_blocks, threads, 0, mfq_current_cuda_stream()>>>(
                        weight.data_ptr<scalar_t>(),
                        token_ids.data_ptr<int64_t>(),
                        output.data_ptr<scalar_t>(),
                        token_count,
                        width,
                        vocabulary);
            }
        });
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint_embedding_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor token_ids,
        int64_t width,
        int64_t group_size) {
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1,
        "NINT embedding bitstream must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT embedding q metadata is invalid");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT embedding subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT embedding neuron metadata is invalid");
    MFQ_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == mfq_tensor_backend::kInt64,
        "NINT embedding token IDs must be contiguous CUDA int64");
    const int vocabulary = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    MFQ_RUNTIME_CHECK(
        vocabulary > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT embedding dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == vocabulary &&
        row_q_bit_offsets.numel() == vocabulary &&
        neuron_scale.numel() == vocabulary &&
        neuron_minimum.numel() == vocabulary,
        "NINT embedding row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device() &&
        bitstream.device() == token_ids.device(),
        "NINT embedding tensors must share one CUDA device");
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(
        shape, bitstream.options().dtype(mfq_tensor_backend::kFloat16));
    if (token_count == 0) {
        return output;
    }
    constexpr int threads = 256;
    const int blocks = launch_blocks(
        static_cast<size_t>(token_count) * width);
    nint_embedding_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            token_ids.data_ptr<int64_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            token_count,
            vocabulary,
            groups,
            static_cast<int>(group_size),
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_embedding_lookup_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor token_ids,
        int64_t width) {
    MFQ_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == mfq_tensor_backend::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 embedding q must be contiguous CUDA uint8 [V,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 embedding scale must be contiguous CUDA fp16 [V,G]");
    MFQ_RUNTIME_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
        token_ids.device() == quantized.device() &&
        scale.device() == quantized.device(),
        "NINT8-0 embedding tensors must share one CUDA device");
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 embedding width is invalid");
    const int vocabulary = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(
        shape, quantized.options().dtype(mfq_tensor_backend::kFloat16));
    if (token_count == 0) {
        return output;
    }
    constexpr int threads = 256;
    const int blocks = launch_blocks(
        static_cast<size_t>(token_count) * width);
    nint8_zero_embedding_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const int8_t *>(
                quantized.data_ptr<uint8_t>()),
            reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
            token_ids.data_ptr<int64_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            token_count,
            vocabulary,
            groups,
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
