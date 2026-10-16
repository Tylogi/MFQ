// Tool-only NINT / NINT8-0 input backward. The native inference library
// compiles nint_matmul.cu alone; the Python calibration extension adds this
// unit.  The NINT8-0 dense weight comes from the shared inference dequant
// entry point and the packed row decoder comes from packed_nint.cuh, so no
// format math is duplicated here.
#include "native/env.h"

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "csrc/backends/cuda/native/tensor/backend.h"
#include "kernels/quant_ops.h"
#include "nint_matmul_impl.cuh"
#include "packed_backward.cuh"
#include "packed_nint.cuh"

namespace {

using namespace mfq::cuda::packed_nint;

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

}  // namespace

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
