// Tool-only MXFP4-SQ input backward. The native inference library compiles
// mxfp4_sq.cu alone; the Python calibration extension and the SQ numeric test
// add this unit.  The wide-M path reuses the inference dequant entry point so
// the packed decode stays one implementation.
#include "native/env.h"
#include "mxfp4_sq.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mxfp4_sq_impl.cuh"
#include "packed_backward.cuh"

namespace {

template<int Rows>
__global__ void __launch_bounds__(128) sq_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int block = static_cast<int>(blockIdx.x) * 4 + warp;
    const int column = block * 32 + lane;
    if (column >= q.width) return;

    const auto* symbols = blob + q.symbols;
    const int split = static_cast<int>(blockIdx.y);
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, q.outputs);
    float accumulators[Rows] = {};
    for (int output = output0; output < output_end; ++output) {
        const int bits = row_q[output];
        const auto auxiliary = std::size_t(row_auxiliary[output]);
        const auto* row_symbols =
            symbols + std::size_t(row_symbol_byte_offsets[output]);
        unsigned state_value = 0;
        if (lane == 0) {
            if (bits == 4) {
                state_value = blob[
                    q.native_scales + auxiliary * (q.width / 32) + block];
            } else {
                const auto selector_index =
                    auxiliary * (q.width / 32) + block;
                const auto state = auxiliary * 8 + block_tag(
                    row_symbols, blob + q.selectors,
                    block, selector_index, bits);
                state_value =
                    (read_bits(blob + q.palettes, state, 5) << 8) |
                    (q.base + read_bits(blob + q.scales, state, 2));
            }
        }
        state_value = __shfl_sync(0xffffffffu, state_value, 0);
        const auto symbol = read_bits(row_symbols, column, bits);
        const float weight = bits == 4
            ? decode_native(symbol, state_value)
            : decode_value(
                state_value >> 8,
                symbol,
                state_value & 255,
                bits);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row < rows
                ? __half2float(output_gradient[
                    std::size_t(row) * q.outputs + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            accumulators[row] = fmaf(gradient, weight, accumulators[row]);
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row < rows) {
            partials[(std::size_t(split) * rows + row) * q.width + column] =
                accumulators[row];
        }
    }
}

template<int Rows>
void launch_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((q.width + 127) / 128),
        static_cast<unsigned>((q.outputs + output_tile - 1) / output_tile));
    sq_backward_matrix<Rows><<<
        grid, 128, 0, stream>>>(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        output_gradient, partials, q, rows, output_tile);
}

void dispatch_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    if (rows == 1) {
        launch_backward_matrix<1>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 2) {
        launch_backward_matrix<2>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 4) {
        launch_backward_matrix<4>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else {
        launch_backward_matrix<8>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    }
}

}  // namespace

mfq_tensor_backend::Tensor mxfp4_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor row_auxiliary,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t bits,
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == outputs &&
        (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kFloat32),
        "MXFP4-SQ backward requires contiguous CUDA FP16/FP32 [M,N]");
    const MfqCudaGuard guard(blob.device());
    auto result = mfq_tensor_backend::empty(
        {output_gradient.size(0), width}, output_gradient.options());
    const auto count = output_gradient.size(0) * width;
    if (!count) return result;
    constexpr int threads = 256;
    const int blocks = int(std::min<std::int64_t>(
        (count + threads - 1) / threads, 65535));
    const auto stream = mfq_current_cuda_stream();
    const auto* data = blob.data_ptr<std::uint8_t>();
    const int rows = int(output_gradient.size(0));
    if (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
            rows <= 8) {
        const int output_tile = rows <= 4 ? 16 : 32;
        const int splits = (int(outputs) + output_tile - 1) / output_tile;
        auto partials = mfq_tensor_backend::empty(
            {splits, rows, width},
            output_gradient.options().dtype(mfq_tensor_backend::kFloat32));
        const auto* gradient = reinterpret_cast<const __half*>(
            output_gradient.data_ptr<mfq_half>());
        auto* destination = reinterpret_cast<__half*>(
            result.data_ptr<mfq_half>());
        dispatch_backward_matrix(
            data,
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            row_auxiliary.data_ptr<std::int32_t>(),
            gradient, partials.data_ptr<float>(), q, rows,
            output_tile, stream);
        mfq_packed_backward::launch_split_float_reduce_to_half(
            partials.data_ptr<float>(), destination,
            rows, int(width), splits, stream);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = mxfp4_sq_dequant_cuda(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows, false);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        rows, int(outputs), int(width), stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
