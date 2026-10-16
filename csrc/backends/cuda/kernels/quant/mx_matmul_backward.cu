// Tool-only MXFP8/MXFP4 input backward. The native inference library compiles
// mx_matmul.cu alone; the Python calibration extension adds this unit.  The
// general path reuses the inference dequant entry points, so the format decode
// stays one implementation.
#include <cuda_fp16.h>
#include "csrc/backends/cuda/native/tensor/backend.h"
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernels/quant_ops.h"
#include "mx_matmul_impl.cuh"
#include "packed_backward.cuh"

namespace {

template <int Rows, int Columns>
__global__ void __launch_bounds__(256) mxfp8_backward_mma_kernel(
        const uint8_t * __restrict__ values,
        const uint8_t * __restrict__ scales,
        const __half * __restrict__ output_gradient,
        __half * __restrict__ input_gradient,
        int rows,
        int outputs,
        int width) {
    using namespace nvcuda;
    constexpr int kWarps = 8;
    __shared__ __align__(16) __half gradient_tile[kWarps][16 * 16];
    __shared__ __align__(16) __half weight_tile[kWarps][16 * 16];
    __shared__ __align__(16) float result_tile[kWarps][16 * 16];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int column0 = static_cast<int>(blockIdx.x) * Columns;
    const int scale_columns = width >> 7;
    for (int index = lane; index < 16 * 16; index += 32) {
        gradient_tile[warp][index] = __float2half_rn(0.0f);
        if constexpr (Columns < 16) {
            weight_tile[warp][index] = __float2half_rn(0.0f);
        }
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int output_group = warp * 128;
         output_group < outputs;
         output_group += kWarps * 128) {
        for (int output0 = output_group;
             output0 < min(output_group + 128, outputs);
             output0 += 16) {
            for (int index = lane; index < Rows * 16; index += 32) {
                const int row = index >> 4;
                const int output = output0 + (index & 15);
                gradient_tile[warp][index] = row < rows && output < outputs
                    ? output_gradient[
                        static_cast<int64_t>(row) * outputs + output]
                    : __float2half_rn(0.0f);
            }
            float scale = lane == 0
                ? decode_e8m0(scales[
                    static_cast<int64_t>(output0 >> 7) * scale_columns +
                    (column0 >> 7)])
                : 0.0f;
            scale = __shfl_sync(0xffffffffu, scale, 0);
            for (int index = lane; index < 16 * Columns; index += 32) {
                const int output = output0 + index / Columns;
                const int local_column = index % Columns;
                const int column = column0 + local_column;
                weight_tile[warp][
                    (index / Columns) * 16 + local_column] = output < outputs
                    ? __float2half_rn(
                        decode_e4m3fn(values[
                            static_cast<int64_t>(output) * width + column]) *
                        scale)
                    : __float2half_rn(0.0f);
            }
            __syncwarp();
            wmma::fragment<
                wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
                    gradient_fragment;
            wmma::fragment<
                wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>
                    weight_fragment;
            wmma::load_matrix_sync(
                gradient_fragment, gradient_tile[warp], 16);
            wmma::load_matrix_sync(
                weight_fragment, weight_tile[warp], 16);
            wmma::mma_sync(
                accumulator, gradient_fragment, weight_fragment, accumulator);
            __syncwarp();
        }
    }
    wmma::store_matrix_sync(
        result_tile[warp], accumulator, 16, wmma::mem_row_major);
    __syncthreads();
    const int index = static_cast<int>(threadIdx.x);
    if (index < rows * Columns) {
        const int row = index / Columns;
        const int local_column = index % Columns;
        const int column = column0 + local_column;
        float value = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
            value += result_tile[source_warp][row * 16 + local_column];
        }
        input_gradient[static_cast<int64_t>(row) * width + column] =
            __float2half_rn(value);
    }
}

template <int Rows, int Columns>
void launch_mxfp8_backward_mma(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    const int blocks = (width + Columns - 1) / Columns;
    mxfp8_backward_mma_kernel<Rows, Columns><<<
        blocks, 256, 0, stream>>>(
        values.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<mfq_half>()),
        reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
        rows, outputs, width);
}

template <int Rows>
__global__ void __launch_bounds__(32) mxfp4_backward_vec4_partial_kernel(
        const uint8_t * __restrict__ values,
        const uint8_t * __restrict__ scales,
        const __half * __restrict__ output_gradient,
        float * __restrict__ partials,
        int rows,
        int outputs,
        int width) {
    constexpr int kOutputTile = 32;
    const int column0 =
        (static_cast<int>(blockIdx.x) * 32 + threadIdx.x) * 4;
    if (column0 >= width) {
        return;
    }
    float accumulators[Rows][4] = {};
    const int output_group = static_cast<int>(blockIdx.y);
    const int output0 = output_group * kOutputTile;
    const int output_end = min(output0 + kOutputTile, outputs);
    const int value_columns = width >> 1;
    const int scale_columns = width >> 5;
    for (int output = output0; output < output_end; ++output) {
        const uint16_t packed = *reinterpret_cast<const uint16_t *>(
            values + static_cast<int64_t>(output) * value_columns +
            (column0 >> 1));
        const float scale = decode_e8m0(scales[
            static_cast<int64_t>(output) * scale_columns +
            (column0 >> 5)]);
        float decoded[4];
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            decoded[component] = decode_mxfp4_e2m1(static_cast<uint8_t>(
                (packed >> (4 * component)) & 15u)) * scale;
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row < rows) {
                const float gradient = __half2float(output_gradient[
                    static_cast<int64_t>(row) * outputs + output]);
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    accumulators[row][component] = fmaf(
                        gradient, decoded[component],
                        accumulators[row][component]);
                }
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row < rows) {
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                const int column = column0 + component;
                if (column < width) {
                    partials[(static_cast<int64_t>(output_group) * rows + row) *
                        width + column] = accumulators[row][component];
                }
            }
        }
    }
}

template <int Rows>
void launch_mxfp4_backward_vec4_small_m(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    constexpr int kOutputTile = 32;
    const int output_groups = (outputs + kOutputTile - 1) / kOutputTile;
    mfq_tensor_backend::Tensor partials = mfq_tensor_backend::empty(
        {output_groups, rows, width},
        values.options().dtype(mfq_tensor_backend::kFloat32));
    const dim3 partial_grid(
        static_cast<unsigned>((width + 127) / 128),
        static_cast<unsigned>(output_groups));
    mxfp4_backward_vec4_partial_kernel<Rows><<<partial_grid, 32, 0, stream>>>(
        values.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<mfq_half>()),
        partials.data_ptr<float>(), rows, outputs, width);
    mfq_packed_backward::launch_split_float_reduce_to_half(
        partials.data_ptr<float>(),
        reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
        rows, width, output_groups, stream);
}

template <int Rows>
__global__ void __launch_bounds__(32) mxfp8_backward_scalar_partial_kernel(
        const uint8_t * __restrict__ values,
        const uint8_t * __restrict__ scales,
        const __half * __restrict__ output_gradient,
        float * __restrict__ partials,
        int rows,
        int outputs,
        int width) {
    const int column = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
    if (column >= width) {
        return;
    }
    float accumulators[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        accumulators[row] = 0.0f;
    }
    const int output_group = static_cast<int>(blockIdx.y);
    const int output0 = output_group * 128;
    const int output_end = min(output0 + 128, outputs);
    const int scale_columns = width >> 7;
    const float scale = decode_e8m0(scales[
        static_cast<int64_t>(output_group) * scale_columns +
        (column >> 7)]);
    const uint8_t * weight = output0 < outputs
        ? values + static_cast<int64_t>(output0) * width + column
        : values;
    for (int output = output0; output < output_end; ++output) {
        const float decoded = decode_e4m3fn(*weight) * scale;
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row < rows) {
                accumulators[row] = fmaf(
                    __half2float(output_gradient[
                        static_cast<int64_t>(row) * outputs + output]),
                    decoded,
                    accumulators[row]);
            }
        }
        weight += width;
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row < rows) {
            partials[(static_cast<int64_t>(output_group) * rows + row) *
                width + column] = accumulators[row];
        }
    }
}

template <int Rows, int Warps>
__global__ void __launch_bounds__(Warps * 32)
mxfp8_backward_block_reduce_kernel(
        const uint8_t * __restrict__ values,
        const uint8_t * __restrict__ scales,
        const __half * __restrict__ output_gradient,
        __half * __restrict__ input_gradient,
        int rows,
        int outputs,
        int width) {
    __shared__ float warp_results[Rows][Warps][32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int column = static_cast<int>(blockIdx.x) * 32 + lane;
    const bool valid = column < width;
    const int output0 = warp * 128;
    const int output_end = min(output0 + 128, outputs);
    const int scale_columns = width >> 7;
    const float scale = valid && output0 < outputs
        ? decode_e8m0(scales[
            static_cast<int64_t>(warp) * scale_columns + (column >> 7)])
        : 0.0f;
    const uint8_t * weight = values +
        static_cast<int64_t>(output0) * width + column;
    float accumulators[Rows] = {};
    if (valid) {
        for (int output = output0; output < output_end; ++output) {
            const float decoded = decode_e4m3fn(*weight) * scale;
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                if (row < rows) {
                    accumulators[row] = fmaf(
                        __half2float(output_gradient[
                            static_cast<int64_t>(row) * outputs + output]),
                        decoded,
                        accumulators[row]);
                }
            }
            weight += width;
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        warp_results[row][warp][lane] = accumulators[row];
    }
    __syncthreads();
    const int result_index = static_cast<int>(threadIdx.x);
    if (result_index < rows * 32) {
        const int row = result_index >> 5;
        const int result_lane = result_index & 31;
        const int result_column = static_cast<int>(blockIdx.x) * 32 + result_lane;
        if (result_column < width) {
            float value = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < Warps; ++source_warp) {
                value += warp_results[row][source_warp][result_lane];
            }
            input_gradient[static_cast<int64_t>(row) * width + result_column] =
                __float2half_rn(value);
        }
    }
}

template <int Rows>
void launch_mxfp8_backward_block_reduce(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    const int output_groups = (outputs + 127) / 128;
    const int blocks = (width + 31) / 32;
#define MFQ_LAUNCH_MXFP8_BLOCK_REDUCE(WARPS) \
    mxfp8_backward_block_reduce_kernel<Rows, WARPS><<< \
        blocks, (WARPS) * 32, 0, stream>>>( \
            values.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), \
            reinterpret_cast<const __half *>( \
                output_gradient.data_ptr<mfq_half>()), \
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()), \
            rows, outputs, width)
    if (output_groups <= 1) {
        MFQ_LAUNCH_MXFP8_BLOCK_REDUCE(1);
    } else if (output_groups <= 2) {
        MFQ_LAUNCH_MXFP8_BLOCK_REDUCE(2);
    } else if (output_groups <= 4) {
        MFQ_LAUNCH_MXFP8_BLOCK_REDUCE(4);
    } else {
        MFQ_LAUNCH_MXFP8_BLOCK_REDUCE(8);
    }
#undef MFQ_LAUNCH_MXFP8_BLOCK_REDUCE
}

template <int Rows>
void launch_mxfp8_backward_scalar_small_m(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    const int output_groups = (outputs + 127) / 128;
    mfq_tensor_backend::Tensor partials = mfq_tensor_backend::empty(
        {output_groups, rows, width},
        values.options().dtype(mfq_tensor_backend::kFloat32));
    const dim3 partial_grid(
        static_cast<unsigned>((width + 31) / 32),
        static_cast<unsigned>(output_groups));
    mxfp8_backward_scalar_partial_kernel<Rows><<<partial_grid, 32, 0, stream>>>(
        values.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<mfq_half>()),
        partials.data_ptr<float>(),
        rows, outputs, width);
    mfq_packed_backward::launch_split_float_reduce_to_half(
        partials.data_ptr<float>(),
        reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
        rows, width, output_groups, stream);
}

template <int Rows>
__global__ void __launch_bounds__(32) mxfp8_backward_vec4_partial_kernel(
        const uint8_t * __restrict__ values,
        const uint8_t * __restrict__ scales,
        const __half * __restrict__ output_gradient,
        float * __restrict__ partials,
        int rows,
        int outputs,
        int width) {
    const int lane = static_cast<int>(threadIdx.x);
    const int column0 = (static_cast<int>(blockIdx.x) * 32 + lane) * 4;
    float accumulators[Rows][4];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            accumulators[row][component] = 0.0f;
        }
    }
    const int output_group = static_cast<int>(blockIdx.y);
    const int output0 = output_group * 32;
    const int output_end = min(output0 + 32, outputs);
    const int scale_columns = width >> 7;
    float scale = lane == 0 ? decode_e8m0(scales[
        static_cast<int64_t>(output0 >> 7) * scale_columns +
        (column0 >> 7)]) : 0.0f;
    scale = __shfl_sync(0xffffffffu, scale, 0);
    const uint8_t * weight = values +
        static_cast<int64_t>(output0) * width + column0;
    for (int output = output0; output < output_end; ++output) {
        const uint32_t packed = *reinterpret_cast<const uint32_t *>(weight);
        float decoded[4];
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            decoded[component] = decode_e4m3fn(static_cast<uint8_t>(
                packed >> (8 * component))) * scale;
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row < rows
                ? __half2float(output_gradient[
                    static_cast<int64_t>(row) * outputs + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                accumulators[row][component] = fmaf(
                    gradient, decoded[component],
                    accumulators[row][component]);
            }
        }
        weight += width;
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row < rows) {
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                partials[(static_cast<int64_t>(output_group) * rows + row) *
                    width + column0 + component] = accumulators[row][component];
            }
        }
    }
}

template <int Rows>
void launch_mxfp8_backward_vec4_small_m(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    const int output_groups = (outputs + 31) / 32;
    mfq_tensor_backend::Tensor partials = mfq_tensor_backend::empty(
        {output_groups, rows, width},
        values.options().dtype(mfq_tensor_backend::kFloat32));
    const dim3 partial_grid(
        static_cast<unsigned>((width + 127) / 128),
        static_cast<unsigned>(output_groups));
    mxfp8_backward_vec4_partial_kernel<Rows><<<partial_grid, 32, 0, stream>>>(
        values.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<mfq_half>()),
        partials.data_ptr<float>(),
        rows, outputs, width);
    mfq_packed_backward::launch_split_float_reduce_to_half(
        partials.data_ptr<float>(),
        reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
        rows, width, output_groups, stream);
}

}  // namespace

mfq_tensor_backend::Tensor mx_backward_input_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor output_gradient,
        bool mxfp4) {
    if (mxfp4) {
        validate_mxfp4_dense(values, scales);
    } else {
        validate_mxfp8(values, scales);
    }
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 && output_gradient.size(1) == values.size(0),
        "MX backward output gradient geometry mismatch");
    MFQ_RUNTIME_CHECK(
        values.device() == scales.device() &&
        values.device() == output_gradient.device(),
        "MX backward tensors must share one CUDA device");
    const auto dtype = output_gradient.scalar_type();
    MFQ_RUNTIME_CHECK(
        dtype == mfq_tensor_backend::kFloat16 ||
        dtype == mfq_tensor_backend::kFloat32 ||
        dtype == mfq_tensor_backend::kBFloat16,
        "MX backward output gradient must be fp16, bf16, or fp32");
    const int rows = static_cast<int>(output_gradient.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(mxfp4 ? values.size(1) * 2 : values.size(1));
    auto result = mfq_tensor_backend::empty({rows, width}, output_gradient.options());
    const int64_t total = static_cast<int64_t>(rows) * width;
    if (total == 0) {
        return result;
    }
    MfqCudaGuard guard(output_gradient.device());
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    const cudaStream_t stream = mfq_current_cuda_stream();
    if (mxfp4 && dtype == mfq_tensor_backend::kFloat16 && rows <= 8) {
        if (rows == 1) {
            launch_mxfp4_backward_vec4_small_m<1>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else if (rows <= 2) {
            launch_mxfp4_backward_vec4_small_m<2>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else if (rows <= 4) {
            launch_mxfp4_backward_vec4_small_m<4>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else {
            launch_mxfp4_backward_vec4_small_m<8>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        }
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    if (!mxfp4 && dtype == mfq_tensor_backend::kFloat16 && rows <= 8) {
        const int output_groups = (outputs + 127) / 128;
        if (rows == 1 && output_groups <= 8) {
            launch_mxfp8_backward_block_reduce<1>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else if (rows == 1) {
            launch_mxfp8_backward_scalar_small_m<1>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else if (rows <= 2) {
            launch_mxfp8_backward_scalar_small_m<2>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else if (rows <= 4) {
            launch_mxfp8_backward_vec4_small_m<4>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        // Wide down projections amortize the per-column WMMA reduction;
        // square and output-wide projections keep the vector path.
        } else if (static_cast<int64_t>(width) >=
                2 * static_cast<int64_t>(outputs)) {
            launch_mxfp8_backward_mma<8, 16>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else if (2 * static_cast<int64_t>(width) >=
                static_cast<int64_t>(outputs)) {
            launch_mxfp8_backward_mma<8, 8>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        } else {
            launch_mxfp8_backward_vec4_small_m<8>(
                values, scales, output_gradient, result,
                rows, outputs, width, stream);
        }
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = mxfp4
        ? mxfp4_dequant_cuda(values, scales)
        : mxfp8_dequant_cuda(values, scales);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result, rows, outputs, width, stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
