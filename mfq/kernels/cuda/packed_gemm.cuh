#pragma once

#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

namespace mfq::packed {

// Decode each weight tile once for 32 activation rows. No dense weight buffer
// is allocated. WideWeight keeps MXFP4's E8M0 exponent range (including weights
// outside FP16) using exact TF32 representations of its E2M1 mantissas.
template <typename Decoder, bool WideWeight = false, int TileM = 32, int TileN = 64, bool VectorDecode = false>
__global__ void gemm_nt(
    Decoder decode, const __half* input, __half* output,
    int rows, int outputs, int width) {
    constexpr int tile_m = TileM, tile_n = TileN, tile_k = 32, stride = 40;
    constexpr int warps_n = tile_n / 16, warps = tile_m * tile_n / 256;
    constexpr int mma_k = WideWeight ? 8 : 16;
    using Storage = std::conditional_t<WideWeight, float, __half>;
    using Operand = std::conditional_t<WideWeight, nvcuda::wmma::precision::tf32, __half>;
    __shared__ __align__(32) Storage a[tile_m * stride];
    __shared__ __align__(32) Storage b[tile_n * stride];
    __shared__ __align__(32) float result[warps * 256];
    const int warp = threadIdx.x / 32;
    const int row_begin = blockIdx.y * tile_m;
    const int output_begin = blockIdx.x * tile_n;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, mma_k, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    for (int begin = 0; begin < width; begin += tile_k) {
        for (int i = threadIdx.x; i < tile_m * tile_k; i += blockDim.x) {
            const int m = i / tile_k, k = i % tile_k;
            const float value = row_begin + m < rows && begin + k < width
                ? __half2float(input[static_cast<std::size_t>(row_begin + m) * width + begin + k]) : 0.0f;
            if constexpr (WideWeight) a[m * stride + k] = value;
            else a[m * stride + k] = __float2half_rn(value);
        }
        if constexpr (VectorDecode) {
            for (int i = threadIdx.x; i < tile_n * tile_k / 8; i += blockDim.x) {
                const int n = i / (tile_k / 8), k = (i % (tile_k / 8)) * 8;
                decode.load8(output_begin + n, begin + k, b + n * stride + k, outputs, width);
            }
        } else for (int i = threadIdx.x; i < tile_n * tile_k; i += blockDim.x) {
            const int n = i / tile_k, k = i % tile_k;
            const float value = output_begin + n < outputs && begin + k < width
                ? decode(output_begin + n, begin + k) : 0.0f;
            if constexpr (WideWeight) b[n * stride + k] = value;
            else b[n * stride + k] = __float2half_rn(value);
        }
        __syncthreads();
        for (int k = 0; k < tile_k; k += mma_k) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, mma_k, Operand,
                nvcuda::wmma::row_major> fa;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, mma_k, Operand,
                nvcuda::wmma::col_major> fb;
            nvcuda::wmma::load_matrix_sync(fa, a + (warp / warps_n) * 16 * stride + k, stride);
            nvcuda::wmma::load_matrix_sync(fb, b + (warp % warps_n) * 16 * stride + k, stride);
            nvcuda::wmma::mma_sync(acc, fa, fb, acc);
        }
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(result + warp * 256, acc, 16, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < tile_m * tile_n; i += blockDim.x) {
        const int m = i / tile_n, n = i % tile_n;
        if (row_begin + m < rows && output_begin + n < outputs) {
            const int owner = (m / 16) * warps_n + n / 16;
            output[static_cast<std::size_t>(row_begin + m) * outputs + output_begin + n] =
                __float2half_rn(result[owner * 256 + (m % 16) * 16 + n % 16]);
        }
    }
}

} // namespace mfq::packed
