#include <cuda_fp16.h>
#include "native/tensor/backend.h"
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "mx_matmul_impl.cuh"

namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

template <
    int TILE_M,
    int OUTPUTS_PER_WARP,
    bool CACHE_ACTIVATION,
    typename Output>
__global__ void mxfp8_small_m_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ x,
        Output * __restrict__ y,
        int rows,
        int outputs,
        int width,
        int groups,
        int outputs_per_group,
        int scale_row_shift,int scale_column_shift) {
    extern __shared__ float shared_activation[];
    constexpr int kWarps = 8;
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int block_first_output =
        int(blockIdx.x) * kWarps * OUTPUTS_PER_WARP;
    const int first_output =
        block_first_output + warp * OUTPUTS_PER_WARP;
    const int first_row = int(blockIdx.y) * TILE_M;
    const int activation_group = block_first_output / outputs_per_group;
    const bool valid_output = first_output < outputs;
    if (first_row >= rows) {
        return;
    }
    if constexpr (CACHE_ACTIVATION) {
        for (int column = int(threadIdx.x); column < width;
             column += int(blockDim.x)) {
            shared_activation[column] = __half2float(x[
                std::size_t(activation_group) * width + column]);
        }
        __syncthreads();
    }
    if (!valid_output) {
        return;
    }

    float accum[TILE_M][OUTPUTS_PER_WARP];
#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            accum[row][output] = 0.0f;
        }
    }

    const int scale_columns = width >> scale_column_shift;
    const int block_columns=1<<scale_column_shift;
    for (int block_column = 0; block_column < width; block_column += block_columns) {
        float scale[OUTPUTS_PER_WARP];
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            const int global_output = first_output + output;
            scale[output] = global_output < outputs
                ? decode_e8m0(scales[
                    (global_output >> scale_row_shift) * scale_columns +
                    (block_column >> scale_column_shift)])
                : 0.0f;
        }
#pragma unroll
        for (int local_column = lane; local_column < block_columns;
             local_column += 32) {
            const int column = block_column + local_column;
            float activation[TILE_M];
#pragma unroll
            for (int row = 0; row < TILE_M; ++row) {
                const int global_row = first_row + row;
                if constexpr (CACHE_ACTIVATION) {
                    activation[row] = shared_activation[column];
                } else {
                    activation[row] = global_row < rows
                        ? __half2float(x[
                            (std::size_t(global_row) * groups + activation_group) *
                            width + column])
                        : 0.0f;
                }
            }
#pragma unroll
            for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
                const int global_output = first_output + output;
                if (global_output >= outputs) {
                    continue;
                }
                const float weight = decode_e4m3fn(
                    values[(std::size_t)global_output * width + column]) *
                    scale[output];
#pragma unroll
                for (int row = 0; row < TILE_M; ++row) {
                    accum[row][output] += activation[row] * weight;
                }
            }
        }
    }

#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
        const int global_row = first_row + row;
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            const int global_output = first_output + output;
            const float total = warp_sum(accum[row][output]);
            if (lane == 0 && global_row < rows && global_output < outputs) {
                if constexpr (std::is_same_v<Output, float>) {
                    y[(std::size_t)global_row * outputs + global_output] =
                        total;
                } else {
                    y[(std::size_t)global_row * outputs + global_output] =
                        __float2half_rn(total);
                }
            }
        }
    }
}

template <int TILE_M, typename Output>
void launch_mxfp8_small_m(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & x,
        mfq_tensor_backend::Tensor & y,
        int groups,
        int outputs_per_group,
        cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    const int outputs = int(values.size(0));
#define MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, CACHE) \
    do { \
        const int outputs_per_block = warps * (OPW); \
        const dim3 grid( \
            unsigned((outputs + outputs_per_block - 1) / outputs_per_block), \
            1u, 1u); \
        const size_t shared_bytes = (CACHE) \
            ? size_t(values.size(1)) * sizeof(float) : 0; \
        mxfp8_small_m_kernel<TILE_M, OPW, CACHE, Output><<< \
            grid, threads, shared_bytes, stream>>>( \
                values.data_ptr<std::uint8_t>(), \
                scales.data_ptr<std::uint8_t>(), \
                reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), \
                reinterpret_cast<Output *>(y.data_ptr()), \
                int(x.size(0)), outputs, int(values.size(1)), \
                groups, outputs_per_group,mxfp8_row_shift(values,scales),mxfp8_column_shift(values,scales)); \
    } while (false)
#define MFQ_LAUNCH_MXFP8_OPW(OPW) \
    do { \
        if constexpr (TILE_M == 1) { \
            /* Cache activations only for wide output projections. */ \
            if (outputs > 8192) { \
                MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, true); \
            } else { \
                MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, false); \
            } \
        } else { \
            MFQ_LAUNCH_MXFP8_OPW_CACHE(OPW, false); \
        } \
    } while (false)
    // One output per warp exposes independent work across the full output
    // axis and avoids a model-shape threshold in the launch policy.
    MFQ_LAUNCH_MXFP8_OPW(1);
#undef MFQ_LAUNCH_MXFP8_OPW
#undef MFQ_LAUNCH_MXFP8_OPW_CACHE
}

__global__ void mxfp8_dequant_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        __half * __restrict__ dense,
        int outputs,
        int width,int scale_row_shift,int scale_column_shift) {
    const std::size_t index =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(outputs) * width;
    if (index >= count) {
        return;
    }
    const int output = int(index / width);
    const int column = int(index - std::size_t(output) * width);
    const int scale_columns = width >> scale_column_shift;
    const float scale = decode_e8m0(
        scales[(output >> scale_row_shift) * scale_columns + (column >> scale_column_shift)]);
    dense[index] = __float2half_rn(decode_e4m3fn(values[index]) * scale);
}

__global__ void mxfp8_embedding_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width,int scale_row_shift,int scale_column_shift) {
    const int64_t total = static_cast<int64_t>(count) * width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_index = static_cast<int>(linear / width);
        const int column = static_cast<int>(linear % width);
        const int64_t token = ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const int scale_columns = width >> scale_column_shift;
            const float scale = decode_e8m0(scales[
                (token >> scale_row_shift) * scale_columns + (column >> scale_column_shift)]);
            value = decode_e4m3fn(values[token * width + column]) * scale;
        }
        output[linear] = __float2half_rn(value);
    }
}

struct alignas(16) PackedPrefillStorage {
    static constexpr int k=64,stride=k+8;
    __align__(16) __half activation[16*stride];
    __align__(16) unsigned char weights[64*stride*sizeof(__half)];
    __device__ __forceinline__ __half* decoded() {return reinterpret_cast<__half*>(weights);}
    __device__ __forceinline__ float* result() {return reinterpret_cast<float*>(weights);}
    template<class Accumulator>
    __device__ __forceinline__ void accumulate(Accumulator& acc,int warp) {
        using namespace nvcuda;
        __syncthreads();
        #pragma unroll
        for(int sub=0;sub<k;sub+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,activation+sub,stride);
            wmma::load_matrix_sync(bf,decoded()+warp*16*stride+sub,stride);
            wmma::mma_sync(acc,af,bf,acc);
        }
        __syncthreads();
    }
};

template <typename Output>
__global__ void __launch_bounds__(128) mxfp8_matmul_kernel(
        const std::uint8_t* __restrict__ values,const std::uint8_t* __restrict__ scales,
        const __half* __restrict__ input,Output* __restrict__ output,
        int rows,int outputs,int width,int scale_row_shift,int scale_column_shift) {
    using namespace nvcuda;
    __shared__ PackedPrefillStorage tile;
    constexpr int tile_k=PackedPrefillStorage::k,stride=PackedPrefillStorage::stride;
    auto* a=tile.activation;auto* b=tile.decoded();auto* c=tile.result();
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int first_row=blockIdx.y*16,first_output=blockIdx.x*64;
    const int scale_columns=width>>scale_column_shift;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.0f);
    for(int k=0;k<width;k+=tile_k) {
        for(int i=threadIdx.x;i<16*tile_k;i+=128) {
            const int row=first_row+i/tile_k,column=k+i%tile_k;
            a[(i/tile_k)*stride+i%tile_k]=row<rows && column<width?
                input[std::size_t(row)*width+column]:__float2half_rn(0);
        }
        for(int i=threadIdx.x;i<64*(tile_k/2);i+=128) {
            const int neuron=first_output+i/(tile_k/2),column=k+2*(i%(tile_k/2));
            float lo=0,hi=0;
            if(neuron<outputs && column<width) {
                const auto offset=std::size_t(neuron)*width+column;
                const float scale=decode_e8m0(scales[std::size_t(neuron>>scale_row_shift)*scale_columns+(column>>scale_column_shift)]);
                lo=decode_e4m3fn(values[offset])*scale;
                hi=decode_e4m3fn(values[offset+1])*scale;
            }
            reinterpret_cast<__half2*>(b)[(i/(tile_k/2))*(stride/2)+i%(tile_k/2)]=__floats2half2_rn(lo,hi);
        }
        tile.accumulate(acc,warp);
    }
    wmma::store_matrix_sync(c+warp*16*16,acc,16,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<16*16;i+=32) {
        const int row=first_row+i/16,neuron=first_output+warp*16+i%16;
        if(row<rows && neuron<outputs) {
            if constexpr(std::is_same_v<Output,float>)output[std::size_t(row)*outputs+neuron]=c[warp*16*16+i];
            else output[std::size_t(row)*outputs+neuron]=__float2half_rn(c[warp*16*16+i]);
        }
    }
}

} // namespace

mfq_tensor_backend::Tensor mxfp8_dequant_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales) {
    validate_mxfp8(values, scales);
    auto dense = mfq_tensor_backend::empty(
        values.sizes(), values.options().dtype(mfq_tensor_backend::kFloat16));
    const std::size_t count = std::size_t(values.numel());
    constexpr int threads = 256;
    const int blocks = int((count + threads - 1) / threads);
    mxfp8_dequant_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<__half *>(dense.data_ptr<mfq_half>()),
            int(values.size(0)), int(values.size(1)),mxfp8_row_shift(values,scales),mxfp8_column_shift(values,scales));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

mfq_tensor_backend::Tensor mxfp8_embedding_lookup_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor token_ids) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP8 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1));
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(shape, values.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(count) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 4096));
    if (blocks > 0) {
        mxfp8_embedding_kernel<<<
            blocks, threads, 0, mfq_current_cuda_stream()>>>(
                values.data_ptr<std::uint8_t>(),
                scales.data_ptr<std::uint8_t>(), token_ids.data_ptr<int64_t>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                count, vocab, width,mxfp8_row_shift(values,scales),mxfp8_column_shift(values,scales));
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

mfq_tensor_backend::Tensor mxfp8_small_m_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2,
                "MXFP8 activation must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(x.size(1) == values.size(1),
                "MXFP8 activation width mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 small-M kernel supports M in [1, 8]");
    auto y = mfq_tensor_backend::empty(
        {x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
#define MFQ_LAUNCH_MXFP8(TILE) \
    launch_mxfp8_small_m<TILE, __half>( \
        values, scales, x, y, 1, int(values.size(0)), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_MXFP8(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_MXFP8(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_MXFP8(4);
    } else {
        MFQ_LAUNCH_MXFP8(8);
    }
#undef MFQ_LAUNCH_MXFP8
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_matmul_f16_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor input) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 && input.size(1) == values.size(1) &&
                    input.size(0) <= std::numeric_limits<int>::max(),
                "MXFP8 activation must be non-empty contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    mxfp8_matmul_kernel<__half><<<dim3((outputs+63)/64,(rows+15)/16),128,0,mfq_current_cuda_stream()>>>(
        values.data_ptr<std::uint8_t>(),scales.data_ptr<std::uint8_t>(),
        reinterpret_cast<const __half*>(input.data_ptr<mfq_half>()),
        reinterpret_cast<__half*>(result.data_ptr<mfq_half>()),rows,outputs,width,
        mxfp8_row_shift(values,scales),mxfp8_column_shift(values,scales));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

mfq_tensor_backend::Tensor mxfp8_small_m_f32_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output activation geometry mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 FP32-output small-M kernel supports M in [1, 8]");
    auto y = mfq_tensor_backend::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(mfq_tensor_backend::kFloat32));
    const cudaStream_t stream = mfq_current_cuda_stream();
#define MFQ_LAUNCH_MXFP8_F32(TILE) \
    launch_mxfp8_small_m<TILE, float>( \
        values, scales, x, y, 1, int(values.size(0)), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_MXFP8_F32(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_MXFP8_F32(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_MXFP8_F32(4);
    } else {
        MFQ_LAUNCH_MXFP8_F32(8);
    }
#undef MFQ_LAUNCH_MXFP8_F32
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_gemm_f32_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output GEMM activation geometry mismatch");
    const int M = int(x.size(0));
    const int N = int(values.size(0));
    auto y = mfq_tensor_backend::empty(
        {M, N}, x.options().dtype(mfq_tensor_backend::kFloat32));
    mxfp8_matmul_kernel<float><<<dim3((N+63)/64,(M+15)/16),128,0,mfq_current_cuda_stream()>>>(
        values.data_ptr<std::uint8_t>(),scales.data_ptr<std::uint8_t>(),
        reinterpret_cast<const __half*>(x.data_ptr<mfq_half>()),y.data_ptr<float>(),M,N,int(x.size(1)),
        mxfp8_row_shift(values,scales),mxfp8_column_shift(values,scales));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3,
                "MXFP8 groupwise activation must be contiguous CUDA fp16 rank-3");
    MFQ_RUNTIME_CHECK(groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise geometry mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise small-M kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    MFQ_RUNTIME_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = mfq_tensor_backend::empty({x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
#define MFQ_LAUNCH_GROUPED_MXFP8(TILE) \
    launch_mxfp8_small_m<TILE, __half>( \
        values, scales, x, y, int(groups), int(outputs_per_group), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_GROUPED_MXFP8(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_GROUPED_MXFP8(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_GROUPED_MXFP8(4);
    } else {
        MFQ_LAUNCH_GROUPED_MXFP8(8);
    }
#undef MFQ_LAUNCH_GROUPED_MXFP8
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

mfq_tensor_backend::Tensor mxfp8_groupwise_small_m_f32_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    MFQ_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3 &&
                    groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise FP32-output geometry mismatch");
    MFQ_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise FP32-output kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    MFQ_RUNTIME_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = mfq_tensor_backend::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(mfq_tensor_backend::kFloat32));
    const cudaStream_t stream = mfq_current_cuda_stream();
#define MFQ_LAUNCH_GROUPED_MXFP8_F32(TILE) \
    launch_mxfp8_small_m<TILE, float>( \
        values, scales, x, y, int(groups), int(outputs_per_group), stream)
    if (x.size(0) == 1) {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(1);
    } else if (x.size(0) <= 2) {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(2);
    } else if (x.size(0) <= 4) {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(4);
    } else {
        MFQ_LAUNCH_GROUPED_MXFP8_F32(8);
    }
#undef MFQ_LAUNCH_GROUPED_MXFP8_F32
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

namespace {

__global__ void mxfp4_dequant_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        __half * __restrict__ dense,
        int outputs,
        int width) {
    const std::size_t index =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(outputs) * width;
    if (index >= count) return;
    const int output = int(index / width);
    const int column = int(index - std::size_t(output) * width);
    const std::uint8_t packed = values[
        std::size_t(output) * (width / 2) + column / 2];
    const std::uint8_t code = static_cast<std::uint8_t>(
        (packed >> ((column & 1) * 4)) & 15u);
    const float scale = decode_e8m0(
        scales[std::size_t(output) * (width / 32) + column / 32]);
    dense[index] = __float2half_rn(decode_mxfp4_e2m1(code) * scale);
}

__global__ void mxfp4_embedding_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width) {
    const int64_t total = static_cast<int64_t>(count) * width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_index = static_cast<int>(linear / width);
        const int column = static_cast<int>(linear % width);
        const int64_t token = ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const std::uint8_t byte = values[
                token * (width / 2) + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (byte >> ((column & 1) * 4)) & 15u);
            value = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[token * (width / 32) + column / 32]);
        }
        output[linear] = __float2half_rn(value);
    }
}

__global__ void __launch_bounds__(256) mxfp4_matmul_f16_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        int rows,
        int outputs,
        int width,
        int row_tiles,
        int output_tiles) {
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int output_tile = static_cast<int>(task % output_tiles);
        const int row_tile = static_cast<int>(task / output_tiles);
        const int neuron = output_tile * kOutputsPerTile + warp;
        if (neuron >= outputs) continue;
        const int first_row = row_tile * kRowsPerTile;
        float accumulators[kRowsPerTile];
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            accumulators[item] = 0.0f;
        }
        const int packed_columns = width / 2;
        const int scale_columns = width / 32;
        for (int column = lane; column < width; column += 32) {
            const std::uint8_t packed = values[
                static_cast<int64_t>(neuron) * packed_columns + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const float weight = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[static_cast<int64_t>(neuron) * scale_columns +
                       column / 32]);
#pragma unroll
            for (int item = 0; item < kRowsPerTile; ++item) {
                const int row = first_row + item;
                if (row < rows) {
                    accumulators[item] = fmaf(
                        __half2float(input[static_cast<int64_t>(row) * width + column]),
                        weight,
                        accumulators[item]);
                }
            }
        }
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            const float value = warp_sum(accumulators[item]);
            const int row = first_row + item;
            if (lane == 0 && row < rows) {
                output[static_cast<int64_t>(row) * outputs + neuron] =
                    __float2half_rn(value);
            }
        }
    }
}

template <bool COMPACT_ROUTE>
__global__ void __launch_bounds__(256) mxfp4_moe_grouped_f16_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ input,
        const std::int32_t * __restrict__ ids,
        const std::int32_t * __restrict__ expert_local,
        const std::int32_t * __restrict__ ids_dst,
        const std::int32_t * __restrict__ expert_bounds,
        const std::int32_t * __restrict__ tile_bounds,
        const std::int32_t * __restrict__ tile_experts,
        __half * __restrict__ output,
        int tokens,
        int routes,
        int global_experts,
        int pool_experts,
        int out_per_expert,
        int width,
        int max_tiles,
        int row_tiles,
        bool routed_input) {
    constexpr int kTileM = 8;
    constexpr int kRowsPerBlock = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int pairs = tokens * routes;
    const int scale_columns = width / 32;
    const int packed_columns = width / 2;
    const int64_t max_tasks = COMPACT_ROUTE
        ? static_cast<int64_t>(max_tiles) * row_tiles
        : static_cast<int64_t>(pairs) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        int expert = -1;
        int first = 0;
        int last = 0;
        int single_pair = -1;
        if constexpr (COMPACT_ROUTE) {
            const int tile = static_cast<int>(task / row_tiles);
            if (tile >= tile_bounds[global_experts]) continue;
            expert = tile_experts[tile];
            const int local_tile = tile - tile_bounds[expert];
            first = expert_bounds[expert] + local_tile * kTileM;
            last = min(first + kTileM, expert_bounds[expert + 1]);
        } else {
            single_pair = static_cast<int>(task / row_tiles);
            expert = ids[single_pair];
            first = single_pair;
            last = single_pair + 1;
        }
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned>(local_expert) >=
                static_cast<unsigned>(pool_experts)) {
            continue;
        }
        const int local_row = row_tile * kRowsPerBlock + warp;
        if (local_row >= out_per_expert) continue;
        const int packed_row = local_expert * out_per_expert + local_row;
        float accum[kTileM];
#pragma unroll
        for (int item = 0; item < kTileM; ++item) accum[item] = 0.0f;
        for (int column = lane; column < width; column += 32) {
            const std::uint8_t packed = values[
                static_cast<int64_t>(packed_row) * packed_columns + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const float weight = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[static_cast<int64_t>(packed_row) * scale_columns +
                       column / 32]);
#pragma unroll
            for (int item = 0; item < kTileM; ++item) {
                int pair = -1;
                if constexpr (COMPACT_ROUTE) {
                    const int compact = first + item;
                    if (compact >= last) continue;
                    pair = ids_dst[compact];
                } else {
                    if (item != 0) continue;
                    pair = single_pair;
                }
                const int source_row = routed_input ? pair : pair / routes;
                accum[item] = fmaf(
                    __half2float(input[static_cast<int64_t>(source_row) * width + column]),
                    weight,
                    accum[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < kTileM; ++item) {
            float value = warp_sum(accum[item]);
            if (lane != 0) continue;
            int pair = -1;
            if constexpr (COMPACT_ROUTE) {
                const int compact = first + item;
                if (compact >= last) continue;
                pair = ids_dst[compact];
            } else {
                if (item != 0) continue;
                pair = single_pair;
            }
            output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                __float2half_rn(value);
        }
    }
}

__global__ void __launch_bounds__(128) mxfp4_moe_grouped_tensor_kernel(
        const std::uint8_t* __restrict__ values,
        const std::uint8_t* __restrict__ scales,
        const __half* __restrict__ input,
        const std::int32_t* __restrict__ expert_local,
        const std::int32_t* __restrict__ ids_dst,
        const std::int32_t* __restrict__ expert_bounds,
        const std::int32_t* __restrict__ tile_bounds,
        const std::int32_t* __restrict__ tile_experts,
        __half* __restrict__ output,
        int routes,int global_experts,int pool_experts,
        int outputs,int width,int max_tiles,int output_tiles,bool routed_input) {
    using namespace nvcuda;
    __shared__ PackedPrefillStorage tile_storage;
    constexpr int tile_k=PackedPrefillStorage::k,stride=PackedPrefillStorage::stride;
    auto* a=tile_storage.activation;auto* b=tile_storage.decoded();auto* c=tile_storage.result();
    __shared__ int pair_ids[16];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int64_t tasks=int64_t(max_tiles)*output_tiles;
    for(int64_t task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const int tile=int(task/output_tiles);
        if(tile>=tile_bounds[global_experts])continue;
        const int expert=tile_experts[tile],local=expert_local[expert];
        if(static_cast<unsigned>(local)>=static_cast<unsigned>(pool_experts))continue;
        const int local_tile=tile-tile_bounds[expert];
        if(local_tile&1)continue;
        const int first=expert_bounds[expert]+local_tile*8;
        const int last=min(first+16,expert_bounds[expert+1]);
        const int first_output=int(task%output_tiles)*64;
        if(threadIdx.x<16)pair_ids[threadIdx.x]=first+threadIdx.x<last?
            ids_dst[first+threadIdx.x]:-1;
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> acc;
        wmma::fill_fragment(acc,0.0f);
        for(int k=0;k<width;k+=tile_k) {
            for(int i=threadIdx.x;i<16*tile_k;i+=128) {
                const int pair=pair_ids[i/tile_k],column=k+i%tile_k;
                const int source_row=routed_input?pair:pair/routes;
                a[(i/tile_k)*stride+i%tile_k]=pair>=0 && column<width?
                    input[int64_t(source_row)*width+column]:__float2half_rn(0);
            }
            for(int i=threadIdx.x;i<64*(tile_k/2);i+=128) {
                const int neuron=first_output+i/(tile_k/2),column=k+2*(i%(tile_k/2));
                float lo=0,hi=0;
                if(neuron<outputs && column<width) {
                    const int64_t packed_row=int64_t(local)*outputs+neuron;
                    const auto packed=values[packed_row*(width/2)+column/2];
                    const float scale=decode_e8m0(scales[packed_row*(width/32)+column/32]);
                    lo=decode_mxfp4_e2m1(packed&15u)*scale;
                    hi=decode_mxfp4_e2m1(packed>>4u)*scale;
                }
                reinterpret_cast<__half2*>(b)[(i/(tile_k/2))*(stride/2)+i%(tile_k/2)]=__floats2half2_rn(lo,hi);
            }
            tile_storage.accumulate(acc,warp);
        }
        wmma::store_matrix_sync(c+warp*16*16,acc,16,wmma::mem_row_major);
        __syncwarp();
        for(int i=lane;i<16*16;i+=32) {
            const int pair=pair_ids[i/16],neuron=first_output+warp*16+i%16;
            if(pair>=0 && neuron<outputs)
                output[int64_t(pair)*outputs+neuron]=__float2half_rn(c[warp*16*16+i]);
        }
        __syncthreads();
    }
}

void validate_mxfp4_moe(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales,
        const mfq_tensor_backend::Tensor & input,
        const mfq_tensor_backend::Tensor & ids,
        const mfq_tensor_backend::Tensor & expert_local,
        const mfq_tensor_backend::Tensor & output,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len) {
    MFQ_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda() && input.is_cuda() &&
                    ids.is_cuda() && expert_local.is_cuda() && output.is_cuda(),
                "MXFP4 routed tensors must be CUDA tensors");
    MFQ_RUNTIME_CHECK(values.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    scales.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    input.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    ids.scalar_type() == mfq_tensor_backend::kInt32 &&
                    expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
                    output.scalar_type() == mfq_tensor_backend::kFloat16,
                "MXFP4 routed tensor dtypes are invalid");
    MFQ_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous() &&
                    input.is_contiguous() && ids.is_contiguous() &&
                    expert_local.is_contiguous() && output.is_contiguous(),
                "MXFP4 routed tensors must be contiguous");
    MFQ_RUNTIME_CHECK(neuron_len > 0 && neuron_len % 32 == 0 &&
                    values.dim() == 2 &&
                    values.size(0) == pool_experts * out_per_expert &&
                    values.size(1) == neuron_len / 2 &&
                    scales.sizes() == mfq_tensor_backend::IntArrayRef(
                        {pool_experts * out_per_expert, neuron_len / 32}),
                "MXFP4 routed weight geometry mismatch");
    MFQ_RUNTIME_CHECK((input.dim() == 2 || input.dim() == 3) &&
                    input.size(-1) == neuron_len && ids.dim() == 2 &&
                    input.size(0) == ids.size(0) &&
                    output.sizes() == mfq_tensor_backend::IntArrayRef(
                        {ids.size(0), ids.size(1), out_per_expert}) &&
                    expert_local.numel() == global_experts,
                "MXFP4 routed activation/route geometry mismatch");
}

} // namespace

mfq_tensor_backend::Tensor mxfp4_dequant_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales) {
    validate_mxfp4_dense(values, scales);
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto dense = mfq_tensor_backend::empty(
        {outputs, width}, values.options().dtype(mfq_tensor_backend::kFloat16));
    const std::size_t count = static_cast<std::size_t>(outputs) * width;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    mxfp4_dequant_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<__half *>(dense.data_ptr<mfq_half>()),
            outputs, width);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

mfq_tensor_backend::Tensor mxfp4_embedding_lookup_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor token_ids) {
    validate_mxfp4_dense(values, scales);
    MFQ_RUNTIME_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == mfq_tensor_backend::kInt64 &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP4 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = mfq_tensor_backend::empty(shape, values.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(count) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 4096));
    if (blocks > 0) {
        mxfp4_embedding_kernel<<<
            blocks, threads, 0, mfq_current_cuda_stream()>>>(
                values.data_ptr<std::uint8_t>(),
                scales.data_ptr<std::uint8_t>(), token_ids.data_ptr<int64_t>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
                count, vocab, width);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

mfq_tensor_backend::Tensor mxfp4_matmul_f16_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor input) {
    validate_mxfp4_dense(values, scales);
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 &&
                    input.size(0) <= std::numeric_limits<int>::max() &&
                    input.size(1) == values.size(1) * 2,
                "MXFP4 activation must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int row_tiles = (rows + kRowsPerTile - 1) / kRowsPerTile;
    const int output_tiles =
        (outputs + kOutputsPerTile - 1) / kOutputsPerTile;
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    const int blocks = static_cast<int>(std::min<int64_t>(tasks, 4096));
    const dim3 threads(32, kOutputsPerTile);
    mxfp4_matmul_f16_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            values.data_ptr<std::uint8_t>(),
            scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            rows, outputs, width, row_tiles, output_tiles);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

mfq_tensor_backend::Tensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
        mfq_tensor_backend::Tensor values,
        mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor expert_local,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len,
        mfq_tensor_backend::Tensor output,
        mfq_tensor_backend::Tensor ids_dst,
        mfq_tensor_backend::Tensor expert_bounds,
        mfq_tensor_backend::Tensor tile_bounds,
        mfq_tensor_backend::Tensor tile_experts) {
    validate_mxfp4_moe(
        values, scales, input, ids, expert_local, output,
        global_experts, pool_experts, out_per_expert, neuron_len);
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const int row_tiles =
        (static_cast<int>(out_per_expert) + 7) / 8;
    const dim3 threads(32, 8);
    const cudaStream_t stream = mfq_current_cuda_stream();
    if (ids_dst.numel() == ids.numel()) {
        MFQ_RUNTIME_CHECK(expert_bounds.is_cuda() && tile_bounds.is_cuda() &&
                        tile_experts.is_cuda() &&
                        expert_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                        tile_bounds.scalar_type() == mfq_tensor_backend::kInt32 &&
                        tile_experts.scalar_type() == mfq_tensor_backend::kInt32 &&
                        expert_bounds.is_contiguous() &&
                        tile_bounds.is_contiguous() && tile_experts.is_contiguous() &&
                        expert_bounds.numel() >= global_experts + 1 &&
                        tile_bounds.numel() >= global_experts + 1,
                    "MXFP4 compact route map is invalid");
        const int max_tiles = (pairs + 7) / 8 + static_cast<int>(global_experts);
        if(pairs>=8*global_experts) {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "mxfp4_moe_prefill_dispatch path=packed_tensor_core tokens=%d routes=%d experts=%lld pool=%lld width=%lld outputs=%lld\n",
                tokens,routes,static_cast<long long>(global_experts),static_cast<long long>(pool_experts),
                static_cast<long long>(neuron_len),static_cast<long long>(out_per_expert));
            const int output_tiles=(static_cast<int>(out_per_expert)+63)/64;
            const int blocks=static_cast<int>(std::min<int64_t>(int64_t(max_tiles)*output_tiles,4096));
            mxfp4_moe_grouped_tensor_kernel<<<blocks,128,0,stream>>>(
                values.data_ptr<std::uint8_t>(),scales.data_ptr<std::uint8_t>(),
                reinterpret_cast<const __half*>(input.data_ptr<mfq_half>()),
                expert_local.data_ptr<std::int32_t>(),ids_dst.data_ptr<std::int32_t>(),
                expert_bounds.data_ptr<std::int32_t>(),tile_bounds.data_ptr<std::int32_t>(),
                tile_experts.data_ptr<std::int32_t>(),reinterpret_cast<__half*>(output.data_ptr<mfq_half>()),
                routes,static_cast<int>(global_experts),static_cast<int>(pool_experts),
                static_cast<int>(out_per_expert),static_cast<int>(neuron_len),max_tiles,
                output_tiles,input.dim()==3);
            MFQ_CUDA_KERNEL_LAUNCH_CHECK();
            return output;
        }
        const int blocks = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(max_tiles) * row_tiles, 4096));
        mxfp4_moe_grouped_f16_kernel<true><<<blocks, threads, 0, stream>>>(
            values.data_ptr<std::uint8_t>(), scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            ids.data_ptr<std::int32_t>(), expert_local.data_ptr<std::int32_t>(),
            ids_dst.data_ptr<std::int32_t>(), expert_bounds.data_ptr<std::int32_t>(),
            tile_bounds.data_ptr<std::int32_t>(), tile_experts.data_ptr<std::int32_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens, routes, static_cast<int>(global_experts),
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),
            static_cast<int>(neuron_len), max_tiles, row_tiles, input.dim() == 3);
    } else {
        const int blocks = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(pairs) * row_tiles, 4096));
        mxfp4_moe_grouped_f16_kernel<false><<<blocks, threads, 0, stream>>>(
            values.data_ptr<std::uint8_t>(), scales.data_ptr<std::uint8_t>(),
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            ids.data_ptr<std::int32_t>(), expert_local.data_ptr<std::int32_t>(),
            nullptr, nullptr, nullptr, nullptr,
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens, routes, static_cast<int>(global_experts),
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),
            static_cast<int>(neuron_len), 0, row_tiles, input.dim() == 3);
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
