#include "mfq_cuda_quant_ops.h"
#define MFQ_NVQ_BACKWARD 1
#include "nvq_matmul_impl.cuh"

mfq_tensor_backend::Tensor nvq_dequant_cuda(
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor aux,
    mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale,
    mfq_tensor_backend::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
                     (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto output = mfq_tensor_backend::empty({N, K}, neuron_scale.options().dtype(mfq_tensor_backend::kFloat16));
    const int block = 128;
    const dim3 grid((nsign + block - 1) / block, std::min(N, 65535));
    cudaStream_t stream = mfq_current_cuda_stream();
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_dequant_kernel<F><<<grid, block, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            N, K, ng, nvec, nsign, static_cast<int>(sub_bits), static_cast<int>(sign_mode));
    });
    MFQ_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ dequant kernel launch failed");
    return output;
}

namespace {

template <int FORMAT, int Rows>
__global__ void __launch_bounds__(32) nvq_backward_quad_partial_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * sub_scale,
        int64_t sub_scale_nbytes,
        const float * neuron_scale,
        const int8_t * codebook,
        const __half * output_gradient,
        __half * partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x);
    const int chunk_linear = static_cast<int>(blockIdx.x) * 32 + lane;
    const int k0 = chunk_linear * 4;
    const bool valid = k0 < K;
    const int group = k0 / kGroupSize;
    const int chunk = (k0 - group * kGroupSize) / 4;
    const int split = static_cast<int>(blockIdx.y);
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, N);
    float accumulators[Rows][4];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            accumulators[row][component] = 0.0f;
        }
    }
    for (int output = output0; output < output_end; ++output) {
        float anchor = lane == 0 ? neuron_scale[output] : 0.0f;
        anchor = __shfl_sync(0xffffffffu, anchor, 0);
        uint32_t state = 0;
        int packed = 0;
        float weight_scale = 0.0f;
        if (valid) {
            const int64_t state_index =
                static_cast<int64_t>(output) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    state_index,sub_bits,ng);
            packed = decode_chunk4<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                output, group, chunk, nvec, nsign, ng,
                sign_mode, state);
            weight_scale = format_scale<FORMAT>(anchor, state, codebook);
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row < M
                ? __half2float(output_gradient[
                    static_cast<int64_t>(row) * N + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            const float factor = gradient * weight_scale;
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                const int value = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * component)) & 0xff));
                accumulators[row][component] = fmaf(
                    factor, static_cast<float>(value),
                    accumulators[row][component]);
            }
        }
    }
    if (valid) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row < M) {
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    const int column = k0 + component;
                    if (column < K) {
                        partials[(static_cast<int64_t>(split) * M + row) *
                            K + column] = __float2half_rn(
                                accumulators[row][component]);
                    }
                }
            }
        }
    }
}

template <int FORMAT, int Rows>
void launch_nvq_backward_quad_small_m(
        const mfq_tensor_backend::Tensor & indices,
        const mfq_tensor_backend::Tensor & aux,
        const mfq_tensor_backend::Tensor & sub_scale,
        const mfq_tensor_backend::Tensor & neuron_scale,
        const mfq_tensor_backend::Tensor & codebook,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((K + 127) / 128),
        static_cast<unsigned>((N + output_tile - 1) / output_tile));
    nvq_backward_quad_partial_kernel<FORMAT, Rows><<<
        grid, 32, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(partials.data_ptr<mfq_half>()),
            M, N, K, ng, nvec, nsign,
            sub_bits, sign_mode, output_tile);
}

template <int FORMAT, int Rows>
__global__ void __launch_bounds__(32) nvq_backward_vec8_partial_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * sub_scale,
        int64_t sub_scale_nbytes,
        const float * neuron_scale,
        const int8_t * codebook,
        const __half * output_gradient,
        __half * partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x);
    const int segment = static_cast<int>(blockIdx.x) * 32 + lane;
    const int k0 = segment * 8;
    const bool valid = k0 < K;
    const int group = k0 / kGroupSize;
    const int split = static_cast<int>(blockIdx.y);
    const int row0 = static_cast<int>(blockIdx.z) * Rows;
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, N);
    float accumulators[Rows][8];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int component = 0; component < 8; ++component) {
            accumulators[row][component] = 0.0f;
        }
    }
    for (int output = output0; output < output_end; ++output) {
        float anchor = lane == 0 ? neuron_scale[output] : 0.0f;
        anchor = __shfl_sync(0xffffffffu, anchor, 0);
        uint32_t state = 0;
        NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
        float weight_scale = 0.0f;
        if (valid) {
            const int64_t state_index =
                static_cast<int64_t>(output) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    state_index,sub_bits,ng);
            decoded = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                output, segment, group, ng, nvec, nsign,
                sign_mode, state);
            weight_scale = format_scale<FORMAT>(anchor, state, codebook);
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row0 + row < M
                ? __half2float(output_gradient[
                    static_cast<int64_t>(row0 + row) * N + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            const float factor = gradient * weight_scale;
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                accumulators[row][component] = fmaf(
                    factor, static_cast<float>(value),
                    accumulators[row][component]);
            }
        }
    }
    if (valid) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row0 + row < M) {
#pragma unroll
                for (int component = 0; component < 8; ++component) {
                    const int column = k0 + component;
                    if (column < K) {
                        partials[(static_cast<int64_t>(split) * M + row0 + row) *
                            K + column] = __float2half_rn(
                                accumulators[row][component]);
                    }
                }
            }
        }
    }
}

template <int FORMAT, int Rows>
void launch_nvq_backward_vec8_small_m(
        const mfq_tensor_backend::Tensor & indices,
        const mfq_tensor_backend::Tensor & aux,
        const mfq_tensor_backend::Tensor & sub_scale,
        const mfq_tensor_backend::Tensor & neuron_scale,
        const mfq_tensor_backend::Tensor & codebook,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((K + 255) / 256),
        static_cast<unsigned>((N + output_tile - 1) / output_tile),
        static_cast<unsigned>((M + Rows - 1) / Rows));
    nvq_backward_vec8_partial_kernel<FORMAT, Rows><<<
        grid, 32, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(partials.data_ptr<mfq_half>()),
            M, N, K, ng, nvec, nsign,
            sub_bits, sign_mode, output_tile);
}

template <int FORMAT>
__global__ void __launch_bounds__(256) nvq_backward_mma_m8_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * sub_scale,
        int64_t sub_scale_nbytes,
        const float * neuron_scale,
        const int8_t * codebook,
        const __half * output_gradient,
        __half * input_gradient,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode) {
    constexpr int kWarps = 8;
    __shared__ __align__(16) __half gradient_tile[kWarps][16 * 16];
    __shared__ __align__(16) __half weight_tile[kWarps][16 * 16];
    __shared__ __align__(16) float result_tile[kWarps][16 * 16];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int column0 = static_cast<int>(blockIdx.x) * 16;
    for (int index = lane; index < 16 * 16; index += 32) {
        gradient_tile[warp][index] = __float2half_rn(0.0f);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int output_group = warp * 128;
         output_group < N;
         output_group += kWarps * 128) {
        for (int output0 = output_group;
             output0 < min(output_group + 128, N);
             output0 += 16) {
            for (int index = lane; index < M * 16; index += 32) {
                const int row = index >> 4;
                const int output = output0 + (index & 15);
                gradient_tile[warp][index] = output < N
                    ? output_gradient[static_cast<int64_t>(row) * N + output]
                    : __float2half_rn(0.0f);
            }
            const int output = output0 + (lane & 15);
            const int half = lane >> 4;
            const int k0 = column0 + half * 8;
            NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
            float weight_scale = 0.0f;
            if (output < N && k0 < K) {
                const int group = k0 / kGroupSize;
                const uint32_t state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    static_cast<int64_t>(output)*ng+group,sub_bits,ng);
                decoded = load_nvq_vec8<FORMAT>(
                    indices, indices_nbytes, aux, aux_nbytes, codebook,
                    output, k0 >> 3, group, ng, nvec, nsign,
                    sign_mode, state);
                weight_scale = format_scale<FORMAT>(
                    neuron_scale[output], state, codebook);
            }
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                const int column = k0 + component;
                weight_tile[warp][
                    (lane & 15) * 16 + half * 8 + component] =
                    decoded.valid && column < K
                    ? __float2half_rn(weight_scale * static_cast<float>(value))
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
    if (index < M * 16) {
        const int row = index >> 4;
        const int column = column0 + (index & 15);
        if (column < K) {
            float value = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
                value += result_tile[source_warp][index];
            }
            input_gradient[static_cast<int64_t>(row) * K + column] =
                __float2half_rn(value);
        }
    }
}

template <int FORMAT>
void launch_nvq_backward_mma_m8(
        const mfq_tensor_backend::Tensor & indices,
        const mfq_tensor_backend::Tensor & aux,
        const mfq_tensor_backend::Tensor & sub_scale,
        const mfq_tensor_backend::Tensor & neuron_scale,
        const mfq_tensor_backend::Tensor & codebook,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        cudaStream_t stream) {
    nvq_backward_mma_m8_kernel<FORMAT><<<
        (K + 15) / 16, 256, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            M, N, K, ng, nvec, nsign, sub_bits, sign_mode);
}

template <int FORMAT>
__global__ void __launch_bounds__(256) nepq_backward_mma_m8_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * state_stream,
        int64_t state_nbytes,
        const float * neuron_scale,
        const int8_t * table_pool,
        const uint8_t * bank_ids,
        const __half * output_gradient,
        __half * input_gradient,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits) {
    constexpr int kWarps = 8;
    __shared__ __align__(16) __half gradient_tile[kWarps][16 * 16];
    __shared__ __align__(16) __half weight_tile[kWarps][16 * 16];
    __shared__ __align__(16) float result_tile[kWarps][16 * 16];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int column0 = static_cast<int>(blockIdx.x) * 16;
    for (int index = lane; index < 16 * 16; index += 32) {
        gradient_tile[warp][index] = __float2half_rn(0.0f);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int output_group = warp * 128;
         output_group < N;
         output_group += kWarps * 128) {
        for (int output0 = output_group;
             output0 < min(output_group + 128, N);
             output0 += 16) {
            for (int index = lane; index < M * 16; index += 32) {
                const int row = index >> 4;
                const int output = output0 + (index & 15);
                gradient_tile[warp][index] = output < N
                    ? output_gradient[static_cast<int64_t>(row) * N + output]
                    : __float2half_rn(0.0f);
            }
            const int output = output0 + (lane & 15);
            const int half = lane >> 4;
            const int k0 = column0 + half * 8;
            NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
            float weight_scale = 0.0f;
            if (output < N && k0 < K) {
                const int group = k0 / kGroupSize;
                const uint32_t state = load_packed_bits(
                    state_stream,
                    (static_cast<int64_t>(output) * ng + group) * state_bits,
                    state_bits,
                    state_nbytes);
                const int8_t * table = nepq_active_table(
                    table_pool, bank_ids, output, group,
                    nsuper, table_stride);
                decoded = load_nepq_vec8<FORMAT>(
                    indices, indices_nbytes, aux, aux_nbytes, table,
                    output, k0 >> 3, group, ng, nvec, nsign, 0, state);
                weight_scale = format_scale<FORMAT>(
                    neuron_scale[output], state, table);
            }
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                const int column = k0 + component;
                weight_tile[warp][
                    (lane & 15) * 16 + half * 8 + component] =
                    decoded.valid && column < K
                    ? __float2half_rn(weight_scale * static_cast<float>(value))
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
    if (index < M * 16) {
        const int row = index >> 4;
        const int column = column0 + (index & 15);
        if (column < K) {
            float value = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
                value += result_tile[source_warp][index];
            }
            input_gradient[static_cast<int64_t>(row) * K + column] =
                __float2half_rn(value);
        }
    }
}

template <int FORMAT>
void launch_nepq_backward_mma_m8(
        const mfq_tensor_backend::Tensor & indices,
        const mfq_tensor_backend::Tensor & aux,
        const mfq_tensor_backend::Tensor & state_stream,
        const mfq_tensor_backend::Tensor & neuron_scale,
        const mfq_tensor_backend::Tensor & table_pool,
        const mfq_tensor_backend::Tensor & bank_ids,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits,
        cudaStream_t stream) {
    nepq_backward_mma_m8_kernel<FORMAT><<<
        (K + 15) / 16, 256, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            state_stream.data_ptr<uint8_t>(), state_stream.numel(),
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),
            bank_ids.data_ptr<uint8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            M, N, K, ng, nvec, nsign, nsuper, table_stride, state_bits);
}

template <int FORMAT, int Rows>
__global__ void __launch_bounds__(32) nepq_backward_partial_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * state_stream,
        int64_t state_nbytes,
        const float * neuron_scale,
        const int8_t * table_pool,
        const uint8_t * bank_ids,
        const __half * output_gradient,
        __half * partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x);
    const int segment = static_cast<int>(blockIdx.x) * 32 + lane;
    const int k0 = segment * 8;
    const bool valid = k0 < K;
    const int group = k0 / kGroupSize;
    const int split = static_cast<int>(blockIdx.y);
    const int row0 = static_cast<int>(blockIdx.z) * Rows;
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, N);
    float accumulators[Rows][8];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int component = 0; component < 8; ++component) {
            accumulators[row][component] = 0.0f;
        }
    }
    for (int output = output0; output < output_end; ++output) {
        float anchor = lane == 0 ? neuron_scale[output] : 0.0f;
        anchor = __shfl_sync(0xffffffffu, anchor, 0);
        uint32_t state = 0;
        NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
        float weight_scale = 0.0f;
        if (valid) {
            const int64_t state_index =
                static_cast<int64_t>(output) * ng + group;
            state = load_packed_bits(
                state_stream, state_index * state_bits,
                state_bits, state_nbytes);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, output, group,
                nsuper, table_stride);
            decoded = load_nepq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                output, segment, group, ng, nvec, nsign, 0, state);
            weight_scale = format_scale<FORMAT>(anchor, state, table);
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row0 + row < M
                ? __half2float(output_gradient[
                    static_cast<int64_t>(row0 + row) * N + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            const float factor = gradient * weight_scale;
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                accumulators[row][component] = fmaf(
                    factor, static_cast<float>(value),
                    accumulators[row][component]);
            }
        }
    }
    if (valid) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row0 + row < M) {
#pragma unroll
                for (int component = 0; component < 8; ++component) {
                    const int column = k0 + component;
                    if (column < K) {
                        partials[(static_cast<int64_t>(split) * M + row0 + row) *
                            K + column] = __float2half_rn(
                                accumulators[row][component]);
                    }
                }
            }
        }
    }
}

template <int FORMAT, int Rows>
void launch_nepq_backward_small_m(
        const mfq_tensor_backend::Tensor & indices,
        const mfq_tensor_backend::Tensor & aux,
        const mfq_tensor_backend::Tensor & state_stream,
        const mfq_tensor_backend::Tensor & neuron_scale,
        const mfq_tensor_backend::Tensor & table_pool,
        const mfq_tensor_backend::Tensor & bank_ids,
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((K + 255) / 256),
        static_cast<unsigned>((N + output_tile - 1) / output_tile),
        static_cast<unsigned>((M + Rows - 1) / Rows));
    nepq_backward_partial_kernel<FORMAT, Rows><<<grid, 32, 0, stream>>>(
        indices.data_ptr<uint8_t>(), indices.numel(),
        aux.data_ptr<uint8_t>(), aux.numel(),
        state_stream.data_ptr<uint8_t>(), state_stream.numel(),
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),
        bank_ids.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<mfq_half>()),
        reinterpret_cast<__half *>(partials.data_ptr<mfq_half>()),
        M, N, K, ng, nvec, nsign,
        nsuper, table_stride, state_bits, output_tile);
}

template <typename T, typename Launch>
void launch_backward_by_dtype(
        const mfq_tensor_backend::Tensor & output_gradient,
        mfq_tensor_backend::Tensor & result,
        Launch && launch) {
    if constexpr (std::is_same_v<T, __half>) {
        launch(
            reinterpret_cast<const T *>(output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<T *>(result.data_ptr<mfq_half>()));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        launch(
            reinterpret_cast<const T *>(output_gradient.data_ptr<mfq_bfloat16>()),
            reinterpret_cast<T *>(result.data_ptr<mfq_bfloat16>()));
    } else {
        launch(output_gradient.data_ptr<T>(), result.data_ptr<T>());
    }
}

}  // namespace

mfq_tensor_backend::Tensor nvq_backward_input_cuda(
        mfq_tensor_backend::Tensor indices,
        mfq_tensor_backend::Tensor aux,
        mfq_tensor_backend::Tensor sub_scale,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor codebook,
        mfq_tensor_backend::Tensor output_gradient,
        int64_t neuron_len,
        int64_t gs,
        int64_t sub_bits,
        int64_t format,
        int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == neuron_scale.numel(),
        "NVQ backward output gradient geometry mismatch");
    MFQ_RUNTIME_CHECK(
        indices.device() == output_gradient.device() &&
        aux.device() == output_gradient.device() &&
        sub_scale.device() == output_gradient.device() &&
        neuron_scale.device() == output_gradient.device() &&
        codebook.device() == output_gradient.device(),
        "NVQ backward tensors must share one CUDA device");
    const auto dtype = output_gradient.scalar_type();
    MFQ_RUNTIME_CHECK(
        dtype == mfq_tensor_backend::kFloat16 ||
        dtype == mfq_tensor_backend::kFloat32 ||
        dtype == mfq_tensor_backend::kBFloat16,
        "NVQ backward output gradient must be fp16, bf16, or fp32");
    const int M = static_cast<int>(output_gradient.size(0));
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto result = mfq_tensor_backend::empty({M, K}, output_gradient.options());
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(M) * K;
    if (total == 0) {
        return result;
    }
    MfqCudaGuard guard(output_gradient.device());
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    const cudaStream_t stream = mfq_current_cuda_stream();
    // Wide down projections expose enough independent K tiles to fuse
    // decoding with WMMA; output-wide projections retain the dense path.
    if (dtype == mfq_tensor_backend::kFloat16 && M >= 5 && M <= 8 &&
            static_cast<int64_t>(K) >= 2 * static_cast<int64_t>(N)) {
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            launch_nvq_backward_mma_m8<F>(
                indices, aux, sub_scale, neuron_scale, codebook,
                output_gradient, result, M, N, K, ng, nvec, nsign,
                static_cast<int>(sub_bits), static_cast<int>(sign_mode),
                stream);
        });
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    if (dtype == mfq_tensor_backend::kFloat16 && M <= 4) {
        constexpr int output_tile = 16;
        const int splits = (N + output_tile - 1) / output_tile;
        auto partials = mfq_tensor_backend::empty(
            {splits, M, K},
            output_gradient.options());
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (M == 1) {
                launch_nvq_backward_vec8_small_m<F, 1>(
                    indices, aux, sub_scale, neuron_scale, codebook,
                    output_gradient, partials, M, N, K, ng, nvec, nsign,
                    static_cast<int>(sub_bits), static_cast<int>(sign_mode),
                    output_tile, stream);
            } else if (M <= 2) {
                launch_nvq_backward_vec8_small_m<F, 2>(
                    indices, aux, sub_scale, neuron_scale, codebook,
                    output_gradient, partials, M, N, K, ng, nvec, nsign,
                    static_cast<int>(sub_bits), static_cast<int>(sign_mode),
                    output_tile, stream);
            } else if (M <= 4) {
                launch_nvq_backward_vec8_small_m<F, 4>(
                    indices, aux, sub_scale, neuron_scale, codebook,
                    output_gradient, partials, M, N, K, ng, nvec, nsign,
                    static_cast<int>(sub_bits), static_cast<int>(sign_mode),
                    output_tile, stream);
            } else if (M <= 8) {
                if constexpr (is_d4_format(F)) {
                    launch_nvq_backward_quad_small_m<F, 8>(
                        indices, aux, sub_scale, neuron_scale, codebook,
                        output_gradient, partials, M, N, K, ng, nvec, nsign,
                        static_cast<int>(sub_bits), static_cast<int>(sign_mode),
                        output_tile, stream);
                } else {
                    launch_nvq_backward_vec8_small_m<F, 4>(
                        indices, aux, sub_scale, neuron_scale, codebook,
                        output_gradient, partials, M, N, K, ng, nvec, nsign,
                        static_cast<int>(sub_bits), static_cast<int>(sign_mode),
                        output_tile, stream);
                }
            }
        });
        mfq_packed_backward::launch_split_half_reduce_to_half(
            reinterpret_cast<const __half *>(partials.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            M, K, splits, stream);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = nvq_dequant_cuda(
        indices, aux, sub_scale, neuron_scale, codebook,
        neuron_len, gs, sub_bits, format, sign_mode);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result, M, N, K, stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

mfq_tensor_backend::Tensor nepq_backward_input_cuda(
        mfq_tensor_backend::Tensor indices,
        mfq_tensor_backend::Tensor aux,
        mfq_tensor_backend::Tensor state_stream,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor table_pool,
        mfq_tensor_backend::Tensor bank_ids,
        mfq_tensor_backend::Tensor output_gradient,
        int64_t neuron_len,
        int64_t state_bits,
        int64_t format) {
    check_nepq_common(
        indices, aux, state_stream, neuron_scale, table_pool, bank_ids,
        neuron_len, state_bits, format);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == neuron_scale.numel(),
        "NEPQ backward output gradient geometry mismatch");
    MFQ_RUNTIME_CHECK(
        indices.device() == output_gradient.device() &&
        aux.device() == output_gradient.device() &&
        state_stream.device() == output_gradient.device() &&
        neuron_scale.device() == output_gradient.device() &&
        table_pool.device() == output_gradient.device() &&
        bank_ids.device() == output_gradient.device(),
        "NEPQ backward tensors must share one CUDA device");
    const auto dtype = output_gradient.scalar_type();
    MFQ_RUNTIME_CHECK(
        dtype == mfq_tensor_backend::kFloat16 ||
        dtype == mfq_tensor_backend::kFloat32 ||
        dtype == mfq_tensor_backend::kBFloat16,
        "NEPQ backward output gradient must be fp16, bf16, or fp32");
    const int M = static_cast<int>(output_gradient.size(0));
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto result = mfq_tensor_backend::empty({M, K}, output_gradient.options());
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(M) * K;
    if (total == 0) {
        return result;
    }
    MfqCudaGuard guard(output_gradient.device());
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    const cudaStream_t stream = mfq_current_cuda_stream();
    // Fuse dynamic table selection, vector decode, and WMMA when K tiles
    // provide enough parallel work; retain split reduction for output-wide MLPs.
    if (dtype == mfq_tensor_backend::kFloat16 && M >= 5 && M <= 8 &&
            static_cast<int64_t>(K) >= 2 * static_cast<int64_t>(N)) {
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            launch_nepq_backward_mma_m8<F>(
                indices, aux, state_stream, neuron_scale, table_pool,
                bank_ids, output_gradient, result, M, N, K, ng,
                nvec, nsign, nsuper, table_stride,
                static_cast<int>(state_bits), stream);
        });
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    if (dtype == mfq_tensor_backend::kFloat16 && M <= 8) {
        const int output_tile = M <= 4 ? 16 : 32;
        const int splits = (N + output_tile - 1) / output_tile;
        auto partials = mfq_tensor_backend::empty(
            {splits, M, K}, output_gradient.options());
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (M == 1) {
                launch_nepq_backward_small_m<F, 1>(
                    indices, aux, state_stream, neuron_scale, table_pool,
                    bank_ids, output_gradient, partials, M, N, K, ng,
                    nvec, nsign, nsuper, table_stride,
                    static_cast<int>(state_bits), output_tile, stream);
            } else if (M <= 2) {
                launch_nepq_backward_small_m<F, 2>(
                    indices, aux, state_stream, neuron_scale, table_pool,
                    bank_ids, output_gradient, partials, M, N, K, ng,
                    nvec, nsign, nsuper, table_stride,
                    static_cast<int>(state_bits), output_tile, stream);
            } else if (M <= 4) {
                launch_nepq_backward_small_m<F, 4>(
                    indices, aux, state_stream, neuron_scale, table_pool,
                    bank_ids, output_gradient, partials, M, N, K, ng,
                    nvec, nsign, nsuper, table_stride,
                    static_cast<int>(state_bits), output_tile, stream);
            } else {
                launch_nepq_backward_small_m<F, 4>(
                    indices, aux, state_stream, neuron_scale, table_pool,
                    bank_ids, output_gradient, partials, M, N, K, ng,
                    nvec, nsign, nsuper, table_stride,
                    static_cast<int>(state_bits), output_tile, stream);
            }
        });
        mfq_packed_backward::launch_split_half_reduce_to_half(
            reinterpret_cast<const __half *>(
                partials.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(result.data_ptr<mfq_half>()),
            M, K, splits, stream);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = nepq_dequant_cuda(
        indices, aux, state_stream, neuron_scale, table_pool, bank_ids,
        neuron_len, state_bits, format);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result, M, N, K, stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
