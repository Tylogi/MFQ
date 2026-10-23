// Copyright 2026 MFQ contributors. SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>
#include <type_traits>

struct ProjectionView {
    const uint32_t* words;
    const uint64_t* descriptors;
    const void* scales;
    const half* relative;
    const float* bias;
    int rows, stored_columns, input_columns, gslog, columns;
};

template<int DT> using Scalar = std::conditional_t<DT == 2, float, uint16_t>;

template<int DT> __device__ float read(Scalar<DT> value) {
    if constexpr (DT == 2) return value;
    else if constexpr (DT == 1) return __bfloat162float(__ushort_as_bfloat16(value));
    else return __half2float(__ushort_as_half(value));
}

template<int DT> __device__ Scalar<DT> rounded(float value) {
    if constexpr (DT == 2) return value;
    else if constexpr (DT == 1) return __bfloat16_as_ushort(__float2bfloat16(value));
    else return __half_as_ushort(__float2half_rn(value));
}

__device__ float weight(ProjectionView m, int row, int col) {
    const uint64_t descriptor = m.descriptors[row];
    const int bits = int((descriptor >> 32) & 7) + 1;
    const int ring_words = bits * 8;
    const uint32_t* ring = m.words + uint32_t(descriptor) + (col / 256) * ring_words;
    const int position = (255 - col % 256) * bits;
    const int word = position / 32, shift = position % 32;
    uint32_t state = ring[word] >> shift;
    if (shift > 16) state |= ring[(word + 1) % ring_words] << (32 - shift);
    const uint32_t product = (state & 65535) * 0x83dcd12du;
    const int sum = (product & 255) + ((product >> 8) & 255) +
                    ((product >> 16) & 255) + (product >> 24);
    const float book = __half2float(__float2half_rn(fmaf(float(sum + 1024),
        __half2float(__ushort_as_half(0x1eee)), __half2float(__ushort_as_half(0xc931)))));
    float scale = __half2float(__ushort_as_half(uint16_t(descriptor >> 40)));
    if (m.columns) scale = __fmul_rn(scale, __half2float(static_cast<const half*>(m.scales)[col]));
    else {
        const int groups = (m.stored_columns + (1 << m.gslog) - 1) >> m.gslog;
        const int group = row * groups + (col >> m.gslog);
        const uint8_t selector = static_cast<const uint8_t*>(m.scales)[group / 2];
        scale = __fmul_rn(scale, __half2float(m.relative[(selector >> (4 * (group & 1))) & 15]));
    }
    // Keep FP32 rounding before the final FP16 conversion.
    volatile float scaled = __fmul_rn(book, scale);
    return __half2float(__float2half_rn(scaled));
}

template<int DT> __global__ void decode(ProjectionView m, const Scalar<DT>* x, Scalar<DT>* y) {
    const int row = blockIdx.x * 4 + threadIdx.x / 64, lane = threadIdx.x % 64;
    if (row >= m.rows) return;
    float sum = 0;
    for (int col = lane; col < m.input_columns; col += 64)
        sum = fmaf(weight(m, row, col), read<DT>(x[col]), sum);
    for (int offset = 32; offset; offset /= 2) sum += __shfl_down(sum, offset, 64);
    if (lane == 0) y[row] = rounded<DT>(sum + (m.bias ? m.bias[row] : 0.f));
}

// Threads share one decoded weight tile across sixteen input tokens.
template<int DT, bool Backward> __global__ void tiled(
        ProjectionView m, const Scalar<DT>* x, Scalar<DT>* y, int tokens) {
    constexpr int Tile = 16;
    __shared__ float weights[Tile][Tile], activations[Tile][Tile];
    const int i = threadIdx.x, j = threadIdx.y;
    const int output_width = Backward ? m.input_columns : m.rows;
    const int reduction = Backward ? m.rows : m.input_columns;
    const int col = blockIdx.x * Tile + i, token = blockIdx.y * Tile + j;
    float sum = 0;
    for (int base = 0; base < reduction; base += Tile) {
        if constexpr (Backward)
            weights[j][i] = base + j < m.rows && col < m.input_columns ? weight(m, base + j, col) : 0;
        else
            weights[j][i] = col < m.rows && base + j < m.input_columns ? weight(m, col, base + j) : 0;
        activations[j][i] = token < tokens && base + i < reduction ?
            read<DT>(x[int64_t(token) * reduction + base + i]) : 0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Tile; ++k) sum = fmaf(activations[j][k], weights[k][i], sum);
        __syncthreads();
    }
    if (token < tokens && col < output_width) {
        if constexpr (!Backward) sum += m.bias ? m.bias[col] : 0.f;
        y[int64_t(token) * output_width + col] = rounded<DT>(sum);
    }
}

template<int DT, bool Backward> int launch(
        ProjectionView m, const void* x, void* y, int tokens, hipStream_t stream) {
    if constexpr (!Backward) {
        if (tokens == 1) {
            decode<DT><<<(m.rows + 3) / 4, 256, 0, stream>>>(m,
                static_cast<const Scalar<DT>*>(x), static_cast<Scalar<DT>*>(y));
            return int(hipGetLastError());
        }
    }
    const int width = Backward ? m.input_columns : m.rows;
    tiled<DT, Backward><<<dim3((width + 15) / 16, (tokens + 15) / 16), dim3(16, 16), 0, stream>>>(m,
        static_cast<const Scalar<DT>*>(x), static_cast<Scalar<DT>*>(y), tokens);
    return int(hipGetLastError());
}

template<bool Backward> int dispatch(
        ProjectionView m, const void* x, void* y, int tokens, int dtype, hipStream_t stream) {
    if (!x || !y || !m.words || !m.descriptors || !m.scales || !m.relative ||
            tokens < 1 || tokens > 65535 * 16 || m.rows < 1 || m.input_columns < 1 ||
            m.input_columns > m.stored_columns) return int(hipErrorInvalidValue);
    switch (dtype) {
        case 0: return launch<0, Backward>(m, x, y, tokens, stream);
        case 1: return launch<1, Backward>(m, x, y, tokens, stream);
        case 2: return launch<2, Backward>(m, x, y, tokens, stream);
        default: return int(hipErrorInvalidValue);
    }
}

extern "C" int mfq_tq_forward(ProjectionView m, const void* x, void* y,
        int tokens, int dtype, hipStream_t stream) {
    return dispatch<false>(m, x, y, tokens, dtype, stream);
}

extern "C" int mfq_tq_backward(ProjectionView m, const void* dy, void* dx,
        int tokens, int dtype, hipStream_t stream) {
    return dispatch<true>(m, dy, dx, tokens, dtype, stream);
}

extern "C" int mfq_tq_gemv(ProjectionView m, const void* x, void* y,
        int dtype, int warps, int rows, hipStream_t stream) {
    if (warps || rows) return int(hipErrorInvalidValue);
    return dispatch<false>(m, x, y, 1, dtype, stream);
}
