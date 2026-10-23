#include "gdn_prefill_chunk.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace tb = mfq_tensor_backend;
namespace {
constexpr int Chunk = 64;
constexpr int Threads = 512;
constexpr int Warps = Threads / 32;
constexpr int Columns = 32;
constexpr int WorkspaceChunks = 256;

__device__ __forceinline__ void gdn_chunk_mma(float (&c)[4], const unsigned (&a)[4],
                                            const unsigned (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// Power-of-two scaling retains FP32 range at the FP16 MMA boundary.
template<int BlockThreads = Threads>
__device__ __forceinline__ int gdn_chunk_exponent(float maximum, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int BlockWarps = BlockThreads / 32;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    maximum = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(maximum)));
#else
    for (int step = 16; step; step /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
#endif
    if (lane == 0) scratch[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        maximum = lane < BlockWarps ? scratch[lane] : 0.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        maximum = __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(maximum)));
#else
        for (int step = 16; step; step /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
#endif
        if (lane == 0) {
            const int exponent = int((__float_as_uint(maximum) >> 23) & 255) - 127;
            scratch[0] = maximum > 0.f ? float(max(-126, exponent - 9)) : 0.f;
        }
    }
    __syncthreads();
    return int(scratch[0]);
}

__device__ __forceinline__ void gdn_chunk_pack(float value, __half* output, int index, int exponent = 0) {
    const float inverse_scale = __int_as_float((127 - exponent) << 23);
    output[index] = __float2half_rn(value * inverse_scale);
}

template<int M, int N, int K, int BlockThreads = Threads>
__device__ __forceinline__ void gdn_chunk_packed_mm(
    const __half* a, const __half* b, float* c, bool add = false, int output_stride = N + 4,
    int a_exponent = 0, int b_exponent = 0) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int Stride = (K + 8) / 2;
    const auto* ap = reinterpret_cast<const unsigned*>(a);
    const auto* bp = reinterpret_cast<const unsigned*>(b);
    for (int tile = warp; tile < (M / 16) * (N / 8); tile += BlockThreads / 32) {
        const int row = (tile / (N / 8)) * 16, col = (tile % (N / 8)) * 8;
        float dot[4] = {};
        for (int kk = 0; kk < K; kk += 16) {
            unsigned av[4], bv[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = row + (lane >> 2) + (i % 2) * 8;
                const int k = kk / 2 + (lane & 3) + (i / 2) * 4;
                av[i] = ap[r * Stride + k];
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int k = kk / 2 + (lane & 3) + i * 4;
                bv[i] = bp[(col + (lane >> 2)) * Stride + k];
            }
            gdn_chunk_mma(dot, av, bv);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = row + (lane >> 2) + (i / 2) * 8;
            const int n = col + (lane & 3) * 2 + i % 2, index = r * output_stride + n;
            const int exponent = a_exponent + b_exponent;
            const int first_exponent = max(-126, min(127, exponent));
            const int second_exponent = exponent - first_exponent;
            const float result = (dot[i] * __int_as_float((127 + first_exponent) << 23))
                * __int_as_float((127 + second_exponent) << 23);
            c[index] = result + (add ? c[index] : 0.f);
        }
    }
}

template<class Input, int D>
__global__ void gdn_chunk_prepare_kernel(
    const Input* q, const Input* k, const Input* v, const float* gate, const float* beta,
    const int64_t* lengths, __half* w, float* u, __half* attention, float* prefix,
    int hq_count, int hv_count, int tokens, int chunks, int window_begin, bool tiled) {
    const int bh = blockIdx.x, head = bh % hv_count, batch = bh / hv_count;
    const int kh = tiled ? head % hq_count : head / (hv_count / hq_count);
    const int64_t first = int64_t(window_begin) + int64_t(blockIdx.y) * Chunk;
    const int tid = threadIdx.x, lane = tid & 31;
    const int length = lengths ? int(lengths[batch]) : tokens;
    if (first >= length) return;
    const int count = min(Chunk, int(length - first));
    const size_t key_offset = (size_t(batch) * hq_count + kh) * tokens * D;
    const size_t value_offset = size_t(bh) * tokens * D, gate_offset = size_t(bh) * tokens;
    const size_t chunk_offset = size_t(bh) * chunks + blockIdx.y;
    constexpr int KeyElements = Chunk * ((D < Chunk ? Chunk : D) + 8);
    constexpr int TileElements = 32 * (Chunk + 8), LowerStride = Chunk + 4;
    extern __shared__ float shared[];
    auto* keys = reinterpret_cast<__half*>(shared);
    auto* operand = keys + KeyElements;
    auto* lower = reinterpret_cast<float*>(operand + TileElements);
    float* inverse = lower + Chunk * LowerStride;
    float* gates = inverse + Chunk * Chunk;
    float* betas = gates + Chunk;
    float* scratch = betas + Chunk;
    float* gate_exp = scratch + Warps;
    if (tid < Chunk) {
        gates[tid] = tid < count ? gate[gate_offset + first + tid] : 0.f;
        betas[tid] = tid < count ? beta[gate_offset + first + tid] : 0.f;
    }
    __syncthreads();
    for (int step = 1; step < Chunk; step *= 2) {
        const float previous = tid < Chunk && tid >= step ? gates[tid - step] : 0.f;
        __syncthreads();
        if (tid < Chunk) gates[tid] += previous;
        __syncthreads();
    }
    if (tid < Chunk) {
        prefix[chunk_offset * (Chunk + 2) + tid] = gates[tid];
        gate_exp[tid] = expf(gates[tid]);
    }
    float key_maximum = 0.f;
    for (int i = tid; i < Chunk * D; i += Threads) {
        const int t = i / D;
        if (t < count) key_maximum = fmaxf(key_maximum,
            fabsf(float(k[key_offset + size_t(first + t) * D + i % D])));
    }
    const int key_exponent = gdn_chunk_exponent(key_maximum, scratch);
    for (int i = tid; i < Chunk * D; i += Threads) {
        const int t = i / D;
        const float value = t < count ? float(k[key_offset + size_t(first + t) * D + i % D]) : 0.f;
        gdn_chunk_pack(value, keys, t * (D + 8) + i % D, key_exponent);
    }
    __syncthreads();
    for (int row = 0; row < Chunk; row += 16) {
        for (int i = tid; i < 16 * D; i += Threads) {
            operand[(i / D) * (D + 8) + i % D] = keys[(row + i / D) * (D + 8) + i % D];
        }
        __syncthreads();
        gdn_chunk_packed_mm<16, Chunk, D>(operand, keys, lower + row * LowerStride,
            false, LowerStride, key_exponent, key_exponent);
        __syncthreads();
    }
    for (int i = tid; i < Chunk * Chunk; i += Threads) {
        const int row = i / Chunk, col = i % Chunk, index = row * LowerStride + col;
        lower[index] = row > col ? lower[index] * betas[row] * expf(gates[row] - gates[col]) : 0.f;
    }
    __syncthreads();
    // Independent warp solves for rows of the unit-lower-triangular inverse.
    for (int row = tid >> 5; row < Chunk; row += Threads / 32) {
        float x0 = lane == row ? 1.f : lane < row ? -lower[row * LowerStride + lane] : 0.f;
        float x1 = lane + 32 == row ? 1.f : lane + 32 < row ? -lower[row * LowerStride + lane + 32] : 0.f;
        for (int j = row - 1; j >= 0; --j) {
            const float pivot = __shfl_sync(0xffffffffu, j < 32 ? x0 : x1, j & 31);
            if (lane < j) x0 -= pivot * lower[j * LowerStride + lane];
            if (lane + 32 < j) x1 -= pivot * lower[j * LowerStride + lane + 32];
        }
        inverse[row * Chunk + lane] = x0;
        inverse[row * Chunk + lane + 32] = x1;
    }
    __syncthreads();
    // Reuse key storage for M. Values and scaled keys stream through a small
    // column tile; preparation fits below 99 KiB even at D128.
    float inverse_maximum = 0.f;
    for (int i = tid; i < Chunk * Chunk; i += Threads) inverse_maximum = fmaxf(inverse_maximum, fabsf(inverse[i]));
    const int inverse_exponent = gdn_chunk_exponent(inverse_maximum, scratch);
    for (int i = tid; i < Chunk * Chunk; i += Threads)
        gdn_chunk_pack(inverse[i], keys, (i / Chunk) * (Chunk + 8) + i % Chunk, inverse_exponent);
    __syncthreads();
    float* rhs = D == 128 ? reinterpret_cast<float*>(keys + Chunk * (Chunk + 8)) : lower;
    float* coefficients = D == 128 ? lower : inverse;
    for (int col = 0; col < D; col += 32) {
        float value_maximum = 0.f;
        for (int i = tid; i < Chunk * 32; i += Threads) {
            const int t = i / 32, d = col + i % 32;
            const float value = t < count ? float(v[value_offset + size_t(first + t) * D + d]) * betas[t] : 0.f;
            rhs[i] = value;
            value_maximum = fmaxf(value_maximum, fabsf(value));
        }
        const int value_exponent = gdn_chunk_exponent(value_maximum, scratch);
        for (int i = tid; i < Chunk * 32; i += Threads)
            gdn_chunk_pack(rhs[i], operand, (i % 32) * (Chunk + 8) + i / 32, value_exponent);
        __syncthreads();
        gdn_chunk_packed_mm<Chunk, 32, Chunk>(keys, operand, u + chunk_offset * Chunk * D + col,
            false, D, inverse_exponent, value_exponent);
        __syncthreads();
    }
    for (int col = 0; col < D; col += 32) {
        float weighted_maximum = 0.f;
        for (int i = tid; i < Chunk * 32; i += Threads) {
            const int t = i / 32, d = col + i % 32;
            const float value = t < count ? float(k[key_offset + size_t(first + t) * D + d])
                * betas[t] * gate_exp[t] : 0.f;
            rhs[i] = value;
            weighted_maximum = fmaxf(weighted_maximum, fabsf(value));
        }
        const int weighted_exponent = gdn_chunk_exponent(weighted_maximum, scratch);
        for (int i = tid; i < Chunk * 32; i += Threads)
            gdn_chunk_pack(rhs[i], operand, (i % 32) * (Chunk + 8) + i / 32, weighted_exponent);
        __syncthreads();
        gdn_chunk_packed_mm<Chunk, 32, Chunk>(keys, operand, coefficients + col,
            false, D, inverse_exponent, weighted_exponent);
        __syncthreads();
    }
    // W and causal QK enter MMA as FP16 in the state kernel. Pack them here
    // once, sharing their exact power-of-two scales across value-column CTAs.
    float w_maximum = 0.f;
    for (int i = tid; i < Chunk * D; i += Threads) w_maximum = fmaxf(w_maximum, fabsf(coefficients[i]));
    const int w_exponent = gdn_chunk_exponent(w_maximum, scratch);
    for (int i = tid; i < Chunk * D; i += Threads)
        gdn_chunk_pack(coefficients[i], w + chunk_offset * Chunk * D, i, w_exponent);
    if (tid == 0) prefix[chunk_offset * (Chunk + 2) + Chunk] = float(w_exponent);
    __syncthreads();
    for (int i = tid; i < Chunk * D; i += Threads) {
        const int t = i / D;
        const float value = t < count ? float(k[key_offset + size_t(first + t) * D + i % D]) : 0.f;
        gdn_chunk_pack(value, keys, t * (D + 8) + i % D, key_exponent);
    }
    __syncthreads();
    for (int row = 0; row < Chunk; row += 16) {
        float query_maximum = 0.f;
        for (int i = tid; i < 16 * D; i += Threads) {
            const int t = row + i / D;
            const float value = t < count ? float(q[key_offset + size_t(first + t) * D + i % D]) : 0.f;
            inverse[i] = value;
            query_maximum = fmaxf(query_maximum, fabsf(value));
        }
        const int query_exponent = gdn_chunk_exponent(query_maximum, scratch);
        for (int i = tid; i < 16 * D; i += Threads)
            gdn_chunk_pack(inverse[i], operand, (i / D) * (D + 8) + i % D, query_exponent);
        __syncthreads();
        gdn_chunk_packed_mm<16, Chunk, D>(operand, keys, lower + row * LowerStride,
            false, LowerStride, query_exponent, key_exponent);
        __syncthreads();
    }
    float attention_maximum = 0.f;
    for (int i = tid; i < Chunk * Chunk; i += Threads) {
        const int row = i / Chunk, col = i % Chunk, index = row * LowerStride + col;
        const float value = row >= col && row < count ? lower[index] * expf(gates[row] - gates[col]) : 0.f;
        lower[index] = value;
        attention_maximum = fmaxf(attention_maximum, fabsf(value));
    }
    const int attention_exponent = gdn_chunk_exponent(attention_maximum, scratch);
    for (int i = tid; i < Chunk * Chunk; i += Threads)
        gdn_chunk_pack(lower[(i / Chunk) * LowerStride + i % Chunk],
            attention + chunk_offset * Chunk * Chunk, i, attention_exponent);
    if (tid == 0) prefix[chunk_offset * (Chunk + 2) + Chunk + 1] = float(attention_exponent);
}

template<class Input, int D>
__global__ __launch_bounds__((D == 128 ? 1024 : Threads), (D == 128 ? 1 : 2)) void gdn_chunk_state_kernel(
    const Input* q, const Input* k, const __half* w, const float* u, const __half* attention,
    const float* prefix, const int64_t* lengths, const float* initial, float* output, float* final_state,
    int hq_count, int hv_count, int tokens, int chunks, int window_begin, bool transposed, bool tiled) {
    constexpr int StateThreads = D == 128 ? 1024 : Threads;
    constexpr int StateColumns = D == 128 ? 64 : Columns;
    const int bh = blockIdx.x, head = bh % hv_count, batch = bh / hv_count, tid = threadIdx.x;
    const int kh = tiled ? head % hq_count : head / (hv_count / hq_count);
    const int column = int(blockIdx.y) * StateColumns;
    const int length = lengths ? int(lengths[batch]) : tokens;
    const size_t key_offset = (size_t(batch) * hq_count + kh) * tokens * D;
    const size_t value_offset = size_t(bh) * tokens * D, state_offset = size_t(bh) * D * D;
    constexpr int LeftElements = (D < Chunk ? Chunk : D) * (Chunk + 8);
    constexpr int StateElements = StateColumns * (D + 8), DeltaElements = StateColumns * (Chunk + 8);
    constexpr int ValueStride = StateColumns + 2, StateItems = D * StateColumns / StateThreads;
    extern __shared__ float shared[];
    auto* left = reinterpret_cast<__half*>(shared);
    auto* state = left + LeftElements;
    auto* delta = state + StateElements;
    auto* temp = reinterpret_cast<float*>(delta + DeltaElements);
    float* gates = temp + (D < Chunk ? Chunk : D) * ValueStride;
    float* scratch = gates + Chunk;
    float* gate_exp = scratch + StateThreads / 32;
    float* terminal_weight = gate_exp + Chunk;
    float state_values[StateItems];
    float state_maximum = 0.f;
    #pragma unroll
    for (int item = 0; item < StateItems; ++item) {
        const int i = tid + item * StateThreads, row = i / StateColumns, col = column + i % StateColumns;
        const size_t index = state_offset + (transposed ? col * D + row : row * D + col);
        state_values[item] = initial ? initial[index] : 0.f;
        state_maximum = fmaxf(state_maximum, fabsf(state_values[item]));
    }
    int state_exponent = gdn_chunk_exponent<StateThreads>(state_maximum, scratch);
    #pragma unroll
    for (int item = 0; item < StateItems; ++item) {
        const int i = tid + item * StateThreads;
        gdn_chunk_pack(state_values[item], state, (i % StateColumns) * (D + 8) + i / StateColumns, state_exponent);
    }
    __syncthreads();
    // FP32 register state advances only at chunk boundaries. Packed operands
    // are scratch representations, never the persistent recurrent state.
    for (int chunk = 0; chunk < chunks && int64_t(window_begin) + chunk * Chunk < length; ++chunk) {
        const int first = window_begin + chunk * Chunk, count = min(Chunk, length - first);
        const size_t chunk_offset = size_t(bh) * chunks + chunk;
        const int w_exponent = int(prefix[chunk_offset * (Chunk + 2) + Chunk]);
        for (int i = tid; i < Chunk * D; i += StateThreads)
            left[(i / D) * (D + 8) + i % D] = w[chunk_offset * Chunk * D + i];
        if (tid < Chunk) gates[tid] = prefix[chunk_offset * (Chunk + 2) + tid];
        __syncthreads();
        // Each token's exponentials are reused by all value columns. W*S
        // does not consume them; its following barrier completes these stores.
        if (tid < Chunk) {
            gate_exp[tid] = expf(gates[tid]);
            terminal_weight[tid] = expf(gates[count - 1] - gates[tid]);
        }
        gdn_chunk_packed_mm<Chunk, StateColumns, D, StateThreads>(left, state, temp, false, ValueStride,
            w_exponent, state_exponent);
        __syncthreads();
        float delta_maximum = 0.f;
        for (int i = tid; i < Chunk * StateColumns; i += StateThreads) {
            const int t = i / StateColumns, col = column + i % StateColumns;
            const float value = u[chunk_offset * Chunk * D + t * D + col] - temp[t * ValueStride + i % StateColumns];
            temp[t * ValueStride + i % StateColumns] = value;
            delta_maximum = fmaxf(delta_maximum, fabsf(value));
        }
        const int delta_exponent = gdn_chunk_exponent<StateThreads>(delta_maximum, scratch);
        for (int i = tid; i < Chunk * StateColumns; i += StateThreads)
            gdn_chunk_pack(temp[(i / StateColumns) * ValueStride + i % StateColumns], delta,
                (i % StateColumns) * (Chunk + 8) + i / StateColumns, delta_exponent);
        __syncthreads();
        float query_maximum = 0.f;
        for (int i = tid; i < Chunk * D; i += StateThreads) {
            const int t = i / D;
            const float value = t < count
                ? float(q[key_offset + size_t(first + t) * D + i % D]) * gate_exp[t] : 0.f;
            if constexpr (D == 128) temp[i] = value;
            query_maximum = fmaxf(query_maximum, fabsf(value));
        }
        const int query_exponent = gdn_chunk_exponent<StateThreads>(query_maximum, scratch);
        for (int i = tid; i < Chunk * D; i += StateThreads) {
            const int t = i / D;
            const float value = t < count
                ? float(q[key_offset + size_t(first + t) * D + i % D]) * gate_exp[t] : 0.f;
            gdn_chunk_pack(D == 128 ? temp[i] : value, left, t * (D + 8) + i % D, query_exponent);
        }
        __syncthreads();
        gdn_chunk_packed_mm<Chunk, StateColumns, D, StateThreads>(left, state, temp, false, ValueStride,
            query_exponent, state_exponent);
        __syncthreads();
        const int attention_exponent = int(prefix[chunk_offset * (Chunk + 2) + Chunk + 1]);
        for (int i = tid; i < Chunk * Chunk; i += StateThreads)
            left[(i / Chunk) * (Chunk + 8) + i % Chunk] = attention[chunk_offset * Chunk * Chunk + i];
        __syncthreads();
        gdn_chunk_packed_mm<Chunk, StateColumns, Chunk, StateThreads>(left, delta, temp,
            true, ValueStride, attention_exponent, delta_exponent);
        __syncthreads();
        for (int i = tid; i < count * StateColumns; i += StateThreads)
            output[value_offset + size_t(first + i / StateColumns) * D + column + i % StateColumns]
                = temp[(i / StateColumns) * ValueStride + i % StateColumns] / sqrtf(float(D));
        __syncthreads();
        float key_maximum = 0.f;
        for (int i = tid; i < Chunk * D; i += StateThreads) {
            const int t = i / D, row = i % D;
            const float value = t < count ? float(k[key_offset + size_t(first + t) * D + row])
                * terminal_weight[t] : 0.f;
            if constexpr (D == 128) temp[i] = value;
            key_maximum = fmaxf(key_maximum, fabsf(value));
        }
        const int key_exponent = gdn_chunk_exponent<StateThreads>(key_maximum, scratch);
        for (int i = tid; i < Chunk * D; i += StateThreads) {
            const int t = i / D, row = i % D;
            const float value = t < count ? float(k[key_offset + size_t(first + t) * D + row])
                * terminal_weight[t] : 0.f;
            gdn_chunk_pack(D == 128 ? temp[i] : value, left, row * (Chunk + 8) + t, key_exponent);
        }
        __syncthreads();
        gdn_chunk_packed_mm<D, StateColumns, Chunk, StateThreads>(left, delta, temp,
            false, ValueStride, key_exponent, delta_exponent);
        __syncthreads();
        const float decay = gate_exp[count - 1];
        state_maximum = 0.f;
        #pragma unroll
        for (int item = 0; item < StateItems; ++item) {
            const int i = tid + item * StateThreads, row = i / StateColumns, col = i % StateColumns;
            state_values[item] = decay * state_values[item] + temp[row * ValueStride + col];
            state_maximum = fmaxf(state_maximum, fabsf(state_values[item]));
        }
        state_exponent = gdn_chunk_exponent<StateThreads>(state_maximum, scratch);
        #pragma unroll
        for (int item = 0; item < StateItems; ++item) {
            const int i = tid + item * StateThreads;
            gdn_chunk_pack(state_values[item], state, (i % StateColumns) * (D + 8) + i / StateColumns, state_exponent);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int item = 0; item < StateItems; ++item) {
        const int i = tid + item * StateThreads, row = i / StateColumns, col = column + i % StateColumns;
        final_state[state_offset + (transposed ? col * D + row : row * D + col)] = state_values[item];
    }
    if (int64_t(window_begin) + chunks * Chunk >= tokens)
        for (int64_t i = tid; i < int64_t(tokens - length) * StateColumns; i += StateThreads)
            output[value_offset + size_t(length + i / StateColumns) * D + column + i % StateColumns] = 0.f;
}

// The WY workspace is transient. Size its window from free device memory so a
// nearly full device covers the prompt with more windows instead of failing to
// allocate; windows carry the full FP32 state, so the window size does not
// change the result.
int gdn_chunk_window(int tokens, int64_t per_chunk_bytes) {
    int wanted = std::min(WorkspaceChunks, (tokens + Chunk - 1) / Chunk);
    // MFQ_GDN_PREFILL_CHUNK_WINDOW caps the window in chunks, for tuning and tests.
    if (const char* option = std::getenv("MFQ_GDN_PREFILL_CHUNK_WINDOW"); option && std::atoi(option) > 0)
        wanted = std::min(wanted, std::atoi(option));
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        cudaGetLastError();
        return wanted;
    }
    const int64_t budget = int64_t(free_bytes) / 4;
    return int(std::clamp<int64_t>(budget / per_chunk_bytes, 1, wanted));
}

template<class Input, int D>
void gdn_chunk_launch(const tb::Tensor& q, const tb::Tensor& k, const tb::Tensor& v,
                      const tb::Tensor& g, const tb::Tensor& beta, const tb::Tensor& lengths,
                      const tb::Tensor& initial, tb::Tensor& output, tb::Tensor& state,
                      bool transposed, bool tiled) {
    const int batch = q.size(0), hq = q.size(1), hv = v.size(1), tokens = q.size(2);
    // W and A are FP16, U and the prefix sums FP32, per chunk and value head.
    const int64_t per_chunk_bytes = int64_t(batch) * hv * (
        int64_t(Chunk) * D * (sizeof(__half) + sizeof(float)) + int64_t(Chunk) * Chunk * sizeof(__half)
        + int64_t(Chunk + 2) * sizeof(float));
    const int chunks = gdn_chunk_window(tokens, per_chunk_bytes);
    auto w = tb::empty({batch, hv, chunks, Chunk, D}, g.options().dtype(tb::kFloat16));
    auto u = tb::empty({batch, hv, chunks, Chunk, D}, g.options());
    auto a = tb::empty({batch, hv, chunks, Chunk, Chunk}, g.options().dtype(tb::kFloat16));
    auto prefix = tb::empty({batch, hv, chunks, Chunk + 2}, g.options());
    constexpr int prepare_bytes = (Chunk * ((D < Chunk ? Chunk : D) + 8) + 32 * (Chunk + 8)) * sizeof(__half)
        + (Chunk * (Chunk + 4) + Chunk * Chunk + 3 * Chunk + Warps) * sizeof(float);
    constexpr int StateColumns = D == 128 ? 64 : Columns, StateThreads = D == 128 ? 1024 : Threads;
    constexpr int state_bytes = ((D < Chunk ? Chunk : D) * (Chunk + 8)
        + StateColumns * (D + 8) + StateColumns * (Chunk + 8)) * sizeof(__half)
        + ((D < Chunk ? Chunk : D) * (StateColumns + 2) + 3 * Chunk + StateThreads / 32) * sizeof(float);
    static_assert(prepare_bytes <= 99 * 1024 && state_bytes <= 99 * 1024);
    const auto prepare = gdn_chunk_prepare_kernel<Input, D>;
    // Scaling protects FP32 state range at the Tensor Core boundary;
    // the persistent state remains in FP32 registers.
    const auto propagate = gdn_chunk_state_kernel<Input, D>;
    static thread_local int configured_device = -1;
    const int device = mfq_current_cuda_device();
    if (configured_device != device) {
        MFQ_CUDA_CHECK(cudaFuncSetAttribute(prepare, cudaFuncAttributeMaxDynamicSharedMemorySize, prepare_bytes));
        MFQ_CUDA_CHECK(cudaFuncSetAttribute(propagate, cudaFuncAttributeMaxDynamicSharedMemorySize, state_bytes));
        configured_device = device;
    }
    const auto stream = mfq_current_cuda_stream();
    const auto* lp = lengths.defined() ? lengths.data_ptr<int64_t>() : nullptr;
    const auto* sp = initial.defined() ? initial.data_ptr<float>() : nullptr;
    // Reuse a bounded WY workspace across long prompts. Window boundaries
    // carry the complete FP32 state and do not truncate the logical sequence.
    for (int64_t first = 0; first < tokens; first += chunks * Chunk) {
        prepare<<<dim3(batch * hv, chunks), Threads, prepare_bytes, stream>>>(
            reinterpret_cast<const Input*>(q.data_ptr()), reinterpret_cast<const Input*>(k.data_ptr()),
            reinterpret_cast<const Input*>(v.data_ptr()), g.data_ptr<float>(), beta.data_ptr<float>(), lp,
            reinterpret_cast<__half*>(w.data_ptr()), u.data_ptr<float>(),
            reinterpret_cast<__half*>(a.data_ptr()), prefix.data_ptr<float>(),
            hq, hv, tokens, chunks, int(first), tiled);
        MFQ_CUDA_CHECK(cudaGetLastError());
        propagate<<<dim3(batch * hv, D / StateColumns), StateThreads, state_bytes, stream>>>(
            reinterpret_cast<const Input*>(q.data_ptr()), reinterpret_cast<const Input*>(k.data_ptr()),
            reinterpret_cast<__half*>(w.data_ptr()), u.data_ptr<float>(),
            reinterpret_cast<__half*>(a.data_ptr()), prefix.data_ptr<float>(), lp,
            first == 0 ? sp : state.data_ptr<float>(),
            output.data_ptr<float>(), state.data_ptr<float>(), hq, hv, tokens, chunks, int(first), transposed, tiled);
        MFQ_CUDA_CHECK(cudaGetLastError());
    }
}

template<class Input>
void gdn_chunk_width(const tb::Tensor& q, const tb::Tensor& k, const tb::Tensor& v,
                     const tb::Tensor& g, const tb::Tensor& beta, const tb::Tensor& lengths,
                     const tb::Tensor& initial, tb::Tensor& output, tb::Tensor& state,
                     bool transposed, bool tiled) {
    switch (q.size(3)) {
    case 32:
        gdn_chunk_launch<Input, 32>(q, k, v, g, beta, lengths, initial, output, state, transposed, tiled);
        break;
    case 64:
        gdn_chunk_launch<Input, 64>(q, k, v, g, beta, lengths, initial, output, state, transposed, tiled);
        break;
    case 128:
        gdn_chunk_launch<Input, 128>(q, k, v, g, beta, lengths, initial, output, state, transposed, tiled);
        break;
    }
}
} // namespace

bool gdn_prefill_chunk_enabled(int tokens, bool scalar_gate) {
    if (!scalar_gate || tokens <= 8) return false;
    const char* option = std::getenv("MFQ_GDN_PREFILL_CHUNK");
    if (option && std::atoi(option) == 0) return false;
    // Keep short projected prefills on the established FP32 recurrence;
    // normalization followed by FP16 projection can amplify small dot errors.
    if (!option && (tokens < 4096 || std::getenv("MFQ_GDN_PREFILL_COLUMNS"))) return false;
    static thread_local int cached_device = -1, major = 0;
    const int device = mfq_current_cuda_device();
    if (cached_device != device) {
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        cached_device = device;
    }
    return major >= 8;
}

std::vector<tb::Tensor> gdn_prefill_chunk_cuda(
    tb::Tensor q, tb::Tensor k, tb::Tensor v, tb::Tensor g, tb::Tensor beta,
    MfqOptional<tb::Tensor> initial_state, bool inplace, bool transposed, bool tiled, tb::Tensor lengths) {
    MFQ_RUNTIME_CHECK(q.dim() == 4 && v.dim() == 4, "GDN chunk: q/v must be [B,H,T,D]");
    const auto dtype = q.scalar_type();
    const int64_t batch = q.size(0), hq = q.size(1), hv = v.size(1), tokens = q.size(2), width = q.size(3);
    MFQ_RUNTIME_CHECK(batch > 0 && hq > 0 && hv > 0 && hv % hq == 0 && batch * hv <= 2147483647 &&
        tokens > 0 && tokens <= std::numeric_limits<int>::max() && (width == 32 || width == 64 || width == 128),
        "GDN chunk: invalid geometry");
    const auto same = [&](const tb::Tensor& x) {
        return x.is_cuda() && x.device() == q.device() && x.is_contiguous();
    };
    MFQ_RUNTIME_CHECK(same(q) && same(k) && same(v) && k.sizes() == q.sizes() &&
        v.sizes().vec() == std::vector<int64_t>({batch, hv, tokens, width}) &&
        (dtype == tb::kFloat32 || dtype == tb::kFloat16 || dtype == tb::kBFloat16) &&
        k.scalar_type() == dtype && v.scalar_type() == dtype, "GDN chunk: q/k/v dtype, shape or device mismatch");
    MFQ_RUNTIME_CHECK(same(g) && same(beta) && g.scalar_type() == tb::kFloat32 && beta.scalar_type() == tb::kFloat32 &&
        g.sizes().vec() == std::vector<int64_t>({batch, hv, tokens}) && beta.sizes() == g.sizes(),
        "GDN chunk: gate/beta must be FP32 [B,Hv,T]");
    // Lengths are validated on the host by the public varlen adapter. Uniform
    // calls need no metadata allocation or device-to-host synchronization.
    MFQ_RUNTIME_CHECK(!lengths.defined() || (same(lengths) && lengths.scalar_type() == tb::kInt64 &&
        lengths.sizes().vec() == std::vector<int64_t>({batch})), "GDN chunk: lengths must be CUDA int64 [B]");
    tb::Tensor initial;
    if (initial_state.has_value() && initial_state->defined() && initial_state->numel() > 0) {
        initial = initial_state.value();
        MFQ_RUNTIME_CHECK(initial.is_cuda() && initial.device() == q.device() &&
            (!inplace || initial.is_contiguous()) && initial.scalar_type() == tb::kFloat32 &&
            initial.sizes().vec() == std::vector<int64_t>({batch, hv, width, width}),
            "GDN chunk: state must be FP32 [B,Hv,D,D]");
        initial = initial.contiguous();
    }
    MFQ_RUNTIME_CHECK(!inplace || initial.defined(), "GDN chunk: inplace state is required");
    auto output = tb::empty({batch, hv, tokens, width}, g.options());
    auto state = inplace ? initial : tb::empty({batch, hv, width, width}, g.options());
    if (dtype == tb::kBFloat16)
        gdn_chunk_width<__nv_bfloat16>(q, k, v, g, beta, lengths, initial, output, state, transposed, tiled);
    else if (dtype == tb::kFloat16)
        gdn_chunk_width<__half>(q, k, v, g, beta, lengths, initial, output, state, transposed, tiled);
    else
        gdn_chunk_width<float>(q, k, v, g, beta, lengths, initial, output, state, transposed, tiled);
    return {output, state};
}

std::vector<tb::Tensor> gdn_prefill_varlen_cuda(
    tb::Tensor q, tb::Tensor k, tb::Tensor v, tb::Tensor g, tb::Tensor beta,
    const std::vector<int64_t>& lengths, MfqOptional<tb::Tensor> state, bool transposed, bool tiled, bool inplace) {
    MFQ_RUNTIME_CHECK(q.is_cuda() && q.dim() == 4 && lengths.size() == size_t(q.size(0)),
        "GDN varlen: one length per padded request is required");
    for (auto length : lengths)
        MFQ_RUNTIME_CHECK(length >= 0 && length <= q.size(2), "GDN varlen: length must be in [0,T]");
    int major = 0;
    MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, mfq_current_cuda_device()));
    MFQ_RUNTIME_CHECK(major >= 8, "GDN varlen chunk prefill requires SM80 or newer");
    auto metadata = tb::tensor(lengths).to(q.device());
    return gdn_prefill_chunk_cuda(q, k, v, g, beta, state, inplace, transposed, tiled, metadata);
}
