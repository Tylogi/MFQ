#pragma once
#include <mma.h>

// Scalar-gate delta rule in chunkwise WY form. Independent chunks prepare the
// triangular solve in parallel; only the small state propagation is recurrent.
// State storage and the triangular solve use FP32; products retain the high
// and low TF32 components to avoid reduced-precision state accumulation.
namespace mfq::gdn_chunked {
constexpr int C = 32;

template <int D>
__global__ void prepare(const float* q, const float* k, const float* v,
    const float* g, const float* beta, float* panels, float* coefficients,
    float* decays, int Hq, int Hv, int T, int heads, int chunks, bool tiled) {
    const int chunk = blockIdx.x, head = blockIdx.y;
    const int hq = (head / Hv) * Hq + (tiled ? head % Hv % Hq : (head % Hv) / (Hv / Hq));
    extern __shared__ float shared[];
    float* ks = shared;
    float* qs = ks + C * D;
    float* ws = qs + C * D;
    float* us = ws + C * D;
    float* lower = us + C * D;
    float* prefix = lower + C * C;
    const auto panel_stride = static_cast<std::size_t>(chunks) * heads * C * D;
    const auto offset = (static_cast<std::size_t>(chunk) * heads + head) * C * D;
    auto* relation = coefficients + (static_cast<std::size_t>(chunk) * heads + head) * C * C;
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < C; ++i) {
            const int t = chunk * C + i;
            if (t < T) sum += g[static_cast<std::size_t>(head) * T + t];
            prefix[i] = sum;
        }
        decays[chunk * heads + head] = expf(sum);
    }
    for (int i = threadIdx.x; i < C * D; i += blockDim.x) {
        const int t = chunk * C + i / D, d = i % D;
        ks[i] = t < T ? k[(static_cast<std::size_t>(hq) * T + t) * D + d] : 0.0f;
        qs[i] = t < T ? q[(static_cast<std::size_t>(hq) * T + t) * D + d] : 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int pair = warp; pair < C * C; pair += 8) {
        const int i = pair / C, j = pair % C;
        float kk = 0.0f, qk = 0.0f;
        if (j <= i) {
            for (int d = lane; d < D; d += 32) {
                kk += ks[i * D + d] * ks[j * D + d];
                qk += qs[i * D + d] * ks[j * D + d];
            }
        }
        for (int delta = 16; delta; delta /= 2) {
            kk += __shfl_down_sync(0xffffffffu, kk, delta);
            qk += __shfl_down_sync(0xffffffffu, qk, delta);
        }
        if (lane == 0) {
            const int t = chunk * C + i;
            const float b = t < T ? beta[static_cast<std::size_t>(head) * T + t] : 0.0f;
            const float decay = j <= i ? expf(prefix[i] - prefix[j]) : 0.0f;
            lower[pair] = j < i ? b * kk * decay : 0.0f;
            relation[pair] = qk * decay;
        }
    }
    __syncthreads();
    for (int i = 0; i < C; ++i) {
        if (threadIdx.x < D) {
            const int d = threadIdx.x, t = chunk * C + i;
            const float b = t < T ? beta[static_cast<std::size_t>(head) * T + t] : 0.0f;
            float w = b * expf(prefix[i]) * ks[i * D + d];
            float u = t < T ? b * v[(static_cast<std::size_t>(head) * T + t) * D + d] : 0.0f;
            for (int j = 0; j < i; ++j) {
                w -= lower[i * C + j] * ws[j * D + d];
                u -= lower[i * C + j] * us[j * D + d];
            }
            ws[i * D + d] = w;
            us[i * D + d] = u;
        }
        __syncthreads();
    }
    for (int index = threadIdx.x; index < C * D; index += blockDim.x) {
        const int i = index / D;
        panels[offset + index] = qs[index] * expf(prefix[i]);
        panels[panel_stride + offset + index] = ws[index];
        panels[2 * panel_stride + offset + index] = us[index];
        panels[3 * panel_stride + offset + index] = ks[index] * expf(prefix[C - 1] - prefix[i]);
    }
}

// Three TF32 products recover the FP32 mantissa to about 1e-7 relative
// accuracy. Dropping either cross term measurably changes recurrent state.
__device__ void multiply(const float* a, const float* b, float* c,
    int m, int n, int k, int lda, int ldb, int ldc, bool add, bool negate = false) {
    namespace wmma = nvcuda::wmma;
    using A = wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major>;
    using B = wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major>;
    using Acc = wmma::fragment<wmma::accumulator,16,16,8,float>;
    for (int tile = threadIdx.x / 32; tile < (m / 16) * (n / 16); tile += 4) {
        const int row = (tile / (n / 16)) * 16, column = (tile % (n / 16)) * 16;
        Acc accumulator;
        if (add) wmma::load_matrix_sync(accumulator, c + row * ldc + column, ldc, wmma::mem_row_major);
        else wmma::fill_fragment(accumulator, 0.0f);
        for (int begin = 0; begin < k; begin += 8) {
            A ah, al; B bh, bl;
            wmma::load_matrix_sync(ah, a + row * lda + begin, lda);
            wmma::load_matrix_sync(bh, b + begin * ldb + column, ldb);
#pragma unroll
            for (int i = 0; i < ah.num_elements; ++i) {
                const float value = negate ? -ah.x[i] : ah.x[i];
                ah.x[i] = wmma::__float_to_tf32(value);
                al.x[i] = wmma::__float_to_tf32(value - ah.x[i]);
            }
#pragma unroll
            for (int i = 0; i < bh.num_elements; ++i) {
                const float value = bh.x[i];
                bh.x[i] = wmma::__float_to_tf32(value);
                bl.x[i] = wmma::__float_to_tf32(value - bh.x[i]);
            }
            wmma::mma_sync(accumulator, ah, bh, accumulator);
            wmma::mma_sync(accumulator, ah, bl, accumulator);
            wmma::mma_sync(accumulator, al, bh, accumulator);
        }
        wmma::store_matrix_sync(c + row * ldc + column, accumulator, ldc, wmma::mem_row_major);
    }
}

template <int D>
__global__ void apply(const float* panels, const float* relations, const float* decays,
    const float* initial, float* output, float* state,
    int heads, int T, int chunks, bool transposed) {
    constexpr int N = 16, ld = 24;
    constexpr int work_size = D * 40 > C * (D + 8) ? D * 40 : C * (D + 8);
    __shared__ __align__(32) float s[D * ld];
    __shared__ __align__(32) float work[work_size];
    __shared__ __align__(32) float u[C * ld];
    __shared__ __align__(32) float o[C * ld];
    const int head = blockIdx.x, first_column = blockIdx.y * N;
    for (int i = threadIdx.x; i < D * N; i += blockDim.x) {
        const int row = i / N, col = i % N + first_column;
        const auto offset = static_cast<std::size_t>(head) * D * D + (transposed ? col * D + row : row * D + col);
        s[row * ld + i % N] = initial ? initial[offset] : 0.0f;
    }
    __syncthreads();
    const auto panel_stride = static_cast<std::size_t>(chunks) * heads * C * D;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const auto offset = (static_cast<std::size_t>(chunk) * heads + head) * C * D;
        const auto* qp = panels + offset;
        const auto* w = qp + panel_stride;
        const auto* base_u = w + panel_stride;
        const auto* kd = base_u + panel_stride;
        for (int i = threadIdx.x; i < C * D; i += blockDim.x) work[(i / D) * (D + 8) + i % D] = w[i];
        for (int i = threadIdx.x; i < C * N; i += blockDim.x) u[(i / N) * ld + i % N] = base_u[(i / N) * D + first_column + i % N];
        __syncthreads();
        multiply(work, s, u, C, N, D, D + 8, ld, ld, true, true);
        __syncthreads();
        for (int i = threadIdx.x; i < C * D; i += blockDim.x) work[(i / D) * (D + 8) + i % D] = qp[i];
        __syncthreads();
        multiply(work, s, o, C, N, D, D + 8, ld, ld, false);
        __syncthreads();
        const auto* relation = relations + (static_cast<std::size_t>(chunk) * heads + head) * C * C;
        for (int i = threadIdx.x; i < C * C; i += blockDim.x) work[(i / C) * 40 + i % C] = relation[i];
        __syncthreads();
        multiply(work, u, o, C, N, C, 40, ld, ld, true);
        __syncthreads();
        for (int i = threadIdx.x; i < C * N; i += blockDim.x) {
            const int token = chunk * C + i / N;
            if (token < T) output[(static_cast<std::size_t>(head) * T + token) * D + first_column + i % N] = o[(i / N) * ld + i % N] * rsqrtf(float(D));
        }
        const float decay = decays[chunk * heads + head];
        for (int i = threadIdx.x; i < D * N; i += blockDim.x) s[(i / N) * ld + i % N] *= decay;
        for (int i = threadIdx.x; i < C * D; i += blockDim.x) work[(i % D) * 40 + i / D] = kd[i];
        __syncthreads();
        multiply(work, u, s, D, N, C, 40, ld, ld, true);
        __syncthreads();
    }
    for (int i = threadIdx.x; i < D * N; i += blockDim.x) {
        const int row = i / N, col = i % N + first_column;
        const auto offset = static_cast<std::size_t>(head) * D * D + (transposed ? col * D + row : row * D + col);
        state[offset] = s[row * ld + i % N];
    }
}

inline void run(const float* q, const float* k, const float* v,
    const float* g, const float* beta, const float* initial, float* output, float* state,
    int B, int Hq, int Hv, int T, int D, bool transposed, bool tiled,
    mfq_tensor_backend::TensorOptions options, cudaStream_t stream) {
    const int heads = B * Hv, chunks = (T + C - 1) / C;
    auto panels = mfq_tensor_backend::empty({4, chunks, heads, C, D}, options);
    auto relations = mfq_tensor_backend::empty({chunks, heads, C, C}, options);
    auto decays = mfq_tensor_backend::empty({chunks, heads}, options);
    const auto shared_bytes = (4 * C * D + C * C + C) * sizeof(float);
#define MFQ_GDN_PREPARE(DIM) \
    MFQ_RUNTIME_CHECK(cudaFuncSetAttribute(prepare<DIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
        static_cast<int>(shared_bytes)) == cudaSuccess, "GDN chunk shared memory configuration failed"); \
    prepare<DIM><<<dim3(chunks, heads), 256, shared_bytes, stream>>>(q, k, v, g, beta, \
        panels.data_ptr<float>(), relations.data_ptr<float>(), decays.data_ptr<float>(), \
        Hq, Hv, T, heads, chunks, tiled)
    if (D == 32) { MFQ_GDN_PREPARE(32); }
    else if (D == 64) { MFQ_GDN_PREPARE(64); }
    else { MFQ_GDN_PREPARE(128); }
#undef MFQ_GDN_PREPARE
#define MFQ_GDN_APPLY(DIM) apply<DIM><<<dim3(heads, DIM / 16), 128, 0, stream>>>(panels.data_ptr<float>(), \
        relations.data_ptr<float>(), decays.data_ptr<float>(), initial, output, state, heads, T, chunks, transposed)
    if (D == 32) { MFQ_GDN_APPLY(32); }
    else if (D == 64) { MFQ_GDN_APPLY(64); }
    else { MFQ_GDN_APPLY(128); }
#undef MFQ_GDN_APPLY
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace mfq::gdn_chunked
