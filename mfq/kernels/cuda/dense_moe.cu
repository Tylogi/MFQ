#include "mfq_cuda_moe_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

// Each warp reads one output row directly from its expert, without a route map
// or a copy of the bank. This avoids padding tiny decode batches into MMA tiles.
template <typename T>
__global__ void dense_route_matvec(const T* bank, const T* input, const int32_t* ids,
                                  T* output, int pairs, int routes, int n, int k, bool routed) {
    const int lane = threadIdx.x % 32;
    for (int64_t task = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
         task < int64_t(pairs) * n; task += int64_t(gridDim.x) * 4) {
        const int column = task % n;
        const int pair = task / n;
        const auto* weight = bank + (int64_t(ids[pair]) * n + column) * k;
        const auto* source = input + int64_t(routed ? pair : pair / routes) * k;
        float sum = 0.0f;
        for (int i = lane; i < k; i += 32)
            sum = fmaf(float(weight[i]), float(source[i]), sum);
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) output[task] = T(sum);
    }
}

// One block handles 16 compact expert rows and 64 output columns. Shared
// memory on sm_120 is 10 KB including FP32 staging, below its 99 KB limit.
template <typename T>
__global__ void dense_grouped_mma(const T* bank, const T* input, T* output,
                                const int32_t* ids_dst, const int32_t* expert_bounds,
                                const int32_t* tile_bounds, const int32_t* tile_experts,
                                int experts, int routes, int n, int k, int tile_m, bool routed) {
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    __shared__ __align__(32) T a[16 * 32];
    __shared__ __align__(32) T b[64 * 32];
    __shared__ __align__(32) float c[16 * 64];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int ntiles = (n + 63) / 64;
    const int subtiles = (tile_m + 15) / 16;
    const int64_t tasks = int64_t(tile_bounds[experts]) * subtiles * ntiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int column = (task % ntiles) * 64;
        const int tile = task / ntiles / subtiles;
        const int expert = tile_experts[tile];
        const int row = expert_bounds[expert] + (tile - tile_bounds[expert]) * tile_m +
            (task / ntiles % subtiles) * 16;
        const int end = min(expert_bounds[expert + 1],
                            expert_bounds[expert] + (tile - tile_bounds[expert] + 1) * tile_m);
        if (row >= end) continue;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int base = 0; base < k; base += 32) {
            for (int i = tid; i < 16 * 32; i += 128) {
                const int r = row + i / 32;
                const int index = base + i % 32;
                const int pair = r < end ? ids_dst[r] : 0;
                a[i] = r < end && index < k
                    ? input[int64_t(routed ? pair : pair / routes) * k + index] : T(0.0f);
            }
            for (int i = tid; i < 64 * 32; i += 128) {
                const int col = column + i / 32;
                const int index = base + i % 32;
                b[i] = col < n && index < k
                    ? bank[(int64_t(expert) * n + col) * k + index] : T(0.0f);
            }
            __syncthreads();
            for (int offset = 0; offset < 32; offset += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, a + offset, 32);
                wmma::load_matrix_sync(fb, b + warp * 16 * 32 + offset, 32);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            __syncthreads();
        }
        wmma::store_matrix_sync(c + warp * 16, acc, 64, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < 16 * 64; i += 128) {
            const int r = row + i / 64;
            const int col = column + i % 64;
            if (r < end && col < n) output[int64_t(ids_dst[r]) * n + col] = T(c[i]);
        }
        __syncthreads();
    }
#else
    asm volatile("trap;");
#endif
}

// F32 uses full-precision SIMT FMA instead of rounding operands to TF32/F16.
__global__ void dense_grouped_float(const float* bank, const float* input, float* output,
                                    const int32_t* ids_dst, const int32_t* expert_bounds,
                                    const int32_t* tile_bounds, const int32_t* tile_experts,
                                    int experts, int routes, int n, int k, int tile_m, bool routed) {
    __shared__ float a[8 * 32];
    __shared__ float b[32 * 33];
    const int tid = threadIdx.x;
    const int ntiles = (n + 31) / 32;
    const int subtiles = (tile_m + 7) / 8;
    const int64_t tasks = int64_t(tile_bounds[experts]) * subtiles * ntiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int column = (task % ntiles) * 32;
        const int tile = task / ntiles / subtiles;
        const int expert = tile_experts[tile];
        const int row = expert_bounds[expert] + (tile - tile_bounds[expert]) * tile_m +
            (task / ntiles % subtiles) * 8;
        const int end = min(expert_bounds[expert + 1],
                            expert_bounds[expert] + (tile - tile_bounds[expert] + 1) * tile_m);
        if (row >= end) continue;
        float sum = 0.0f;
        for (int base = 0; base < k; base += 32) {
            const int r = row + tid / 32;
            const int index = base + tid % 32;
            const int pair = r < end ? ids_dst[r] : 0;
            a[tid] = r < end && index < k
                ? input[int64_t(routed ? pair : pair / routes) * k + index] : 0.0f;
            for (int i = tid; i < 32 * 32; i += 256) {
                const int col = column + i / 32;
                const int ki = base + i % 32;
                b[i / 32 * 33 + i % 32] = col < n && ki < k
                    ? bank[(int64_t(expert) * n + col) * k + ki] : 0.0f;
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 32; ++i)
                sum = fmaf(a[tid / 32 * 32 + i], b[tid % 32 * 33 + i], sum);
            __syncthreads();
        }
        if (row + tid / 32 < end && column + tid % 32 < n)
            output[int64_t(ids_dst[row + tid / 32]) * n + column + tid % 32] = sum;
        __syncthreads();
    }
}

template <typename T>
void launch_dense(const mfq_tensor_backend::Tensor& bank, const mfq_tensor_backend::Tensor& input,
                  const mfq_tensor_backend::Tensor& ids, mfq_tensor_backend::Tensor& output,
                  const mfq_tensor_backend::Tensor& ids_dst, const mfq_tensor_backend::Tensor& bounds,
                  const mfq_tensor_backend::Tensor& tiles, const mfq_tensor_backend::Tensor& experts,
                  int tile_m) {
    const int n = bank.size(1);
    const int k = bank.size(2);
    const int routes = ids.size(1);
    const auto stream = mfq_current_cuda_stream();
    const auto* weights = reinterpret_cast<const T*>(bank.data_ptr());
    const auto* source = reinterpret_cast<const T*>(input.data_ptr());
    auto* destination = reinterpret_cast<T*>(output.data_ptr());
    if (ids.size(0) <= 8) {
        const int blocks = std::min<int64_t>(65535, (ids.numel() * n + 3) / 4);
        dense_route_matvec<<<blocks, 128, 0, stream>>>(
            weights, source, ids.data_ptr<int32_t>(), destination, ids.numel(), routes, n, k, input.dim() == 3);
    } else {
        constexpr bool fp32 = std::is_same_v<T, float>;
        constexpr int bm = fp32 ? 8 : 16;
        constexpr int bn = fp32 ? 32 : 64;
        const int64_t max_tiles = (ids.numel() + tile_m - 1) / tile_m + bank.size(0);
        const int blocks = std::min<int64_t>(4096, max_tiles * ((tile_m + bm - 1) / bm) * ((n + bn - 1) / bn));
        if constexpr (fp32) {
            dense_grouped_float<<<blocks, 256, 0, stream>>>(weights, source, destination,
                ids_dst.data_ptr<int32_t>(), bounds.data_ptr<int32_t>(), tiles.data_ptr<int32_t>(),
                experts.data_ptr<int32_t>(), bank.size(0), routes, n, k, tile_m, input.dim() == 3);
        } else {
            dense_grouped_mma<<<blocks, 128, 0, stream>>>(weights, source, destination,
                ids_dst.data_ptr<int32_t>(), bounds.data_ptr<int32_t>(), tiles.data_ptr<int32_t>(),
                experts.data_ptr<int32_t>(), bank.size(0), routes, n, k, tile_m, input.dim() == 3);
        }
    }
}
} // namespace

mfq_tensor_backend::Tensor dense_moe_grouped_matmul_cuda(
        mfq_tensor_backend::Tensor bank, mfq_tensor_backend::Tensor input, mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor ids_dst, mfq_tensor_backend::Tensor expert_bounds,
        mfq_tensor_backend::Tensor tile_bounds, mfq_tensor_backend::Tensor tile_experts, int64_t tile_m) {
    namespace tb = mfq_tensor_backend;
    const MfqCudaGuard guard(bank.device());
    MFQ_RUNTIME_CHECK(bank.is_cuda() && bank.is_contiguous() && bank.dim() == 3 &&
        input.is_cuda() && input.is_contiguous() && input.device() == bank.device() &&
        input.scalar_type() == bank.scalar_type() && ids.is_cuda() && ids.device() == bank.device() &&
        ids.dim() == 2 && ids.scalar_type() == tb::kInt32 && ids.is_contiguous(),
        "dense grouped tensors require matching CUDA devices, contiguous storage and int32 IDs");
    MFQ_RUNTIME_CHECK(bank.scalar_type() == tb::kFloat16 || bank.scalar_type() == tb::kBFloat16 ||
        bank.scalar_type() == tb::kFloat32, "dense grouped bank must be F16, BF16 or F32");
    MFQ_RUNTIME_CHECK((input.dim() == 2 || (input.dim() == 3 && input.size(1) == ids.size(1))) &&
        input.size(0) == ids.size(0) && input.size(-1) == bank.size(2), "dense grouped input shape mismatch");
    if (ids.size(0) > 8) {
        for (const auto& map : {ids_dst, expert_bounds, tile_bounds, tile_experts})
            MFQ_RUNTIME_CHECK(map.is_cuda() && map.device() == bank.device() && map.is_contiguous() &&
                map.scalar_type() == tb::kInt32, "dense grouped route map must be contiguous CUDA int32");
        MFQ_RUNTIME_CHECK(ids_dst.numel() == ids.numel() && expert_bounds.numel() == bank.size(0) + 1 &&
            tile_bounds.numel() == bank.size(0) + 1 && tile_experts.numel() >= ids.numel() &&
            (tile_m == 8 || tile_m == 16 || tile_m == 32 || tile_m == 64 || tile_m == 128),
            "dense grouped route map geometry mismatch");
    }
    auto output = tb::empty({ids.size(0), ids.size(1), bank.size(1)}, bank.options());
    if (bank.scalar_type() == tb::kFloat16)
        launch_dense<__half>(bank, input, ids, output, ids_dst, expert_bounds, tile_bounds, tile_experts, tile_m);
    else if (bank.scalar_type() == tb::kBFloat16)
        launch_dense<__nv_bfloat16>(bank, input, ids, output, ids_dst, expert_bounds,
                                   tile_bounds, tile_experts, tile_m);
    else
        launch_dense<float>(bank, input, ids, output, ids_dst, expert_bounds, tile_bounds, tile_experts, tile_m);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
