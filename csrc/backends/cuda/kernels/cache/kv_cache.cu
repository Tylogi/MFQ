// KV cache writer for decode/prefill plumbing.
// Layout: k/v input [B, H, T, D], cache [B, H, max_seq, D].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "kernels/kernels.h"

namespace {

constexpr int KV_WARP_SIZE = 32;
constexpr int KV_WARPS_PER_BLOCK = 4;
constexpr int KV_BLOCK_SIZE = KV_WARP_SIZE * KV_WARPS_PER_BLOCK;
constexpr size_t KV_ROW_KERNEL_MIN_ROWS = 4096;

template <typename scalar_t, bool Vectorized>
__device__ __forceinline__ void copy_kv_row(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    size_t src,
    size_t dst,
    int D,
    int lane)
{
    if constexpr (Vectorized) {
        constexpr int elements_per_vector = sizeof(uint2) / sizeof(scalar_t);
        const int vectors = D / elements_per_vector;
        const auto* k_vector = reinterpret_cast<const uint2*>(k + src);
        const auto* v_vector = reinterpret_cast<const uint2*>(v + src);
        auto* k_cache_vector = reinterpret_cast<uint2*>(k_cache + dst);
        auto* v_cache_vector = reinterpret_cast<uint2*>(v_cache + dst);
        for (int index = lane; index < vectors; index += KV_WARP_SIZE) {
            k_cache_vector[index] = k_vector[index];
            v_cache_vector[index] = v_vector[index];
        }
    } else {
        for (int d = lane; d < D; d += KV_WARP_SIZE) {
            k_cache[dst + d] = k[src + d];
            v_cache[dst + d] = v[src + d];
        }
    }
}

}  // namespace

template <typename scalar_t>
__global__ void kv_cache_write_flat_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim)
{
    size_t n = (size_t)B * H * T * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        size_t t0 = idx / D;
        int t = (int)(t0 % T);
        size_t h0 = t0 / T;
        int h = (int)(h0 % H);
        int b = (int)(h0 / H);
        int64_t p = pos_dim == 1 ? positions[t] : positions[(size_t)b * T + t];
        if (p < 0 || p >= max_seq) {
            continue;
        }
        size_t src = (((size_t)b * H + h) * T + t) * D + d;
        size_t dst = (((size_t)b * H + h) * max_seq + (size_t)p) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

template <typename scalar_t, bool Vectorized>
__global__ void kv_cache_write_rows_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim)
{
    const int warp = threadIdx.x / KV_WARP_SIZE;
    const int lane = threadIdx.x % KV_WARP_SIZE;
    const size_t row =
        static_cast<size_t>(blockIdx.x) * KV_WARPS_PER_BLOCK + warp;
    const size_t rows = static_cast<size_t>(B) * H * T;
    if (row >= rows) return;

    const int t = static_cast<int>(row % T);
    const size_t head_row = row / T;
    const int h = static_cast<int>(head_row % H);
    const int b = static_cast<int>(head_row / H);
    const int64_t p = pos_dim == 1
        ? positions[t]
        : positions[static_cast<size_t>(b) * T + t];
    if (p < 0 || p >= max_seq) return;

    const size_t src = row * D;
    const size_t dst =
        ((static_cast<size_t>(b) * H + h) * max_seq +
         static_cast<size_t>(p)) * D;
    copy_kv_row<scalar_t, Vectorized>(
        k, v, k_cache, v_cache, src, dst, D, lane);
}

template <typename scalar_t>
__global__ void kv_cache_write_ring_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    int64_t position_start,
    int B, int H, int T, int D, int capacity, int source_start)
{
    const int kept = T - source_start;
    const size_t n = (size_t)B * H * kept * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int d = (int)(idx % D);
        const size_t t0 = idx / D;
        const int local_t = (int)(t0 % kept);
        const size_t h0 = t0 / kept;
        const int h = (int)(h0 % H);
        const int b = (int)(h0 / H);
        const int t = source_start + local_t;
        const int slot = (int)((position_start + t) % capacity);
        const size_t src = (((size_t)b * H + h) * T + t) * D + d;
        const size_t dst = (((size_t)b * H + h) * capacity + slot) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

template <typename scalar_t>
__global__ void kv_cache_write_ring_positions_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B, int H, int T, int D, int capacity, int source_start)
{
    const int kept = T - source_start;
    const size_t n = (size_t)B * H * kept * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int d = (int)(idx % D);
        const size_t t0 = idx / D;
        const int local_t = (int)(t0 % kept);
        const int t = source_start + local_t;
        const size_t h0 = t0 / kept;
        const int h = (int)(h0 % H);
        const int b = (int)(h0 / H);
        const int slot = (int)(positions[t] % capacity);
        const size_t src = (((size_t)b * H + h) * T + t) * D + d;
        const size_t dst = (((size_t)b * H + h) * capacity + slot) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

namespace {

template <typename scalar_t>
void launch_kv_cache_write(
    scalar_t* k_cache,
    scalar_t* v_cache,
    const scalar_t* k,
    const scalar_t* v,
    const int64_t* positions,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim,
    cudaStream_t stream)
{
    const size_t rows = (size_t)B * H * T;
    if (rows >= KV_ROW_KERNEL_MIN_ROWS &&
        sizeof(scalar_t) < sizeof(float)) {
        constexpr int elements_per_vector =
            sizeof(uint2) / sizeof(scalar_t);
        const auto vector_addresses =
            reinterpret_cast<std::uintptr_t>(k) |
            reinterpret_cast<std::uintptr_t>(v) |
            reinterpret_cast<std::uintptr_t>(k_cache) |
            reinterpret_cast<std::uintptr_t>(v_cache);
        const bool vector_aligned =
            (vector_addresses & (alignof(uint2) - 1)) == 0;
        const int grid = static_cast<int>(
            (rows + KV_WARPS_PER_BLOCK - 1) / KV_WARPS_PER_BLOCK);
        if (D % elements_per_vector == 0 && vector_aligned) {
            kv_cache_write_rows_kernel<scalar_t, true><<<
                grid, KV_BLOCK_SIZE, 0, stream>>>(
                k, v, k_cache, v_cache, positions, B, H, T, D, max_seq,
                pos_dim);
        } else {
            kv_cache_write_rows_kernel<scalar_t, false><<<
                grid, KV_BLOCK_SIZE, 0, stream>>>(
                k, v, k_cache, v_cache, positions, B, H, T, D, max_seq,
                pos_dim);
        }
    } else {
        constexpr int block = 256;
        const size_t elements = rows * D;
        const int grid = static_cast<int>(std::min<size_t>(
            4096, (elements + block - 1) / block));
        kv_cache_write_flat_kernel<scalar_t><<<
            grid, block, 0, stream>>>(
            k, v, k_cache, v_cache, positions, B, H, T, D, max_seq, pos_dim);
    }
}

template <typename scalar_t>
void launch_kv_cache_write_ring(
    scalar_t* k_cache,
    scalar_t* v_cache,
    const scalar_t* k,
    const scalar_t* v,
    int64_t position_start,
    int B, int H, int T, int D, int capacity,
    cudaStream_t stream)
{
    // Keeping only the latest capacity entries makes every destination unique,
    // even when one prefill chunk is larger than the physical ring.
    const int source_start = std::max(0, T - capacity);
    const size_t n = (size_t)B * H * (T - source_start) * D;
    constexpr int block = 256;
    const int grid = (int)std::min<size_t>(4096, (n + block - 1) / block);
    kv_cache_write_ring_kernel<scalar_t><<<grid, block, 0, stream>>>(
        k, v, k_cache, v_cache, position_start, B, H, T, D, capacity,
        source_start);
}

template <typename scalar_t>
void launch_kv_cache_write_ring_positions(
    scalar_t* k_cache,
    scalar_t* v_cache,
    const scalar_t* k,
    const scalar_t* v,
    const int64_t* positions,
    int B, int H, int T, int D, int capacity,
    cudaStream_t stream)
{
    const int source_start = std::max(0, T - capacity);
    const size_t n = (size_t)B * H * (T - source_start) * D;
    constexpr int block = 256;
    const int grid = (int)std::min<size_t>(4096, (n + block - 1) / block);
    kv_cache_write_ring_positions_kernel<scalar_t><<<grid, block, 0, stream>>>(
        k, v, k_cache, v_cache, positions, B, H, T, D, capacity, source_start);
}

}  // namespace

namespace mfq::cuda::kernels {

void kv_cache_write(
    void* k_cache,
    void* v_cache,
    const void* k,
    const void* v,
    const std::int64_t* positions,
    ScalarType dtype,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim,
    cudaStream_t stream)
{
    if (dtype == ScalarType::float32) {
        launch_kv_cache_write(
            static_cast<float*>(k_cache), static_cast<float*>(v_cache),
            static_cast<const float*>(k), static_cast<const float*>(v),
            positions, B, H, T, D, max_seq, pos_dim, stream);
    } else if (dtype == ScalarType::float16) {
        launch_kv_cache_write(
            static_cast<__half*>(k_cache), static_cast<__half*>(v_cache),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            positions, B, H, T, D, max_seq, pos_dim, stream);
    } else if (dtype == ScalarType::bfloat16) {
        launch_kv_cache_write(
            static_cast<__nv_bfloat16*>(k_cache),
            static_cast<__nv_bfloat16*>(v_cache),
            static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v),
            positions, B, H, T, D, max_seq, pos_dim, stream);
    } else {
        throw std::invalid_argument(
            "kv_cache_write: dtype must be f16, bf16 or f32");
    }
}

void kv_cache_write_ring(
    void* k_cache,
    void* v_cache,
    const void* k,
    const void* v,
    ScalarType dtype,
    std::int64_t position_start,
    int B,
    int H,
    int T,
    int D,
    int capacity,
    cudaStream_t stream)
{
    if (dtype == ScalarType::float32) {
        launch_kv_cache_write_ring(
            static_cast<float*>(k_cache), static_cast<float*>(v_cache),
            static_cast<const float*>(k), static_cast<const float*>(v),
            position_start, B, H, T, D, capacity, stream);
    } else if (dtype == ScalarType::float16) {
        launch_kv_cache_write_ring(
            static_cast<__half*>(k_cache), static_cast<__half*>(v_cache),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            position_start, B, H, T, D, capacity, stream);
    } else if (dtype == ScalarType::bfloat16) {
        launch_kv_cache_write_ring(
            static_cast<__nv_bfloat16*>(k_cache),
            static_cast<__nv_bfloat16*>(v_cache),
            static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v),
            position_start, B, H, T, D, capacity, stream);
    } else {
        throw std::invalid_argument(
            "kv_cache_write_ring: dtype must be f16, bf16 or f32");
    }
}

void kv_cache_write_ring_positions(
    void* k_cache,
    void* v_cache,
    const void* k,
    const void* v,
    const std::int64_t* positions,
    ScalarType dtype,
    int B,
    int H,
    int T,
    int D,
    int capacity,
    cudaStream_t stream)
{
    if (dtype == ScalarType::float32) {
        launch_kv_cache_write_ring_positions(
            static_cast<float*>(k_cache), static_cast<float*>(v_cache),
            static_cast<const float*>(k), static_cast<const float*>(v),
            positions, B, H, T, D, capacity, stream);
    } else if (dtype == ScalarType::float16) {
        launch_kv_cache_write_ring_positions(
            static_cast<__half*>(k_cache), static_cast<__half*>(v_cache),
            static_cast<const __half*>(k), static_cast<const __half*>(v),
            positions, B, H, T, D, capacity, stream);
    } else if (dtype == ScalarType::bfloat16) {
        launch_kv_cache_write_ring_positions(
            static_cast<__nv_bfloat16*>(k_cache),
            static_cast<__nv_bfloat16*>(v_cache),
            static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v),
            positions, B, H, T, D, capacity, stream);
    } else {
        throw std::invalid_argument(
            "kv_cache_write_ring_positions: dtype must be f16, bf16 or f32");
    }
}

}  // namespace mfq::cuda::kernels
