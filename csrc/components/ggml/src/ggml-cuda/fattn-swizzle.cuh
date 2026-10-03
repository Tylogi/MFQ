#pragma once

#include "common.cuh"
#include "mma.cuh"

// XOR swizzle for K/V SMEM tiles to avoid bank conflicts without row padding
// on Turing and newer CUDA GPUs. Strides that are not multiples of 32 half2
// columns retain the padded layout.

namespace ggml_cuda_fattn_smem_swizzle {

static __host__ __device__ constexpr bool bank_aligned(const int nbatch_2) {
    return nbatch_2 >= 32 && nbatch_2 % 32 == 0;
}

static __device__ constexpr bool enabled(const int nbatch_2) {
#if defined(TURING_MMA_AVAILABLE)
    return bank_aligned(nbatch_2);
#else
    GGML_UNUSED(nbatch_2);
    return false;
#endif
}

static __host__ bool enabled(const int nbatch_2, const int cc) {
#ifdef GGML_USE_HIP
    GGML_UNUSED(nbatch_2);
    GGML_UNUSED(cc);
    return false;
#else
    return turing_mma_available(cc) && bank_aligned(nbatch_2);
#endif
}

static __device__ constexpr int tile_stride(const int nbatch_2) {
    return enabled(nbatch_2) ? nbatch_2 : nbatch_2 + 4;
}

static __host__ int tile_stride(const int nbatch_2, const int cc) {
    return enabled(nbatch_2, cc) ? nbatch_2 : nbatch_2 + 4;
}

template<int stride_h2>
static __device__ __forceinline__ int bytes_rc(
        const int row, const int col_h2) {
    static_assert(bank_aligned(stride_h2),
                  "swizzled tile needs a stride that is a multiple of 32");
    return ((row * stride_h2 + col_h2) * static_cast<int>(sizeof(half2))) ^
        ((row & 7) << 4);
}

static __device__ __forceinline__ void ldmatrix_x4(
        int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ldmatrix_x4_trans(
        int * xi, const half2 * addr) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
        : "l"(addr));
#else
    GGML_UNUSED_VARS(xi, addr);
    NO_DEVICE_CODE;
#endif
}

template<int stride_h2>
static __device__ __forceinline__ const half2 * lane_addr(
        const half2 * tile_base, const int base_row,
        const int base_col_h2, const int I, const int J) {
    static_assert(bank_aligned(stride_h2),
                  "swizzled tile needs a stride that is a multiple of 32");
    const int lane_row = threadIdx.x % I;
    const int lane_col = (threadIdx.x / I) * (J / 2);
    uint32_t byte_offset = static_cast<uint32_t>(
        (base_row + lane_row) * stride_h2 + base_col_h2 + lane_col) *
        static_cast<uint32_t>(sizeof(half2));
    byte_offset ^= static_cast<uint32_t>(
        ((base_row + lane_row) & 7) << 4);
    return reinterpret_cast<const half2 *>(
        reinterpret_cast<const char *>(tile_base) + byte_offset);
}

template<int stride_h2, bool swizzled, typename Tile>
static __device__ __forceinline__ void load_ldmatrix(
        Tile & tile, const half2 * tile_base,
        const int base_row, const int base_col_h2) {
    if constexpr (swizzled) {
        static_assert(std::is_same_v<Tile,
                          ggml_cuda_mma::tile<16, 8, half2>>,
                      "unsupported swizzled matrix tile");
        ldmatrix_x4(
            reinterpret_cast<int *>(tile.x),
            lane_addr<stride_h2>(tile_base, base_row, base_col_h2,
                                 Tile::I, Tile::J));
    } else {
        ggml_cuda_mma::load_ldmatrix(
            tile, tile_base + base_row * stride_h2 + base_col_h2,
            stride_h2);
    }
}

template<int stride_h2, bool swizzled, typename Tile>
static __device__ __forceinline__ void load_ldmatrix(
        Tile & tile, const half2 * tile_base, const int offset_h2) {
    if constexpr (swizzled) {
        load_ldmatrix<stride_h2, swizzled>(
            tile, tile_base, offset_h2 / stride_h2,
            offset_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix(
            tile, tile_base + offset_h2, stride_h2);
    }
}

template<int stride_h2, bool swizzled, typename Tile>
static __device__ __forceinline__ void load_ldmatrix_trans(
        Tile & tile, const half2 * tile_base,
        const int base_row, const int base_col_h2) {
    if constexpr (swizzled) {
        static_assert(std::is_same_v<Tile,
                          ggml_cuda_mma::tile<16, 8, half2>>,
                      "unsupported swizzled matrix tile");
        ldmatrix_x4_trans(
            reinterpret_cast<int *>(tile.x),
            lane_addr<stride_h2>(tile_base, base_row, base_col_h2,
                                 Tile::I, Tile::J));
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(
            tile, tile_base + base_row * stride_h2 + base_col_h2,
            stride_h2);
    }
}

template<int stride_h2, bool swizzled, typename Tile>
static __device__ __forceinline__ void load_ldmatrix_trans(
        Tile & tile, const half2 * tile_base, const int offset_h2) {
    if constexpr (swizzled) {
        load_ldmatrix_trans<stride_h2, swizzled>(
            tile, tile_base, offset_h2 / stride_h2,
            offset_h2 % stride_h2);
    } else {
        ggml_cuda_mma::load_ldmatrix_trans(
            tile, tile_base + offset_h2, stride_h2);
    }
}

}  // namespace ggml_cuda_fattn_smem_swizzle
