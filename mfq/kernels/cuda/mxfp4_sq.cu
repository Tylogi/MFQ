#include "mxfp4_sq.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstdlib>

#include "packed_backward.cuh"
#include "packed_gemm.cuh"

namespace {

// Exact frozen catalogs; source tests compare every entry to Metal.
__device__ __constant__ std::uint8_t kSq1Palette[64] = {
    15, 6, 14, 7, 13, 7, 15, 5, 15, 7, 11, 6, 14, 3, 12, 6,
    14, 6, 14, 4, 10, 6, 14, 2, 13, 6, 14, 5, 9, 6, 14, 1,
    15, 2, 11, 7, 10, 7, 15, 3, 9, 7, 15, 1, 14, 0, 13, 5,
    12, 4, 11, 3, 10, 2, 9, 1, 13, 4, 12, 5, 11, 4, 12, 3,
};

__device__ __constant__ std::uint8_t kSq2Palette[128] = {
    15, 13, 0,  5,  15, 13, 1,  6,  15, 12, 2,  6,  14, 11, 0,  3,  14, 11, 1,
    5,  14, 11, 2,  6,  14, 10, 1,  4,  14, 10, 1,  5,  14, 10, 3,  6,  14, 10,
    4,  7,  14, 9,  5,  7,  13, 10, 1,  4,  13, 9,  3,  6,  13, 0,  5,  7,  12,
    9,  2,  5,  12, 9,  2,  6,  15, 12, 3,  7,  11, 0,  3,  6,  12, 9,  3,  6,
    13, 9,  2,  6,  14, 10, 3,  7,  15, 11, 4,  7,  15, 13, 9,  4,  15, 11, 1,
    5,  15, 11, 2,  6,  15, 12, 1,  6,  15, 12, 2,  7,  15, 13, 0,  6,  14, 11,
    0,  4,  13, 1,  5,  7,  15, 14, 13, 12, 15, 14, 13, 11,
};

__device__ __constant__ std::uint8_t kSq3Palette[256] = {
    15, 14, 13, 11, 0,  3,  5,  7,  15, 13, 11, 0,  3,  5,  6,  7,  15, 14, 13,
    11, 1,  4,  6,  7,  13, 11, 9,  0,  1,  2,  3,  6,  15, 14, 12, 9,  3,  5,
    6,  7, 15, 13, 12, 10, 0,  2,  4,  5,  13, 12, 10, 0,  2,  4,  5,  7,  14,
    12, 10, 0,  2,  4,  5,  7,  15, 14, 12, 10, 0,  3,  5,  7,  15, 13, 11, 0,
    2,  4,  5,  6,  15, 13, 11, 0,  2,  4,  6,  7,  15, 14, 12, 10, 0,  2,  4,
    5,  15, 14, 12, 10, 0,  2,  4,  6,  14, 13, 11, 0,  2,  4,  5,  6,  13, 12,
    10, 0,  3,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  6,  14, 13, 12, 10, 0,
    3,  5,  6,  15, 13, 12, 10, 0,  3,  5,  6,  13, 12, 10, 0,  2,  4,  6,  7,
    14, 13, 11, 0,  2,  4,  5,  7,  13, 12, 9,  0,  2,  4,  5,  6,  14, 12, 10,
    0,  2,  4,  6,  7,  15, 14, 13, 11, 0,  3,  6,  7,  15, 14, 11, 0,  3,  5,
    6,  7,  14, 13, 12, 10, 0,  2,  4,  5,  14, 13, 12, 10, 0,  3,  5,  7,  15,
    14, 11, 0,  2,  4,  6,  7,  14, 11, 10, 9,  0,  1,  2,  3,  13, 11, 0,  2,
    4,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  7,  15, 12, 10, 0,  2,  4,  5,
    7,  15, 13, 12, 10, 1,  4,  6,  7,
};

__device__ __forceinline__ unsigned read_bits(
        const std::uint8_t* data, std::size_t index, int bits) {
    const auto bit = index * bits;
    const unsigned shift = bit & 7;
    unsigned value = data[bit >> 3];
    if (shift + bits > 8) value |= unsigned(data[(bit >> 3) + 1]) << 8;
    return (value >> shift) & ((1u << bits) - 1);
}

__device__ __forceinline__ unsigned block_tag(
        const std::uint8_t* row_symbols, const std::uint8_t* selectors,
        std::size_t block, std::size_t selector_index, int bits) {
    const auto* words = reinterpret_cast<const unsigned*>(
        row_symbols + block * (bits * 4));
    unsigned low;
    if (bits == 1) {
        const unsigned word = words[0];
        low = (__popc(word & 0x55555555u) & 1u) |
            ((__popc(word & 0xaaaaaaaau) & 1u) << 1u);
    } else if (bits == 2) {
        unsigned fold = words[0] ^ words[1];
        fold ^= fold >> 16; fold ^= fold >> 8; fold ^= fold >> 4; fold ^= fold >> 2;
        low = fold & 3;
    } else {
        constexpr unsigned m0 = 0x49249249u, m1 = 0x92492492u, m2 = 0x24924924u;
        const unsigned a = words[0], b = words[1], c = words[2];
        const unsigned lo = __popc(a & m0) + __popc(b & m1) + __popc(c & m2);
        const unsigned hi = __popc(a & m1) + __popc(b & m2) + __popc(c & m0);
        low = (lo & 1) | ((hi & 1) << 1);
    }
    return low | (((selectors[selector_index >> 3] >> (selector_index & 7)) & 1) << 2);
}

__device__ __forceinline__ float decode_native(
        unsigned nibble, unsigned exponent) {
    float magnitude = float((0xc8643210u >> ((nibble & 7) * 4)) & 15) * 0.5f;
    if (nibble & 8) magnitude = -magnitude;
    const float scale = __uint_as_float(exponent == 0 ? 0x00400000u : exponent << 23);
    float result;
    // No .ftz: preserve base-zero E8M0 in --use_fast_math builds too.
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(magnitude), "f"(scale));
    return result;
}

__device__ __forceinline__ float decode_value(
        unsigned palette, unsigned symbol, unsigned exponent, int bits) {
    const unsigned nibble = bits == 1
        ? kSq1Palette[palette * 2 + symbol]
        : bits == 2
        ? kSq2Palette[palette * 4 + symbol]
        : kSq3Palette[palette * 8 + symbol];
    return decode_native(nibble, exponent);
}

struct SqPackedDecoder {
    const std::uint8_t *blob, *row_q;
    const std::int32_t *offsets, *auxiliary;
    mfq::sq::Layout q;
    __device__ void load8(int row, int column, float* destination, int outputs, int width) const {
        if (row >= outputs || column >= width) {
#pragma unroll
            for (int i = 0; i < 8; ++i) destination[i] = 0;
            return;
        }
        const int bits = row_q[row];
        const auto aux = static_cast<std::size_t>(auxiliary[row]);
        const auto* symbols = blob + q.symbols + offsets[row];
        const auto block = column / 32;
        unsigned exponent, palette = 0;
        if (bits == 4) exponent = blob[q.native_scales + aux * (q.width / 32) + block];
        else {
            const auto selector = aux * (q.width / 32) + block;
            const auto state = aux * 8 + block_tag(symbols, blob + q.selectors, block, selector, bits);
            exponent = q.base + read_bits(blob + q.scales, state, 2);
            palette = read_bits(blob + q.palettes, state, 5);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const auto symbol = read_bits(symbols, column + i, bits);
            destination[i] = bits == 4 ? decode_native(symbol, exponent) : decode_value(palette, symbol, exponent, bits);
        }
    }

};

template<typename T> __device__ __forceinline__ float as_float(T x) { return float(x); }
template<> __device__ __forceinline__ float as_float(__half x) { return __half2float(x); }
template<typename T> __device__ __forceinline__ T from_float(float x) { return T(x); }
template<> __device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }

template<typename T>
__global__ void sq_dequant(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        T* out,
        mfq::sq::Layout q) {
    const auto* symbols = blob + q.symbols;
    const std::size_t count = std::size_t(q.outputs) * q.width;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += std::size_t(gridDim.x) * blockDim.x) {
        const auto output = index / q.width;
        const auto column = index - output * q.width;
        const int bits = row_q[output];
        const auto auxiliary = std::size_t(row_auxiliary[output]);
        const auto* row_symbols =
            symbols + std::size_t(row_symbol_byte_offsets[output]);
        const auto block = column / 32;
        const auto symbol = read_bits(row_symbols, column, bits);
        if (bits == 4) {
            const auto exponent = blob[
                q.native_scales + auxiliary * (q.width / 32) + block];
            out[index] = from_float<T>(decode_native(symbol, exponent));
        } else {
            const auto selector_index =
                auxiliary * (q.width / 32) + block;
            const auto state = auxiliary * 8 + block_tag(
                row_symbols, blob + q.selectors, block, selector_index, bits);
            const auto exponent = q.base + read_bits(blob + q.scales, state, 2);
            out[index] = from_float<T>(decode_value(
                read_bits(blob + q.palettes, state, 5),
                symbol, exponent, bits));
        }
    }
}

// One warp per output, coalesced K loads, decode reuse across TILE_M rows.
// All batch sizes use packed weights; M=1..6 have exact tile specializations.
template<int TILE_M, typename T, bool ROUTED = false>
__global__ void sq_mmq(
                       const std::uint8_t* blob,
                       const std::uint8_t* row_q,
                       const std::int32_t* row_symbol_byte_offsets,
                       const std::int32_t* row_auxiliary,
                       const T* x, T* y,
                       mfq::sq::Layout q, int rows,
                       const std::int32_t* expert_ids,
                       const std::int32_t* expert_local,
                       int n_experts,
                       int local_experts,
                       int out_per_expert,
                       int routes,
                       bool shared_input) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int logical_outputs = ROUTED ? out_per_expert : q.outputs;
    const std::int64_t output_tiles =
        (std::int64_t(logical_outputs) + 3) / 4;
    const std::int64_t row_tiles = (std::int64_t(rows) + TILE_M - 1) / TILE_M;
    const auto* symbols = blob + q.symbols;
    for (std::int64_t task = blockIdx.x; task < output_tiles * row_tiles; task += gridDim.x) {
        const auto logical_output = (task % output_tiles) * 4 + warp;
        if (logical_output >= logical_outputs) continue;
        const auto first_row = (task / output_tiles) * TILE_M;
        int local_expert = 0;
        if constexpr (ROUTED) {
            const int expert = expert_ids[first_row];
            if (expert < 0 || expert >= n_experts) continue;
            local_expert = expert_local[expert];
            if (local_expert < 0 || local_expert >= local_experts) continue;
        }
        const auto output = ROUTED
            ? std::int64_t(local_expert) * out_per_expert + logical_output
            : logical_output;
        const int bits = row_q[output];
        const auto auxiliary = std::size_t(row_auxiliary[output]);
        const auto* row_symbols =
            symbols + std::size_t(row_symbol_byte_offsets[output]);
        float accum[TILE_M] = {};
        for (int column = 0; column < q.width; column += 32) {
            const auto block = std::size_t(column / 32);
            unsigned state_value = 0;
            if (lane == 0) {
                if (bits == 4) {
                    state_value = blob[
                        q.native_scales + auxiliary * (q.width / 32) + block];
                } else {
                    const auto selector_index =
                        auxiliary * (q.width / 32) + block;
                    const auto state = auxiliary * 8 + block_tag(
                        row_symbols, blob + q.selectors,
                        block, selector_index, bits);
                    state_value = (read_bits(blob + q.palettes, state, 5) << 8) |
                                  (q.base + read_bits(blob + q.scales, state, 2));
                }
            }
            state_value = __shfl_sync(0xffffffffu, state_value, 0);
            const auto symbol = read_bits(
                row_symbols, std::size_t(column + lane), bits);
            const float w = bits == 4
                ? decode_native(symbol, state_value)
                : decode_value(
                    state_value >> 8,
                    symbol,
                    state_value & 255,
                    bits);
#pragma unroll
            for (int m = 0; m < TILE_M; ++m)
                if (first_row + m < rows)
                    accum[m] = fmaf(
                        w,
                        as_float(x[((ROUTED && shared_input)
                            ? (first_row + m) / routes
                            : first_row + m) * q.width + column + lane]),
                        accum[m]);
        }
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1)
                accum[m] += __shfl_down_sync(0xffffffffu, accum[m], delta);
            if (lane == 0 && first_row + m < rows)
                y[(first_row + m) * logical_outputs + logical_output] =
                    from_float<T>(accum[m]);
        }
    }
}

template<int Rows>
__global__ void __launch_bounds__(128) sq_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int block = static_cast<int>(blockIdx.x) * 4 + warp;
    const int column = block * 32 + lane;
    if (column >= q.width) return;

    const auto* symbols = blob + q.symbols;
    const int split = static_cast<int>(blockIdx.y);
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, q.outputs);
    float accumulators[Rows] = {};
    for (int output = output0; output < output_end; ++output) {
        const int bits = row_q[output];
        const auto auxiliary = std::size_t(row_auxiliary[output]);
        const auto* row_symbols =
            symbols + std::size_t(row_symbol_byte_offsets[output]);
        unsigned state_value = 0;
        if (lane == 0) {
            if (bits == 4) {
                state_value = blob[
                    q.native_scales + auxiliary * (q.width / 32) + block];
            } else {
                const auto selector_index =
                    auxiliary * (q.width / 32) + block;
                const auto state = auxiliary * 8 + block_tag(
                    row_symbols, blob + q.selectors,
                    block, selector_index, bits);
                state_value =
                    (read_bits(blob + q.palettes, state, 5) << 8) |
                    (q.base + read_bits(blob + q.scales, state, 2));
            }
        }
        state_value = __shfl_sync(0xffffffffu, state_value, 0);
        const auto symbol = read_bits(row_symbols, column, bits);
        const float weight = bits == 4
            ? decode_native(symbol, state_value)
            : decode_value(
                state_value >> 8,
                symbol,
                state_value & 255,
                bits);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row < rows
                ? __half2float(output_gradient[
                    std::size_t(row) * q.outputs + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            accumulators[row] = fmaf(gradient, weight, accumulators[row]);
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row < rows) {
            partials[(std::size_t(split) * rows + row) * q.width + column] =
                accumulators[row];
        }
    }
}

mfq::sq::Layout validate(
        const mfq_tensor_backend::Tensor& blob,
        const mfq_tensor_backend::Tensor& row_q,
        const mfq_tensor_backend::Tensor& row_symbol_byte_offsets,
        const mfq_tensor_backend::Tensor& row_auxiliary,
        std::int64_t bits,
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows) {
    const auto q = bits == 0
        ? mfq::sq::adaptive_layout(outputs, width, base, q_sum, sq4_rows)
        : mfq::sq::layout(bits, outputs, width, base);
    MFQ_RUNTIME_CHECK(blob.is_cuda() && blob.scalar_type() == mfq_tensor_backend::kUInt8 &&
        blob.dim() == 1 && blob.is_contiguous() && std::size_t(blob.numel()) == q.bytes,
        "MXFP4-SQ blob requires contiguous rank-1 CUDA uint8 and exact payload length");
    MFQ_RUNTIME_CHECK(
        row_q.is_cuda() && row_q.get_device() == blob.get_device() &&
        row_q.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q.dim() == 1 && row_q.is_contiguous() && row_q.numel() == outputs &&
        row_symbol_byte_offsets.is_cuda() &&
        row_symbol_byte_offsets.get_device() == blob.get_device() &&
        row_symbol_byte_offsets.scalar_type() == mfq_tensor_backend::kInt32 &&
        row_symbol_byte_offsets.dim() == 1 &&
        row_symbol_byte_offsets.is_contiguous() &&
        row_symbol_byte_offsets.numel() == outputs &&
        row_auxiliary.is_cuda() && row_auxiliary.get_device() == blob.get_device() &&
        row_auxiliary.scalar_type() == mfq_tensor_backend::kInt32 &&
        row_auxiliary.dim() == 1 && row_auxiliary.is_contiguous() &&
        row_auxiliary.numel() == outputs,
        "MXFP4-SQ row metadata requires matching contiguous CUDA arrays");
    MFQ_RUNTIME_CHECK(reinterpret_cast<std::uintptr_t>(blob.data_ptr<std::uint8_t>()) % 4 == 0,
        "MXFP4-SQ blob must be 4-byte aligned");
    return q;
}

template<int M, typename T>
void launch_mmq(
                const std::uint8_t* blob,
                const std::uint8_t* row_q,
                const std::int32_t* row_symbol_byte_offsets,
                const std::int32_t* row_auxiliary,
                const T* x, T* y,
                mfq::sq::Layout q, int rows, cudaStream_t stream) {
    const auto tasks = ((std::int64_t(q.outputs) + 3) / 4) * ((std::int64_t(rows) + M - 1) / M);
    const int blocks = int(std::min<std::int64_t>(tasks, 65535));
    sq_mmq<M, T><<<blocks, 128, 0, stream>>>(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        x, y, q, rows,
        nullptr, nullptr, 0, 0, q.outputs, 1, false);
}

void launch_routed_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* x,
        __half* y,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        mfq::sq::Layout q,
        int route_count,
        int routes,
        int n_experts,
        int local_experts,
        int out_per_expert,
        bool shared_input,
        cudaStream_t stream) {
    const auto tasks =
        std::int64_t(route_count) * ((std::int64_t(out_per_expert) + 3) / 4);
    const int blocks = int(std::min<std::int64_t>(tasks, 65535));
    sq_mmq<1, __half, true><<<blocks, 128, 0, stream>>>(
        blob,
        row_q,
        row_symbol_byte_offsets,
        row_auxiliary,
        x,
        y,
        q,
        route_count,
        expert_ids,
        expert_local,
        n_experts,
        local_experts,
        out_per_expert,
        routes,
        shared_input);
}

template<typename T>
void dispatch_mmq(
                  const std::uint8_t* blob,
                  const std::uint8_t* row_q,
                  const std::int32_t* row_symbol_byte_offsets,
                  const std::int32_t* row_auxiliary,
                  const T* x, T* y,
                  mfq::sq::Layout q, int rows, cudaStream_t stream) {
#define MFQ_SQ_M_CASE(M) case M: launch_mmq<M>(blob, row_q, row_symbol_byte_offsets, row_auxiliary, x, y, q, rows, stream); break
    switch (rows) {
        MFQ_SQ_M_CASE(1); MFQ_SQ_M_CASE(2); MFQ_SQ_M_CASE(3);
        MFQ_SQ_M_CASE(4); MFQ_SQ_M_CASE(5); MFQ_SQ_M_CASE(6);
        default: launch_mmq<8>(blob, row_q, row_symbol_byte_offsets, row_auxiliary, x, y, q, rows, stream); break;
    }
#undef MFQ_SQ_M_CASE
}

template<int Rows>
void launch_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((q.width + 127) / 128),
        static_cast<unsigned>((q.outputs + output_tile - 1) / output_tile));
    sq_backward_matrix<Rows><<<
        grid, 128, 0, stream>>>(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        output_gradient, partials, q, rows, output_tile);
}

void dispatch_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        mfq::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    if (rows == 1) {
        launch_backward_matrix<1>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 2) {
        launch_backward_matrix<2>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 4) {
        launch_backward_matrix<4>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else {
        launch_backward_matrix<8>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    }
}

} // namespace

mfq_tensor_backend::Tensor mxfp4_sq_dequant_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor row_auxiliary,
        std::int64_t bits, std::int64_t outputs,
        std::int64_t width, std::int64_t base,
        std::int64_t q_sum, std::int64_t sq4_rows,
        bool fp32) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    const MfqCudaGuard guard(blob.device());
    auto out = mfq_tensor_backend::empty({outputs, width}, blob.options().dtype(
        fp32 ? mfq_tensor_backend::kFloat32 : mfq_tensor_backend::kFloat16));
    const auto count = outputs * width;
    const int blocks = int(std::min<std::int64_t>((count + 255) / 256, 65535));
    const auto* data = blob.data_ptr<std::uint8_t>();
    const auto* q_data = row_q.data_ptr<std::uint8_t>();
    const auto* symbol_offsets = row_symbol_byte_offsets.data_ptr<std::int32_t>();
    const auto* auxiliary = row_auxiliary.data_ptr<std::int32_t>();
    const auto stream = mfq_current_cuda_stream();
#define MFQ_SQ_DEQUANT(T, PTR) sq_dequant<T><<<blocks, 256, 0, stream>>>(data, q_data, symbol_offsets, auxiliary, PTR, q)
    if (fp32) {
        MFQ_SQ_DEQUANT(float, out.data_ptr<float>());
    } else {
        auto* ptr = reinterpret_cast<__half*>(out.data_ptr<mfq_half>());
        MFQ_SQ_DEQUANT(__half, ptr);
    }
#undef MFQ_SQ_DEQUANT
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

mfq_tensor_backend::Tensor mxfp4_sq_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor row_auxiliary,
        mfq_tensor_backend::Tensor input,
        std::int64_t bits, std::int64_t outputs,
        std::int64_t width, std::int64_t base,
        std::int64_t q_sum, std::int64_t sq4_rows) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() && input.size(1) == width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == mfq_tensor_backend::kFloat16 ||
         input.scalar_type() == mfq_tensor_backend::kFloat32),
        "MXFP4-SQ activation requires contiguous rank-2 CUDA FP16/FP32 on the weight device");
    const MfqCudaGuard guard(blob.device());
    auto out = mfq_tensor_backend::empty({input.size(0), outputs}, input.options());
    const int rows = int(input.size(0));
    if (!rows) return out;
    const auto* data = blob.data_ptr<std::uint8_t>();
    const auto* q_data = row_q.data_ptr<std::uint8_t>();
    const auto* symbol_offsets = row_symbol_byte_offsets.data_ptr<std::int32_t>();
    const auto* auxiliary = row_auxiliary.data_ptr<std::int32_t>();
    const auto stream = mfq_current_cuda_stream();
    if (input.scalar_type() == mfq_tensor_backend::kFloat32) {
        dispatch_mmq(
            data, q_data, symbol_offsets, auxiliary,
            input.data_ptr<float>(), out.data_ptr<float>(), q, rows, stream);
    } else {
        const auto* x = reinterpret_cast<const __half*>(input.data_ptr<mfq_half>());
        auto* y = reinterpret_cast<__half*>(out.data_ptr<mfq_half>());
        const char* disable_tensor_core = std::getenv("MFQ_DISABLE_SQ_TENSOR_CORE");
        const char* force_tensor_core = std::getenv("MFQ_FORCE_SQ_TENSOR_CORE");
        // Retained 3090 Ti comparisons show a crossover at M*N=131072 for
        // M>=32; smaller projections lose to the direct packed kernel.
        const bool profitable = rows >= 32 && std::int64_t(rows) * outputs >= 131072;
        if (rows > 8 && (profitable || (force_tensor_core != nullptr && force_tensor_core[0] == '1')) &&
            (disable_tensor_core == nullptr || disable_tensor_core[0] != '1')) {
            mfq::packed::gemm_nt<SqPackedDecoder, true, 32, 64, true>
                <<<dim3((q.outputs + 63) / 64, (rows + 31) / 32), 256, 0, stream>>>(
                    {data, q_data, symbol_offsets, auxiliary, q}, x, y, rows, q.outputs, q.width);
        } else {
            dispatch_mmq(data, q_data, symbol_offsets, auxiliary, x, y, q, rows, stream);
        }
    }
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void mxfp4_sq_moe_matmul_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor row_auxiliary,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        std::int64_t bits,
        std::int64_t n_experts,
        std::int64_t local_experts,
        std::int64_t out_per_expert,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows,
        mfq_tensor_backend::Tensor output) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, local_experts * out_per_expert, width, base,
        q_sum, sq4_rows);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.is_contiguous() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        (input.dim() == 2 || input.dim() == 3) && input.size(-1) == width &&
        expert_ids.is_cuda() && expert_ids.get_device() == blob.get_device() &&
        expert_ids.is_contiguous() && expert_ids.scalar_type() == mfq_tensor_backend::kInt32 &&
        expert_ids.dim() == 2 && expert_ids.size(0) == input.size(0) &&
        expert_local.is_cuda() && expert_local.get_device() == blob.get_device() &&
        expert_local.is_contiguous() && expert_local.scalar_type() == mfq_tensor_backend::kInt32 &&
        expert_local.dim() == 1 && expert_local.numel() == n_experts &&
        output.is_cuda() && output.get_device() == blob.get_device() &&
        output.is_contiguous() && output.scalar_type() == mfq_tensor_backend::kFloat16 &&
        output.dim() == 3 && output.size(0) == expert_ids.size(0) &&
        output.size(1) == expert_ids.size(1) && output.size(2) == out_per_expert &&
        (input.dim() == 2 || input.size(1) == expert_ids.size(1)) &&
        n_experts > 0 && local_experts > 0 && out_per_expert > 0 &&
        n_experts <= std::numeric_limits<int>::max() &&
        local_experts <= std::numeric_limits<int>::max() &&
        out_per_expert <= std::numeric_limits<int>::max(),
        "MXFP4-SQ routed matmul requires matching contiguous CUDA FP16 tensors");
    const auto route_count64 = expert_ids.numel();
    MFQ_RUNTIME_CHECK(
        route_count64 <= std::numeric_limits<int>::max(),
        "MXFP4-SQ route count exceeds CUDA limits");
    if (route_count64 == 0) return;
    const MfqCudaGuard guard(blob.device());
    const auto stream = mfq_current_cuda_stream();
    const auto* data = blob.data_ptr<std::uint8_t>();
    const auto* x = reinterpret_cast<const __half*>(
        input.data_ptr<mfq_half>());
    auto* y = reinterpret_cast<__half*>(output.data_ptr<mfq_half>());
    const auto* ids = expert_ids.data_ptr<std::int32_t>();
    const auto* local = expert_local.data_ptr<std::int32_t>();
    const bool shared_input = input.dim() == 2;
    const int route_count = static_cast<int>(route_count64);
    const int routes = static_cast<int>(expert_ids.size(1));
    launch_routed_mmq(
        data,
        row_q.data_ptr<std::uint8_t>(),
        row_symbol_byte_offsets.data_ptr<std::int32_t>(),
        row_auxiliary.data_ptr<std::int32_t>(),
        x, y, ids, local, q, route_count, routes,
        static_cast<int>(n_experts), static_cast<int>(local_experts),
        static_cast<int>(out_per_expert), shared_input, stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}

mfq_tensor_backend::Tensor mxfp4_sq_backward_input_cuda(
        mfq_tensor_backend::Tensor blob,
        mfq_tensor_backend::Tensor row_q,
        mfq_tensor_backend::Tensor row_symbol_byte_offsets,
        mfq_tensor_backend::Tensor row_auxiliary,
        mfq_tensor_backend::Tensor output_gradient,
        std::int64_t bits,
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == outputs &&
        (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
         output_gradient.scalar_type() == mfq_tensor_backend::kFloat32),
        "MXFP4-SQ backward requires contiguous CUDA FP16/FP32 [M,N]");
    const MfqCudaGuard guard(blob.device());
    auto result = mfq_tensor_backend::empty(
        {output_gradient.size(0), width}, output_gradient.options());
    const auto count = output_gradient.size(0) * width;
    if (!count) return result;
    constexpr int threads = 256;
    const int blocks = int(std::min<std::int64_t>(
        (count + threads - 1) / threads, 65535));
    const auto stream = mfq_current_cuda_stream();
    const auto* data = blob.data_ptr<std::uint8_t>();
    const int rows = int(output_gradient.size(0));
    if (output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 &&
            rows <= 8) {
        const int output_tile = rows <= 4 ? 16 : 32;
        const int splits = (int(outputs) + output_tile - 1) / output_tile;
        auto partials = mfq_tensor_backend::empty(
            {splits, rows, width},
            output_gradient.options().dtype(mfq_tensor_backend::kFloat32));
        const auto* gradient = reinterpret_cast<const __half*>(
            output_gradient.data_ptr<mfq_half>());
        auto* destination = reinterpret_cast<__half*>(
            result.data_ptr<mfq_half>());
        dispatch_backward_matrix(
            data,
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            row_auxiliary.data_ptr<std::int32_t>(),
            gradient, partials.data_ptr<float>(), q, rows,
            output_tile, stream);
        mfq_packed_backward::launch_split_float_reduce_to_half(
            partials.data_ptr<float>(), destination,
            rows, int(width), splits, stream);
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = mxfp4_sq_dequant_cuda(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows, false);
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        rows, int(outputs), int(width), stream);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
