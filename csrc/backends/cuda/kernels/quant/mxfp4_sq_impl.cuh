#pragma once

// Shared by the MXFP4-SQ inference unit and the tool-only backward unit: the
// frozen palettes, the packed bit reader they feed, and the blob validation.

#include "kernels/quant/mxfp4_sq.h"

#include <cstddef>
#include <cstdint>

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

}  // namespace
