#pragma once
#include "mfq/mxfp4_sq_blob.h"
#include <cmath>

namespace mfq::sq {
// Frozen catalogs shared with the CUDA and Metal wire-format contract.
inline constexpr std::uint8_t kSq1Palette[64] = {
    15, 6, 14, 7, 13, 7, 15, 5, 15, 7, 11, 6, 14, 3, 12, 6,
    14, 6, 14, 4, 10, 6, 14, 2, 13, 6, 14, 5, 9, 6, 14, 1,
    15, 2, 11, 7, 10, 7, 15, 3, 9, 7, 15, 1, 14, 0, 13, 5,
    12, 4, 11, 3, 10, 2, 9, 1, 13, 4, 12, 5, 11, 4, 12, 3,
};
inline constexpr std::uint8_t kSq2Palette[128] = {
    15, 13, 0,  5,  15, 13, 1,  6,  15, 12, 2,  6,  14, 11, 0,  3,  14, 11, 1,
    5,  14, 11, 2,  6,  14, 10, 1,  4,  14, 10, 1,  5,  14, 10, 3,  6,  14, 10,
    4,  7,  14, 9,  5,  7,  13, 10, 1,  4,  13, 9,  3,  6,  13, 0,  5,  7,  12,
    9,  2,  5,  12, 9,  2,  6,  15, 12, 3,  7,  11, 0,  3,  6,  12, 9,  3,  6,
    13, 9,  2,  6,  14, 10, 3,  7,  15, 11, 4,  7,  15, 13, 9,  4,  15, 11, 1,
    5,  15, 11, 2,  6,  15, 12, 1,  6,  15, 12, 2,  7,  15, 13, 0,  6,  14, 11,
    0,  4,  13, 1,  5,  7,  15, 14, 13, 12, 15, 14, 13, 11,
};
inline constexpr std::uint8_t kSq3Palette[256] = {
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
inline unsigned host_bits(const std::uint8_t* bytes, std::size_t index, int bits) {
    const auto bit = index * bits;
    unsigned word = bytes[bit / 8];
    if (bit % 8 + bits > 8) word |= unsigned(bytes[bit / 8 + 1]) << 8;
    return (word >> (bit % 8)) & ((1u << bits) - 1);
}
inline float decode_cpu(const std::uint8_t* blob, const Layout& layout,
                        int bits, int symbol_offset, int auxiliary, int column) {
    const auto* symbols = blob + layout.symbols + symbol_offset;
    const int block = column / 32;
    unsigned code = host_bits(symbols, column, bits), exponent;
    if (bits == 4) exponent = blob[layout.native_scales + std::size_t(auxiliary) * (layout.width / 32) + block];
    else {
        unsigned low = 0;
        for (int i = 0; i < 32; ++i) {
            const auto symbol = host_bits(symbols, block * 32 + i, bits);
            low ^= bits == 1 ? symbol << (i & 1) : symbol & 3;
        }
        const auto selector = std::size_t(auxiliary) * (layout.width / 32) + block;
        const auto state = std::size_t(auxiliary) * 8 + low + (host_bits(blob + layout.selectors, selector, 1) << 2);
        exponent = layout.base + host_bits(blob + layout.scales, state, 2);
        const auto palette = host_bits(blob + layout.palettes, state, 5);
        code = bits == 1 ? kSq1Palette[palette * 2 + code] : bits == 2 ? kSq2Palette[palette * 4 + code] : kSq3Palette[palette * 8 + code];
    }
    constexpr float magnitudes[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
    const float magnitude = (code & 8 ? -1.0f : 1.0f) * magnitudes[code & 7];
    return magnitude * std::ldexp(1.0f, exponent == 0 ? -127 : int(exponent) - 127);
}
// Resolve selector-derived state once per native 32-column block on CPU.
inline void decode_cpu_row(const std::uint8_t* blob, const Layout& layout,
                           int bits, int symbol_offset, int auxiliary, float* output) {
    const auto* symbols = blob + layout.symbols + symbol_offset;
    constexpr float magnitudes[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (std::size_t block = 0; block < layout.width / 32; ++block) {
        unsigned exponent, palette = 0;
        if (bits == 4) exponent = blob[layout.native_scales + std::size_t(auxiliary) * (layout.width / 32) + block];
        else {
            unsigned low = 0;
            for (int i = 0; i < 32; ++i) {
                const auto symbol = host_bits(symbols, block * 32 + i, bits);
                low ^= bits == 1 ? symbol << (i & 1) : symbol & 3;
            }
            const auto selector = std::size_t(auxiliary) * (layout.width / 32) + block;
            const auto state = std::size_t(auxiliary) * 8 + low + (host_bits(blob + layout.selectors, selector, 1) << 2);
            exponent = layout.base + host_bits(blob + layout.scales, state, 2);
            palette = host_bits(blob + layout.palettes, state, 5);
        }
        const float scale = std::ldexp(1.0f, int(exponent) - 127);
        for (int i = 0; i < 32; ++i) {
            auto code = host_bits(symbols, block * 32 + i, bits);
            if (bits < 4) code = bits == 1 ? kSq1Palette[palette * 2 + code] : bits == 2 ? kSq2Palette[palette * 4 + code] : kSq3Palette[palette * 8 + code];
            output[block * 32 + i] = (code & 8 ? -1.0f : 1.0f) * magnitudes[code & 7] * scale;
        }
    }
}
} // namespace mfq::sq
