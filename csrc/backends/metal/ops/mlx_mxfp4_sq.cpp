#include "mlx_mxfp4_sq.h"

#include "mfq/mxfp4_sq_blob.h"
#include "mfq_format_compat.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;

// Every q profile shares one decoder.  The loader expands each two-bit q
// descriptor and its variable-width symbol offset once; execution therefore
// needs no profile-specific pipeline.  The immutable palette catalog is one
// small shared input.
constexpr const char* kSqHeader = R"METAL(
inline uint mfq_sq_read_bits(
    device const uchar* stream,
    uint value_index,
    uint bits
) {
    uint bit_offset = value_index * bits;
    uint byte_index = bit_offset >> 3u;
    uint shift = bit_offset & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

inline uint mfq_sq_block_tag(
    device const uchar* row_symbols,
    device const uchar* selectors,
    uint block,
    uint selector_index,
    uint bits
) {
    device const uint* words = (device const uint*)(
        row_symbols + block * bits * 4u);
    uint low = 0u;
    if (bits == 1u) {
        uint word = words[0];
        low = (popcount(word & 0x55555555u) & 1u)
            | ((popcount(word & 0xAAAAAAAAu) & 1u) << 1u);
    } else if (bits == 2u) {
        uint folded = words[0] ^ words[1];
        folded ^= folded >> 16u;
        folded ^= folded >> 8u;
        folded ^= folded >> 4u;
        folded ^= folded >> 2u;
        low = folded & 3u;
    } else {
        constexpr uint MASK0 = 0x49249249u;
        constexpr uint MASK1 = 0x92492492u;
        constexpr uint MASK2 = 0x24924924u;
        uint word0 = words[0];
        uint word1 = words[1];
        uint word2 = words[2];
        uint low_count =
            popcount(word0 & MASK0)
            + popcount(word1 & MASK1)
            + popcount(word2 & MASK2);
        uint high_count =
            popcount(word0 & MASK1)
            + popcount(word1 & MASK2)
            + popcount(word2 & MASK0);
        low = (low_count & 1u) | ((high_count & 1u) << 1u);
    }
    uint explicit_high =
        (uint(selectors[selector_index >> 3u])
            >> (selector_index & 7u)) & 1u;
    return low | (explicit_high << 2u);
}

inline float mfq_sq_scale(uint exponent) {
    uint raw = exponent == 0u ? 0x00400000u : exponent << 23u;
    return as_type<float>(raw);
}

inline float mfq_sq_decode_native(uint nibble, uint exponent) {
    constexpr float MAGNITUDES[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    float magnitude = MAGNITUDES[nibble & 7u];
    if ((nibble & 8u) != 0u) {
        magnitude = -magnitude;
    }
    return magnitude * mfq_sq_scale(exponent);
}

inline float mfq_sq_decode(
    device const uchar* palette_catalog,
    uint palette,
    uint symbol,
    uint exponent,
    uint bits
) {
    uint catalog_offset = bits == 1u ? 0u : (bits == 2u ? 64u : 192u);
    uint nibble = uint(palette_catalog[
        catalog_offset + palette * (1u << bits) + symbol]);
    return mfq_sq_decode_native(nibble, exponent);
}
)METAL";

constexpr const char* kSqDequantize = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint block_index = thread_position_in_grid.x >> 5u;
    if (block_index >= uint(TOTAL_BLOCKS)) {
        return;
    }
    device const uchar* symbols = blob + uint(SYMBOLS_OFFSET);
    device const uchar* selectors = blob + uint(SELECTORS_OFFSET);
    device const uchar* state_scales = blob + uint(SCALES_OFFSET);
    device const uchar* state_palettes = blob + uint(PALETTES_OFFSET);
    uint output = block_index / uint(BLOCKS);
    uint block = block_index - output * uint(BLOCKS);
    uint bits = uint(row_q[output]);
    device const uchar* row_symbols =
        symbols + row_symbol_byte_offsets[output];
    uint auxiliary = row_auxiliary[output];
    uint state_value = 0u;
    if (lane == 0u) {
        if (bits == 4u) {
            state_value = uint(
                blob[uint(NATIVE_SCALES_OFFSET)
                    + auxiliary * uint(BLOCKS) + block]);
        } else {
            uint selector_index = auxiliary * uint(BLOCKS) + block;
            uint state = auxiliary * 8u + mfq_sq_block_tag(
                row_symbols, selectors, block, selector_index, bits);
            state_value =
                (mfq_sq_read_bits(state_palettes, state, 5u) << 8u)
                | (uint(MATRIX_SCALE_BASE)
                    + mfq_sq_read_bits(state_scales, state, 2u));
        }
    }
    state_value = simd_broadcast_first(state_value);
    uint symbol = mfq_sq_read_bits(
        row_symbols, block * 32u + lane, bits);
    float decoded = bits == 4u
        ? mfq_sq_decode_native(symbol, state_value)
        : mfq_sq_decode(
            palette, state_value >> 8u, symbol, state_value & 255u, bits);
    y[block_index * 32u + lane] = T(decoded);
)METAL";

// One metadata-driven packed matrix kernel for every SQ profile and small M.
// A lane subgroup owns one output and whole block-32 weight vectors.  Decoded
// weights are reused across TILE_M input rows; the eight row states are cached
// once per output rather than reread for every block.
constexpr const char* kSqMatmul = R"METAL(
    constexpr uint K_LANES_VALUE = uint(K_LANES);
    constexpr uint SIMD_GROUPS_VALUE = uint(SIMD_GROUPS);
    constexpr uint THREADS = SIMD_GROUPS_VALUE * 32u;
    constexpr uint OUTPUTS_PER_SIMD = 32u / K_LANES_VALUE;
    constexpr uint OUTPUTS_PER_TG =
        SIMD_GROUPS_VALUE * OUTPUTS_PER_SIMD;

    threadgroup float row_scales[OUTPUTS_PER_TG * 8u];
    threadgroup uchar row_palettes[OUTPUTS_PER_TG * 8u];

    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES_VALUE - 1u);
    uint simd_output = lane / K_LANES_VALUE;
    uint output_slot =
        simd_group * OUTPUTS_PER_SIMD + simd_output;
    uint first_row =
        threadgroup_position_in_grid.x * uint(TILE_M);
    uint output_index =
        threadgroup_position_in_grid.y * OUTPUTS_PER_TG + output_slot;
    int local_expert = 0;
    bool route_valid = true;
    if (uint(ROUTED) != 0u) {
        int expert = expert_ids[first_row];
        route_valid = expert >= 0 && expert < int(EXPERT_MAP_SIZE);
        local_expert = route_valid ? expert_map[expert] : -1;
        route_valid = route_valid && local_expert >= 0;
    }
    uint output = uint(ROUTED) != 0u
        ? (route_valid
            ? uint(local_expert) * uint(OUT_PER_EXPERT) + output_index
            : 0u)
        : output_index;
    output = min(output, uint(OUT) - 1u);

    device const uchar* symbols = blob + uint(SYMBOLS_OFFSET);
    device const uchar* selectors = blob + uint(SELECTORS_OFFSET);
    device const uchar* state_scales = blob + uint(SCALES_OFFSET);
    device const uchar* state_palettes = blob + uint(PALETTES_OFFSET);

    for (uint metadata = local_thread;
         metadata < OUTPUTS_PER_TG * 8u;
         metadata += THREADS) {
        uint local_output = metadata >> 3u;
        uint state = metadata & 7u;
        uint row_index =
            threadgroup_position_in_grid.y * OUTPUTS_PER_TG
            + local_output;
        uint row = uint(ROUTED) != 0u
            ? (route_valid
                ? uint(local_expert) * uint(OUT_PER_EXPERT) + row_index
                : 0u)
            : row_index;
        row = min(row, uint(OUT) - 1u);
        uint bits = uint(row_q[row]);
        uint auxiliary = row_auxiliary[row];
        if (bits == 4u) {
            row_scales[metadata] = 1.0f;
            row_palettes[metadata] = 0;
        } else {
            uint state_index = auxiliary * 8u + state;
            row_scales[metadata] = mfq_sq_scale(
                uint(MATRIX_SCALE_BASE)
                + mfq_sq_read_bits(state_scales, state_index, 2u));
            row_palettes[metadata] = uchar(
                mfq_sq_read_bits(state_palettes, state_index, 5u));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulators[TILE_M];
    for (uint local_row = 0u;
         local_row < uint(TILE_M);
         ++local_row) {
        accumulators[local_row] = 0.0f;
    }

    for (uint block = k_lane;
         block < uint(BLOCKS);
         block += K_LANES_VALUE) {
        uint bits = uint(row_q[output]);
        uint auxiliary = row_auxiliary[output];
        device const uchar* row_symbols =
            symbols + row_symbol_byte_offsets[output];
        uint tag = bits == 4u ? 0u : mfq_sq_block_tag(
            row_symbols,
            selectors,
            block,
            auxiliary * uint(BLOCKS) + block,
            bits);
        uint cached_state = output_slot * 8u + tag;
        float scale = bits == 4u
            ? mfq_sq_scale(uint(blob[uint(NATIVE_SCALES_OFFSET)
                + auxiliary * uint(BLOCKS) + block]))
            : row_scales[cached_state];
        uint palette_index = uint(row_palettes[cached_state]);
        uint column_base = block * 32u;

        if (bits == 1u) {
            for (uint packed_index = 0u;
                 packed_index < 4u;
                 ++packed_index) {
                uint column = column_base + packed_index * 8u;
                uint packed = uint(row_symbols[block * 4u + packed_index]);
                float4 weight0 = float4(
                    mfq_sq_decode(
                        palette, palette_index, packed & 1u, 127u, 1u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 1u) & 1u, 127u, 1u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 2u) & 1u, 127u, 1u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 3u) & 1u, 127u, 1u))
                    * scale;
                float4 weight1 = float4(
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 4u) & 1u, 127u, 1u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 5u) & 1u, 127u, 1u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 6u) & 1u, 127u, 1u),
                    mfq_sq_decode(
                        palette, palette_index, packed >> 7u, 127u, 1u))
                    * scale;
                for (uint local_row = 0u;
                     local_row < uint(TILE_M);
                     ++local_row) {
                    uint row = min(first_row + local_row, uint(M) - 1u);
                    uint input_row = uint(ROUTED) != 0u
                        && uint(SHARED_INPUT) != 0u
                        ? row / uint(ROUTES)
                        : row;
                    half4 activation0 = *(device const half4*)(
                        x + input_row * uint(K) + column);
                    half4 activation1 = *(device const half4*)(
                        x + input_row * uint(K) + column + 4u);
                    accumulators[local_row] +=
                        dot(float4(activation0), weight0)
                        + dot(float4(activation1), weight1);
                }
            }
        } else if (bits == 2u) {
            for (uint packed_index = 0u;
                 packed_index < 8u;
                 ++packed_index) {
                uint column = column_base + packed_index * 4u;
                uint packed = uint(
                    row_symbols[block * 8u + packed_index]);
                float4 weight = float4(
                    mfq_sq_decode(
                        palette, palette_index, packed & 3u, 127u, 2u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 2u) & 3u, 127u, 2u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 4u) & 3u, 127u, 2u),
                    mfq_sq_decode(
                        palette, palette_index, packed >> 6u, 127u, 2u)) * scale;
                for (uint local_row = 0u;
                     local_row < uint(TILE_M);
                     ++local_row) {
                    uint row = min(
                        first_row + local_row,
                        uint(M) - 1u);
                    uint input_row = uint(ROUTED) != 0u
                        && uint(SHARED_INPUT) != 0u
                        ? row / uint(ROUTES)
                        : row;
                    half4 activation = *(device const half4*)(
                        x + input_row * uint(K) + column);
                    accumulators[local_row] += dot(
                        float4(activation), weight);
                }
            }
        } else if (bits == 3u) {
            for (uint group = 0u; group < 4u; ++group) {
                uint column = column_base + group * 8u;
                uint byte_offset = block * 12u + group * 3u;
                uint packed = uint(row_symbols[byte_offset])
                    | (uint(row_symbols[byte_offset + 1u]) << 8u)
                    | (uint(row_symbols[byte_offset + 2u]) << 16u);
                float4 weight0 = float4(
                    mfq_sq_decode(
                        palette, palette_index, packed & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 3u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 6u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 9u) & 7u, 127u, 3u))
                    * scale;
                float4 weight1 = float4(
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 12u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 15u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 18u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 21u) & 7u, 127u, 3u))
                    * scale;
                for (uint local_row = 0u;
                     local_row < uint(TILE_M);
                     ++local_row) {
                    uint row = min(
                        first_row + local_row,
                        uint(M) - 1u);
                    uint input_row = uint(ROUTED) != 0u
                        && uint(SHARED_INPUT) != 0u
                        ? row / uint(ROUTES)
                        : row;
                    half4 activation0 = *(device const half4*)(
                        x + input_row * uint(K) + column);
                    half4 activation1 = *(device const half4*)(
                        x + input_row * uint(K) + column + 4u);
                    accumulators[local_row] +=
                        dot(float4(activation0), weight0)
                        + dot(float4(activation1), weight1);
                }
            }
        } else {
            for (uint group = 0u; group < 8u; ++group) {
                uint column = column_base + group * 4u;
                uint byte_offset = block * 16u + group * 2u;
                uint packed = uint(row_symbols[byte_offset])
                    | (uint(row_symbols[byte_offset + 1u]) << 8u);
                float4 weight = float4(
                    mfq_sq_decode_native(packed & 15u, 127u),
                    mfq_sq_decode_native((packed >> 4u) & 15u, 127u),
                    mfq_sq_decode_native((packed >> 8u) & 15u, 127u),
                    mfq_sq_decode_native(packed >> 12u, 127u)) * scale;
                for (uint local_row = 0u;
                     local_row < uint(TILE_M);
                     ++local_row) {
                    uint row = min(first_row + local_row, uint(M) - 1u);
                    uint input_row = uint(ROUTED) != 0u
                        && uint(SHARED_INPUT) != 0u
                        ? row / uint(ROUTES)
                        : row;
                    half4 activation = *(device const half4*)(
                        x + input_row * uint(K) + column);
                    accumulators[local_row] += dot(
                        float4(activation), weight);
                }
            }
        }
    }

    for (uint local_row = 0u;
         local_row < uint(TILE_M);
         ++local_row) {
        if (K_LANES_VALUE >= 32u) {
            accumulators[local_row] += simd_shuffle_down(
                accumulators[local_row], 16u);
        }
        if (K_LANES_VALUE >= 16u) {
            accumulators[local_row] += simd_shuffle_down(
                accumulators[local_row], 8u);
        }
        if (K_LANES_VALUE >= 8u) {
            accumulators[local_row] += simd_shuffle_down(
                accumulators[local_row], 4u);
        }
        accumulators[local_row] += simd_shuffle_down(
            accumulators[local_row], 2u);
        accumulators[local_row] += simd_shuffle_down(
            accumulators[local_row], 1u);
        uint row = first_row + local_row;
        if (k_lane == 0u && row < uint(M)
            && output_index < uint(LOGICAL_OUT)) {
            y[row * uint(LOGICAL_OUT) + output_index] = T(
                route_valid ? accumulators[local_row] : 0.0f);
        }
    }
)METAL";

// Packed input-gradient kernel shared by every per-neuron SQ1..SQ4 mixture.
// It stages one [64,32] weight tile directly from the variable-width stream
// and multiplies an eight-row cotangent tile without materializing the full
// dequantized matrix.
constexpr const char* kSqBackwardMatrix = R"METAL(
    constexpr uint BM = 8u;
    constexpr uint BN = 64u;
    constexpr uint BK = 32u;
    constexpr uint BN_PAD = BN + 8u;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint row_base = threadgroup_position_in_grid.y * BM;
    uint block = threadgroup_position_in_grid.x;
    uint column_base = block * BK;

    device const uchar* symbols = blob + uint(SYMBOLS_OFFSET);
    device const uchar* selectors = blob + uint(SELECTORS_OFFSET);
    device const uchar* state_scales = blob + uint(SCALES_OFFSET);
    device const uchar* state_palettes = blob + uint(PALETTES_OFFSET);

    threadgroup half gradient_tile[BM * BN_PAD];
    threadgroup half weight_tile[BN * BK];
    threadgroup float cached_scales[BN];
    threadgroup uchar cached_palettes[BN];
    threadgroup uchar cached_bits[BN];
    threadgroup uint cached_symbol_offsets[BN];

    metal::simdgroup_matrix<float, 8, 8> result;
    result.thread_elements()[0] = 0.0f;
    result.thread_elements()[1] = 0.0f;
    uint quadrant = lane / 4u;
    uint fragment_row = (quadrant & 4u) + ((lane / 2u) & 3u);
    uint fragment_col = (quadrant & 2u) * 2u + (lane & 1u) * 2u;
    uint simd_col = simd_group * 8u;

    for (uint chunk = 0u;
         chunk < (uint(OUT) + BN - 1u) / BN;
         ++chunk) {
        uint output_base = chunk * BN;
        for (uint index = local_thread;
             index < BM * BN;
             index += 128u) {
            uint local_row = index / BN;
            uint local_output = index - local_row * BN;
            uint row = row_base + local_row;
            uint output = output_base + local_output;
            gradient_tile[local_row * BN_PAD + local_output] =
                row < uint(M) && output < uint(OUT)
                ? half(x[row * uint(OUT) + output])
                : half(0.0f);
        }

        if (local_thread < BN) {
            uint output = output_base + local_thread;
            if (output < uint(OUT) && block < uint(BLOCKS)) {
                uint bits = uint(row_q[output]);
                uint auxiliary = row_auxiliary[output];
                uint symbol_offset = row_symbol_byte_offsets[output];
                device const uchar* row_symbols = symbols + symbol_offset;
                cached_bits[local_thread] = uchar(bits);
                cached_symbol_offsets[local_thread] = symbol_offset;
                if (bits == 4u) {
                    cached_scales[local_thread] = mfq_sq_scale(uint(
                        blob[uint(NATIVE_SCALES_OFFSET)
                            + auxiliary * uint(BLOCKS) + block]));
                    cached_palettes[local_thread] = 0u;
                } else {
                    uint selector_index =
                        auxiliary * uint(BLOCKS) + block;
                    uint state = auxiliary * 8u + mfq_sq_block_tag(
                        row_symbols,
                        selectors,
                        block,
                        selector_index,
                        bits);
                    cached_scales[local_thread] = mfq_sq_scale(
                        uint(MATRIX_SCALE_BASE)
                        + mfq_sq_read_bits(state_scales, state, 2u));
                    cached_palettes[local_thread] = uchar(
                        mfq_sq_read_bits(state_palettes, state, 5u));
                }
            } else {
                cached_scales[local_thread] = 0.0f;
                cached_palettes[local_thread] = 0u;
                cached_bits[local_thread] = 1u;
                cached_symbol_offsets[local_thread] = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint task = local_thread;
             task < BN * 8u;
             task += 128u) {
            uint local_output = task >> 3u;
            uint vector = task & 7u;
            uint output = output_base + local_output;
            uint column = vector * 4u;
            half4 decoded = half4(0.0h);
            if (output < uint(OUT) && block < uint(BLOCKS)) {
                uint bits = uint(cached_bits[local_output]);
                uint palette_index = uint(cached_palettes[local_output]);
                float scale = cached_scales[local_output];
                device const uchar* row_symbols =
                    symbols + cached_symbol_offsets[local_output];
                float4 values;
#pragma unroll
                for (uint element = 0u; element < 4u; ++element) {
                    uint symbol = mfq_sq_read_bits(
                        row_symbols,
                        block * 32u + column + element,
                        bits);
                    values[element] = (bits == 4u
                        ? mfq_sq_decode_native(symbol, 127u)
                        : mfq_sq_decode(
                            palette,
                            palette_index,
                            symbol,
                            127u,
                            bits)) * scale;
                }
                decoded = half4(values);
            }
            *(threadgroup half4*)(
                weight_tile + local_output * BK + column) = decoded;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BN; kk += 8u) {
            metal::simdgroup_matrix<half, 8, 8> a;
            metal::simdgroup_matrix<half, 8, 8> b;
            a.thread_elements()[0] = gradient_tile[
                fragment_row * BN_PAD + kk + fragment_col];
            a.thread_elements()[1] = gradient_tile[
                fragment_row * BN_PAD + kk + fragment_col + 1u];
            b.thread_elements()[0] = weight_tile[
                (kk + fragment_row) * BK + simd_col + fragment_col];
            b.thread_elements()[1] = weight_tile[
                (kk + fragment_row) * BK + simd_col + fragment_col + 1u];
            simdgroup_multiply_accumulate(result, a, b, result);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint row = row_base + fragment_row;
    uint column = column_base + simd_col + fragment_col;
    if (row < uint(M) && column < uint(K)) {
        y[row * uint(K) + column] = T(result.thread_elements()[0]);
    }
    if (row < uint(M) && column + 1u < uint(K)) {
        y[row * uint(K) + column + 1u] =
            T(result.thread_elements()[1]);
    }
)METAL";

const array& palette_catalog() {
    static const auto catalog = [] {
        std::vector<std::uint8_t> values;
        values.reserve(
            kMxfp4Sq1PaletteNibbles.size() +
            kMxfp4Sq2PaletteNibbles.size() +
            kMxfp4Sq3PaletteNibbles.size());
        values.insert(
            values.end(),
            kMxfp4Sq1PaletteNibbles.begin(),
            kMxfp4Sq1PaletteNibbles.end());
        values.insert(
            values.end(),
            kMxfp4Sq2PaletteNibbles.begin(),
            kMxfp4Sq2PaletteNibbles.end());
        values.insert(
            values.end(),
            kMxfp4Sq3PaletteNibbles.begin(),
            kMxfp4Sq3PaletteNibbles.end());
        return array(values.begin(), Shape{static_cast<int>(values.size())});
    }();
    return catalog;
}

const mlx::core::fast::CustomKernelFunction& dequantize_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_mxfp4_sq_dequantize",
            {
                "blob",
                "palette",
                "row_q",
                "row_symbol_byte_offsets",
                "row_auxiliary",
            },
            {"y"},
            kSqDequantize,
            kSqHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& matmul_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_mxfp4_sq_matmul",
            {
                "blob",
                "palette",
                "row_q",
                "row_symbol_byte_offsets",
                "row_auxiliary",
                "x",
                "expert_ids",
                "expert_map",
            },
            {"y"},
            kSqMatmul,
            kSqHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& backward_matrix_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_mxfp4_sq_backward_matrix",
            {
                "blob",
                "palette",
                "row_q",
                "row_symbol_byte_offsets",
                "row_auxiliary",
                "x",
            },
            {"y"},
            kSqBackwardMatrix,
            kSqHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

bool identifier_character(char value) noexcept {
    return std::isalnum(static_cast<unsigned char>(value)) != 0 ||
        value == '_';
}

void replace_identifier(
    std::string& source,
    std::string_view from,
    std::string_view to) {
    std::size_t position = 0;
    while ((position = source.find(from, position)) != std::string::npos) {
        const bool left_boundary = position == 0 ||
            !identifier_character(source[position - 1]);
        const auto end = position + from.size();
        const bool right_boundary = end == source.size() ||
            !identifier_character(source[end]);
        if (left_boundary && right_boundary) {
            source.replace(position, from.size(), to);
            position += to.size();
        } else {
            position = end;
        }
    }
}

void replace_all(
    std::string& source,
    std::string_view from,
    std::string_view to) {
    std::size_t position = 0;
    while ((position = source.find(from, position)) != std::string::npos) {
        source.replace(position, from.size(), to);
        position += to.size();
    }
}

std::vector<std::string> projection_group_input_names(
    std::size_t projections) {
    std::vector<std::string> names;
    names.reserve(projections * 4 + 4);
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        names.push_back("blob_" + suffix);
        names.push_back("row_q_" + suffix);
        names.push_back("row_symbol_byte_offsets_" + suffix);
        names.push_back("row_auxiliary_" + suffix);
    }
    names.emplace_back("palette");
    names.emplace_back("x");
    names.emplace_back("expert_ids");
    names.emplace_back("expert_map");
    return names;
}

std::string make_projection_group_source(std::size_t projections) {
    std::string source;
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        const auto prefix = "P" + suffix + "_";
        source += projection == 0 ? "    if (" : "    else if (";
        source += "threadgroup_position_in_grid.y < uint(" + prefix
            + "TILE_END)) {\n";

        std::string body(kSqMatmul);
        replace_all(
            body,
            "threadgroup_position_in_grid.y",
            "(threadgroup_position_in_grid.y - uint(" + prefix
                + "TILE_BEGIN))");
        for (const auto name : {
                 "MATRIX_SCALE_BASE",
                 "SYMBOLS_OFFSET",
                 "SELECTORS_OFFSET",
                 "SCALES_OFFSET",
                 "PALETTES_OFFSET",
                 "NATIVE_SCALES_OFFSET",
                 "OUT_PER_EXPERT",
                 "LOGICAL_OUT",
                 "BLOCKS",
                 "OUT",
             }) {
            replace_identifier(body, name, prefix + name);
        }
        replace_identifier(body, "blob", "blob_" + suffix);
        replace_identifier(body, "row_q", "row_q_" + suffix);
        replace_identifier(
            body,
            "row_symbol_byte_offsets",
            "row_symbol_byte_offsets_" + suffix);
        replace_identifier(
            body,
            "row_auxiliary",
            "row_auxiliary_" + suffix);
        replace_all(
            body,
            "y[row * uint(" + prefix
                + "LOGICAL_OUT) + output_index]",
            "y[row * uint(TOTAL_OUT) + uint(" + prefix
                + "OUT_OFFSET) + output_index]");
        source += body;
        source += "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction projection_group_kernel(
    std::size_t projections) {
    static std::mutex mutex;
    static std::unordered_map<
        std::size_t,
        mlx::core::fast::CustomKernelFunction> kernels;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(projections); found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_mxfp4_sq_projection_group_p"
            + std::to_string(projections),
        projection_group_input_names(projections),
        {"y"},
        make_projection_group_source(projections),
        kSqHeader,
        true,
        false,
        options);
    kernels.emplace(projections, kernel);
    return kernel;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
templates(const MlxMxfp4SqWeight& weight, Dtype dtype) {
    const auto& layout = weight.wire_layout();
    const auto checked_int = [](std::size_t value, const char* name) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                std::string("MXFP4-SQ ") + name + " exceeds MLX limits");
        }
        return static_cast<int>(value);
    };
    return {
        {"T", dtype},
        {"MATRIX_SCALE_BASE", layout.base},
        {"K", weight.input_size()},
        {"OUT", weight.output_size()},
        {"BLOCKS", weight.input_size() / 32},
        {"TOTAL_BLOCKS", checked_int(
            static_cast<std::size_t>(weight.input_size() / 32) *
                static_cast<std::size_t>(weight.output_size()),
            "block count")},
        {"WEIGHTS", checked_int(
            static_cast<std::size_t>(weight.input_size()) *
                static_cast<std::size_t>(weight.output_size()),
            "weight count")},
        {"SYMBOLS_OFFSET", checked_int(layout.symbols, "symbol offset")},
        {"SELECTORS_OFFSET", checked_int(
            layout.selectors, "selector offset")},
        {"SCALES_OFFSET", checked_int(layout.scales, "scale offset")},
        {"PALETTES_OFFSET", checked_int(
            layout.palettes, "palette offset")},
        {"NATIVE_SCALES_OFFSET", checked_int(
            layout.native_scales, "native-scale offset")},
    };
}

} // namespace

bool is_mxfp4_sq_dtype(std::string_view dtype) noexcept {
    return mfq::canonical_format_dtype(dtype) == mfq::kMxfp4SqDtype;
}

MlxMxfp4SqWeight::MlxMxfp4SqWeight(
    array blob,
    array row_q,
    array row_symbol_byte_offsets,
    array row_auxiliary,
    mfq::sq::Layout layout,
    Mxfp4SqDescriptor descriptor)
    : blob_(std::move(blob)),
      row_q_(std::move(row_q)),
      row_symbol_byte_offsets_(std::move(row_symbol_byte_offsets)),
      row_auxiliary_(std::move(row_auxiliary)),
      layout_(std::move(layout)),
      descriptor_(descriptor) {}

MlxMxfp4SqWeight MlxMxfp4SqWeight::from_blob(
    const std::vector<std::uint8_t>& blob) {
    return from_blob(std::span<const std::uint8_t>(blob));
}

MlxMxfp4SqWeight MlxMxfp4SqWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    const auto layout = mfq::sq::parse(blob.data(), blob.size());
    if (blob.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("MXFP4-SQ payload exceeds MLX limits");
    }
    const auto rows = mfq::sq::row_metadata(blob.data(), layout);
    Mxfp4SqDescriptor descriptor;
    descriptor.format_version = layout.version;
    descriptor.aggregate_bpw = static_cast<double>(
        layout.bytes - 24) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<std::size_t, 4> q_counts{};
    for (const auto q : rows.q) {
        ++q_counts[static_cast<std::size_t>(q - 1)];
    }
    for (const auto count : q_counts) {
        if (count == 0) {
            continue;
        }
        const double probability = static_cast<double>(count) / layout.outputs;
        descriptor.distribution_entropy -= probability * std::log2(probability);
    }
    return MlxMxfp4SqWeight(
        array(blob.begin(), Shape{static_cast<int>(blob.size())}),
        array(rows.q.begin(), Shape{layout.outputs}),
        array(rows.symbol_byte_offsets.begin(), Shape{layout.outputs}),
        array(rows.auxiliary_rows.begin(), Shape{layout.outputs}),
        layout,
        descriptor);
}

array MlxMxfp4SqWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error(
            "MXFP4-SQ dequantization requires float16 or float32");
    }
    const auto count =
        static_cast<std::size_t>(input_size()) *
        static_cast<std::size_t>(output_size());
    if (count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "MXFP4-SQ dequantization grid exceeds MLX limits");
    }
    constexpr std::size_t threads = 256;
    const auto grid = (count + threads - 1) / threads * threads;
    auto outputs = dequantize_kernel()(
        {
            blob_,
            palette_catalog(),
            row_q_,
            row_symbol_byte_offsets_,
            row_auxiliary_,
        },
        {Shape{output_size(), input_size()}},
        {dtype},
        {static_cast<int>(grid), 1, 1},
        {static_cast<int>(threads), 1, 1},
        templates(*this, dtype),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxMxfp4SqWeight::embedding(
    const array& rows,
    Dtype dtype) const {
    return mlx::core::take(dequantize(dtype), rows, 0);
}

array MlxMxfp4SqWeight::matmul(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size()) {
        throw std::runtime_error(
            "MXFP4-SQ input width does not match packed weight");
    }
    const auto rows =
        input.size() / static_cast<std::size_t>(input_size());
    if (rows == 0 || rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("unsupported MXFP4-SQ input row count");
    }
    Shape output_shape = input.shape();
    output_shape.back() = output_size();
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<int>(rows), input_size()});

    if (rows > 48 || source.dtype() == mlx::core::float32) {
        auto dense = dequantize(source.dtype());
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        return mlx::core::reshape(
            std::move(result),
            std::move(output_shape));
    }

    const bool first_bucket = rows <= 6;
    // A uniform profile can pack several output neurons into one SIMD group.
    // With heterogeneous q, using the whole SIMD for one neuron prevents the
    // q=1/2/3/4 decode arms from serializing each other lane-wise.
    const bool mixed_q = !has_uniform_q();
    const int k_lanes = mixed_q
        ? 32
        : (first_bucket && rows <= 3 ? 16 : 8);
    const int simd_groups = mixed_q
        ? 4
        : (first_bucket && rows <= 3 ? 4 : 2);
    const int threads = simd_groups * 32;
    const int row_tiles = first_bucket
        ? 1
        : (static_cast<int>(rows) + 7) / 8;
    const int tile_rows =
        (static_cast<int>(rows) + row_tiles - 1) / row_tiles;
    const int outputs_per_threadgroup =
        simd_groups * 32 / k_lanes;
    const int output_tiles =
        (output_size() + outputs_per_threadgroup - 1) /
        outputs_per_threadgroup;
    auto arguments = templates(*this, source.dtype());
    arguments.emplace_back("M", static_cast<int>(rows));
    arguments.emplace_back("TILE_M", tile_rows);
    arguments.emplace_back("K_LANES", k_lanes);
    arguments.emplace_back("SIMD_GROUPS", simd_groups);
    arguments.emplace_back("ROUTED", 0);
    arguments.emplace_back("ROUTES", 1);
    arguments.emplace_back("SHARED_INPUT", 0);
    arguments.emplace_back("EXPERT_MAP_SIZE", 1);
    arguments.emplace_back("OUT_PER_EXPERT", output_size());
    arguments.emplace_back("LOGICAL_OUT", output_size());
    const auto unused_ids = mlx::core::zeros(
        Shape{1}, mlx::core::int32);
    auto outputs = matmul_kernel()(
        {
            blob_,
            palette_catalog(),
            row_q_,
            row_symbol_byte_offsets_,
            row_auxiliary_,
            std::move(source),
            unused_ids,
            unused_ids,
        },
        {Shape{static_cast<int>(rows), output_size()}},
        {input.dtype() == mlx::core::float32
            ? mlx::core::float32
            : mlx::core::float16},
        {row_tiles * threads, output_tiles, 1},
        {threads, 1, 1},
        std::move(arguments),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

std::vector<array> MlxMxfp4SqWeight::projection_group_matmul(
    std::span<const MlxMxfp4SqWeight> weights,
    const array& input) {
    if (weights.size() < 2 || weights.size() > 5) {
        throw std::invalid_argument(
            "MXFP4-SQ projection group requires two through five weights");
    }
    const int input_size = weights.front().input_size();
    if (input.ndim() == 0 || input.shape(-1) != input_size) {
        throw std::invalid_argument(
            "MXFP4-SQ projection group input width mismatch");
    }
    if (input.dtype() != mlx::core::float16 &&
        input.dtype() != mlx::core::float32) {
        throw std::invalid_argument(
            "MXFP4-SQ projection group requires FP16 or FP32 input");
    }
    for (const auto& weight : weights) {
        if (weight.input_size() != input_size) {
            throw std::invalid_argument(
                "MXFP4-SQ projection group requires one input width");
        }
    }
    if (input.dtype() == mlx::core::float32) {
        std::vector<array> outputs;
        outputs.reserve(weights.size());
        for (const auto& weight : weights) {
            outputs.push_back(weight.matmul(input));
        }
        return outputs;
    }

    const auto checked_int = [](std::size_t value, const char* name) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                std::string("MXFP4-SQ ") + name + " exceeds MLX limits");
        }
        return static_cast<int>(value);
    };
    const auto rows = input.size() / static_cast<std::size_t>(input_size);
    if (rows < 1 || rows > 16) {
        throw std::invalid_argument(
            "MXFP4-SQ projection group supports one through sixteen rows");
    }

    std::vector<int> output_sizes;
    output_sizes.reserve(weights.size());
    int total_output = 0;
    bool mixed_q = false;
    for (const auto& weight : weights) {
        total_output = checked_int(
            static_cast<std::size_t>(total_output) +
                static_cast<std::size_t>(weight.output_size()),
            "projection-group output width");
        output_sizes.push_back(weight.output_size());
        mixed_q = mixed_q || !weight.has_uniform_q();
    }

    const bool first_bucket = rows <= 6;
    const int k_lanes = mixed_q
        ? 32
        : (first_bucket && rows <= 3 ? 16 : 8);
    const int simd_groups = mixed_q
        ? 4
        : (first_bucket && rows <= 3 ? 4 : 2);
    const int threads = simd_groups * 32;
    const int row_tiles = first_bucket
        ? 1
        : (static_cast<int>(rows) + 7) / 8;
    const int tile_rows =
        (static_cast<int>(rows) + row_tiles - 1) / row_tiles;
    const int outputs_per_threadgroup =
        simd_groups * 32 / k_lanes;

    Shape prefix(input.shape().begin(), input.shape().end() - 1);
    auto source = mlx::core::contiguous(mlx::core::reshape(
        input,
        Shape{checked_int(rows, "projection-group row count"), input_size}));
    std::vector<array> inputs;
    inputs.reserve(weights.size() * 4 + 4);
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> arguments{
        {"T", source.dtype()},
        {"M", checked_int(rows, "projection-group row count")},
        {"TILE_M", tile_rows},
        {"K", input_size},
        {"K_LANES", k_lanes},
        {"SIMD_GROUPS", simd_groups},
        {"ROUTED", 0},
        {"ROUTES", 1},
        {"SHARED_INPUT", 0},
        {"EXPERT_MAP_SIZE", 1},
        {"TOTAL_OUT", total_output},
    };
    int output_offset = 0;
    int tile_offset = 0;
    for (std::size_t projection = 0;
         projection < weights.size();
         ++projection) {
        const auto& weight = weights[projection];
        const auto& layout = weight.layout_;
        const auto prefix_name = "P" + std::to_string(projection) + "_";
        inputs.push_back(weight.blob_);
        inputs.push_back(weight.row_q_);
        inputs.push_back(weight.row_symbol_byte_offsets_);
        inputs.push_back(weight.row_auxiliary_);
        arguments.emplace_back(prefix_name + "MATRIX_SCALE_BASE", layout.base);
        arguments.emplace_back(prefix_name + "OUT", weight.output_size());
        arguments.emplace_back(prefix_name + "BLOCKS", input_size / 32);
        arguments.emplace_back(
            prefix_name + "SYMBOLS_OFFSET",
            checked_int(layout.symbols, "symbol offset"));
        arguments.emplace_back(
            prefix_name + "SELECTORS_OFFSET",
            checked_int(layout.selectors, "selector offset"));
        arguments.emplace_back(
            prefix_name + "SCALES_OFFSET",
            checked_int(layout.scales, "scale offset"));
        arguments.emplace_back(
            prefix_name + "PALETTES_OFFSET",
            checked_int(layout.palettes, "palette offset"));
        arguments.emplace_back(
            prefix_name + "NATIVE_SCALES_OFFSET",
            checked_int(layout.native_scales, "native-scale offset"));
        arguments.emplace_back(
            prefix_name + "OUT_PER_EXPERT", weight.output_size());
        arguments.emplace_back(
            prefix_name + "LOGICAL_OUT", weight.output_size());
        arguments.emplace_back(prefix_name + "OUT_OFFSET", output_offset);
        arguments.emplace_back(prefix_name + "TILE_BEGIN", tile_offset);
        output_offset += weight.output_size();
        tile_offset +=
            (weight.output_size() + outputs_per_threadgroup - 1) /
            outputs_per_threadgroup;
        arguments.emplace_back(prefix_name + "TILE_END", tile_offset);
    }
    const auto unused = mlx::core::zeros(Shape{1}, mlx::core::int32);
    inputs.push_back(palette_catalog());
    inputs.push_back(source);
    inputs.push_back(unused);
    inputs.push_back(unused);
    auto combined = projection_group_kernel(weights.size())(
        std::move(inputs),
        {Shape{checked_int(rows, "projection-group row count"), total_output}},
        {source.dtype()},
        {row_tiles * threads, tile_offset, 1},
        {threads, 1, 1},
        std::move(arguments),
        std::nullopt,
        false,
        {}).front();

    std::vector<array> outputs;
    outputs.reserve(output_sizes.size());
    int offset = 0;
    for (const int width : output_sizes) {
        auto shape = prefix;
        shape.push_back(width);
        outputs.push_back(mlx::core::reshape(
            mlx::core::slice(
                combined,
                Shape{0, offset},
                Shape{checked_int(rows, "projection-group row count"),
                      offset + width}),
            std::move(shape)));
        offset += width;
    }
    return outputs;
}

array MlxMxfp4SqWeight::routed_matmul(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    int out_per_expert) const {
    if (out_per_expert <= 0 || output_size() % out_per_expert != 0) {
        throw std::invalid_argument(
            "MXFP4-SQ routed output width is inconsistent");
    }
    if (expert_ids.ndim() != 2 || expert_map.ndim() != 1 ||
        expert_ids.dtype() != mlx::core::int32 ||
        expert_map.dtype() != mlx::core::int32) {
        throw std::invalid_argument(
            "MXFP4-SQ routing metadata must be contiguous int32 IDs");
    }
    const int tokens = expert_ids.shape(0);
    const int routes = expert_ids.shape(1);
    const bool shared_input = input.ndim() == 2 &&
        input.shape(0) == tokens && input.shape(1) == input_size();
    if (!shared_input && (
        input.ndim() != 3 || input.shape(0) != tokens ||
        input.shape(1) != routes || input.shape(2) != input_size())) {
        throw std::invalid_argument(
            "MXFP4-SQ routed input must be [tokens,K] or "
            "[tokens,routes,K]");
    }
    const auto route_count =
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(routes);
    if (route_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("MXFP4-SQ route count exceeds MLX limits");
    }
    const Shape output_shape{tokens, routes, out_per_expert};
    if (route_count == 0) {
        return mlx::core::zeros(output_shape, mlx::core::float16);
    }
    auto source = input.dtype() == mlx::core::float16
        ? input
        : mlx::core::astype(input, mlx::core::float16);
    source = mlx::core::contiguous(source);
    auto ids = mlx::core::contiguous(expert_ids);
    auto map = mlx::core::contiguous(expert_map);

    const int kLanes = has_uniform_q() ? 8 : 32;
    const int kSimdGroups = has_uniform_q() ? 2 : 4;
    const int kThreads = kSimdGroups * 32;
    const int kOutputsPerThreadgroup =
        kSimdGroups * 32 / kLanes;
    const int output_tiles =
        (out_per_expert + kOutputsPerThreadgroup - 1) /
        kOutputsPerThreadgroup;
    auto arguments = templates(*this, source.dtype());
    arguments.emplace_back("M", static_cast<int>(route_count));
    arguments.emplace_back("TILE_M", 1);
    arguments.emplace_back("K_LANES", kLanes);
    arguments.emplace_back("SIMD_GROUPS", kSimdGroups);
    arguments.emplace_back("ROUTED", 1);
    arguments.emplace_back("ROUTES", routes);
    arguments.emplace_back("SHARED_INPUT", static_cast<int>(shared_input));
    arguments.emplace_back("EXPERT_MAP_SIZE", expert_map.shape(0));
    arguments.emplace_back("OUT_PER_EXPERT", out_per_expert);
    arguments.emplace_back("LOGICAL_OUT", out_per_expert);
    auto outputs = matmul_kernel()(
        {
            blob_,
            palette_catalog(),
            row_q_,
            row_symbol_byte_offsets_,
            row_auxiliary_,
            source,
            ids,
            map,
        },
        {Shape{static_cast<int>(route_count), out_per_expert}},
        {mlx::core::float16},
        {static_cast<int>(route_count) * kThreads, output_tiles, 1},
        {kThreads, 1, 1},
        std::move(arguments),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()), output_shape);
}

array MlxMxfp4SqWeight::backward_input(
    const array& output_gradient) const {
    if (output_gradient.ndim() == 0 ||
        output_gradient.shape(-1) != output_size()) {
        throw std::runtime_error(
            "MXFP4-SQ output-gradient width does not match packed weight");
    }
    Shape output_shape = output_gradient.shape();
    output_shape.back() = input_size();
    auto source = output_gradient;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    const auto rows =
        source.size() / static_cast<std::size_t>(output_size());
    if (rows == 0 || rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "unsupported MXFP4-SQ backward row count");
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<int>(rows), output_size()});
    if (rows <= 8 && source.dtype() == mlx::core::float16) {
        auto arguments = templates(*this, source.dtype());
        arguments.emplace_back("M", static_cast<int>(rows));
        auto outputs = backward_matrix_kernel()(
            {
                blob_,
                palette_catalog(),
                row_q_,
                row_symbol_byte_offsets_,
                row_auxiliary_,
                source,
            },
            {Shape{static_cast<int>(rows), input_size()}},
            {source.dtype()},
            {(input_size() / 32) * 128,
             (static_cast<int>(rows) + 7) / 8,
             1},
            {128, 1, 1},
            std::move(arguments),
            std::nullopt,
            false,
            {});
        return mlx::core::reshape(
            std::move(outputs.front()),
            std::move(output_shape));
    }
    auto result = mlx::core::matmul(
        source,
        dequantize(source.dtype()));
    return mlx::core::reshape(
        std::move(result),
        std::move(output_shape));
}

} // namespace mfq::metal
