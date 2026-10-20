#include "mlx_fp8_sq.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;

constexpr const char* kFp8SqHeader = R"METAL(
template <typename Stream>
inline uint mfq_fp8_sq_read_bits(
    Stream stream,
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

inline float mfq_fp8_sq_e4m3(uchar raw) {
    uint magnitude = uint(raw & 0x7fu);
    uint exponent = magnitude >> 3u;
    uint mantissa = magnitude & 7u;
    float value;
    if (exponent == 0u) {
        value = float(mantissa) * 0x1p-9f;
    } else {
        ushort half_bits = ushort((magnitude << 7u) + 0x2000u);
        value = float(as_type<half>(half_bits));
    }
    return (raw & 0x80u) == 0u ? value : -value;
}

template <typename Scales>
inline float mfq_fp8_sq_scale(
    Scales scales,
    uint index,
    uint scale_kind
) {
    if (scale_kind == 1u) {
        uchar raw = scales[index];
        uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
        return as_type<float>(bits);
    }
    if (scale_kind == 2u) {
        uint byte = index * 2u;
        uint word = uint(as_type<ushort>(*(device const packed_uchar2*)(scales + byte)));
        return as_type<float>(word << 16u);
    }
    if (scale_kind == 3u) {
        uint byte = index * 2u;
        ushort word = as_type<ushort>(*(device const packed_uchar2*)(scales + byte));
        return float(as_type<half>(word));
    }
    uint byte = index * 4u;
    uint word = as_type<uint>(*(device const packed_uchar4*)(scales + byte));
    return as_type<float>(word);
}

template <typename Blob, typename RowQ, typename RowOffsets>
inline uchar mfq_fp8_sq_code(
    Blob blob,
    RowQ row_q,
    RowOffsets row_symbol_byte_offsets,
    uint output,
    uint column,
    uint symbols_offset,
    uint palettes_offset
) {
    uint bits = uint(row_q[output]);
    auto row_symbols = blob + symbols_offset
        + row_symbol_byte_offsets[output];
    if (bits == 8u) {
        return row_symbols[column];
    }
    uint symbol = mfq_fp8_sq_read_bits(row_symbols, column, bits);
    uint palette_offset = (1u << bits) - 2u;
    return blob[palettes_offset + palette_offset + symbol];
}

template <typename Blob, typename RowQ, typename RowOffsets>
inline float mfq_fp8_sq_weight(
    Blob blob,
    RowQ row_q,
    RowOffsets row_symbol_byte_offsets,
    uint output,
    uint column,
    uint palettes_offset,
    uint symbols_offset,
    uint scales_offset,
    uint scale_kind,
    uint block_rows,
    uint block_columns,
    uint scale_columns
) {
    uchar code = mfq_fp8_sq_code(
        blob, row_q, row_symbol_byte_offsets, output, column,
        symbols_offset, palettes_offset);
    uint scale_row = output / block_rows;
    uint scale_column = column / block_columns;
    uint scale_index = scale_row * scale_columns + scale_column;
    float scale = mfq_fp8_sq_scale(
        blob + scales_offset, scale_index, scale_kind);
    return mfq_fp8_sq_e4m3(code) * scale;
}

template <uint WIDTH, typename T>
__attribute__((always_inline)) inline float4 mfq_fp8_sq_load4(
    device const T* input, uint offset, uint column) {
    if (WIDTH % 4u == 0u || column + 4u <= WIDTH) {
        if constexpr (sizeof(T) == 2u) {
            return float4(*(device const packed_half4*)(input + offset));
        } else {
            return float4(*(device const packed_float4*)(input + offset));
        }
    }
    return float4(float(input[offset]),
        column + 1u < WIDTH ? float(input[offset + 1u]) : 0.0f,
        column + 2u < WIDTH ? float(input[offset + 2u]) : 0.0f,
        column + 3u < WIDTH ? float(input[offset + 3u]) : 0.0f);
}

template <uint WIDTH, typename T>
inline float4 mfq_fp8_sq_load4(constant const T* input, uint offset, uint column) {
    return float4(float(input[offset]),
        column + 1u < WIDTH ? float(input[offset + 1u]) : 0.0f,
        column + 2u < WIDTH ? float(input[offset + 2u]) : 0.0f,
        column + 3u < WIDTH ? float(input[offset + 3u]) : 0.0f);
}

template <uint WIDTH, uint INPUT_ROWS, typename Blob, typename RowQ, typename RowOffsets, typename Palette>
__attribute__((always_inline)) inline float4 mfq_fp8_sq_weight4(
    Blob blob, RowQ row_q, RowOffsets row_offsets, Palette palette_values,
    uint output, uint column,
    uint palettes_offset, uint symbols_offset, uint scales_offset,
    uint scale_kind, uint block_rows, uint block_columns, uint scale_columns
) {
    if (WIDTH % 4u != 0u && column + 4u > WIDTH) {
        float4 result = float4(0.0f);
        for (uint element = 0u; element < 4u && column + element < WIDTH; ++element) {
            result[element] = mfq_fp8_sq_weight(blob, row_q, row_offsets, output,
                column + element, palettes_offset, symbols_offset, scales_offset,
                scale_kind, block_rows, block_columns, scale_columns);
        }
        return result;
    }
    uint bits = uint(row_q[output]);
    auto symbols = blob + symbols_offset + row_offsets[output];
    uchar4 codes;
    if (bits == 8u) {
        codes = *(device const packed_uchar4*)(symbols + column);
    } else {
        uint bit = column * bits;
        uint byte = bit >> 3u;
        uint shift = bit & 7u;
        uint packed;
        if (bits == 7u) {
            packed = as_type<uint>(*(device const packed_uchar4*)(symbols + byte));
        } else {
            uint span = (INPUT_ROWS == 1u ? 0u : shift) + bits * 4u;
            packed = uint(symbols[byte]);
            if (span > 8u) packed |= uint(symbols[byte + 1u]) << 8u;
            if (span > 16u) packed |= uint(symbols[byte + 2u]) << 16u;
        }
        uint4 indices = (uint4(packed >> shift) >> (uint4(0u, 1u, 2u, 3u) * bits))
            & ((1u << bits) - 1u);
        uint palette = (1u << bits) - 2u;
        float4 values = float4(palette_values[palette + indices.x],
            palette_values[palette + indices.y], palette_values[palette + indices.z],
            palette_values[palette + indices.w]);
        uint scale_index = (output / block_rows) * scale_columns + column / block_columns;
        return values * mfq_fp8_sq_scale(blob + scales_offset, scale_index, scale_kind);
    }
    uint4 raw = uint4(codes);
    uint4 magnitudes = raw & 127u;
    ushort4 half_bits = ushort4(((magnitudes << 7u) + 0x2000u)
        | ((raw & 128u) << 8u));
    float4 values = float4(as_type<half4>(half_bits));
    float4 subnormal = float4(raw & 7u) * 0x1p-9f;
    subnormal = select(subnormal, -subnormal, (raw & 128u) != 0u);
    values = select(values, subnormal, magnitudes < 8u);
    uint scale_index = (output / block_rows) * scale_columns + column / block_columns;
    return values * mfq_fp8_sq_scale(blob + scales_offset, scale_index, scale_kind);
}

inline float2 mfq_fp8_sq_pair(device const half* values, uint index) {
    return float2(*(device const packed_half2*)(values + index * 2u));
}

inline float2 mfq_fp8_sq_pair(constant const half* values, uint index) {
    return float2(values[index * 2u], values[index * 2u + 1u]);
}

template <uint BITS, uint WIDTH, uint INPUT_ROWS, uint LANES,
          typename Symbols, typename Scales, typename Palette, typename Pairs, typename XStream>
__attribute__((always_inline)) inline void mfq_fp8_sq_high_dot(
    Symbols symbols, Scales scales, Palette palette_values, Pairs palette_pairs, XStream x,
    uint first_row, uint rows, uint input_divisor, uint lane,
    uint scale_base, uint scale_kind, uint block_columns, thread float* accum
) {
    for (uint column = lane * 4u; column < WIDTH; column += LANES * 4u) {
        uint bit = column * BITS;
        uint byte = bit >> 3u;
        uint packed = uint(symbols[byte]) | (uint(symbols[byte + 1u]) << 8u);
        if constexpr (BITS >= 5u) packed |= uint(symbols[byte + 2u]) << 16u;
        packed >>= bit & 7u;
        float4 weight;
        if constexpr (BITS == 6u) {
            uint4 indices = (uint4(packed) >> uint4(0u, 6u, 12u, 18u)) & 63u;
            weight = float4(palette_values[62u + indices.x], palette_values[62u + indices.y],
                palette_values[62u + indices.z], palette_values[62u + indices.w]);
        } else {
            constexpr uint MASK = (1u << (2u * BITS)) - 1u;
            auto pairs = palette_pairs + 2u * ((1u << (2u * BITS)) - 64u) / 3u;
            weight = float4(mfq_fp8_sq_pair(pairs, packed & MASK),
                mfq_fp8_sq_pair(pairs, (packed >> (2u * BITS)) & MASK));
        }
        float scale = mfq_fp8_sq_scale(scales, scale_base + column / block_columns, scale_kind);
        if constexpr (INPUT_ROWS > 4u) weight *= scale;
        for (uint local = 0u; local < INPUT_ROWS; ++local) {
            uint row = first_row + local;
            if (row < rows) {
                uint input = (row / input_divisor) * WIDTH + column;
                float value = dot(mfq_fp8_sq_load4<WIDTH>(x, input, column), weight);
                if constexpr (INPUT_ROWS <= 4u) value *= scale;
                accum[local] += value;
            }
        }
    }
}

template <uint WIDTH, uint INPUT_ROWS, uint LANES, bool HAS_LOW_Q, uint FAST_Q_MAX,
          typename Blob, typename RowQ, typename RowOffsets, typename Palette,
          typename Quads, typename Pairs, typename XStream>
__attribute__((always_inline)) inline void mfq_fp8_sq_dot(
    Blob blob, RowQ row_q, RowOffsets row_offsets, Palette palette_values,
    Quads palette_quads, Pairs palette_pairs,
    XStream x, uint output, uint first_row, uint rows, uint input_divisor, uint lane,
    uint palettes_offset, uint symbols_offset, uint scales_offset,
    uint scale_kind, uint block_rows, uint block_columns, uint scale_columns,
    thread float* accum
) {
    if constexpr (HAS_LOW_Q && WIDTH % 8u == 0u) {
        constexpr bool WIDE_Q2 = LANES == 16u && WIDTH % 16u == 0u;
        uint bits = uint(row_q[output]);
        if (WIDE_Q2 && bits == 2u) {
            auto symbols = blob + symbols_offset + row_offsets[output];
            for (uint column = lane * 16u; column < WIDTH; column += LANES * 16u) {
                uint packed = as_type<uint>(*(device const packed_uchar4*)(symbols + column / 4u));
                float4 weight0 = mfq_fp8_sq_load4<4u>(palette_quads, (packed & 255u) * 4u, 0u);
                float4 weight1 = mfq_fp8_sq_load4<4u>(palette_quads, ((packed >> 8u) & 255u) * 4u, 0u);
                float4 weight2 = mfq_fp8_sq_load4<4u>(palette_quads, ((packed >> 16u) & 255u) * 4u, 0u);
                float4 weight3 = mfq_fp8_sq_load4<4u>(palette_quads, (packed >> 24u) * 4u, 0u);
                uint scale_index = (output / block_rows) * scale_columns + column / block_columns;
                float scale = mfq_fp8_sq_scale(blob + scales_offset, scale_index, scale_kind);
                for (uint local = 0u; local < INPUT_ROWS; ++local) {
                    uint row = first_row + local;
                    if (row < rows) {
                        uint input = (row / input_divisor) * WIDTH + column;
                        float value = dot(mfq_fp8_sq_load4<WIDTH>(x, input, column), weight0)
                            + dot(mfq_fp8_sq_load4<WIDTH>(x, input + 4u, column + 4u), weight1)
                            + dot(mfq_fp8_sq_load4<WIDTH>(x, input + 8u, column + 8u), weight2)
                            + dot(mfq_fp8_sq_load4<WIDTH>(x, input + 12u, column + 12u), weight3);
                        accum[local] += scale * value;
                    }
                }
            }
            return;
        }
        if (bits == 3u || (!WIDE_Q2 && bits == 2u)) {
            auto symbols = blob + symbols_offset + row_offsets[output];
            for (uint column = lane * 8u; column < WIDTH; column += LANES * 8u) {
                uint byte = column / 8u * (WIDE_Q2 ? 3u : bits);
                uint packed = uint(symbols[byte]) | (uint(symbols[byte + 1u]) << 8u);
                if (WIDE_Q2 || bits == 3u) packed |= uint(symbols[byte + 2u]) << 16u;
                float4 weight0, weight1;
                if (!WIDE_Q2 && bits == 2u) {
                    weight0 = mfq_fp8_sq_load4<4u>(palette_quads, (packed & 255u) * 4u, 0u);
                    weight1 = mfq_fp8_sq_load4<4u>(palette_quads, ((packed >> 8u) & 255u) * 4u, 0u);
                } else {
                    weight0 = float4(mfq_fp8_sq_pair(palette_pairs, packed & 63u),
                        mfq_fp8_sq_pair(palette_pairs, (packed >> 6u) & 63u));
                    weight1 = float4(mfq_fp8_sq_pair(palette_pairs, (packed >> 12u) & 63u),
                        mfq_fp8_sq_pair(palette_pairs, (packed >> 18u) & 63u));
                }
                uint scale_index = (output / block_rows) * scale_columns + column / block_columns;
                float scale = mfq_fp8_sq_scale(blob + scales_offset, scale_index, scale_kind);
                for (uint local = 0u; local < INPUT_ROWS; ++local) {
                    uint row = first_row + local;
                    if (row < rows) {
                        uint input = (row / input_divisor) * WIDTH + column;
                        accum[local] += scale * (dot(mfq_fp8_sq_load4<WIDTH>(x, input, column), weight0)
                            + dot(mfq_fp8_sq_load4<WIDTH>(x, input + 4u, column + 4u), weight1));
                    }
                }
            }
            return;
        }
    }
    if constexpr (FAST_Q_MAX >= 4u && WIDTH % 4u == 0u) {
        uint bits = uint(row_q[output]);
        auto symbols = blob + symbols_offset + row_offsets[output];
        auto scales = blob + scales_offset;
        uint scale_base = (output / block_rows) * scale_columns;
        if (bits == 4u) {
            mfq_fp8_sq_high_dot<4u, WIDTH, INPUT_ROWS, LANES>(symbols, scales,
                palette_values, palette_pairs, x, first_row, rows, input_divisor,
                lane, scale_base, scale_kind, block_columns, accum);
            return;
        }
        if (FAST_Q_MAX >= 5u && bits == 5u) {
            mfq_fp8_sq_high_dot<5u, WIDTH, INPUT_ROWS, LANES>(symbols, scales,
                palette_values, palette_pairs, x, first_row, rows, input_divisor,
                lane, scale_base, scale_kind, block_columns, accum);
            return;
        }
        if (FAST_Q_MAX >= 6u && bits == 6u) {
            mfq_fp8_sq_high_dot<6u, WIDTH, INPUT_ROWS, LANES>(symbols, scales,
                palette_values, palette_pairs, x, first_row, rows, input_divisor,
                lane, scale_base, scale_kind, block_columns, accum);
            return;
        }
    }
    if (uint(row_q[output]) == 1u) {
        float low = palette_values[0];
        float high = palette_values[1];
        auto symbols = blob + symbols_offset + row_offsets[output];
        for (uint column = lane * 8u; column < WIDTH; column += LANES * 8u) {
            uint packed = uint(symbols[column >> 3u]);
            uint scale_index = (output / block_rows) * scale_columns + column / block_columns;
            float scale = mfq_fp8_sq_scale(blob + scales_offset, scale_index, scale_kind);
            float4 scaled_low = float4(low * scale);
            float4 scaled_high = float4(high * scale);
            float4 weight0 = select(scaled_low, scaled_high,
                (uint4(packed) & uint4(1u, 2u, 4u, 8u)) != 0u);
            float4 weight1 = select(scaled_low, scaled_high,
                (uint4(packed >> 4u) & uint4(1u, 2u, 4u, 8u)) != 0u);
            for (uint local = 0u; local < INPUT_ROWS; ++local) {
                uint row = first_row + local;
                if (row < rows) {
                    uint input = (row / input_divisor) * WIDTH + column;
                    float value = dot(mfq_fp8_sq_load4<WIDTH>(x, input, column), weight0);
                    if (WIDTH % 8u == 0u || column + 4u < WIDTH) {
                        value += dot(mfq_fp8_sq_load4<WIDTH>(x, input + 4u, column + 4u), weight1);
                    }
                    accum[local] += value;
                }
            }
        }
    } else if (WIDTH % 4u == 0u && uint(row_q[output]) == 8u) {
        auto symbols = blob + symbols_offset + row_offsets[output];
        for (uint column = lane * 4u; column < WIDTH; column += LANES * 4u) {
            uchar4 codes = *(device const packed_uchar4*)(symbols + column);
            uint scale_index = (output / block_rows) * scale_columns + column / block_columns;
            float scale = mfq_fp8_sq_scale(blob + scales_offset, scale_index, scale_kind);
            float4 weight = float4(palette_values[256u + uint(codes.x)],
                palette_values[256u + uint(codes.y)], palette_values[256u + uint(codes.z)],
                palette_values[256u + uint(codes.w)]) * scale;
            for (uint local = 0u; local < INPUT_ROWS; ++local) {
                uint row = first_row + local;
                if (row < rows) {
                    uint input = (row / input_divisor) * WIDTH + column;
                    accum[local] += dot(mfq_fp8_sq_load4<WIDTH>(x, input, column), weight);
                }
            }
        }
    } else {
        for (uint column = lane * 4u; column < WIDTH; column += LANES * 4u) {
            float4 weight = mfq_fp8_sq_weight4<WIDTH, INPUT_ROWS>(
                blob, row_q, row_offsets, palette_values, output, column,
                palettes_offset, symbols_offset, scales_offset, scale_kind,
                block_rows, block_columns, scale_columns);
            for (uint local = 0u; local < INPUT_ROWS; ++local) {
                uint row = first_row + local;
                if (row < rows) {
                    uint input = (row / input_divisor) * WIDTH + column;
                    accum[local] += dot(mfq_fp8_sq_load4<WIDTH>(x, input, column), weight);
                }
            }
        }
    }
}
)METAL";

constexpr const char* kFp8SqDequantize = R"METAL(
    uint index = thread_position_in_grid.x * 2u;
    if (index >= uint(WEIGHTS)) {
        return;
    }
    uint output = index / uint(K);
    uint column = index - output * uint(K);
    float first = mfq_fp8_sq_weight(
        blob, row_q, row_symbol_byte_offsets, output, column,
        uint(PALETTES_OFFSET), uint(SYMBOLS_OFFSET), uint(SCALES_OFFSET),
        uint(SCALE_KIND), uint(BLOCK_ROWS), uint(BLOCK_COLUMNS),
        uint(SCALE_COLUMNS));
    if (index + 1u >= uint(WEIGHTS)) {
        y[index] = T(first);
        return;
    }
    uint next_output = (index + 1u) / uint(K);
    uint next_column = index + 1u - next_output * uint(K);
    float second = mfq_fp8_sq_weight(
        blob, row_q, row_symbol_byte_offsets, next_output, next_column,
        uint(PALETTES_OFFSET), uint(SYMBOLS_OFFSET), uint(SCALES_OFFSET),
        uint(SCALE_KIND), uint(BLOCK_ROWS), uint(BLOCK_COLUMNS),
        uint(SCALE_COLUMNS));
    if constexpr (sizeof(T) == 2u) {
        *(device packed_half2*)(y + index) = packed_half2(first, second);
    } else {
        *(device packed_float2*)(y + index) = packed_float2(first, second);
    }
)METAL";

constexpr const char* kFp8SqMatmul = R"METAL(
    const int M = ROUTED != 0 ? expert_ids_shape[0] * ROUTES : x_shape[0];
    constexpr uint OUTPUTS_PER_SIMD = 32u / uint(K_LANES);
    constexpr uint OUTPUT_TILES = (uint(LOGICAL_OUT) + OUTPUTS_PER_SIMD - 1u) / OUTPUTS_PER_SIMD;
    uint lane = thread_index_in_simdgroup % uint(K_LANES);
    uint workgroup = thread_position_in_grid.x >> 5u;
    uint logical_output = workgroup % OUTPUT_TILES * OUTPUTS_PER_SIMD
        + thread_index_in_simdgroup / uint(K_LANES);
    uint row_tile = workgroup / OUTPUT_TILES;
    uint first_row = row_tile * uint(TILE_M);
    if (first_row >= uint(M) || logical_output >= uint(LOGICAL_OUT)) {
        return;
    }
    int local_expert = 0;
    bool route_valid = true;
    uint output = logical_output;
    if (uint(ROUTED) != 0u) {
        int expert = expert_ids[first_row];
        route_valid = expert >= 0 && expert < int(EXPERT_MAP_SIZE);
        local_expert = route_valid ? expert_map[expert] : -1;
        route_valid = route_valid && local_expert >= 0;
        output = route_valid
            ? uint(local_expert) * uint(OUT_PER_EXPERT) + logical_output
            : 0u;
    }
    float accum[TILE_M];
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        accum[local] = 0.0f;
    }
    mfq_fp8_sq_dot<uint(K), uint(TILE_M), uint(K_LANES), bool(HAS_LOW_Q), uint(FAST_Q_MAX)>(
        blob, row_q, row_symbol_byte_offsets, palette_values, palette_quads, palette_pairs, x, output,
        first_row, uint(M), uint(ROUTED) != 0u && uint(SHARED_INPUT) != 0u
            ? uint(ROUTES) : 1u, lane,
        uint(PALETTES_OFFSET), uint(SYMBOLS_OFFSET), uint(SCALES_OFFSET),
        uint(SCALE_KIND), uint(BLOCK_ROWS), uint(BLOCK_COLUMNS), uint(SCALE_COLUMNS), accum);
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        uint row = first_row + local;
        float total = accum[local];
        if constexpr (K_LANES == 32) {
            total = simd_sum(total);
        } else {
            for (uint offset = uint(K_LANES) / 2u; offset > 0u; offset >>= 1u) {
                total += simd_shuffle_down(total, offset);
            }
        }
        if (lane == 0u && row < uint(M)) {
            y[row * uint(LOGICAL_OUT) + logical_output] = T(
                route_valid ? total : 0.0f);
        }
    }
)METAL";


// One metadata-driven packed input-gradient kernel body is compiled under
// separate MXFP8-SQ and FP8-128SQ entry points.  It consumes q=1..8 rows
// directly and never materializes a dense dequantized weight matrix.
constexpr const char* kFp8SqBackwardMatrix = R"METAL(
    constexpr uint BM = 8u;
    constexpr uint BN = 64u;
    constexpr uint BK = 32u;
    constexpr uint BN_PAD = BN + 8u;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint local_thread = thread_index_in_threadgroup;
    uint row_base = threadgroup_position_in_grid.y * BM;
    uint column_base = threadgroup_position_in_grid.x * BK;

    threadgroup half gradient_tile[BM * BN_PAD];
    threadgroup half weight_tile[BN * BK];
    threadgroup float cached_scales[BN];
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
            if (output < uint(OUT) && column_base < uint(K)) {
                cached_bits[local_thread] = row_q[output];
                cached_symbol_offsets[local_thread] =
                    row_symbol_byte_offsets[output];
                uint scale_row = output / uint(BLOCK_ROWS);
                uint scale_column = column_base / uint(BLOCK_COLUMNS);
                uint scale_index =
                    scale_row * uint(SCALE_COLUMNS) + scale_column;
                cached_scales[local_thread] = mfq_fp8_sq_scale(
                    blob + uint(SCALES_OFFSET),
                    scale_index,
                    uint(SCALE_KIND));
            } else {
                cached_bits[local_thread] = 1u;
                cached_symbol_offsets[local_thread] = 0u;
                cached_scales[local_thread] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint task = local_thread;
             task < BN * 8u;
             task += 128u) {
            uint local_output = task >> 3u;
            uint vector = task & 7u;
            uint output = output_base + local_output;
            uint local_column = vector * 4u;
            uint column = column_base + local_column;
            half4 decoded = half4(0.0h);
            if (output < uint(OUT) && column < uint(K)) {
                uint bits = uint(cached_bits[local_output]);
                float scale = cached_scales[local_output];
                device const uchar* row_symbols =
                    blob + uint(SYMBOLS_OFFSET)
                    + cached_symbol_offsets[local_output];
                float4 values = float4(0.0f);
#pragma unroll
                for (uint element = 0u; element < 4u; ++element) {
                    uint current_column = column + element;
                    if (current_column < uint(K)) {
                        uchar code;
                        if (bits == 8u) {
                            code = row_symbols[current_column];
                        } else {
                            uint symbol = mfq_fp8_sq_read_bits(
                                row_symbols, current_column, bits);
                            uint palette_offset = (1u << bits) - 2u;
                            code = blob[
                                uint(PALETTES_OFFSET)
                                + palette_offset + symbol];
                        }
                        values[element] = mfq_fp8_sq_e4m3(code) * scale;
                    }
                }
                decoded = half4(values);
            }
            *(threadgroup half4*)(
                weight_tile + local_output * BK + local_column) = decoded;
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

mlx::core::fast::CustomKernelFunction make_kernel(
    std::string name,
    const char* source,
    bool matmul) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        std::move(name),
        matmul
            ? std::vector<std::string>{
                  "blob", "row_q", "row_symbol_byte_offsets", "x",
                  "expert_ids", "expert_map", "palette_values", "palette_quads", "palette_pairs"}
            : std::vector<std::string>{
                  "blob", "row_q", "row_symbol_byte_offsets"},
        {"y"},
        source,
        kFp8SqHeader,
        true,
        false,
        options);
}

mlx::core::fast::CustomKernelFunction make_backward_kernel(
    std::string name) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        std::move(name),
        {"blob", "row_q", "row_symbol_byte_offsets", "x"},
        {"y"},
        kFp8SqBackwardMatrix,
        kFp8SqHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& mxfp8_dequantize_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mxfp8_sq_dequantize", kFp8SqDequantize, false);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& fp8_128_dequantize_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_fp8_128_sq_dequantize", kFp8SqDequantize, false);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mxfp8_matmul_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_mxfp8_sq_matmul", kFp8SqMatmul, true);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& fp8_128_matmul_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_fp8_128_sq_matmul", kFp8SqMatmul, true);
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& mxfp8_backward_kernel() {
    static const auto kernel = make_backward_kernel(
        "mfq_cpp_mxfp8_sq_backward_matrix");
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& fp8_128_backward_kernel() {
    static const auto kernel = make_backward_kernel(
        "mfq_cpp_fp8_128_sq_backward_matrix");
    return kernel;
}

int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string("FP8-SQ ") + name + " exceeds MLX limits");
    }
    return static_cast<int>(value);
}

unsigned maximum_fast_q(unsigned q_mask, bool grouped = false) {
    if (grouped && (q_mask & (q_mask - 1u)) != 0u) return (q_mask & 4u) != 0u ? 3 : 0;
    for (unsigned bits = 6; bits >= 3; --bits) {
        if ((q_mask & (1u << (bits - 1))) != 0) return bits;
    }
    return 0;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
templates(const MlxFp8SqWeight& weight, Dtype dtype) {
    const auto& layout = weight.wire_layout();
    return {
        {"T", dtype},
        {"HAS_LOW_Q", (weight.descriptor().q_mask & 6u) != 0},
        {"FAST_Q_MAX", static_cast<int>(maximum_fast_q(weight.descriptor().q_mask))},
        {"K", layout.width},
        {"OUT", layout.outputs},
        {"WEIGHTS", checked_int(
            static_cast<std::size_t>(layout.width) * layout.outputs,
            "weight count")},
        {"PALETTES_OFFSET", checked_int(layout.palettes, "palette offset")},
        {"SYMBOLS_OFFSET", checked_int(layout.symbols, "symbol offset")},
        {"SCALES_OFFSET", checked_int(layout.scales, "scale offset")},
        {"SCALE_KIND", static_cast<int>(layout.scale_kind)},
        {"BLOCK_ROWS", layout.block_rows},
        {"BLOCK_COLUMNS", layout.block_columns},
        {"SCALE_COLUMNS", layout.scale_columns},
    };
}

const mlx::core::fast::CustomKernelFunction& dequantize_kernel(
    const MlxFp8SqWeight& weight) {
    return weight.dtype() == "MXFP8-SQ"
        ? mxfp8_dequantize_kernel()
        : fp8_128_dequantize_kernel();
}

const mlx::core::fast::CustomKernelFunction& matmul_kernel(
    const MlxFp8SqWeight& weight) {
    return weight.dtype() == "MXFP8-SQ"
        ? mxfp8_matmul_kernel()
        : fp8_128_matmul_kernel();
}

const mlx::core::fast::CustomKernelFunction& backward_kernel(
    const MlxFp8SqWeight& weight) {
    return weight.dtype() == "MXFP8-SQ"
        ? mxfp8_backward_kernel()
        : fp8_128_backward_kernel();
}

std::vector<std::string> projection_group_input_names(
    std::size_t projections) {
    std::vector<std::string> names;
    names.reserve(projections * 6 + 1);
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        names.push_back("blob_" + suffix);
        names.push_back("row_q_" + suffix);
        names.push_back("row_symbol_byte_offsets_" + suffix);
        names.push_back("palette_values_" + suffix);
        names.push_back("palette_quads_" + suffix);
        names.push_back("palette_pairs_" + suffix);
    }
    names.emplace_back("x");
    return names;
}

std::string make_projection_group_source(std::size_t projections) {
    std::string source = R"METAL(
    const int M = x_shape[0];
    constexpr uint OUTPUTS_PER_SIMD = 32u / uint(K_LANES);
    constexpr uint OUTPUT_TILES = (uint(TOTAL_OUT) + OUTPUTS_PER_SIMD - 1u) / OUTPUTS_PER_SIMD;
    uint lane = thread_index_in_simdgroup % uint(K_LANES);
    uint workgroup = thread_position_in_grid.x >> 5u;
    uint logical_output = workgroup % OUTPUT_TILES * OUTPUTS_PER_SIMD
        + thread_index_in_simdgroup / uint(K_LANES);
    uint row_tile = workgroup / OUTPUT_TILES;
    uint first_row = row_tile * uint(TILE_M);
    if (first_row >= uint(M) || logical_output >= uint(TOTAL_OUT)) {
        return;
    }

    float accum[TILE_M];
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        accum[local] = 0.0f;
    }
)METAL";
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source += projection == 0 ? "    if (" : "    else if (";
        source += "logical_output < uint(P" + suffix + "_END)) {\n";
        source += "        uint local_output = logical_output - uint(P" + suffix
            + "_OFFSET);\n";
        source +=
            "            mfq_fp8_sq_dot<uint(K), uint(TILE_M), uint(K_LANES), bool(P" + suffix
            + "_HAS_LOW_Q), uint(P" + suffix + "_FAST_Q_MAX)>(\n"
            "                blob_" + suffix + ", row_q_" + suffix
            + ", row_symbol_byte_offsets_" + suffix + ", palette_values_" + suffix
            + ", palette_quads_" + suffix + ", palette_pairs_" + suffix + ",\n"
            "                x, local_output, first_row, uint(M), 1u, lane,\n"
            "                uint(P" + suffix + "_PALETTES_OFFSET),\n"
            "                uint(P" + suffix + "_SYMBOLS_OFFSET),\n"
            "                uint(P" + suffix + "_SCALES_OFFSET),\n"
            "                uint(P" + suffix + "_SCALE_KIND),\n"
            "                uint(P" + suffix + "_BLOCK_ROWS),\n"
            "                uint(P" + suffix + "_BLOCK_COLUMNS),\n"
            "                uint(P" + suffix + "_SCALE_COLUMNS), accum);\n"
            "        }\n";
    }
    source += R"METAL(
    for (uint local = 0u; local < uint(TILE_M); ++local) {
        uint row = first_row + local;
        float total = accum[local];
        if constexpr (K_LANES == 32) {
            total = simd_sum(total);
        } else {
            for (uint offset = uint(K_LANES) / 2u; offset > 0u; offset >>= 1u) {
                total += simd_shuffle_down(total, offset);
            }
        }
        if (lane == 0u && row < uint(M)) {
            y[row * uint(TOTAL_OUT) + logical_output] = T(total);
        }
    }
)METAL";
    return source;
}

mlx::core::fast::CustomKernelFunction projection_group_kernel(
    std::string_view dtype,
    std::size_t projections) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const std::string family = dtype == "MXFP8-SQ"
        ? "mxfp8"
        : "fp8_128";
    const auto key = family + "_p" + std::to_string(projections);
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = kernels.find(key); found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_" + family + "_sq_projection_group_p"
            + std::to_string(projections),
        projection_group_input_names(projections),
        {"y"},
        make_projection_group_source(projections),
        kFp8SqHeader,
        true,
        false,
        options);
    kernels.emplace(key, kernel);
    return kernel;
}

} // namespace

bool is_fp8_sq_dtype(std::string_view dtype) noexcept {
    return mfq::fp8sq::is_dtype(dtype);
}

MlxFp8SqWeight::MlxFp8SqWeight(
    std::string dtype,
    array blob,
    array row_q,
    array row_symbol_byte_offsets,
    array palette_values,
    array palette_quads,
    array palette_pairs,
    mfq::fp8sq::Layout layout,
    Fp8SqDescriptor descriptor)
    : dtype_(std::move(dtype)),
      blob_(std::move(blob)),
      row_q_(std::move(row_q)),
      row_symbol_byte_offsets_(std::move(row_symbol_byte_offsets)),
      palette_values_(std::move(palette_values)),
      palette_quads_(std::move(palette_quads)),
      palette_pairs_(std::move(palette_pairs)),
      layout_(std::move(layout)),
      descriptor_(descriptor) {}

MlxFp8SqWeight MlxFp8SqWeight::from_blob(
    std::string_view dtype,
    const std::vector<std::uint8_t>& blob) {
    return from_blob(dtype, std::span<const std::uint8_t>(blob));
}

MlxFp8SqWeight MlxFp8SqWeight::from_blob(
    std::string_view dtype,
    std::span<const std::uint8_t> blob) {
    const auto layout = mfq::fp8sq::parse(dtype, blob.data(), blob.size());
    if (blob.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("FP8-SQ payload exceeds MLX limits");
    }
    const auto rows = mfq::fp8sq::row_metadata(blob.data(), layout);
    std::array<float, mfq::fp8sq::kPaletteBytes + 256> palette_values{};
    for (std::size_t index = 0; index < palette_values.size(); ++index) {
        const auto code = index < mfq::fp8sq::kPaletteBytes ? blob[layout.palettes + index]
            : static_cast<std::uint8_t>(index - mfq::fp8sq::kPaletteBytes);
        const int exponent = (code & 127u) >> 3u;
        const int mantissa = code & 7u;
        const auto magnitude = exponent == 0 ? float(mantissa) * 0x1p-9f
            : std::ldexp(1.0f + float(mantissa) / 8.0f, exponent - 7);
        palette_values[index] = (code & 128u) != 0u ? -magnitude : magnitude;
    }
    Fp8SqDescriptor descriptor;
    descriptor.format_version = layout.version;
    descriptor.aggregate_bpw = static_cast<double>(
        layout.bytes - mfq::fp8sq::kHeaderBytes) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<std::size_t, 8> q_counts{};
    for (const auto q : rows.q) {
        descriptor.q_mask |= 1u << (q - 1);
        ++q_counts[static_cast<std::size_t>(q - 1)];
    }
    for (const auto count : q_counts) {
        if (count == 0) {
            continue;
        }
        const double probability = static_cast<double>(count) / layout.outputs;
        descriptor.distribution_entropy -= probability * std::log2(probability);
    }
    std::vector<mlx::core::float16_t> palette_quads;
    if ((descriptor.q_mask & 2u) != 0) {
        palette_quads.reserve(1024);
        for (unsigned code = 0; code < 256; ++code) {
            for (unsigned element = 0; element < 4; ++element) {
                const auto index = (code >> (element * 2u)) & 3u;
                palette_quads.push_back(mlx::core::float16_t(palette_values[2u + index]));
            }
        }
    } else {
        palette_quads.push_back(mlx::core::float16_t(0.0f));
    }
    std::vector<mlx::core::float16_t> palette_pairs;
    const unsigned maximum_q = maximum_fast_q(descriptor.q_mask & 31u);
    if (maximum_q != 0) {
        const unsigned count = 2u * ((1u << (2u * (maximum_q + 1u))) - 64u) / 3u;
        palette_pairs.reserve(count);
        for (unsigned bits = 3; bits <= maximum_q; ++bits) {
            const unsigned palette = (1u << bits) - 2u;
            for (unsigned code = 0; code < (1u << (2u * bits)); ++code) {
                palette_pairs.push_back(mlx::core::float16_t(
                    palette_values[palette + (code & ((1u << bits) - 1u))]));
                palette_pairs.push_back(mlx::core::float16_t(palette_values[palette + (code >> bits)]));
            }
        }
    } else {
        palette_pairs.push_back(mlx::core::float16_t(0.0f));
    }
    return MlxFp8SqWeight(
        std::string(dtype),
        array(blob.begin(), Shape{static_cast<int>(blob.size())}),
        array(rows.q.begin(), Shape{layout.outputs}),
        array(rows.symbol_byte_offsets.begin(), Shape{layout.outputs}),
        array(palette_values.begin(), Shape{static_cast<int>(palette_values.size())}),
        array(palette_quads.begin(), Shape{static_cast<int>(palette_quads.size())}),
        array(palette_pairs.begin(), Shape{static_cast<int>(palette_pairs.size())}),
        layout,
        descriptor);
}

array MlxFp8SqWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error("FP8-SQ dequantization requires float16 or float32");
    }
    const auto count = static_cast<std::size_t>(input_size()) * output_size();
    checked_int(count, "dequantization grid");
    constexpr int threads = 256;
    const auto work = (count + 1) / 2;
    const auto grid = (work + threads - 1) / threads * threads;
    auto outputs = dequantize_kernel(*this)(
        {blob_, row_q_, row_symbol_byte_offsets_},
        {Shape{output_size(), input_size()}},
        {dtype},
        {checked_int(grid, "dequantization grid"), 1, 1},
        {threads, 1, 1},
        templates(*this, dtype),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxFp8SqWeight::embedding(const array& rows, Dtype dtype) const {
    return mlx::core::take(dequantize(dtype), rows, 0);
}

array MlxFp8SqWeight::matmul(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size()) {
        throw std::runtime_error("FP8-SQ input width does not match packed weight");
    }
    const auto rows = input.size() / static_cast<std::size_t>(input_size());
    if (rows == 0 || rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("unsupported FP8-SQ input row count");
    }
    Shape output_shape = input.shape();
    output_shape.back() = output_size();
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source, Shape{static_cast<int>(rows), input_size()});
    if (rows > 48 || source.dtype() == mlx::core::float32) {
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dequantize(source.dtype())));
        return mlx::core::reshape(std::move(result), std::move(output_shape));
    }
    const int tile_rows = rows <= 6 ? static_cast<int>(rows) : 8;
    const auto row_tiles = (rows + tile_rows - 1) / tile_rows;
    const int k_lanes = has_uniform_q() && (descriptor_.q_mask & 7u) != 0 ? 16 : 32;
    const int outputs_per_simd = 32 / k_lanes;
    const auto workgroups = row_tiles * ((static_cast<std::size_t>(output_size())
        + outputs_per_simd - 1) / outputs_per_simd);
    auto arguments = templates(*this, source.dtype());
    arguments.emplace_back("K_LANES", k_lanes);
    arguments.emplace_back("TILE_M", tile_rows);
    arguments.emplace_back("ROUTED", 0);
    arguments.emplace_back("ROUTES", 1);
    arguments.emplace_back("SHARED_INPUT", 0);
    arguments.emplace_back("EXPERT_MAP_SIZE", 1);
    arguments.emplace_back("OUT_PER_EXPERT", output_size());
    arguments.emplace_back("LOGICAL_OUT", output_size());
    const auto unused = mlx::core::zeros(Shape{1}, mlx::core::int32);
    auto outputs = matmul_kernel(*this)(
        {blob_, row_q_, row_symbol_byte_offsets_, source, unused, unused, palette_values_, palette_quads_, palette_pairs_},
        {Shape{static_cast<int>(rows), output_size()}},
        {source.dtype()},
        {checked_int(workgroups * 32, "matmul grid"), 1, 1},
        {32, 1, 1},
        std::move(arguments),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(std::move(outputs.front()), std::move(output_shape));
}

std::vector<array> MlxFp8SqWeight::projection_group_matmul(
    std::span<const MlxFp8SqWeight> weights,
    const array& input) {
    if (weights.size() < 2 || weights.size() > 10) {
        throw std::invalid_argument(
            "FP8-SQ projection group requires two through ten weights");
    }
    const int input_size = weights.front().input_size();
    const auto& dtype = weights.front().dtype();
    if (input.ndim() == 0 || input.shape(-1) != input_size) {
        throw std::invalid_argument(
            "FP8-SQ projection group input width mismatch");
    }
    if (input.dtype() != mlx::core::float16 &&
        input.dtype() != mlx::core::float32) {
        throw std::invalid_argument(
            "FP8-SQ projection group requires FP16 or FP32 input");
    }

    std::vector<int> output_sizes;
    output_sizes.reserve(weights.size());
    int total_output = 0;
    for (const auto& weight : weights) {
        if (weight.input_size() != input_size || weight.dtype() != dtype) {
            throw std::invalid_argument(
                "FP8-SQ projection group requires one width and scale family");
        }
        total_output = checked_int(
            static_cast<std::size_t>(total_output) +
                static_cast<std::size_t>(weight.output_size()),
            "projection-group output width");
        output_sizes.push_back(weight.output_size());
    }

    const auto rows = input.size() / static_cast<std::size_t>(input_size);
    if (rows < 1 || rows > 16) {
        throw std::invalid_argument(
            "FP8-SQ projection group supports one through sixteen rows");
    }
    Shape prefix(input.shape().begin(), input.shape().end() - 1);
    auto source = mlx::core::contiguous(mlx::core::reshape(
        input,
        Shape{checked_int(rows, "projection-group row count"), input_size}));
    const int tile_rows = rows <= 6 ? static_cast<int>(rows) : 8;
    const auto row_tiles = (rows + static_cast<std::size_t>(tile_rows) - 1) /
        static_cast<std::size_t>(tile_rows);

    std::vector<array> inputs;
    inputs.reserve(weights.size() * 6 + 1);
    const int k_lanes = std::all_of(weights.begin(), weights.end(), [](const auto& weight) {
        return weight.has_uniform_q() && (weight.descriptor().q_mask & 7u) != 0;
    }) ? 16 : 32;
    const int outputs_per_simd = 32 / k_lanes;
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> arguments{
        {"T", source.dtype()},
        {"K_LANES", k_lanes},
        {"TILE_M", tile_rows},
        {"K", input_size},
        {"TOTAL_OUT", total_output},
    };
    int output_offset = 0;
    for (std::size_t projection = 0;
         projection < weights.size();
         ++projection) {
        const auto& weight = weights[projection];
        const auto& layout = weight.layout_;
        const auto prefix_name = "P" + std::to_string(projection) + "_";
        inputs.push_back(weight.blob_);
        inputs.push_back(weight.row_q_);
        inputs.push_back(weight.row_symbol_byte_offsets_);
        inputs.push_back(weight.palette_values_);
        inputs.push_back(weight.palette_quads_);
        inputs.push_back(weight.palette_pairs_);
        arguments.emplace_back(prefix_name + "HAS_LOW_Q", (weight.descriptor().q_mask & 6u) != 0);
        arguments.emplace_back(prefix_name + "FAST_Q_MAX",
            static_cast<int>(maximum_fast_q(weight.descriptor().q_mask, true)));
        arguments.emplace_back(prefix_name + "OFFSET", output_offset);
        output_offset += weight.output_size();
        arguments.emplace_back(prefix_name + "END", output_offset);
        arguments.emplace_back(
            prefix_name + "PALETTES_OFFSET",
            checked_int(layout.palettes, "palette offset"));
        arguments.emplace_back(
            prefix_name + "SYMBOLS_OFFSET",
            checked_int(layout.symbols, "symbol offset"));
        arguments.emplace_back(
            prefix_name + "SCALES_OFFSET",
            checked_int(layout.scales, "scale offset"));
        arguments.emplace_back(
            prefix_name + "SCALE_KIND",
            static_cast<int>(layout.scale_kind));
        arguments.emplace_back(prefix_name + "BLOCK_ROWS", layout.block_rows);
        arguments.emplace_back(
            prefix_name + "BLOCK_COLUMNS", layout.block_columns);
        arguments.emplace_back(
            prefix_name + "SCALE_COLUMNS", layout.scale_columns);
    }
    inputs.push_back(source);
    const auto workgroups = row_tiles * ((static_cast<std::size_t>(total_output)
        + outputs_per_simd - 1) / outputs_per_simd);
    auto combined = projection_group_kernel(dtype, weights.size())(
        std::move(inputs),
        {Shape{checked_int(rows, "projection-group row count"), total_output}},
        {source.dtype()},
        {checked_int(workgroups * 32, "projection-group Metal grid"), 1, 1},
        {32, 1, 1},
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

array MlxFp8SqWeight::routed_matmul(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    int out_per_expert) const {
    if (out_per_expert <= 0 || output_size() % out_per_expert != 0) {
        throw std::invalid_argument("FP8-SQ routed output width is inconsistent");
    }
    if (expert_ids.ndim() != 2 || expert_map.ndim() != 1 ||
        expert_ids.dtype() != mlx::core::int32 ||
        expert_map.dtype() != mlx::core::int32) {
        throw std::invalid_argument("FP8-SQ routing metadata must be int32");
    }
    const int tokens = expert_ids.shape(0);
    const int routes = expert_ids.shape(1);
    const bool shared_input = input.ndim() == 2 &&
        input.shape(0) == tokens && input.shape(1) == input_size();
    if (!shared_input && (input.ndim() != 3 || input.shape(0) != tokens ||
        input.shape(1) != routes || input.shape(2) != input_size())) {
        throw std::invalid_argument(
            "FP8-SQ routed input must be [tokens,K] or [tokens,routes,K]");
    }
    const auto route_count = static_cast<std::size_t>(tokens) * routes;
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
    const int k_lanes = has_uniform_q() && (descriptor_.q_mask & 7u) != 0 ? 16 : 32;
    const int outputs_per_simd = 32 / k_lanes;
    auto arguments = templates(*this, source.dtype());
    arguments.emplace_back("K_LANES", k_lanes);
    arguments.emplace_back("TILE_M", 1);
    arguments.emplace_back("ROUTED", 1);
    arguments.emplace_back("ROUTES", routes);
    arguments.emplace_back("SHARED_INPUT", static_cast<int>(shared_input));
    arguments.emplace_back("EXPERT_MAP_SIZE", expert_map.shape(0));
    arguments.emplace_back("OUT_PER_EXPERT", out_per_expert);
    arguments.emplace_back("LOGICAL_OUT", out_per_expert);
    const auto workgroups = route_count * ((static_cast<std::size_t>(out_per_expert)
        + outputs_per_simd - 1) / outputs_per_simd);
    auto outputs = matmul_kernel(*this)(
        {blob_, row_q_, row_symbol_byte_offsets_, source, ids, map, palette_values_, palette_quads_, palette_pairs_},
        {Shape{checked_int(route_count, "route count"), out_per_expert}},
        {mlx::core::float16},
        {checked_int(workgroups * 32, "routed matmul grid"), 1, 1},
        {32, 1, 1},
        std::move(arguments),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(std::move(outputs.front()), output_shape);
}

array MlxFp8SqWeight::backward_input(const array& output_gradient) const {
    if (output_gradient.ndim() == 0 ||
        output_gradient.shape(-1) != output_size()) {
        throw std::runtime_error(
            "FP8-SQ output-gradient width does not match packed weight");
    }
    Shape output_shape = output_gradient.shape();
    output_shape.back() = input_size();
    auto source = output_gradient;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    const auto rows = source.size() / static_cast<std::size_t>(output_size());
    if (rows == 0 || rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("unsupported FP8-SQ backward row count");
    }
    source = mlx::core::reshape(
        source, Shape{checked_int(rows, "backward row count"), output_size()});
    if (rows <= 8 && source.dtype() == mlx::core::float16) {
        auto arguments = templates(*this, source.dtype());
        arguments.emplace_back("M", static_cast<int>(rows));
        const auto column_blocks =
            (static_cast<std::size_t>(input_size()) + 31) / 32;
        auto outputs = backward_kernel(*this)(
            {blob_, row_q_, row_symbol_byte_offsets_, source},
            {Shape{static_cast<int>(rows), input_size()}},
            {source.dtype()},
            {checked_int(column_blocks * 128, "backward grid"),
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
    auto result = mlx::core::matmul(source, dequantize(source.dtype()));
    return mlx::core::reshape(std::move(result), std::move(output_shape));
}

} // namespace mfq::metal
