#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

#ifndef MFQ_GROUPED_FAMILY_MASK
#define MFQ_GROUPED_FAMILY_MASK 127
#endif

#ifndef MFQ_GROUPED_NINT_GROUP_SIZE
#define MFQ_GROUPED_NINT_GROUP_SIZE 0
#endif

#ifndef MFQ_GROUPED_MMQ_BM
#define MFQ_GROUPED_MMQ_BM 32
#endif

#ifndef MFQ_GROUPED_NAX_BK
#define MFQ_GROUPED_NAX_BK 96
#endif

#ifndef MFQ_GROUPED_NAX_ALIGNED_INPUT
#define MFQ_GROUPED_NAX_ALIGNED_INPUT 0
#endif

#ifndef MFQ_GROUPED_NAX_SIMD_ROWS
#define MFQ_GROUPED_NAX_SIMD_ROWS 16
#endif

#ifndef MFQ_GROUPED_NAX_INPUT_WIDTH
#define MFQ_GROUPED_NAX_INPUT_WIDTH 0
#endif

#ifndef MFQ_GROUPED_NAX_OUTPUT_WIDTH
#define MFQ_GROUPED_NAX_OUTPUT_WIDTH 0
#endif

#ifndef MFQ_GROUPED_NAX_MATRIX_OUTPUT_WIDTH
#define MFQ_GROUPED_NAX_MATRIX_OUTPUT_WIDTH 0
#endif

#if defined(MFQ_ENABLE_DSV4_MXFP4_BLOCKS)
// The block geometry and quantized Steel loader below are adapted from the
// Apache-2.0 oMLX DeepSeek-V4 pair projection kernel (Apple Inc., 2026).
// MFQ adds independent physical-slot maps so raw-HF virtual expert pools do
// not need to be concatenated or copied into logical expert order.
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool Paired>
[[kernel]] void mfq_dsv4_mxfp4_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w0 [[buffer(1)]],
    const device uint8_t* scales0 [[buffer(2)]],
    const device int32_t* slots0 [[buffer(3)]],
    const device uint32_t* w1 [[buffer(4)]],
    const device uint8_t* scales1 [[buffer(5)]],
    const device int32_t* slots1 [[buffer(6)]],
    const device int32_t* block_meta [[buffer(7)]],
    const device int32_t* block_count [[buffer(8)]],
    device T* y [[buffer(9)]],
    const constant int& max_blocks [[buffer(10)]],
    const constant int& N [[buffer(11)]],
    const constant int& K [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
    constexpr int group_size = 32;
    constexpr int bits = 4;
    constexpr int pack_factor = get_pack_factor<8, bits>();
    constexpr int bytes_per_pack = get_bytes_per_pack();
    constexpr int BK_padded = BK + 16 / sizeof(T);

    using mma_t = mlx::steel::BlockMMA<
        T,
        T,
        BM,
        BN,
        BK,
        WM,
        WN,
        false,
        true,
        BK_padded,
        BK_padded>;
    using loader_x_t = mlx::steel::BlockLoader<
        T,
        BM,
        BK,
        BK_padded,
        1,
        WM * WN * SIMD_SIZE>;
    using loader_w_t = QuantizedBlockLoader<
        T,
        BN,
        BK,
        BK_padded,
        true,
        WM * WN * SIMD_SIZE,
        group_size,
        bits>;

    threadgroup T Xs[BM * BK_padded];
    threadgroup T Ws[BN * BK_padded];

    const int block_id = int(tid.y);
    const int nblocks = block_count[0];
    if (block_id >= max_blocks || block_id >= nblocks) {
        return;
    }

    const int concatenated_column = int(tid.x) * BN;
    constexpr int projections = Paired ? 2 : 1;
    if (concatenated_column >= projections * N) {
        return;
    }
    const int projection = Paired && concatenated_column >= N ? 1 : 0;
    const int output_column = projection == 0
        ? concatenated_column
        : concatenated_column - N;

    const int row_start = block_meta[block_id * 3 + 0];
    const int logical_expert = block_meta[block_id * 3 + 1];
    const int rows = block_meta[block_id * 3 + 2];
    if (rows <= 0 || logical_expert < 0) {
        return;
    }
    const int physical_slot = projection == 0
        ? slots0[logical_expert]
        : slots1[logical_expert];
    if (physical_slot < 0) {
        return;
    }

    const short valid_m = short(min(BM, rows));
    const short valid_n = short(min(BN, N - output_column));
    const int iterations = K / BK;
    const int remainder = K - iterations * BK;
    const short2 x_tail = short2(remainder, valid_m);
    const short2 w_tail = short2(remainder, valid_n);

    const int packed_row = K * bytes_per_pack / pack_factor;
    const int scale_row = K / group_size;
    const size_t packed_expert_stride = size_t(N) * packed_row;
    const size_t scale_expert_stride = size_t(N) * scale_row;
    const device uint32_t* selected_w = projection == 0 ? w0 : w1;
    const device uint8_t* selected_scales =
        projection == 0 ? scales0 : scales1;

    const device T* x_base = x + size_t(row_start) * K;
    device T* y_base = y + size_t(row_start) * (projections * N)
        + size_t(projection) * N + output_column;
    const device uint8_t* w_base =
        reinterpret_cast<const device uint8_t*>(selected_w)
        + size_t(physical_slot) * packed_expert_stride
        + size_t(output_column) * packed_row;
    const device uint8_t* scale_base = selected_scales
        + size_t(physical_slot) * scale_expert_stride
        + size_t(output_column) * scale_row;

    thread mma_t mma(simd_group_id, simd_lane_id);
    thread loader_x_t load_x(
        x_base,
        K,
        Xs,
        simd_group_id,
        simd_lane_id);
    thread loader_w_t load_w(
        w_base,
        scale_base,
        K,
        Ws,
        simd_group_id,
        simd_lane_id);

    if (rows == BM && valid_n == BN) {
        gemm_loop_aligned(Xs, Ws, mma, load_x, load_w, iterations);
        if (remainder != 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gemm_loop_finalize(
                Xs, Ws, mma, load_x, load_w, x_tail, w_tail);
        }
        mma.store_result(y_base, projections * N);
    } else if (valid_n == BN) {
        gemm_loop_unaligned<false, true, true>(
            Xs,
            Ws,
            mma,
            load_x,
            load_w,
            iterations,
            valid_m,
            valid_n,
            BK);
        if (remainder != 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gemm_loop_finalize(
                Xs, Ws, mma, load_x, load_w, x_tail, w_tail);
        }
        mma.store_result_slice(
            y_base,
            projections * N,
            short2(0, 0),
            short2(BN, valid_m));
    } else if (rows == BM) {
        gemm_loop_unaligned<true, false, true>(
            Xs,
            Ws,
            mma,
            load_x,
            load_w,
            iterations,
            valid_m,
            valid_n,
            BK);
        if (remainder != 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gemm_loop_finalize(
                Xs, Ws, mma, load_x, load_w, x_tail, w_tail);
        }
        mma.store_result_slice(
            y_base,
            projections * N,
            short2(0, 0),
            short2(valid_n, BM));
    } else {
        gemm_loop_unaligned<false, false, true>(
            Xs,
            Ws,
            mma,
            load_x,
            load_w,
            iterations,
            valid_m,
            valid_n,
            BK);
        if (remainder != 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gemm_loop_finalize(
                Xs, Ws, mma, load_x, load_w, x_tail, w_tail);
        }
        mma.store_result_slice(
            y_base,
            projections * N,
            short2(0, 0),
            short2(valid_n, valid_m));
    }
}

#define instantiate_mfq_dsv4_mxfp4_blocks(name, bm, paired) \
    template [[host_name(name)]] [[kernel]] \
    decltype(mfq_dsv4_mxfp4_blocks_rhs< \
        half, bm, 32, 32, 1, 2, paired>) \
    mfq_dsv4_mxfp4_blocks_rhs< \
        half, bm, 32, 32, 1, 2, paired>;

instantiate_mfq_dsv4_mxfp4_blocks(
    "mfq_dsv4_mxfp4_single_f16_bm32_bn32_bk32",
    32,
    false)

instantiate_mfq_dsv4_mxfp4_blocks(
    "mfq_dsv4_mxfp4_pair_concat_f16_bm32_bn32_bk32",
    32,
    true)
#endif

namespace {

inline uint2 vq_bit_cursor(uint value_index, uint bits) {
    uint residual = (value_index & 7u) * bits;
    return uint2((value_index >> 3u) * bits + (residual >> 3u), residual & 7u);
}

inline uint read_bits(
    const device uchar* stream,
    uint value_index,
    uint bits) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3u) * bits + (residual_bits >> 3u);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8u;
    }
    if (shift + bits > 16u) {
        packed |= uint(stream[byte_index + 2u]) << 16u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

#if defined(MFQ_ENABLE_LEGACY_VQ_VECTOR) \
    || defined(MFQ_ENABLE_JSC_EXTENDED_VECTOR)
template <uint BITS>
inline uint3 read_vq_group_indices(
    const device uchar* stream,
    uint row,
    uint group,
    uint vectors) {
    uint first = group * 3u;
    if (first + 2u >= vectors) {
        return uint3(
            first < vectors
                ? read_bits(stream, row * vectors + first, BITS)
                : 0u,
            first + 1u < vectors
                ? read_bits(stream, row * vectors + first + 1u, BITS)
                : 0u,
            0u);
    }
    uint2 cursor = vq_bit_cursor(row * vectors + first, BITS);
    uint byte = cursor.x;
    uint shift = cursor.y;
    ulong packed = ulong(stream[byte])
        | (ulong(stream[byte + 1u]) << 8u)
        | (ulong(stream[byte + 2u]) << 16u);
    if (shift + 3u * BITS > 24u) {
        packed |= ulong(stream[byte + 3u]) << 24u;
    }
    if (shift + 3u * BITS > 32u) {
        packed |= ulong(stream[byte + 4u]) << 32u;
    }
    ulong mask = (1ul << BITS) - 1ul;
    return uint3(
        uint((packed >> shift) & mask),
        uint((packed >> (shift + BITS)) & mask),
        uint((packed >> (shift + 2u * BITS)) & mask));
}
#endif

inline uint read_nint_row_value(
    const device uchar* stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    uint shift = row_relative_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

template <uint FIXED_BITS = 0u>
inline ushort4 read_nint_row_quad(
    const device uchar* stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint row_bits) {
    uint bits = FIXED_BITS != 0u ? FIXED_BITS : row_bits;
    uint row_relative_bits = row_bit_shift + value_index * bits;
    uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    uint shift = row_relative_bits & 7u;
    uint required_bits = shift + 4u * bits;
    packed_uchar4 bytes = *reinterpret_cast<device const packed_uchar4*>(
        stream + byte_index);
    uint packed = as_type<uint>(bytes);
    if (shift != 0u) {
        packed = (packed >> shift)
            | (required_bits > 32u
                ? uint(stream[byte_index + 4u]) << (32u - shift)
                : 0u);
    }
    uint mask = (1u << bits) - 1u;
    return ushort4(
        packed & mask,
        (packed >> bits) & mask,
        (packed >> (2u * bits)) & mask,
        (packed >> (3u * bits)) & mask);
}

inline float decode_mxfp4_value(uchar raw) {
    uchar magnitude = raw & 7u;
    float value = magnitude == 0u ? 0.0f
        : (magnitude == 1u ? 0.5f
        : (magnitude == 2u ? 1.0f
        : (magnitude == 3u ? 1.5f
        : (magnitude == 4u ? 2.0f
        : (magnitude == 5u ? 3.0f
        : (magnitude == 6u ? 4.0f : 6.0f))))));
    return (raw & 8u) == 0u ? value : -value;
}

constant float kMxfp4DecodeLut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

inline float decode_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

inline float decode_mxfp8_value(uchar raw) {
    uint magnitude = uint(raw & 0x7fu);
    uint exponent = magnitude >> 3u;
    uint mantissa = magnitude & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return NAN;
    }
    float value = exponent == 0u
        ? float(mantissa) * 0.001953125f
        : as_type<float>((exponent + 120u) << 23u)
            * (1.0f + float(mantissa) * 0.125f);
    return (raw & 0x80u) == 0u ? value : -value;
}

inline void decode_mxfp4_group32(
    const device int* d,
    const device uchar* values,
    const device uchar* scales,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    uint groups = uint(d[4]);
    uint value_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    float scale = decode_e8m0(
        scales[scale_offset + row * groups + group]);
    uint packed_row = value_offset + row * (k_size >> 1u);
    uint packed_column = group * 16u;
    // Four vector loads replace sixteen scalar loads.  The constant E2M1
    // table is exact, so this preserves the previous FP32 multiply -> FP16
    // conversion while avoiding the nested magnitude branch per nibble.
#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 4u; ++chunk) {
        uchar4 packed = *reinterpret_cast<device const uchar4*>(
            values + packed_row + packed_column + chunk * 4u);
        half4 first = half4(
            scale * kMxfp4DecodeLut[packed.x & 15u],
            scale * kMxfp4DecodeLut[packed.x >> 4u],
            scale * kMxfp4DecodeLut[packed.y & 15u],
            scale * kMxfp4DecodeLut[packed.y >> 4u]);
        half4 second = half4(
            scale * kMxfp4DecodeLut[packed.z & 15u],
            scale * kMxfp4DecodeLut[packed.z >> 4u],
            scale * kMxfp4DecodeLut[packed.w & 15u],
            scale * kMxfp4DecodeLut[packed.w >> 4u]);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = first;
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = second;
    }
}

inline half decode_mxfp8_value_at(
    const device int* d,
    const device uchar* values,
    const device uchar* scales,
    uint row,
    uint column,
    uint k_size) {
    uint groups = uint(d[4]);
    uint value_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    uint group = column >> 7u;
    float scale = decode_e8m0(
        scales[scale_offset + (row >> 7u) * groups + group]);
    return half(scale * decode_mxfp8_value(
        values[value_offset + row * k_size + column]));
}

inline half4 decode_mxfp8_quad_at(
    const device int* d,
    const device uchar* values,
    const device uchar* scales,
    uint row,
    uint column,
    uint k_size) {
    uint groups = uint(d[4]);
    uint value_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    uint group = column >> 7u;
    float scale = decode_e8m0(
        scales[scale_offset + (row >> 7u) * groups + group]);
    uchar4 packed = *reinterpret_cast<device const uchar4*>(
        values + value_offset + row * k_size + column);
    float4 decoded = float4(
        decode_mxfp8_value(packed.x),
        decode_mxfp8_value(packed.y),
        decode_mxfp8_value(packed.z),
        decode_mxfp8_value(packed.w));
    return half4(scale * decoded);
}

inline half decode_dense_value_at(
    const device int* d,
    const device int8_t* values,
    uint family,
    uint row,
    uint column,
    uint k_size) {
    ulong byte_offset = ulong(uint(d[4]))
        + (ulong(row) * ulong(k_size) + ulong(column)) * 2u;
    ushort raw = *reinterpret_cast<device const ushort*>(
        values + byte_offset);
    return family == 5u
        ? half(as_type<float>(uint(raw) << 16u))
        : as_type<half>(raw);
}

inline half4 decode_dense_quad_at(
    const device int* d,
    const device int8_t* values,
    uint family,
    uint row,
    uint column,
    uint k_size) {
    ulong byte_offset = ulong(uint(d[4]))
        + (ulong(row) * ulong(k_size) + ulong(column)) * 2u;
    if (family == 5u) {
        ushort4 raw = *reinterpret_cast<device const ushort4*>(
            values + byte_offset);
        return half4(as_type<float4>(uint4(raw) << 16u));
    }
    return *reinterpret_cast<device const half4*>(values + byte_offset);
}

inline half4 decode_nint_row_quad_at(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    uint row,
    uint column,
    uint k_size) {
    uint group_size = MFQ_GROUPED_NINT_GROUP_SIZE > 0
        ? uint(MFQ_GROUPED_NINT_GROUP_SIZE) : uint(d[5]);
    uint q_offset = uint(d[7]);
    uint sub_offset = uint(d[8]);
    const device uint* row_metadata =
        reinterpret_cast<const device uint*>(
            values + uint(d[11]));
    const MfqNintRow metadata = mfq_nint_row(row_metadata, row);
    ushort4 quantized = read_nint_row_quad(
        values + q_offset,
        metadata.q_offset,
        metadata.q_shift,
        column,
        metadata.q_bits);
    const uint first_group = column / group_size;
    if (column + 3u < k_size
        && first_group == (column + 3u) / group_size) {
        const float scale = metadata.scale
            * float(mfq_nint_sub_value(sub_scales,
                sub_offset + metadata.sub_offset, metadata.sub_shift,
                metadata.sub_bits, first_group));
        const float minimum = metadata.minimum
            * float(mfq_nint_sub_value(sub_mins,
                sub_offset + metadata.sub_offset, metadata.sub_shift,
                metadata.sub_bits, first_group));
        return half4(scale * float4(quantized) - minimum);
    }
    half4 decoded = half4(0.0h);
#pragma clang loop unroll(full)
    for (uint lane = 0u; lane < 4u; ++lane) {
        uint input_column = column + lane;
        if (input_column < k_size) {
            const uint group = input_column / group_size;
            float scale = metadata.scale
                * float(mfq_nint_sub_value(sub_scales,
                    sub_offset + metadata.sub_offset, metadata.sub_shift,
                    metadata.sub_bits, group));
            float minimum = metadata.minimum
                * float(mfq_nint_sub_value(sub_mins,
                    sub_offset + metadata.sub_offset, metadata.sub_shift,
                    metadata.sub_bits, group));
            decoded[lane] = half(
                scale * float(quantized[lane]) - minimum);
        }
    }
    return decoded;
}

inline void decode_nint_row_pair_at(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint column,
    uint k_size) {
    uint group_size = MFQ_GROUPED_NINT_GROUP_SIZE > 0
        ? uint(MFQ_GROUPED_NINT_GROUP_SIZE) : uint(d[5]);
    uint first_group = column / group_size;
    if (column + 7u >= k_size
        || first_group != (column + 7u) / group_size) {
        *reinterpret_cast<threadgroup half4*>(target) = decode_nint_row_quad_at(
            d, values, sub_scales, sub_mins, anchor_scales, anchor_mins,
            row, column, k_size);
        *reinterpret_cast<threadgroup half4*>(target + 4u) = column + 4u < k_size
            ? decode_nint_row_quad_at(
                d, values, sub_scales, sub_mins, anchor_scales, anchor_mins,
                row, column + 4u, k_size)
            : half4(0.0h);
        return;
    }
    const MfqNintRow metadata = mfq_nint_row(
        reinterpret_cast<const device uint*>(values + uint(d[11])), row);
    uint offset = metadata.q_offset;
    uint bits = metadata.q_bits;
    uint shift = metadata.q_shift;
    const device uchar* stream = values + uint(d[7]);
    ushort4 first = read_nint_row_quad(stream, offset, shift, column, bits);
    ushort4 second = read_nint_row_quad(stream, offset, shift, column + 4u, bits);
    float scale = metadata.scale
        * float(mfq_nint_sub_value(sub_scales,
            uint(d[8]) + metadata.sub_offset, metadata.sub_shift,
            metadata.sub_bits, first_group));
    float minimum = metadata.minimum
        * float(mfq_nint_sub_value(sub_mins,
            uint(d[8]) + metadata.sub_offset, metadata.sub_shift,
            metadata.sub_bits, first_group));
    *reinterpret_cast<threadgroup half4*>(target) = half4(scale * float4(first) - minimum);
    *reinterpret_cast<threadgroup half4*>(target + 4u) = half4(scale * float4(second) - minimum);
}

template <uint GROUP_SIZE, uint TILE_K, uint GROUP_LANES = 1u>
inline void decode_nint_group_tile_at(
    const device int* d,
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    const device float* anchor_scales,
    const device float* anchor_mins,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_base,
    uint k_size,
    uint group_lane = 0u) {
    constexpr uint QUADS_PER_LANE = (GROUP_SIZE / 4u + GROUP_LANES - 1u) / GROUP_LANES;
    const MfqNintRow metadata = mfq_nint_row(
        reinterpret_cast<const device uint*>(values + uint(d[11])), row);
    uint offset = metadata.q_offset;
    uint bits = metadata.q_bits;
    uint shift = metadata.q_shift;
    const device uchar* stream = values + uint(d[7]);
    float scale = metadata.scale
        * float(mfq_nint_sub_value(sub_scales,
            uint(d[8]) + metadata.sub_offset, metadata.sub_shift,
            metadata.sub_bits, group));
    float minimum = metadata.minimum
        * float(mfq_nint_sub_value(sub_mins,
            uint(d[8]) + metadata.sub_offset, metadata.sub_shift,
            metadata.sub_bits, group));
    if constexpr (TILE_K % GROUP_SIZE == 0u) {
        if ((group + 1u) * GROUP_SIZE <= k_size) {
            constexpr uint COMMON_BITS = GROUP_SIZE == 24u ? 6u : 5u;
            if (simd_all(bits == COMMON_BITS)) {
#pragma clang loop unroll(full)
                for (uint quad = 0u; quad < QUADS_PER_LANE; ++quad) {
                    uint inner = (group_lane * QUADS_PER_LANE + quad) * 4u;
                    if (inner >= GROUP_SIZE) continue;
                    uint column = group * GROUP_SIZE + inner;
                    ushort4 quantized = read_nint_row_quad<COMMON_BITS>(
                        stream, offset, shift, column, bits);
                    *reinterpret_cast<threadgroup half4*>(target + column - k_base)
                        = half4(scale * float4(quantized) - minimum);
                }
            } else {
#pragma clang loop unroll(full)
                for (uint quad = 0u; quad < QUADS_PER_LANE; ++quad) {
                    uint inner = (group_lane * QUADS_PER_LANE + quad) * 4u;
                    if (inner >= GROUP_SIZE) continue;
                    uint column = group * GROUP_SIZE + inner;
                    ushort4 quantized = read_nint_row_quad(
                        stream, offset, shift, column, bits);
                    *reinterpret_cast<threadgroup half4*>(target + column - k_base)
                        = half4(scale * float4(quantized) - minimum);
                }
            }
            return;
        }
    }
#pragma clang loop unroll(full)
    for (uint quad = 0u; quad < QUADS_PER_LANE; ++quad) {
        uint inner = (group_lane * QUADS_PER_LANE + quad) * 4u;
        if (inner >= GROUP_SIZE) continue;
        uint column = group * GROUP_SIZE + inner;
        int local_column = int(column) - int(k_base);
        if ((TILE_K % GROUP_SIZE == 0u
                || (local_column >= 0 && local_column + 4 <= int(TILE_K)))
            && column < k_size) {
            ushort4 quantized = read_nint_row_quad(stream, offset, shift, column, bits);
            half4 decoded = half4(scale * float4(quantized) - minimum);
            if (column + 4u > k_size) {
#pragma clang loop unroll(full)
                for (uint lane = 0u; lane < 4u; ++lane) {
                    if (column + lane >= k_size) decoded[lane] = half(0.0h);
                }
            }
            *reinterpret_cast<threadgroup half4*>(target + local_column) = decoded;
        }
    }
}

inline void decode_nint8_zero_group32(
    const device int* d,
    const device int8_t* values,
    const device half* scales,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[4]);
    uint q_offset = uint(d[5]);
    uint scale_offset = uint(d[6]);
    float scale = float(scales[scale_offset + row * groups + group]);
    uint value_base = q_offset + (row * groups + group) * 32u;
#pragma clang loop unroll(full)
    for (uint column = 0u; column < 32u; column += 4u) {
        char4 packed = *reinterpret_cast<device const char4*>(
            values + value_base + column);
        *reinterpret_cast<threadgroup half4*>(target + column) =
            half4(scale * float4(packed));
    }
}

// Layout 3 stores one or two wide indices followed by the parity-completed
// eight-bit sign mask for each eight weights. Decode uses the same record.
inline uint3 read_jsc_wide_record(
    const device uchar* indices,
    uint offset,
    uint sign_index,
    uint vector_size,
    uint index_bits) {
    uint index_count = vector_size == 4u ? 2u : 1u;
    uint bytes = index_count == 2u ? 4u : 3u;
    uint record_offset = offset + sign_index * bytes;
    uint record = uint(indices[record_offset])
        | (uint(indices[record_offset + 1u]) << 8u)
        | (uint(indices[record_offset + 2u]) << 16u);
    if (index_count == 2u) {
        record |= uint(indices[record_offset + 3u]) << 24u;
    }
    uint mask = (1u << index_bits) - 1u;
    return uint3(
        record & mask,
        index_count == 2u ? (record >> index_bits) & mask : 0u,
        record >> (index_count * index_bits));
}

template <uint JSC_VECTOR, uint INDEX_BITS>
inline void decode_jsc_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device uchar* state_to_bank,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    constexpr bool DUAL_INDEX = JSC_VECTOR == 4u;
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state_bank_offset = uint(d[24]);
    uint state_index = row * groups + group;
    uint signs = (k_size + 7u) / 8u;
    uint packed_state =
        uint(state_stream[state_offset + (state_index >> 1u)]);
    uint state =
        (packed_state >> ((state_index & 1u) * 4u)) & 15u;
    uint selected_bank = uint(state_to_bank[state_bank_offset + state]);
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];
    uint sign_base = row * signs + group * 3u;
    uint vector_base = row * vectors + group * (24u / JSC_VECTOR);
    uint2 packed_indices;
    uint packed_signs;
    uint2 cursor = vq_bit_cursor(vector_base, INDEX_BITS);
    uint offset = indices_offset + cursor.x;
    uint low = as_type<uint>(*reinterpret_cast<device const packed_uchar4*>(indices + offset));
    uint high = 0u;
    if constexpr ((24u / JSC_VECTOR) * INDEX_BITS + (INDEX_BITS == 8u ? 0u : 7u) > 32u)
        high = as_type<uint>(*reinterpret_cast<device const packed_uchar4*>(indices + offset + 4u));
    uint shift = cursor.y;
    uint upper = 0u;
    if constexpr (JSC_VECTOR == 4u && INDEX_BITS == 10u)
        upper = uint(indices[offset + 8u]);
    packed_indices = uint2((low >> shift) | (shift != 0u ? high << (32u - shift) : 0u),
        (high >> shift) | (shift != 0u ? upper << (32u - shift) : 0u));
    uint2 sign_cursor = vq_bit_cursor(sign_base, 7u);
    packed_signs = as_type<uint>(*reinterpret_cast<device const packed_uchar4*>(
        aux + aux_offset + sign_cursor.x)) >> sign_cursor.y;

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        if (group * 24u + chunk * 8u >= k_size) {
            *reinterpret_cast<threadgroup half4*>(target + chunk * 8u) = half4(0.0h);
            *reinterpret_cast<threadgroup half4*>(target + chunk * 8u + 4u) = half4(0.0h);
            continue;
        }
        uint index_shift = chunk * (8u / JSC_VECTOR) * INDEX_BITS;
        uint record = index_shift == 0u ? packed_indices.x : index_shift < 32u
            ? (packed_indices.x >> index_shift) | (packed_indices.y << (32u - index_shift))
            : packed_indices.y >> (index_shift - 32u);
        uint index0 = record & ((1u << INDEX_BITS) - 1u);
        uint index1 = DUAL_INDEX ? (record >> INDEX_BITS) & ((1u << INDEX_BITS) - 1u) : 0u;
        uint sign_bits = (packed_signs >> (chunk * 7u)) & 127u;
        uint parity = (popcount(sign_bits) & 1u)
            ^ (d[13] == 2 ? (index0 >> 7u) & 1u : 0u);
        sign_bits |= parity << 7u;
        uint first_code_base = codebook_offset
            + (selected_bank * (1u << INDEX_BITS) + index0) * JSC_VECTOR;
        uint second_code_base = 0u;
        if constexpr (DUAL_INDEX) {
            second_code_base = codebook_offset
                + (selected_bank * (1u << INDEX_BITS) + index1) * JSC_VECTOR;
        } else {
            second_code_base = first_code_base + 4u;
        }
        char4 first_code = *reinterpret_cast<device const char4*>(
            codebooks + first_code_base);
        char4 second_code = *reinterpret_cast<device const char4*>(
            codebooks + second_code_base);
        float4 first_value = float4(first_code);
        float4 second_value = float4(second_code);
        first_value = select(
            first_value,
            -first_value,
            (uint4(sign_bits) & uint4(1u, 2u, 4u, 8u))
                != uint4(0u));
        second_value = select(
            second_value,
            -second_value,
            (uint4(sign_bits) & uint4(16u, 32u, 64u, 128u))
                != uint4(0u));
        if (group * 24u + chunk * 8u + 4u >= k_size) second_value = float4(0.0f);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = half4(scale * first_value);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = half4(scale * second_value);
    }
}

#ifdef MFQ_ENABLE_JSC_EXTENDED_VECTOR
template <uint FIXED_EXECUTION = 0xffffffffu>
inline void decode_jsc_extended_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device uchar* state_to_bank,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint index_bits = uint(d[8]);
    uint entries = uint(d[11]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state_bank_offset = uint(d[24]);
    uint execution = FIXED_EXECUTION != 0xffffffffu
        ? FIXED_EXECUTION : uint(d[29]);
    uint state_index = row * groups + group;
    uint signs = (k_size + 7u) / 8u;
    uint state;
    uint3 group_indices;
    uint3 sign_values;
    if (execution == 2u) {
        ulong record = *reinterpret_cast<device const ulong*>(
            indices + indices_offset + state_index * 8u);
        group_indices = uint3(
            uint(record & 4095ul),
            uint((record >> 20u) & 4095ul),
            uint((record >> 40u) & 4095ul));
        sign_values = uint3(
            uint((record >> 12u) & 255ul),
            uint((record >> 32u) & 255ul),
            uint((record >> 52u) & 255ul));
        state = uint(record >> 60u);
    } else if (execution == 3u) {
        uint sign_base = row * signs + group * 3u;
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint3 record = group * 24u + chunk * 8u < k_size
                ? read_jsc_wide_record(
                      indices, indices_offset, sign_base + chunk, 8u,
                      index_bits)
                : uint3(0u);
            group_indices[chunk] = record.x;
            sign_values[chunk] = record.z;
        }
        state = read_bits(
            state_stream + state_offset, state_index, 4u);
    } else {
        group_indices = index_bits == 10u
            ? read_vq_group_indices<10u>(
                indices + indices_offset, row, group, vectors)
            : read_vq_group_indices<12u>(
                indices + indices_offset, row, group, vectors);
        uint sign_base = row * signs + group * 3u;
        sign_values = uint3(
            read_bits(aux + aux_offset, sign_base, 7u),
            read_bits(aux + aux_offset, sign_base + 1u, 7u),
            read_bits(aux + aux_offset, sign_base + 2u, 7u));
        state = read_bits(
            state_stream + state_offset,
            state_index,
            4u);
    }
    uint selected_bank = uint(
        state_to_bank[state_bank_offset + state]);
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint column_base = group * 24u + chunk * 8u;
        if (column_base >= k_size) {
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u) = half4(0.0h);
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u + 4u) = half4(0.0h);
            continue;
        }
        uint sign_bits = sign_values[chunk];
        if (execution == 0u) {
            sign_bits |= (popcount(sign_bits) & 1u) << 7u;
        }
        uint code_base = codebook_offset
            + (selected_bank * entries + group_indices[chunk]) * 8u;
        float4 first_code = float4(
            *reinterpret_cast<device const char4*>(
                codebooks + code_base));
        float4 second_code = float4(
            *reinterpret_cast<device const char4*>(
                codebooks + code_base + 4u));
        first_code = select(
            first_code,
            -first_code,
            (uint4(sign_bits) & uint4(1u, 2u, 4u, 8u))
                != uint4(0u));
        second_code = select(
            second_code,
            -second_code,
            (uint4(sign_bits) & uint4(16u, 32u, 64u, 128u))
                != uint4(0u));
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = half4(scale * first_code);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = half4(scale * second_code);
    }
}
#endif

#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
template <uint STATE_WIDTH, uint INDEX_WIDTH, uint TABLE_SIZE>
inline void decode_npq_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state = read_bits(
        state_stream + state_offset,
        row * groups + group,
        STATE_WIDTH);
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];
    uint3 group_indices = read_vq_group_indices<INDEX_WIDTH>(
        indices + indices_offset, row, group, vectors);

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint index = group_indices[chunk];
        uint code_base = codebook_offset
            + (state * TABLE_SIZE + index) * 8u;
        char4 first_code = *reinterpret_cast<device const char4*>(
            codebooks + code_base);
        char4 second_code = *reinterpret_cast<device const char4*>(
            codebooks + code_base + 4u);
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = half4(
                scale * float4(first_code));
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = half4(
                scale * float4(second_code));
    }
}

template <
    uint STATE_WIDTH,
    uint INDEX_WIDTH,
    uint ENTRIES_PER_BANK,
    bool DUAL_BANK>
inline void decode_nvq1_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device float* parameters,
    threadgroup half* target,
    uint row,
    uint group) {
    uint groups = uint(d[5]);
    uint vectors = uint(d[7]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint parameter_offset = uint(d[26]);
    uint state_index = row * groups + group;
    uint state = read_bits(state_stream + state_offset, state_index, STATE_WIDTH);
    uint bank = read_bits(aux + aux_offset, state_index, 1u);
    uint2 cursor = vq_bit_cursor(row * vectors + group * 3u, INDEX_WIDTH);
    uint offset = indices_offset + cursor.x;
    uint low = as_type<uint>(*(device const packed_uchar4*)(indices + offset));
    uint high = uint(indices[offset + 4u]);
    uint shift = cursor.y;
    uint first = (low >> shift) | (shift != 0u ? high << (32u - shift) : 0u);
    uint second = high >> shift;
    constexpr uint MASK = (1u << INDEX_WIDTH) - 1u;
    uint3 group_indices(first & MASK, (first >> INDEX_WIDTH) & MASK,
        ((first >> (2u * INDEX_WIDTH)) | (second << (32u - 2u * INDEX_WIDTH))) & MASK);
    float delta = parameters[parameter_offset];
    float signed_delta = bank != 0u ? -delta : delta;
    float scale = anchors[anchor_offset + row]
        * scales[scale_offset + state];
    half negative_value = half(scale * (-1.0f + signed_delta));
    half zero_value = half(scale * (0.0f + signed_delta));
    half positive_value = half(scale * (1.0f + signed_delta));

#pragma clang loop unroll(full)
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        uint entry = group_indices[chunk];
        if constexpr (DUAL_BANK) {
            entry += bank * ENTRIES_PER_BANK;
        }
        uint code_base = codebook_offset + entry * 8u;
        char4 first_code =
            *reinterpret_cast<device const char4*>(
                codebooks + code_base);
        char4 second_code =
            *reinterpret_cast<device const char4*>(
                codebooks + code_base + 4u);
        half4 first_value = select(half4(zero_value), half4(positive_value), first_code > char4(0));
        half4 second_value = select(half4(zero_value), half4(positive_value), second_code > char4(0));
        first_value = select(first_value, half4(negative_value), first_code < char4(0));
        second_value = select(second_value, half4(negative_value), second_code < char4(0));
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u) = first_value;
        *reinterpret_cast<threadgroup half4*>(
            target + chunk * 8u + 4u) = second_value;
    }
}
#endif

template <
    uint FIXED_PROFILE = 0xffffffffu,
    uint FIXED_EXECUTION = 0xffffffffu>
inline void decode_vq_group24(
    const device int* d,
    const device uchar* indices,
    const device uchar* state_stream,
    const device uchar* aux,
    const device float* anchors,
    const device int8_t* codebooks,
    const device float* scales,
    const device uchar* state_to_bank,
    const device uchar* banks,
    const device float* parameters,
    threadgroup half* target,
    uint row,
    uint group,
    uint k_size) {
    uint groups = uint(d[5]);
    uint vector_size = uint(d[6]);
    uint vectors = uint(d[7]);
    uint index_bits = uint(d[8]);
    uint state_bits = uint(d[9]);
    uint state_count = uint(d[10]);
    uint entries = uint(d[11]);
    uint code_banks = uint(d[12]);
    uint aux_mode = uint(d[13]);
    uint code_bank_mode = uint(d[14]);
    uint has_table_banks = uint(d[15]);
    uint groups_per_super = uint(d[16]);
    uint supergroups = uint(d[17]);
    uint indices_offset = uint(d[18]);
    uint state_offset = uint(d[19]);
    uint aux_offset = uint(d[20]);
    uint anchor_offset = uint(d[21]);
    uint codebook_offset = uint(d[22]);
    uint scale_offset = uint(d[23]);
    uint state_bank_offset = uint(d[24]);
    uint bank_offset = uint(d[25]);
    uint parameter_offset = uint(d[26]);
    uint execution = FIXED_EXECUTION != 0xffffffffu
        ? FIXED_EXECUTION : uint(d[29]);
    uint profile = FIXED_PROFILE != 0xffffffffu
        ? FIXED_PROFILE : uint(d[28]);
    uint state_index = row * groups + group;
    uint signs = (k_size + 7u) / 8u;
    float anchor = anchors[anchor_offset + row];

    if (profile == 1u) {
        decode_jsc_group24<4u, 8u>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            state_to_bank, target, row, group, k_size);
        return;
    }
    if (profile == 4u) {
        decode_jsc_group24<8u, 8u>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            state_to_bank, target, row, group, k_size);
        return;
    }
    if ((profile == 7u || profile == 8u) && execution == 0u) {
        #define MFQ_DECODE_BANKED_GROUP(VECTOR, BITS) \
            decode_jsc_group24<VECTOR, BITS>( \
                d, indices, state_stream, aux, anchors, codebooks, scales, \
                state_to_bank, target, row, group, k_size)
        if (profile == 8u) {
            if (index_bits == 9u) { MFQ_DECODE_BANKED_GROUP(4u, 9u); }
            else { MFQ_DECODE_BANKED_GROUP(4u, 10u); }
        } else {
            if (index_bits == 10u) { MFQ_DECODE_BANKED_GROUP(8u, 10u); }
            else { MFQ_DECODE_BANKED_GROUP(8u, 12u); }
        }
        #undef MFQ_DECODE_BANKED_GROUP
        return;
    }
#ifdef MFQ_ENABLE_JSC_EXTENDED_VECTOR
    if (profile == 7u) {
        decode_jsc_extended_group24<FIXED_EXECUTION>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            state_to_bank, target, row, group, k_size);
        return;
    }
#endif

#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
    if (profile == 2u) {
        decode_npq_group24<2u, 6u, 64u>(
            d, indices, state_stream, anchors, codebooks, scales,
            target, row, group);
        return;
    }
    if (profile == 5u) {
        decode_npq_group24<3u, 7u, 128u>(
            d, indices, state_stream, anchors, codebooks, scales,
            target, row, group);
        return;
    }

    if (profile == 3u) {
        decode_nvq1_group24<3u, 11u, 2048u, false>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            parameters, target, row, group);
        return;
    }
    if (profile == 6u) {
        decode_nvq1_group24<4u, 9u, 512u, true>(
            d, indices, state_stream, aux, anchors, codebooks, scales,
            parameters, target, row, group);
        return;
    }
#else
    if (profile == 2u || profile == 5u) {
        uint state_width = profile == 2u ? 2u : 3u;
        uint index_width = profile == 2u ? 6u : 7u;
        uint state = read_bits(
            state_stream + state_offset,
            state_index,
            state_width);
        uint table_size = profile == 2u ? 64u : 128u;
        float scale = anchor * scales[scale_offset + state];
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + group * 3u + chunk,
                index_width);
            uint code_base = codebook_offset
                + (state * table_size + index) * 8u;
            for (uint inner = 0u; inner < 8u; ++inner) {
                target[chunk * 8u + inner] = half(
                    scale * float(codebooks[code_base + inner]));
            }
        }
        return;
    }

    if (profile == 3u) {
        uint state = read_bits(state_stream + state_offset, state_index, state_bits);
        uint sign = read_bits(aux + aux_offset, state_index, 1u);
        float delta = parameters[parameter_offset];
        float scale = anchor * scales[scale_offset + state];
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint index = read_bits(indices + indices_offset,
                    row * vectors + group * 3u + chunk, 11u);
            uint code_base = codebook_offset + index * 8u;
            for (uint inner = 0u; inner < 8u; ++inner) {
                float code = float(codebooks[code_base + inner]);
                code += sign != 0u ? -delta : delta;
                target[chunk * 8u + inner] = half(scale * code);
            }
        }
        return;
    }
#endif

    uint state = read_bits(
        state_stream + state_offset,
        state_index,
        state_bits);
    uint table_bank = has_table_banks != 0u
        ? uint(banks[
              bank_offset + row * supergroups
              + group / groups_per_super])
        : 0u;
    uint delta_value = aux_mode == 3u
        ? read_bits(aux + aux_offset, state_index, 1u)
        : 0u;
    uint selected_bank = code_bank_mode == 1u
        ? uint(state_to_bank[state_bank_offset + state])
        : (code_bank_mode == 2u ? delta_value : 0u);
    float scale = anchor * scales[
        scale_offset + table_bank * state_count + state];
#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
    if (profile == 0u && vector_size == 8u) {
#pragma clang loop unroll(full)
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint column_base = group * 24u + chunk * 8u;
            if (column_base >= k_size) {
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u) = half4(0.0h);
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u + 4u) = half4(0.0h);
                continue;
            }
            uint sign_value = 0u;
            if (aux_mode == 1u || aux_mode == 2u) {
                sign_value = read_bits(
                    aux + aux_offset,
                    row * signs + group * 3u + chunk,
                    7u);
            }
            uint vector = group * 3u + chunk;
            uint index = read_bits(
                indices + indices_offset,
                row * vectors + vector,
                index_bits);
            uint code_base = codebook_offset
                + (((table_bank * code_banks + selected_bank) * entries
                  + index) * 8u);
            float4 first_code = float4(
                *reinterpret_cast<device const char4*>(
                    codebooks + code_base));
            float4 second_code = float4(
                *reinterpret_cast<device const char4*>(
                    codebooks + code_base + 4u));
            if (aux_mode == 1u || aux_mode == 2u) {
                first_code = select(
                    first_code,
                    -first_code,
                    (uint4(sign_value) & uint4(1u, 2u, 4u, 8u))
                        != uint4(0u));
                uint parity = popcount(sign_value) & 1u;
                if (aux_mode == 2u) {
                    parity ^= (index >> 7u) & 1u;
                }
                second_code = select(
                    second_code,
                    -second_code,
                    bool4(
                        (sign_value & 16u) != 0u,
                        (sign_value & 32u) != 0u,
                        (sign_value & 64u) != 0u,
                        parity != 0u));
            } else if (aux_mode == 3u) {
                float delta = parameters[parameter_offset];
                float signed_delta = delta_value != 0u
                    ? -delta : delta;
                first_code += signed_delta;
                second_code += signed_delta;
            }
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u) = half4(scale * first_code);
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u + 4u) = half4(
                    scale * second_code);
        }
        return;
    }
    if (profile == 0u && vector_size == 4u) {
#pragma clang loop unroll(full)
        for (uint chunk = 0u; chunk < 3u; ++chunk) {
            uint column_base = group * 24u + chunk * 8u;
            if (column_base >= k_size) {
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u) = half4(0.0h);
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u + 4u) = half4(0.0h);
                continue;
            }
            uint sign_value = 0u;
            if (aux_mode == 1u || aux_mode == 2u) {
                sign_value = read_bits(
                    aux + aux_offset,
                    row * signs + group * 3u + chunk,
                    7u);
            }
#pragma clang loop unroll(full)
            for (uint local = 0u; local < 2u; ++local) {
                uint vector = group * 6u + chunk * 2u + local;
                if (vector >= vectors) {
                    *reinterpret_cast<threadgroup half4*>(
                        target + chunk * 8u + local * 4u) = half4(0.0h);
                    continue;
                }
                uint index = read_bits(
                    indices + indices_offset,
                    row * vectors + vector,
                    index_bits);
                uint code_base = codebook_offset
                    + (((table_bank * code_banks + selected_bank) * entries
                      + index) * 4u);
                float4 code = float4(
                    *reinterpret_cast<device const char4*>(
                        codebooks + code_base));
                if (aux_mode == 1u || aux_mode == 2u) {
                    if (local == 0u) {
                        code = select(
                            code,
                            -code,
                            (uint4(sign_value)
                                & uint4(1u, 2u, 4u, 8u))
                                != uint4(0u));
                    } else {
                        uint parity = popcount(sign_value) & 1u;
                        if (aux_mode == 2u) {
                            parity ^= (index >> 7u) & 1u;
                        }
                        code = select(
                            code,
                            -code,
                            bool4(
                                (sign_value & 16u) != 0u,
                                (sign_value & 32u) != 0u,
                                (sign_value & 64u) != 0u,
                                parity != 0u));
                    }
                } else if (aux_mode == 3u) {
                    float delta = parameters[parameter_offset];
                    code += delta_value != 0u ? -delta : delta;
                }
                *reinterpret_cast<threadgroup half4*>(
                    target + chunk * 8u + local * 4u) = half4(
                        scale * code);
            }
        }
        return;
    }
#endif
    uint vectors_per_chunk = 8u / vector_size;
    for (uint chunk = 0u; chunk < 3u; ++chunk) {
        if (group * 24u + chunk * 8u >= k_size) {
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u) = half4(0.0h);
            *reinterpret_cast<threadgroup half4*>(
                target + chunk * 8u + 4u) = half4(0.0h);
            continue;
        }
        uint sign_value = 0u;
        uint3 execution_record = uint3(0u);
        if (execution == 3u) {
            execution_record = read_jsc_wide_record(
                indices, indices_offset,
                row * signs + group * 3u + chunk,
                vector_size, index_bits);
            // The generic path reconstructs the parity bit below.
            sign_value = execution_record.z & 127u;
        } else if (aux_mode == 1u || aux_mode == 2u) {
            sign_value = read_bits(
                aux + aux_offset,
                row * signs + group * 3u + chunk,
                7u);
        }
        for (uint local = 0u; local < vectors_per_chunk; ++local) {
            uint vector =
                group * (24u / vector_size)
                + chunk * vectors_per_chunk + local;
            uint index = execution == 3u
                ? execution_record[local]
                : read_bits(
                      indices + indices_offset,
                      row * vectors + vector,
                      index_bits);
            uint code_base = codebook_offset +
                (((table_bank * code_banks + selected_bank) * entries
                  + index) * vector_size);
            for (uint component = 0u;
                 component < vector_size;
                 ++component) {
                uint inner = local * vector_size + component;
                float code = float(codebooks[code_base + component]);
                uint negative = inner < 7u
                    ? ((sign_value >> inner) & 1u)
                    : (popcount(sign_value) & 1u);
                if (aux_mode == 2u && inner == 7u) {
                    negative ^= (index >> 7u) & 1u;
                }
                if (aux_mode == 1u || aux_mode == 2u) {
                    if (negative != 0u) {
                        code = -code;
                    }
                } else if (aux_mode == 3u) {
                    float delta = parameters[parameter_offset];
                    code += delta_value != 0u ? -delta : delta;
                }
                target[chunk * 8u + inner] = half(scale * code);
            }
        }
    }
}

inline void add_nepq_residual_group24(
    const device int* d,
    const device float* codebooks,
    const device short* first_records,
    const device short* second_records,
    threadgroup half* target,
    uint row,
    uint group,
    uint vectors) {
    uint residual_profile = uint(d[28]);
    uint position_bits = (residual_profile >> 8u) & 255u;
    uint block_vectors = (residual_profile >> 16u) & 255u;
    if (position_bits == 0u || block_vectors == 0u) {
        return;
    }
    uint residual_blocks =
        (vectors + block_vectors - 1u) / block_vectors;
    uint codebook_offset = uint(d[30]);
    uint record_offset = uint(d[31]);
    uint position_mask = (1u << position_bits) - 1u;
    uint first_vector = group * 3u;
    uint last_vector = min(first_vector + 2u, vectors - 1u);
    uint first_block = first_vector / block_vectors;
    uint last_block = last_vector / block_vectors;
    for (uint block = first_block; block <= last_block; ++block) {
        uint record_index =
            record_offset + row * residual_blocks + block;
        short records[2] = {
            first_records[record_index],
            second_records[record_index],
        };
        for (uint stream = 0u; stream < 2u; ++stream) {
            int record = int(records[stream]);
            if (record < 0) {
                continue;
            }
            uint position = uint(record) & position_mask;
            uint dictionary_id = uint(record) >> position_bits;
            uint vector = block * block_vectors + position;
            if (
                dictionary_id >= 1024u
                || vector < first_vector
                || vector > last_vector
                || vector >= vectors
            ) {
                continue;
            }
            uint target_offset = (vector - first_vector) * 8u;
            uint dictionary_offset =
                codebook_offset + dictionary_id * 8u;
            for (uint component = 0u; component < 8u; ++component) {
                target[target_offset + component] += half(
                    codebooks[dictionary_offset + component]);
            }
        }
    }
}

} // namespace

struct MfqGroupedMmqParams {
    int route_count;
    int tokens;
    int routes;
    int experts;
    int output_width;
    int matrix_output_width;
    int projections;
    int input_width;
    int descriptor_size;
    int variant_stride;
    int shared_input;
    int input_sorted;
    float swiglu_limit;
};

#if !defined(MFQ_ENABLE_NAX) && !defined(MFQ_ENABLE_DSV4_MXFP4_BLOCKS)
template <int BM, bool FUSED_SWIGLU, bool HAS_NEPQ_RESIDUAL>
[[kernel]] void mfq_grouped_mmq_f16(
    const device int* descriptors [[buffer(0)]],
    const device uchar* vq_indices [[buffer(1)]],
    const device uchar* vq_state [[buffer(2)]],
    const device uchar* vq_aux [[buffer(3)]],
    const device float* vq_anchors [[buffer(4)]],
    const device int8_t* vq_codebooks [[buffer(5)]],
    const device float* vq_scales [[buffer(6)]],
    const device uchar* vq_state_to_bank [[buffer(7)]],
    const device uchar* vq_banks [[buffer(8)]],
    const device float* vq_parameters [[buffer(9)]],
    const device half* x [[buffer(10)]],
    const device int* expert_ids [[buffer(11)]],
    const device int* route_order [[buffer(12)]],
    device half* y [[buffer(13)]],
    constant MfqGroupedMmqParams& params [[buffer(14)]],
    const device float* vq_residual_codebooks [[buffer(15)]],
    const device short* vq_residual_first [[buffer(16)]],
    const device short* vq_residual_second [[buffer(17)]],
    const device int* block_meta [[buffer(18)]],
    const device int* block_count [[buffer(19)]],
    const device uchar* mx_values [[buffer(20)]],
    const device uchar* mx_scales [[buffer(21)]],
    const device uchar* nint_q [[buffer(22)]],
    const device uchar* nint_sub_scale [[buffer(23)]],
    const device uchar* nint_sub_min [[buffer(24)]],
    const device float* nint_anchor_scale [[buffer(25)]],
    const device float* nint_anchor_min [[buffer(26)]],
    const device int8_t* q8_q [[buffer(27)]],
    const device half* q8_scales [[buffer(28)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr bool WIDE_TILE = BM == 80 || BM == 96;
    constexpr bool WIDE_NINT = WIDE_TILE && MFQ_GROUPED_FAMILY_MASK == 1
        && (MFQ_GROUPED_NINT_GROUP_SIZE == 24 || MFQ_GROUPED_NINT_GROUP_SIZE == 28);
    constexpr bool WIDE_VQ = WIDE_TILE && MFQ_GROUPED_FAMILY_MASK == 2;
    constexpr int BN = WIDE_TILE && !WIDE_NINT && !WIDE_VQ ? 32 : 64;
    constexpr int BK = WIDE_NINT ? 2 * MFQ_GROUPED_NINT_GROUP_SIZE
        : WIDE_VQ ? 72
        : MFQ_GROUPED_FAMILY_MASK == 1 && MFQ_GROUPED_NINT_GROUP_SIZE == 28 ? 112 : 96;
    constexpr int BK_padded = BK + (BK / 8 % 2 == 0 ? 8 : 0);
    constexpr int WM = WIDE_NINT || WIDE_VQ ? 2 : BM == 48 || BM == 96 ? 3 : BM == 64 ? 4 : 2;
    constexpr int WN = WIDE_NINT || WIDE_VQ || BM == 32 || BM == 80 ? 4 : 2;
    constexpr uint TGP_SIZE = 256u;
    const bool computes_matrix = simd_group_id < uint(WM * WN);
    const int route_count = params.route_count;
    const int tokens = params.tokens;
    const int routes = params.routes;
    const int experts = params.experts;
    const int output_width = params.output_width;
    const int matrix_output_width = params.matrix_output_width;
    const int input_width = params.input_width;
    const int descriptor_size = params.descriptor_size;
    const int variant_stride = params.variant_stride;
    const int shared_input = params.shared_input;
    const int input_sorted = params.input_sorted;
    const float swiglu_limit = params.swiglu_limit;
    using mma_t = mlx::steel::BlockMMA<
        half,
        half,
        BM,
        BN,
        BK,
        WM,
        WN,
        false,
        true,
        BK_padded,
        BK_padded>;

    int output_base = int(tid.x) * BN;
    int block_id = int(tid.y);
    int nblocks = block_count[0];
    if (output_base >= output_width || block_id >= nblocks) {
        return;
    }

    int row_base = block_meta[block_id * 3 + 0];
    int expert = block_meta[block_id * 3 + 1];
    int row_count = block_meta[block_id * 3 + 2];
    if (row_count <= 0 || expert < 0 || expert >= experts) {
        return;
    }
    if constexpr (MFQ_GROUPED_MMQ_BM >= 64) {
        constexpr int MID_BM = MFQ_GROUPED_MMQ_BM > 64 ? 64 : 48;
        if ((BM == 32 && row_count > 32)
            || (BM == MID_BM && (row_count <= 32 || row_count > MID_BM))
            || (BM == 80 && MFQ_GROUPED_MMQ_BM > 80 && (row_count <= 64 || row_count > 80))
            || (BM == MFQ_GROUPED_MMQ_BM
                && row_count <= (MFQ_GROUPED_MMQ_BM > 80 ? 80 : MID_BM))) return;
    }

    const device int* base_descriptor =
        descriptors + expert * params.projections * descriptor_size;
    if ((MFQ_GROUPED_FAMILY_MASK & (1 << uint(base_descriptor[0]))) == 0) return;
    if (MFQ_GROUPED_NINT_GROUP_SIZE > 0 && base_descriptor[0] == 0
        && base_descriptor[5] != MFQ_GROUPED_NINT_GROUP_SIZE) return;
    uint rotation = uint(base_descriptor[27]);
    short valid_n = short(min(BN, output_width - output_base));
    threadgroup half Xs[BM * BK_padded];
    threadgroup half Ws[BN * BK_padded];

    thread mma_t gate_mma(simd_group_id, simd_lane_id);
    thread mma_t up_mma(simd_group_id, simd_lane_id);
    // Give each thread one contiguous fragment of a routed activation row.
    // The route/source lookup is invariant across K tiles, and packed_half4
    // keeps the gather at its natural two-byte alignment.
    constexpr uint X_LOAD_ROWS = WIDE_TILE ? 128u
        : BM == 48 || (BK == 112 && BM == 32) ? 64u : uint(BM);
    constexpr uint X_LOAD_LANES = TGP_SIZE / X_LOAD_ROWS;
    constexpr uint X_VALUES_PER_LANE = uint(BK) / X_LOAD_LANES;
    static_assert(TGP_SIZE % X_LOAD_ROWS == 0u);
    static_assert(uint(BK) % X_LOAD_LANES == 0u);
    static_assert(X_VALUES_PER_LANE % 4u == 0u);
    uint row = thread_id / X_LOAD_LANES;
    uint load_lane = thread_id - row * X_LOAD_LANES;
    uint local_column = load_lane * X_VALUES_PER_LANE;
    bool valid_row = int(row) < row_count && row < uint(BM);
    uint source_offset = 0u;
    if (valid_row) {
        uint route_index = uint(route_order[row_base + int(row)]);
        uint source_row = input_sorted != 0
            ? uint(row_base) + row
            : (shared_input != 0
                ? route_index / uint(routes)
                : route_index);
        source_offset =
            (rotation * uint(variant_stride) + source_row)
            * uint(input_width);
    }
    for (int k_base = 0; k_base < input_width; k_base += BK) {
#pragma clang loop unroll(full)
        for (uint column = 0u;
             row < uint(BM) && column < X_VALUES_PER_LANE;
             column += 4u) {
            uint input_column = uint(k_base) + local_column + column;
            half4 value = half4(0.0h);
            if (valid_row && input_column + 4u <= uint(input_width)) {
                packed_half4 packed =
                    *reinterpret_cast<device const packed_half4*>(
                        x + source_offset + input_column);
                value = half4(packed);
            } else if (valid_row && input_column < uint(input_width)) {
#pragma clang loop unroll(full)
                for (uint lane = 0u; lane < 4u; ++lane) {
                    if (input_column + lane < uint(input_width)) {
                        value[lane] = x[source_offset + input_column + lane];
                    }
                }
            }
            *reinterpret_cast<threadgroup half4*>(
                Xs + row * uint(BK_padded) + local_column + column) = value;
        }
        constexpr int PROJECTIONS = FUSED_SWIGLU ? 2 : 1;
        for (int projection = 0;
             projection < PROJECTIONS;
             ++projection) {
                uint descriptor_projection = params.projections == 2
                    ? uint(projection)
                    : 0u;
                const device int* descriptor = base_descriptor
                    + descriptor_projection * uint(descriptor_size);
                uint family = MFQ_GROUPED_FAMILY_MASK == 1 ? 0u
                    : MFQ_GROUPED_FAMILY_MASK == 2 ? 1u
                    : MFQ_GROUPED_FAMILY_MASK == 4 ? 2u
                    : MFQ_GROUPED_FAMILY_MASK == 8 ? 3u
                    : MFQ_GROUPED_FAMILY_MASK == 16 ? 4u
                    : MFQ_GROUPED_FAMILY_MASK == 32 ? 5u
                    : MFQ_GROUPED_FAMILY_MASK == 64 ? 6u : uint(descriptor[0]);
                uint local_expert = uint(descriptor[1]);
                uint projection_row_offset = params.projections == 1
                    ? uint(projection * output_width)
                    : 0u;
                if (valid_n != BN || k_base + BK > input_width) {
                    for (uint item = thread_id;
                         item < uint(BN * BK_padded);
                         item += TGP_SIZE) {
                        Ws[item] = half(0.0f);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (family == 0u) {
                    uint group_size = MFQ_GROUPED_NINT_GROUP_SIZE > 0
                        ? uint(MFQ_GROUPED_NINT_GROUP_SIZE) : uint(descriptor[5]);
                    if (group_size == 24u || group_size == 28u) {
                        constexpr uint GROUP_LANES = WIDE_TILE ? 2u : 1u;
                        uint tile_groups = (uint(BK) + group_size - 1u) / group_size
                            + uint(uint(BK) % group_size != 0u);
                        uint first_group = uint(k_base) / group_size;
                        for (uint item = thread_id;
                             item < uint(BN) * tile_groups * GROUP_LANES;
                             item += TGP_SIZE) {
                            uint output_row = item / (tile_groups * GROUP_LANES);
                            uint group = first_group + (item / GROUP_LANES) % tile_groups;
                            if (output_row < uint(valid_n) && group < uint(descriptor[6])
                                && group * group_size < uint(k_base + BK)) {
                                uint pool_row = local_expert * uint(matrix_output_width)
                                    + uint(output_base) + output_row + projection_row_offset;
                                threadgroup half* target = Ws + output_row * uint(BK_padded);
                                if (group_size == 24u) {
                                    decode_nint_group_tile_at<24u, BK, GROUP_LANES>(
                                        descriptor, nint_q, nint_sub_scale, nint_sub_min,
                                        nint_anchor_scale, nint_anchor_min, target, pool_row,
                                        group, uint(k_base), uint(input_width), item % GROUP_LANES);
                                } else {
                                    decode_nint_group_tile_at<28u, BK, GROUP_LANES>(
                                        descriptor, nint_q, nint_sub_scale, nint_sub_min,
                                        nint_anchor_scale, nint_anchor_min, target, pool_row,
                                        group, uint(k_base), uint(input_width), item % GROUP_LANES);
                                }
                            }
                        }
                    } else {
                        constexpr uint VALUES_PER_ITEM = 4u;
                        constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                        for (uint item = thread_id;
                             item < uint(BN) * ITEMS_PER_ROW;
                             item += TGP_SIZE) {
                            uint output_row = item / ITEMS_PER_ROW;
                            uint local_item = item - output_row * ITEMS_PER_ROW;
                            uint local_column = local_item * VALUES_PER_ITEM;
                            uint input_column = uint(k_base) + local_column;
                            if (output_row < uint(valid_n)
                                && input_column < uint(input_width)) {
                                uint pool_row =
                                    local_expert * uint(matrix_output_width)
                                    + uint(output_base) + output_row
                                    + projection_row_offset;
                                *reinterpret_cast<threadgroup half4*>(
                                    Ws + output_row * uint(BK_padded)
                                        + local_column) = decode_nint_row_quad_at(
                                    descriptor,
                                    nint_q,
                                    nint_sub_scale,
                                    nint_sub_min,
                                    nint_anchor_scale,
                                    nint_anchor_min,
                                    pool_row,
                                    input_column,
                                    uint(input_width));
                            }
                        }
                    }
                } else if (family == 4u) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row < uint(valid_n)
                            && input_column < uint(input_width)) {
                            uint pool_row =
                                local_expert * uint(matrix_output_width)
                                + uint(output_base) + output_row
                                + projection_row_offset;
                            *reinterpret_cast<threadgroup half4*>(
                                Ws + output_row * uint(BK_padded)
                                    + local_column) = decode_mxfp8_quad_at(
                                descriptor, mx_values, mx_scales,
                                pool_row, input_column, uint(input_width));
                        }
                    }
                } else if (family == 5u || family == 6u) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row >= uint(valid_n)
                            || input_column >= uint(input_width)) {
                            continue;
                        }
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + projection_row_offset;
                        threadgroup half* target =
                            Ws + output_row * uint(BK_padded) + local_column;
                        if (input_column + VALUES_PER_ITEM
                            <= uint(input_width)) {
                            *reinterpret_cast<threadgroup half4*>(target) =
                                decode_dense_quad_at(
                                    descriptor, q8_q, family, pool_row,
                                    input_column, uint(input_width));
                        } else {
#pragma clang loop unroll(full)
                            for (uint lane = 0u;
                                 lane < VALUES_PER_ITEM;
                                 ++lane) {
                                if (input_column + lane < uint(input_width)) {
                                    target[lane] = decode_dense_value_at(
                                        descriptor, q8_q, family, pool_row,
                                        input_column + lane,
                                        uint(input_width));
                                }
                            }
                        }
                    }
                } else {
                constexpr uint GROUPS_PER_TILE = BK / 24;
                for (uint group_item = thread_id;
                     group_item < uint(BN) * GROUPS_PER_TILE;
                     group_item += TGP_SIZE) {
                    uint output_row = group_item / GROUPS_PER_TILE;
                    uint local_group =
                        group_item - output_row * GROUPS_PER_TILE;
                    if (output_row < uint(valid_n) && family == 2u
                        && local_group < uint(BK / 32)) {
                        uint input_column =
                            uint(k_base) + local_group * 32u;
                        if (input_column >= uint(input_width)) {
                            continue;
                        }
                        uint group = input_column / 32u;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + projection_row_offset;
                        decode_nint8_zero_group32(
                            descriptor,
                            q8_q,
                            q8_scales,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 32u,
                            pool_row,
                            group);
                    } else if (output_row < uint(valid_n) && family == 3u
                        && local_group < uint(BK / 32)) {
                        uint input_column =
                            uint(k_base) + local_group * 32u;
                        if (input_column >= uint(input_width)) {
                            continue;
                        }
                        uint group = input_column / 32u;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + projection_row_offset;
                        decode_mxfp4_group32(
                            descriptor,
                            mx_values,
                            mx_scales,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 32u,
                            pool_row,
                            group,
                            uint(input_width));
                    } else if (output_row < uint(valid_n)
                        && family == 1u) {
                        uint input_column =
                            uint(k_base) + local_group * 24u;
                        if (input_column >= uint(input_width)) {
                            continue;
                        }
                        uint group = input_column / 24u;
                        uint pool_row =
                            local_expert * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + projection_row_offset;
                        decode_vq_group24(
                            descriptor,
                            vq_indices,
                            vq_state,
                            vq_aux,
                            vq_anchors,
                            vq_codebooks,
                            vq_scales,
                            vq_state_to_bank,
                            vq_banks,
                            vq_parameters,
                            Ws + output_row * uint(BK_padded)
                                + local_group * 24u,
                            pool_row,
                            group,
                            uint(input_width));
                        if constexpr (HAS_NEPQ_RESIDUAL) {
                            add_nepq_residual_group24(
                                descriptor,
                                vq_residual_codebooks,
                                vq_residual_first,
                                vq_residual_second,
                                Ws + output_row * uint(BK_padded)
                                    + local_group * 24u,
                                pool_row,
                                group,
                                uint(descriptor[7]));
                        }
                    }
                }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (computes_matrix) {
                    if (projection == 0) {
                        gate_mma.mma(Xs, Ws);
                    } else {
                        up_mma.mma(Xs, Ws);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (computes_matrix) {
        if constexpr (FUSED_SWIGLU) {
            for (short item = 0;
                 item < decltype(gate_mma.Ctile)::kElemsPerTile;
                 ++item) {
                float gate = gate_mma.Ctile.elems()[item];
                float up = up_mma.Ctile.elems()[item];
                if (swiglu_limit > 0.0f) {
                    gate = min(gate, swiglu_limit);
                    up = clamp(up, -swiglu_limit, swiglu_limit);
                }
                gate_mma.Ctile.elems()[item] =
                    gate / (1.0f + exp(-gate)) * up;
            }
        }
        device half* destination =
            y + row_base * output_width + output_base;
        gate_mma.store_result_slice(
            destination,
            output_width,
            short2(0, 0),
            short2(valid_n, short(row_count)));
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#define instantiate_mfq_grouped_mmq(name, bm, fused, residual) \
    template [[host_name(name)]] [[kernel]] \
    decltype(mfq_grouped_mmq_f16<bm, fused, residual>) \
    mfq_grouped_mmq_f16<bm, fused, residual>;

instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_f16_specialized",
    MFQ_GROUPED_MMQ_BM,
    false,
    false)
instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_swiglu_f16_specialized",
    MFQ_GROUPED_MMQ_BM,
    true,
    false)
instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_f16_specialized_nr",
    MFQ_GROUPED_MMQ_BM,
    false,
    true)
instantiate_mfq_grouped_mmq(
    "mfq_grouped_mmq_swiglu_f16_specialized_nr",
    MFQ_GROUPED_MMQ_BM,
    true,
    true)
#if MFQ_GROUPED_MMQ_BM >= 64
#if MFQ_GROUPED_MMQ_BM > 64
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_mid", 64, false, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_mid", 64, true, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_nr_mid", 64, false, true)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_nr_mid", 64, true, true)
#if MFQ_GROUPED_MMQ_BM > 80
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_mid80", 80, false, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_mid80", 80, true, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_nr_mid80", 80, false, true)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_nr_mid80", 80, true, true)
#endif
#else
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_mid", 48, false, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_mid", 48, true, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_nr_mid", 48, false, true)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_nr_mid", 48, true, true)
#endif
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_tail", 32, false, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_tail", 32, true, false)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_f16_specialized_nr_tail", 32, false, true)
instantiate_mfq_grouped_mmq("mfq_grouped_mmq_swiglu_f16_specialized_nr_tail", 32, true, true)
#endif
#endif

#ifdef MFQ_ENABLE_NAX
template <
    int BM,
    int BN,
    int X_STRIDE,
    int W_STRIDE,
    bool FUSED_SWIGLU,
    bool DIRECT_PACKED>
[[kernel]] void mfq_grouped_mfe_nax_f16(
    const device int* descriptors [[buffer(0)]],
    const device uchar* vq_indices [[buffer(1)]],
    const device uchar* vq_state [[buffer(2)]],
    const device uchar* vq_aux [[buffer(3)]],
    const device float* vq_anchors [[buffer(4)]],
    const device int8_t* vq_codebooks [[buffer(5)]],
    const device float* vq_scales [[buffer(6)]],
    const device uchar* vq_state_to_bank [[buffer(7)]],
    const device uchar* vq_banks [[buffer(8)]],
    const device float* vq_parameters [[buffer(9)]],
    const device half* x [[buffer(10)]],
    const device int* expert_ids [[buffer(11)]],
    const device int* route_order [[buffer(12)]],
    device half* y [[buffer(13)]],
    constant MfqGroupedMmqParams& params [[buffer(14)]],
    const device float* vq_residual_codebooks [[buffer(15)]],
    const device short* vq_residual_first [[buffer(16)]],
    const device short* vq_residual_second [[buffer(17)]],
    const device int* block_meta [[buffer(18)]],
    const device int* block_count [[buffer(19)]],
    const device uchar* mx_values [[buffer(20)]],
    const device uchar* mx_scales [[buffer(21)]],
    const device uchar* nint_q [[buffer(22)]],
    const device uchar* nint_sub_scale [[buffer(23)]],
    const device uchar* nint_sub_min [[buffer(24)]],
    const device float* nint_anchor_scale [[buffer(25)]],
    const device float* nint_anchor_min [[buffer(26)]],
    const device int8_t* q8_q [[buffer(27)]],
    const device half* q8_scales [[buffer(28)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint3 grid_size [[threadgroups_per_grid]]) {
    constexpr int BK = MFQ_GROUPED_NAX_BK;
    const int input_width = MFQ_GROUPED_NAX_INPUT_WIDTH > 0
        ? MFQ_GROUPED_NAX_INPUT_WIDTH : params.input_width;
    const int output_width = MFQ_GROUPED_NAX_OUTPUT_WIDTH > 0
        ? MFQ_GROUPED_NAX_OUTPUT_WIDTH : params.output_width;
    const int matrix_output_width = MFQ_GROUPED_NAX_MATRIX_OUTPUT_WIDTH > 0
        ? MFQ_GROUPED_NAX_MATRIX_OUTPUT_WIDTH : params.matrix_output_width;
    static_assert(BM % MFQ_GROUPED_NAX_SIMD_ROWS == 0);
    constexpr int WM = BM / MFQ_GROUPED_NAX_SIMD_ROWS;
    constexpr int WN = BN / 32;
    constexpr uint TGP_SIZE = uint(WM * WN * 32);
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;
    constexpr int PROJECTIONS = FUSED_SWIGLU ? 2 : 1;
    const int columns = (output_width + BN - 1) / BN;
    int output_base = int(tid.y % uint(columns)) * BN;
    int block_id = int(tid.x + (tid.y / uint(columns)) * grid_size.x);
    int nblocks = block_count[0];
    if (output_base >= output_width || block_id >= nblocks) {
        return;
    }
    int row_base = block_meta[block_id * 3 + 0];
    int expert = block_meta[block_id * 3 + 1];
    int row_count = block_meta[block_id * 3 + 2];
    if (row_count <= 0 || expert < 0 || expert >= params.experts) {
        return;
    }

    const device int* base_descriptor = descriptors
        + expert * params.projections * params.descriptor_size;
    if ((MFQ_GROUPED_FAMILY_MASK & (1 << uint(base_descriptor[0]))) == 0) {
        return;
    }
    if (MFQ_GROUPED_NINT_GROUP_SIZE > 0
        && base_descriptor[5] != MFQ_GROUPED_NINT_GROUP_SIZE) {
        return;
    }
    constexpr bool HAS_NINT_FAMILY =
        (MFQ_GROUPED_FAMILY_MASK & 1) != 0;
    constexpr bool HAS_Q8_FAMILY =
        (MFQ_GROUPED_FAMILY_MASK & 4) != 0;
    constexpr bool HAS_MXFP4_FAMILY =
        (MFQ_GROUPED_FAMILY_MASK & 8) != 0;
    constexpr bool HAS_MXFP8_FAMILY =
        (MFQ_GROUPED_FAMILY_MASK & 16) != 0;
    constexpr bool HAS_DENSE_FAMILY =
        (MFQ_GROUPED_FAMILY_MASK & (32 | 64)) != 0;
    constexpr bool HAS_VQ_FAMILY =
        (MFQ_GROUPED_FAMILY_MASK & 2) != 0;
    uint rotation = uint(base_descriptor[27]);
    short valid_n = short(min(BN, output_width - output_base));
    short tm = short(SM * int(simd_group_id / WN));
    short tn = short(SN * int(simd_group_id % WN));
    short simd_m = short(max(0, min(int(SM), row_count - int(tm))));
    short simd_n = short(max(0, min(int(SN), int(valid_n) - int(tn))));

    constexpr bool DIRECT_ACTIVATION = BN <= 64 && !DIRECT_PACKED;
    constexpr bool ALIGNED_INPUT = MFQ_GROUPED_NAX_ALIGNED_INPUT != 0;
    constexpr bool PING_PONG = DIRECT_ACTIVATION && BK == 96;
    threadgroup half Xs[DIRECT_ACTIVATION ? 1 : BM * X_STRIDE];
    threadgroup half weight_storage[
        DIRECT_PACKED ? 1 : (PING_PONG ? 2 : 1) * BN * W_STRIDE];
    mlx::steel::NAXTile<float, TM, TN> gate_tile;
    mlx::steel::NAXTile<float, TM, TN> up_tile;
    gate_tile.clear();
    up_tile.clear();

    // Match activation-loader lanes to this NAX tile's thread geometry.  This
    // computes the routed source once, then moves contiguous FP16 quads.
    constexpr uint X_LOAD_LANES = TGP_SIZE / uint(BM);
    constexpr uint X_VALUES_PER_LANE = uint(BK) / X_LOAD_LANES;
    static_assert(TGP_SIZE % uint(BM) == 0u);
    static_assert(uint(BK) % X_LOAD_LANES == 0u);
    static_assert(X_VALUES_PER_LANE % 4u == 0u);
    uint row = thread_id / X_LOAD_LANES;
    uint load_lane = thread_id - row * X_LOAD_LANES;
    uint local_column = load_lane * X_VALUES_PER_LANE;
    bool valid_row = int(row) < row_count;
    uint source_offset = 0u;
    if (valid_row) {
        uint route_index = uint(route_order[row_base + int(row)]);
        uint source_row = params.input_sorted != 0
            ? uint(row_base) + row
            : (params.shared_input != 0
                ? route_index / uint(params.routes)
                : route_index);
        source_offset =
            (rotation * uint(params.variant_stride) + source_row)
            * uint(input_width);
    }
    uint fragment_offsets[TM][2];
    bool fragment_valid[TM][2];
    if constexpr (DIRECT_ACTIVATION) {
        short2 coord = mlx::steel::BaseNAXFrag::get_coord();
#pragma clang loop unroll(full)
        for (short fragment_m = 0; fragment_m < TM; ++fragment_m) {
#pragma clang loop unroll(full)
        for (uint half_row = 0u; half_row < 2u; ++half_row) {
            uint fragment_row = uint(tm) + uint(fragment_m) * 16u
                + uint(coord.y) + half_row * 8u;
            if constexpr (ALIGNED_INPUT) {
                fragment_row = min(fragment_row, uint(row_count - 1));
            }
            fragment_valid[fragment_m][half_row] =
                ALIGNED_INPUT || fragment_row < uint(row_count);
            fragment_offsets[fragment_m][half_row] = 0u;
            if (fragment_valid[fragment_m][half_row]) {
                uint route_index = uint(route_order[row_base + int(fragment_row)]);
                uint source_row = params.input_sorted != 0
                    ? uint(row_base) + fragment_row
                    : (params.shared_input != 0
                        ? route_index / uint(params.routes)
                        : route_index);
                fragment_offsets[fragment_m][half_row] =
                    (rotation * uint(params.variant_stride) + source_row)
                        * uint(input_width) + uint(coord.x);
            }
        }
        }
    }
#if (MFQ_GROUPED_FAMILY_MASK & 2) != 0
    constexpr bool FIXED_VQ_GEOMETRY = !FUSED_SWIGLU && !DIRECT_PACKED
        && DIRECT_ACTIVATION && BK % 24 == 0
        && uint(BN * (BK / 24)) == TGP_SIZE;
    uint vq_output_row = 0u;
    uint vq_local_group = 0u;
    uint vq_pool_row = 0u;
    bool vq_row_valid = false;
    if constexpr (FIXED_VQ_GEOMETRY) {
        constexpr uint GROUPS_PER_TILE = uint(BK / 24);
        vq_output_row = thread_id / GROUPS_PER_TILE;
        vq_local_group = thread_id % GROUPS_PER_TILE;
        vq_pool_row = uint(base_descriptor[1]) * uint(matrix_output_width)
            + uint(output_base) + vq_output_row;
        vq_row_valid = vq_output_row < uint(valid_n);
    }
    auto execute_k = [&](auto profile, auto execution) {
        constexpr int VQ_PROFILE = decltype(profile)::value;
        constexpr uint VQ_EXECUTION = decltype(execution)::value;
#endif
    for (int k_base = 0; k_base < input_width; k_base += BK) {
        if constexpr (!DIRECT_ACTIVATION) {
#pragma clang loop unroll(full)
        for (uint column = 0u;
             column < X_VALUES_PER_LANE;
             column += 4u) {
            uint input_column = uint(k_base) + local_column + column;
            half4 value = half4(0.0h);
            if (valid_row
                && input_column + 4u <= uint(input_width)) {
                packed_half4 packed =
                    *reinterpret_cast<device const packed_half4*>(
                        x + source_offset + input_column);
                value = half4(packed);
            } else if (valid_row
                && input_column < uint(input_width)) {
#pragma clang loop unroll(full)
                for (uint lane = 0u; lane < 4u; ++lane) {
                    if (input_column + lane < uint(input_width)) {
                        value[lane] = x[source_offset + input_column + lane];
                    }
                }
            }
            *reinterpret_cast<threadgroup half4*>(
                Xs + row * uint(X_STRIDE) + local_column + column) = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int projection = 0; projection < PROJECTIONS; ++projection) {
            threadgroup half* Ws = weight_storage + (PING_PONG
                ? ((k_base / BK * PROJECTIONS + projection) & 1) * BN * W_STRIDE
                : 0);
            uint descriptor_projection = params.projections == 2
                ? uint(projection)
                : 0u;
            const device int* descriptor = base_descriptor
                + descriptor_projection * uint(params.descriptor_size);
#if MFQ_GROUPED_FAMILY_MASK == 1
            constexpr uint family = 0u;
#elif MFQ_GROUPED_FAMILY_MASK == 2
            constexpr uint family = 1u;
#elif MFQ_GROUPED_FAMILY_MASK == 4
            constexpr uint family = 2u;
#elif MFQ_GROUPED_FAMILY_MASK == 8
            constexpr uint family = 3u;
#elif MFQ_GROUPED_FAMILY_MASK == 16
            constexpr uint family = 4u;
#elif MFQ_GROUPED_FAMILY_MASK == 32
            constexpr uint family = 5u;
#elif MFQ_GROUPED_FAMILY_MASK == 64
            constexpr uint family = 6u;
#else
            uint family = uint(descriptor[0]);
#endif
            uint local_expert = uint(descriptor[1]);
            uint projection_row_offset = params.projections == 1
                ? uint(projection * output_width)
                : 0u;
#if (MFQ_GROUPED_FAMILY_MASK & 2) != 0
            if constexpr (VQ_PROFILE != 0) {
                uint group = uint(k_base / 24) + vq_local_group;
                threadgroup half* target = Ws + vq_output_row * uint(W_STRIDE)
                    + vq_local_group * 24u;
                if (vq_row_valid && group * 24u < uint(input_width)) {
#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
                    if constexpr (VQ_PROFILE == 6) {
                        decode_nvq1_group24<4u, 9u, 512u, true>(
                            descriptor, vq_indices, vq_state, vq_aux,
                            vq_anchors, vq_codebooks, vq_scales, vq_parameters,
                            target, vq_pool_row, group);
                    }
#endif
                    if constexpr (VQ_PROFILE != 6) {
                        decode_vq_group24<uint(VQ_PROFILE), VQ_EXECUTION>(
                            descriptor, vq_indices, vq_state, vq_aux,
                            vq_anchors, vq_codebooks, vq_scales, vq_state_to_bank,
                            vq_banks, vq_parameters, target, vq_pool_row, group,
                            uint(input_width));
                    }
                } else {
#pragma clang loop unroll(full)
                    for (uint inner = 0u; inner < 24u; inner += 4u) {
                        *reinterpret_cast<threadgroup half4*>(target + inner)
                            = half4(0.0h);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
#endif
            if constexpr (!DIRECT_PACKED) {
                // Complete tiles overwrite all weight elements. Clear only a
                // boundary tile, where an absent tail would otherwise leave
                // undefined threadgroup data multiplied by padded zeros.
                if (k_base + BK > input_width || valid_n < BN) {
                    for (uint item = thread_id;
                         item < uint(BN * W_STRIDE);
                         item += TGP_SIZE) {
                        Ws[item] = half(0.0f);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (HAS_NINT_FAMILY && family == 0u) {
                    uint group_size = MFQ_GROUPED_NINT_GROUP_SIZE > 0
                        ? uint(MFQ_GROUPED_NINT_GROUP_SIZE) : uint(descriptor[5]);
                    if (group_size == 24u || group_size == 28u) {
                        uint tile_groups = (uint(BK) + group_size - 1u)
                            / group_size + uint(uint(BK) % group_size != 0u);
                        uint first_group = uint(k_base) / group_size;
                        for (uint item = thread_id;
                             item < uint(BN) * tile_groups;
                             item += TGP_SIZE) {
                            uint output_row = item / tile_groups;
                            uint group = first_group + item % tile_groups;
                            if (output_row < uint(valid_n)
                                && group < uint(descriptor[6])
                                && group * group_size < uint(k_base + BK)) {
                                uint pool_row = local_expert
                                        * uint(matrix_output_width)
                                    + uint(output_base) + output_row
                                    + projection_row_offset;
                                threadgroup half* target = Ws + output_row * uint(W_STRIDE);
                                if (group_size == 24u) {
                                    decode_nint_group_tile_at<24u, BK>(
                                        descriptor, nint_q, nint_sub_scale, nint_sub_min,
                                        nint_anchor_scale, nint_anchor_min, target, pool_row,
                                        group, uint(k_base), uint(input_width));
                                } else {
                                    decode_nint_group_tile_at<28u, BK>(
                                        descriptor, nint_q, nint_sub_scale, nint_sub_min,
                                        nint_anchor_scale, nint_anchor_min, target, pool_row,
                                        group, uint(k_base), uint(input_width));
                                }
                            }
                        }
                    } else {
                        constexpr uint VALUES_PER_ITEM = 8u;
                        constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                        for (uint item = thread_id;
                             item < uint(BN) * ITEMS_PER_ROW;
                             item += TGP_SIZE) {
                            uint output_row = item / ITEMS_PER_ROW;
                            uint local_item = item - output_row * ITEMS_PER_ROW;
                            uint local_column = local_item * VALUES_PER_ITEM;
                            uint input_column = uint(k_base) + local_column;
                            if (output_row < uint(valid_n)
                                && input_column < uint(input_width)) {
                                uint pool_row = local_expert
                                        * uint(matrix_output_width)
                                    + uint(output_base) + output_row
                                    + projection_row_offset;
                                decode_nint_row_pair_at(
                                    descriptor,
                                    nint_q,
                                    nint_sub_scale,
                                    nint_sub_min,
                                    nint_anchor_scale,
                                    nint_anchor_min,
                                    Ws + output_row * uint(W_STRIDE)
                                        + local_column,
                                    pool_row,
                                    input_column,
                                    uint(input_width));
                            }
                        }
                    }
                } else if (HAS_MXFP8_FAMILY && family == 4u) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row < uint(valid_n)
                            && input_column < uint(input_width)) {
                            uint pool_row = local_expert
                                    * uint(matrix_output_width)
                                + uint(output_base) + output_row
                                + projection_row_offset;
                            *reinterpret_cast<threadgroup half4*>(
                                Ws + output_row * uint(W_STRIDE)
                                    + local_column) = decode_mxfp8_quad_at(
                                descriptor, mx_values, mx_scales,
                                pool_row, input_column,
                                uint(input_width));
                        }
                    }
                } else if (HAS_DENSE_FAMILY
                    && (family == 5u || family == 6u)) {
                    constexpr uint VALUES_PER_ITEM = 4u;
                    constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
                    for (uint item = thread_id;
                         item < uint(BN) * ITEMS_PER_ROW;
                         item += TGP_SIZE) {
                        uint output_row = item / ITEMS_PER_ROW;
                        uint local_item = item - output_row * ITEMS_PER_ROW;
                        uint local_column = local_item * VALUES_PER_ITEM;
                        uint input_column = uint(k_base) + local_column;
                        if (output_row >= uint(valid_n)
                            || input_column >= uint(input_width)) {
                            continue;
                        }
                        uint pool_row = local_expert
                                * uint(matrix_output_width)
                            + uint(output_base) + output_row
                            + projection_row_offset;
                        threadgroup half* target =
                            Ws + output_row * uint(W_STRIDE) + local_column;
                        if (input_column + VALUES_PER_ITEM
                            <= uint(input_width)) {
                            *reinterpret_cast<threadgroup half4*>(target) =
                                decode_dense_quad_at(
                                    descriptor, q8_q, family, pool_row,
                                    input_column, uint(input_width));
                        } else {
#pragma clang loop unroll(full)
                            for (uint lane = 0u;
                                 lane < VALUES_PER_ITEM;
                                 ++lane) {
                                if (input_column + lane
                                    < uint(input_width)) {
                                    target[lane] = decode_dense_value_at(
                                        descriptor, q8_q, family, pool_row,
                                        input_column + lane,
                                        uint(input_width));
                                }
                            }
                        }
                    }
                } else {
                    constexpr uint GROUPS_PER_TILE = BK / 24;
                    for (uint item = thread_id;
                         item < uint(BN) * GROUPS_PER_TILE;
                         item += TGP_SIZE) {
                        uint output_row = item / GROUPS_PER_TILE;
                        uint local_group = item - output_row * GROUPS_PER_TILE;
                        if (output_row < uint(valid_n) && HAS_Q8_FAMILY
                            && family == 2u
                            && local_group < uint(BK / 32)) {
                            uint input_column =
                                uint(k_base) + local_group * 32u;
                            if (input_column >= uint(input_width)) {
                                continue;
                            }
                            uint group = input_column / 32u;
                            uint pool_row = local_expert
                                    * uint(matrix_output_width)
                                + uint(output_base) + output_row
                                + projection_row_offset;
                            decode_nint8_zero_group32(
                                descriptor, q8_q, q8_scales,
                                Ws + output_row * uint(W_STRIDE)
                                    + local_group * 32u,
                                pool_row, group);
                        } else if (output_row < uint(valid_n)
                            && HAS_MXFP4_FAMILY && family == 3u
                            && local_group < uint(BK / 32)) {
                            uint input_column =
                                uint(k_base) + local_group * 32u;
                            if (input_column >= uint(input_width)) {
                                continue;
                            }
                            uint group = input_column / 32u;
                            uint pool_row = local_expert
                                    * uint(matrix_output_width)
                                + uint(output_base) + output_row
                                + projection_row_offset;
                            decode_mxfp4_group32(
                                descriptor, mx_values, mx_scales,
                                Ws + output_row * uint(W_STRIDE)
                                    + local_group * 32u,
                                pool_row, group,
                                uint(input_width));
                        } else if (output_row < uint(valid_n)
                            && HAS_VQ_FAMILY && family == 1u) {
                            uint input_column =
                                uint(k_base) + local_group * 24u;
                            if (input_column >= uint(input_width)) {
                                continue;
                            }
                            uint group = input_column / 24u;
                            uint pool_row = local_expert
                                    * uint(matrix_output_width)
                                + uint(output_base) + output_row
                                + projection_row_offset;
                            decode_vq_group24(
                                descriptor, vq_indices, vq_state, vq_aux,
                                vq_anchors, vq_codebooks, vq_scales,
                                vq_state_to_bank, vq_banks, vq_parameters,
                                Ws + output_row * uint(W_STRIDE)
                                    + local_group * 24u,
                                pool_row, group,
                                uint(input_width));
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

#if (MFQ_GROUPED_FAMILY_MASK & 2) != 0
            }
#endif
            const int k_limit = BK == 96 && !ALIGNED_INPUT ? BK
                : min(BK, input_width - k_base);
            for (int kk = 0; kk < k_limit && simd_m > 0 && simd_n > 0; kk += SK) {
                mlx::steel::NAXTile<half, TM, TK> input_tile;
                mlx::steel::NAXTile<half, TN, TK> weight_tile;
                if constexpr (DIRECT_ACTIVATION) {
                    short2 coord = mlx::steel::BaseNAXFrag::get_coord();
#pragma clang loop unroll(full)
                    for (short fragment_m = 0; fragment_m < TM; ++fragment_m) {
#pragma clang loop unroll(full)
                    for (short fragment_k = 0; fragment_k < TK; ++fragment_k) {
#pragma clang loop unroll(full)
                        for (short half_row = 0; half_row < 2; ++half_row) {
                            uint column = uint(k_base + kk + fragment_k * 16)
                                + uint(coord.x);
                            half4 values = half4(0.0h);
                            if constexpr (ALIGNED_INPUT) {
                                values = half4(*reinterpret_cast<device const packed_half4*>(
                                    x + fragment_offsets[fragment_m][half_row]
                                        + uint(k_base + kk + fragment_k * 16)));
                            } else if (fragment_valid[fragment_m][half_row]
                                && column + 4u <= uint(input_width)) {
                                values = half4(*reinterpret_cast<device const packed_half4*>(
                                    x + fragment_offsets[fragment_m][half_row]
                                        + uint(k_base + kk + fragment_k * 16)));
                            } else if (fragment_valid[fragment_m][half_row]) {
#pragma clang loop unroll(full)
                                for (short lane = 0; lane < 4; ++lane) {
                                    if (column + uint(lane) < uint(input_width)) {
                                        values[lane] = x[fragment_offsets[fragment_m][half_row]
                                            + uint(k_base + kk + fragment_k * 16 + lane)];
                                    }
                                }
                            }
#pragma clang loop unroll(full)
                            for (short lane = 0; lane < 4; ++lane) {
                                input_tile.frag_at(fragment_m, fragment_k)[half_row * 4 + lane]
                                    = values[lane];
                            }
                        }
                    }
                    }
                } else {
                    input_tile.template load<half, X_STRIDE, 1>(
                        Xs + int(tm) * X_STRIDE + kk);
                }
                if constexpr (DIRECT_PACKED) {
                    // BaseNAXFrag assigns every SIMD lane two rows and four
                    // contiguous columns from each 16x16 fragment.  Decode
                    // those packed NINT values straight into the fragment;
                    // no FP16 weight tile or threadgroup round trip exists.
                    short2 fragment_coord =
                        mlx::steel::BaseNAXFrag::get_coord();
#pragma clang loop unroll(full)
                    for (short fragment_n = 0;
                         fragment_n < TN;
                         ++fragment_n) {
#pragma clang loop unroll(full)
                    for (short fragment_k = 0;
                         fragment_k < TK;
                         ++fragment_k) {
                        thread auto& weight_fragment =
                            weight_tile.frag_at(fragment_n, fragment_k);
#pragma clang loop unroll(full)
                        for (short fragment_row = 0;
                             fragment_row < 2;
                             ++fragment_row) {
                            int local_output = int(tn)
                                + int(fragment_n) * 16
                                + int(fragment_coord.y)
                                + int(fragment_row) * 8;
                            int input_column = k_base + kk
                                + int(fragment_k) * 16
                                + int(fragment_coord.x);
                            half4 decoded = half4(0.0h);
                            if (local_output < int(valid_n)
                                && input_column < input_width) {
                                uint pool_row = local_expert
                                        * uint(matrix_output_width)
                                    + uint(output_base + local_output)
                                    + projection_row_offset;
                                decoded = decode_nint_row_quad_at(
                                    descriptor,
                                    nint_q,
                                    nint_sub_scale,
                                    nint_sub_min,
                                    nint_anchor_scale,
                                    nint_anchor_min,
                                    pool_row,
                                    uint(input_column),
                                    uint(input_width));
                            }
                            weight_fragment[fragment_row * 4] = decoded[0];
                            weight_fragment[fragment_row * 4 + 1] = decoded[1];
                            weight_fragment[fragment_row * 4 + 2] = decoded[2];
                            weight_fragment[fragment_row * 4 + 3] = decoded[3];
                        }
                    }
                    }
                } else {
                    weight_tile.template load<half, W_STRIDE, 1>(
                        Ws + int(tn) * W_STRIDE + kk);
                }
                if (projection == 0) {
                    mlx::steel::tile_matmad_nax(
                        gate_tile,
                        input_tile,
                        metal::bool_constant<false>{},
                        weight_tile,
                        metal::bool_constant<true>{});
                } else {
                    mlx::steel::tile_matmad_nax(
                        up_tile,
                        input_tile,
                        metal::bool_constant<false>{},
                        weight_tile,
                        metal::bool_constant<true>{});
                }
            }
            if constexpr (!PING_PONG) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

#if (MFQ_GROUPED_FAMILY_MASK & 2) != 0
    };
    if constexpr (FIXED_VQ_GEOMETRY) {
        bool eligible = params.projections == 1 && input_width % 32 == 0
            && uint(base_descriptor[0]) == 1u;
        if (eligible && (uint(base_descriptor[28]) & 255u) == 8u
            && base_descriptor[29] == 6) {
            execute_k(metal::integral_constant<int, 8>{},
                metal::integral_constant<uint, 6u>{});
#ifdef MFQ_ENABLE_LEGACY_VQ_VECTOR
        } else if (eligible && (uint(base_descriptor[28]) & 255u) == 6u
            && base_descriptor[29] == 4) {
            execute_k(metal::integral_constant<int, 6>{},
                metal::integral_constant<uint, 4u>{});
#endif
        } else if (eligible && base_descriptor[28] == 1 && base_descriptor[29] == 1) {
            execute_k(metal::integral_constant<int, 1>{},
                metal::integral_constant<uint, 1u>{});
        } else if (eligible && base_descriptor[28] == 4 && base_descriptor[29] == 1) {
            execute_k(metal::integral_constant<int, 4>{},
                metal::integral_constant<uint, 1u>{});
        } else if (eligible && base_descriptor[28] == 3 && base_descriptor[29] == 5) {
            execute_k(metal::integral_constant<int, 3>{},
                metal::integral_constant<uint, 5u>{});
        } else if (eligible && base_descriptor[28] == 7 && base_descriptor[29] == 2) {
            execute_k(metal::integral_constant<int, 7>{},
                metal::integral_constant<uint, 2u>{});
        } else if (eligible && base_descriptor[28] == 7 && base_descriptor[29] == 3) {
            execute_k(metal::integral_constant<int, 7>{},
                metal::integral_constant<uint, 3u>{});
        } else if (eligible && base_descriptor[28] == 8 && base_descriptor[29] == 3) {
            execute_k(metal::integral_constant<int, 8>{},
                metal::integral_constant<uint, 3u>{});
        } else {
            execute_k(metal::integral_constant<int, 0>{},
                metal::integral_constant<uint, 0xffffffffu>{});
        }
    } else {
        execute_k(metal::integral_constant<int, 0>{},
            metal::integral_constant<uint, 0xffffffffu>{});
    }
#endif
    if constexpr (FUSED_SWIGLU) {
        for (short item = 0;
             item < decltype(gate_tile)::kElemsPerTile;
             ++item) {
            float gate = gate_tile.elems()[item];
            float up = up_tile.elems()[item];
            if (params.swiglu_limit > 0.0f) {
                gate = min(gate, params.swiglu_limit);
                up = clamp(up, -params.swiglu_limit, params.swiglu_limit);
            }
            gate_tile.elems()[item] =
                gate / (1.0f + exp(-gate)) * up;
        }
    }
    if (simd_m > 0 && simd_n > 0) {
        device half* destination =
            y + (row_base + int(tm)) * output_width
                + output_base + int(tn);
        gate_tile.store_safe(
            destination,
            output_width,
            short2(simd_n, simd_m));
    }
}

#define instantiate_mfq_grouped_mfe_nax( \
    name, bm, bn, x_stride, w_stride, fused, direct) \
    template [[host_name(name)]] [[kernel]] \
    decltype(mfq_grouped_mfe_nax_f16< \
        bm, bn, x_stride, w_stride, fused, direct>) \
    mfq_grouped_mfe_nax_f16< \
        bm, bn, x_stride, w_stride, fused, direct>;

instantiate_mfq_grouped_mfe_nax(
    "mfq_grouped_mfe_nax_f16_specialized",
    MFQ_GROUPED_NAX_BM,
    MFQ_GROUPED_NAX_BN,
    MFQ_GROUPED_NAX_X_STRIDE,
    MFQ_GROUPED_NAX_W_STRIDE,
    MFQ_GROUPED_NAX_FUSED,
    MFQ_GROUPED_NAX_DIRECT)
#endif
