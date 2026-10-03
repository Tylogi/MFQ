#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;

struct MfqNintPrefillParams {
    int rows;
    int output_width;
    int input_width;
    int group_size;
    int groups;
};

namespace {

inline ushort4 nint_prefill_read_row_quad(
    const device uchar* stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits) {
    const uint row_relative_bits = row_bit_shift + value_index * bits;
    const uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    const uint shift = row_relative_bits & 7u;
    const uint required_bits = shift + 4u * bits;
    const packed_uchar4 bytes =
        *reinterpret_cast<device const packed_uchar4*>(stream + byte_index);
    uint packed = as_type<uint>(bytes);
    if (shift != 0u) {
        packed = (packed >> shift)
            | (required_bits > 32u
                ? uint(stream[byte_index + 4u]) << (32u - shift)
                : 0u);
    }
    const uint mask = (1u << bits) - 1u;
    return ushort4(
        packed & mask,
        (packed >> bits) & mask,
        (packed >> (2u * bits)) & mask,
        (packed >> (3u * bits)) & mask);
}

inline half4 nint_prefill_decode_row_quad(
    const device uchar* values,
    const device uchar* sub_scales,
    const device uchar* sub_mins,
    uint row,
    uint column,
    uint layout,
    uint row_byte_offset,
    float anchor_scale,
    float anchor_minimum,
    constant MfqNintPrefillParams& params) {
    const uint bits = layout & 15u;
    const uint row_bit_shift = layout >> 4u;
    const ushort4 quantized = nint_prefill_read_row_quad(
        values,
        row_byte_offset,
        row_bit_shift,
        column,
        bits);
    const uint group_size = uint(params.group_size);
    const uint first_group = column / group_size;
    if (column + 3u < uint(params.input_width) &&
        first_group == (column + 3u) / group_size) {
        const uint metadata = row * uint(params.groups) + first_group;
        const float scale = anchor_scale * float(sub_scales[metadata]);
        const float minimum = anchor_minimum * float(sub_mins[metadata]);
        return half4(scale * float4(quantized) - minimum);
    }
    half4 decoded = half4(0.0h);
#pragma clang loop unroll(full)
    for (uint lane = 0u; lane < 4u; ++lane) {
        const uint input_column = column + lane;
        if (input_column < uint(params.input_width)) {
            const uint metadata = row * uint(params.groups)
                + input_column / group_size;
            const float scale = anchor_scale * float(sub_scales[metadata]);
            const float minimum = anchor_minimum * float(sub_mins[metadata]);
            decoded[lane] = half(scale * float(quantized[lane]) - minimum);
        }
    }
    return decoded;
}

} // namespace

[[kernel]] void mfq_nint_prefill_mmq_f16_bm128_bn64_bk48(
    const device uchar* q [[buffer(0)]],
    const device uint* row_metadata [[buffer(1)]],
    const device uchar* sub_scale [[buffer(2)]],
    const device uchar* sub_min [[buffer(3)]],
    const device half* x [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant MfqNintPrefillParams& params [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 48;
    constexpr int BK_padded = 56;
    constexpr uint TGP_SIZE = 256u;
    using mma_t = mlx::steel::BlockMMA<
        half,
        half,
        BM,
        BN,
        BK,
        4,
        2,
        false,
        true,
        BK_padded,
        BK_padded>;

    const int output_base = int(tid.x) * BN;
    const int row_base = int(tid.y) * BM;
    if (output_base >= params.output_width || row_base >= params.rows) return;
    const short valid_n = short(min(BN, params.output_width - output_base));
    const short valid_m = short(min(BM, params.rows - row_base));
    threadgroup half Xs[BM * BK_padded];
    threadgroup half Ws[BN * BK_padded];
    threadgroup uint4 row_state[BN];
    thread mma_t mma(simd_group_id, simd_lane_id);

    if (thread_id < uint(BN)) {
        row_state[thread_id] = thread_id < uint(valid_n)
            ? *reinterpret_cast<const device uint4*>(
                  row_metadata + (uint(output_base) + thread_id) * 4u)
            : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k_base = 0; k_base < params.input_width; k_base += BK) {
        for (uint item = thread_id;
             item < uint(BM * BK_padded);
             item += TGP_SIZE) {
            const uint row = item / uint(BK_padded);
            const uint column = item - row * uint(BK_padded);
            const int input_column = k_base + int(column);
            Xs[item] = row < uint(valid_m) && column < uint(BK) &&
                    input_column < params.input_width
                ? x[(uint(row_base) + row) * uint(params.input_width)
                    + uint(input_column)]
                : half(0.0h);
        }
        for (uint item = thread_id;
             item < uint(BN * BK_padded);
             item += TGP_SIZE) {
            Ws[item] = half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        constexpr uint VALUES_PER_ITEM = 4u;
        constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
        for (uint item = thread_id;
             item < uint(BN) * ITEMS_PER_ROW;
             item += TGP_SIZE) {
            const uint output_row = item / ITEMS_PER_ROW;
            const uint local_item = item - output_row * ITEMS_PER_ROW;
            if (output_row < uint(valid_n)) {
                const uint weight_row = uint(output_base) + output_row;
                const uint4 metadata = row_state[output_row];
                const uint local_column = local_item * VALUES_PER_ITEM;
                const uint input_column = uint(k_base) + local_column;
                if (input_column < uint(params.input_width)) {
                    *reinterpret_cast<threadgroup half4*>(
                        Ws + output_row * uint(BK_padded) + local_column) =
                        nint_prefill_decode_row_quad(
                            q,
                            sub_scale,
                            sub_min,
                            weight_row,
                            input_column,
                            metadata.x,
                            metadata.y,
                            as_type<float>(metadata.z),
                            as_type<float>(metadata.w),
                            params);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.mma(Xs, Ws);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mma.store_result_slice(
        y + row_base * params.output_width + output_base,
        params.output_width,
        short2(0, 0),
        short2(valid_n, valid_m));
}

#ifdef MFQ_ENABLE_NAX
template <int BM, int BN, int BK, int WM, int WN>
inline void mfq_nint_prefill_nax_f16_impl(
    const device uchar* q,
    const device uint* row_metadata,
    const device uchar* sub_scale,
    const device uchar* sub_min,
    const device half* x,
    device half* y,
    constant MfqNintPrefillParams& params,
    threadgroup half* Ws,
    threadgroup uint4* row_state,
    uint3 tid,
    uint simd_group_id,
    uint thread_id) {
    constexpr int BK_padded = BK + 8;
    constexpr uint TGP_SIZE = uint(WM * WN * 32);
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short SK = 32;
    constexpr short TM = SM / 16;
    constexpr short TN = SN / 16;
    constexpr short TK = SK / 16;

    const int output_base = int(tid.x) * BN;
    const int row_base = int(tid.y) * BM;
    if (output_base >= params.output_width || row_base >= params.rows) return;
    const short valid_n = short(min(BN, params.output_width - output_base));
    const short valid_m = short(min(BM, params.rows - row_base));
    const short tm = short(SM * int(simd_group_id / WN));
    const short tn = short(SN * int(simd_group_id % WN));
    const short simd_m = short(max(
        0, min(int(SM), int(valid_m) - int(tm))));
    const short simd_n = short(max(
        0, min(int(SN), int(valid_n) - int(tn))));

    mlx::steel::NAXTile<float, TM, TN> output_tile;
    output_tile.clear();
    if (thread_id < uint(BN)) {
        row_state[thread_id] = thread_id < uint(valid_n)
            ? *reinterpret_cast<const device uint4*>(
                  row_metadata + (uint(output_base) + thread_id) * 4u)
            : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k_base = 0; k_base < params.input_width; k_base += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_base + BK > params.input_width) {
            for (uint item = thread_id;
                 item < uint(BN * BK_padded);
                 item += TGP_SIZE) {
                Ws[item] = half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        constexpr uint VALUES_PER_ITEM = 4u;
        constexpr uint ITEMS_PER_ROW = uint(BK) / VALUES_PER_ITEM;
        for (uint item = thread_id;
             item < uint(BN) * ITEMS_PER_ROW;
             item += TGP_SIZE) {
            const uint output_row = item / ITEMS_PER_ROW;
            const uint local_item = item - output_row * ITEMS_PER_ROW;
            if (output_row < uint(valid_n)) {
                const uint weight_row = uint(output_base) + output_row;
                const uint4 metadata = row_state[output_row];
                const uint local_column = local_item * VALUES_PER_ITEM;
                const uint input_column = uint(k_base) + local_column;
                if (input_column < uint(params.input_width)) {
                    *reinterpret_cast<threadgroup half4*>(
                        Ws + output_row * uint(BK_padded) + local_column) =
                        nint_prefill_decode_row_quad(
                            q,
                            sub_scale,
                            sub_min,
                            weight_row,
                            input_column,
                            metadata.x,
                            metadata.y,
                            as_type<float>(metadata.z),
                            as_type<float>(metadata.w),
                            params);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(disable)
        for (int kk = 0; kk < BK; kk += SK) {
            const int input_column = k_base + kk;
            const short valid_k = short(max(
                0, min(int(SK), params.input_width - input_column)));
            mlx::steel::NAXTile<half, TM, TK> input_tile;
            mlx::steel::NAXTile<half, TN, TK> weight_tile;
            const device half* input_base =
                x + (row_base + int(tm)) * params.input_width + input_column;
            if (simd_m == SM && valid_k == SK) {
                input_tile.load(input_base, params.input_width);
            } else {
                input_tile.load_safe(
                    input_base,
                    params.input_width,
                    short2(valid_k, simd_m));
            }
            weight_tile.template load<half, BK_padded, 1>(
                Ws + int(tn) * BK_padded + kk);
            mlx::steel::tile_matmad_nax(
                output_tile,
                input_tile,
                metal::bool_constant<false>{},
                weight_tile,
                metal::bool_constant<true>{});
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* destination =
        y + (row_base + int(tm)) * params.output_width
            + output_base + int(tn);
    if (simd_m == SM && simd_n == SN) {
        output_tile.store(destination, params.output_width);
    } else {
        output_tile.store_safe(
            destination,
            params.output_width,
            short2(simd_n, simd_m));
    }
}

[[kernel]] void mfq_nint_prefill_nax_f16_bm128_bn64_bk96(
    const device uchar* q [[buffer(0)]],
    const device uint* row_metadata [[buffer(1)]],
    const device uchar* sub_scale [[buffer(2)]],
    const device uchar* sub_min [[buffer(3)]],
    const device half* x [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant MfqNintPrefillParams& params [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    threadgroup half Ws[64 * 104];
    threadgroup uint4 row_state[64];
    mfq_nint_prefill_nax_f16_impl<128, 64, 96, 4, 2>(
        q,
        row_metadata,
        sub_scale,
        sub_min,
        x,
        y,
        params,
        Ws,
        row_state,
        tid,
        simd_group_id,
        thread_id);
}
#endif
