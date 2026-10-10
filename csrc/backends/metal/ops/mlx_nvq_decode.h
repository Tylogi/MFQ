#pragma once

namespace mfq::metal {

inline constexpr const char* kNvqVectorInputHeader = R"METAL(
template <typename T> struct MfqNvqVectorInput {
    device const T* values;
    T operator[](uint index) const { return values[index]; }
};
template <uint K, typename T>
inline MfqNvqVectorInput<T> mfq_nvq_input_row(MfqNvqVectorInput<T> x, uint row) {
    return {x.values + row * K};
}
template <uint K, typename T>
__attribute__((always_inline)) inline float4 mfq_nvq_load_input4(
    MfqNvqVectorInput<T> x, uint offset, uint column) {
    static_assert(K % 4u == 0u);
    return float4(*(device const vec<T, 4>*)(x.values + offset));
}
)METAL";

inline constexpr const char* kNvqDecodeHeader = R"METAL(
inline float4 mfq_nvq_load_code4(device const int8_t* stream, uint offset) {
    return float4(*(device const char4*)(stream + offset));
}

inline float4 mfq_nvq_load_code4(constant const int8_t* stream, uint offset) {
    return float4(*(constant const char4*)(stream + offset));
}

inline uint mfq_nvq_load_record4(device const uchar* stream, uint offset) {
    return as_type<uint>(*(device const packed_uchar4*)(stream + offset));
}

inline uint mfq_nvq_load_record4(constant const uchar* stream, uint offset) {
    return as_type<uint>(*(constant const packed_uchar4*)(stream + offset));
}

inline uint mfq_nvq_load_record2(device const uchar* stream, uint offset) {
    return uint(as_type<ushort>(*(device const packed_uchar2*)(stream + offset)));
}

inline uint mfq_nvq_load_record2(constant const uchar* stream, uint offset) {
    return uint(as_type<ushort>(*(constant const packed_uchar2*)(stream + offset)));
}

inline uint2 mfq_nvq_load_record8(device const uchar* stream, uint offset) {
    return *(device const uint2*)(stream + offset);
}

inline uint2 mfq_nvq_load_record8(constant const uchar* stream, uint offset) {
    return *(constant const uint2*)(stream + offset);
}

inline uint2 mfq_nvq_bit_cursor(uint value_index, uint bits) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index = (value_index >> 3u) * bits + (residual_bits >> 3u);
    return uint2(byte_index, residual_bits & 7u);
}

template <typename Stream>
inline uint mfq_nvq_read_bits(Stream stream, uint value_index, uint bits) {
    uint2 cursor = mfq_nvq_bit_cursor(value_index, bits);
    uint byte_index = cursor.x, shift = cursor.y;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) packed |= uint(stream[byte_index + 1u]) << 8u;
    if (shift + bits > 16u) packed |= uint(stream[byte_index + 2u]) << 16u;
    return (packed >> shift) & ((1u << bits) - 1u);
}

template <uint K, typename XStream>
inline auto mfq_nvq_input_row(XStream x, uint row) {
    return x + row * K;
}

template <
    uint VECTOR_SIZE,
    uint K,
    uint MATRIX_ROWS,
    uint EXECUTION_LAYOUT,
    bool EXPANSION,
    uint FIXED_INDEX_BITS,
    uint INPUT_ROWS = 1u,
    typename XStream,
    typename IndexStream,
    typename StateStream,
    typename AuxStream,
    typename ScaleStream,
    typename StateBankStream,
    typename CodebookStream
>
inline void mfq_nvq_jsc_profile(
    XStream x,
    IndexStream indices_stream,
    StateStream state_stream,
    AuxStream aux_stream,
    ScaleStream scale_stream,
    StateBankStream state_bank_stream,
    CodebookStream codebook_stream,
    uint x_offset,
    thread const uint* outputs,
    thread const float* row_anchors,
    thread float* accumulators,
    uint index_bits,
    uint entries,
    uint indices_offset,
    uint state_offset,
    uint aux_offset,
    uint codebook_offset,
    uint scale_offset,
    uint state_bank_offset,
    uint k_lane,
    uint k_lanes,
    uint execution_layout
) {
    constexpr uint groups = (K + 23u) / 24u;
    constexpr uint vectors = (K + VECTOR_SIZE - 1u) / VECTOR_SIZE;
    constexpr uint signs = (K + 7u) / 8u;
    constexpr uint VECTOR_SHIFT =
        VECTOR_SIZE == 4u ? 2u : 3u;
    constexpr uint BYTES_PER_SIGN =
        VECTOR_SIZE == 4u ? 3u : 2u;
    constexpr uint WIDE_BYTES_PER_SIGN =
        VECTOR_SIZE == 4u ? 4u : 3u;
    uint index_width = FIXED_INDEX_BITS != 0u ? FIXED_INDEX_BITS : index_bits;
    uint entry_count = FIXED_INDEX_BITS != 0u ? (1u << FIXED_INDEX_BITS) : entries;
    for (uint work = k_lane;
         work < (EXPANSION ? signs : groups);
         work += k_lanes) {
        uint group = EXPANSION ? work / 3u : work;
        uint selected_code_banks[MATRIX_ROWS];
        float weight_scales[MATRIX_ROWS];
        uint2 group_records[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint state_index = outputs[row] * groups + group;
            uint state;
            if (EXECUTION_LAYOUT != 0u && FIXED_INDEX_BITS != 8u && execution_layout == 2u) {
                uint record_offset = indices_offset + state_index * 8u;
                group_records[row] = mfq_nvq_load_record8(indices_stream, record_offset);
                state = group_records[row].y >> 28u;
            } else {
                uint state_byte = uint(state_stream[
                    state_offset + (state_index >> 1)
                ]);
                state = (
                    state_byte >> ((state_index & 1u) * 4u)
                ) & 15u;
            }
            selected_code_banks[row] = uint(
                state_bank_stream[state_bank_offset + state]);
            weight_scales[row] =
                row_anchors[row]
                * scale_stream[scale_offset + state];
        }
        for (uint block = 0u; block < (EXPANSION ? 1u : 3u); ++block) {
            uint sign_block = EXPANSION ? work % 3u : block;
            uint column_base = group * 24u + sign_block * 8u;
            if constexpr (!EXPANSION && K % 24u != 0u) {
                if (column_base >= K) break;
            }
            float4 activation0 = INPUT_ROWS == 1u
                ? float4(float(x[x_offset + column_base]),
                    float(x[x_offset + column_base + 1u]),
                    float(x[x_offset + column_base + 2u]),
                    float(x[x_offset + column_base + 3u])) : float4(0.0f);
            float4 activation1 = INPUT_ROWS != 1u || (K & 7u) != 0u
                && column_base + 4u >= K ? float4(0.0f)
                : float4(float(x[x_offset + column_base + 4u]),
                    float(x[x_offset + column_base + 5u]),
                    float(x[x_offset + column_base + 6u]),
                    float(x[x_offset + column_base + 7u]));
            uint first_vector = column_base >> VECTOR_SHIFT;
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint index0;
                uint index1 = 0u;
                uint sign_value;
                if (EXECUTION_LAYOUT != 0u && FIXED_INDEX_BITS != 8u && execution_layout == 2u) {
                    uint2 record = group_records[row];
                    uint segment = sign_block == 0u
                        ? record.x & 0xfffffu
                        : (sign_block == 1u
                            ? ((record.x >> 20u) | (record.y << 12u))
                                & 0xfffffu
                            : (record.y >> 8u) & 0xfffffu);
                    index0 = segment & 4095u;
                    sign_value = segment >> 12u;
                } else if (EXECUTION_LAYOUT != 0u && FIXED_INDEX_BITS != 8u && execution_layout == 3u) {
                    uint execution_offset = indices_offset + (
                        outputs[row] * signs + column_base / 8u
                    ) * WIDE_BYTES_PER_SIGN;
                    uint record = VECTOR_SIZE == 4u
                        ? mfq_nvq_load_record4(indices_stream, execution_offset)
                        : uint(indices_stream[execution_offset])
                            | (uint(indices_stream[execution_offset + 1u]) << 8u)
                            | (uint(indices_stream[execution_offset + 2u]) << 16u);
                    uint index_mask = (1u << index_width) - 1u;
                    index0 = record & index_mask;
                    if (VECTOR_SIZE == 4u) {
                        index1 = (record >> index_width) & index_mask;
                    }
                    sign_value = record >> (
                        index_width * (VECTOR_SIZE == 4u ? 2u : 1u));
                } else if (EXECUTION_LAYOUT != 0u && execution_layout == 1u) {
                    uint execution_offset = indices_offset + (
                        outputs[row] * signs + column_base / 8u
                    ) * BYTES_PER_SIGN;
                    if (VECTOR_SIZE == 4u) {
                        uint record = mfq_nvq_load_record2(indices_stream, execution_offset);
                        index0 = record & 255u;
                        index1 = record >> 8u;
                        sign_value = uint(indices_stream[execution_offset + 2u]);
                    } else {
                        uint record = mfq_nvq_load_record2(indices_stream, execution_offset);
                        index0 = record & 255u;
                        sign_value = record >> 8u;
                    }
                } else {
                    uint index_linear =
                        outputs[row] * vectors + first_vector;
                    index0 = index_width == 8u
                        ? uint(indices_stream[
                              indices_offset + index_linear])
                        : mfq_nvq_read_bits(
                              indices_stream + indices_offset,
                              index_linear,
                              index_width);
                    if (VECTOR_SIZE == 4u) {
                        index1 = index_width == 8u
                            ? uint(indices_stream[
                                  indices_offset
                                  + index_linear + 1u])
                            : mfq_nvq_read_bits(
                                  indices_stream + indices_offset,
                                  index_linear + 1u,
                                  index_width);
                    }
                    sign_value = mfq_nvq_read_bits(
                        aux_stream + aux_offset,
                        outputs[row] * signs + column_base / 8u,
                        7u);
                }
                uint bank_base =
                    selected_code_banks[row] * entry_count;
                uint code0_offset =
                    (bank_base + index0) * VECTOR_SIZE;
                uint code1_offset = VECTOR_SIZE == 4u
                    ? (bank_base + index1) * 4u
                    : code0_offset + 4u;
                bool4 negative0 = bool4(
                    (sign_value & 1u) != 0u,
                    (sign_value & 2u) != 0u,
                    (sign_value & 4u) != 0u,
                    (sign_value & 8u) != 0u);
                bool4 negative1 = bool4(
                    (sign_value & 16u) != 0u,
                    (sign_value & 32u) != 0u,
                    (sign_value & 64u) != 0u,
                    EXECUTION_LAYOUT != 0u && execution_layout != 0u
                        ? (sign_value & 128u) != 0u
                        : (popcount(sign_value) & 1u) != 0u);
                float4 code0 = mfq_nvq_load_code4(
                    codebook_stream,
                    codebook_offset + code0_offset);
                float4 code1 = mfq_nvq_load_code4(
                    codebook_stream,
                    codebook_offset + code1_offset);
                code0 = select(code0, -code0, negative0);
                code1 = select(code1, -code1, negative1);
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
                    float4 input0 = activation0;
                    float4 input1 = activation1;
                    if constexpr (INPUT_ROWS > 1u) {
                        auto input = mfq_nvq_input_row<K>(x, input_row) + x_offset + column_base;
                        input0 = float4(float(input[0]), float(input[1]),
                            float(input[2]), float(input[3]));
                        input1 = (K & 7u) != 0u && column_base + 4u >= K
                            ? float4(0.0f) : float4(float(input[4]), float(input[5]),
                                float(input[6]), float(input[7]));
                    }
                    uint target = row * INPUT_ROWS + input_row;
                    accumulators[target] = fma(weight_scales[row],
                        dot(input0, code0) + dot(input1, code1), accumulators[target]);
                }
            }
        }
    }
}

template <uint K, typename XStream>
__attribute__((always_inline)) inline float4 mfq_nvq_load_input4(XStream x, uint offset, uint column) {
    if constexpr (K % 4u == 0u) {
        return float4(float(x[offset]), float(x[offset + 1u]),
            float(x[offset + 2u]), float(x[offset + 3u]));
    }
    return float4(float(x[offset]),
        column + 1u < K ? float(x[offset + 1u]) : 0.0f,
        column + 2u < K ? float(x[offset + 2u]) : 0.0f,
        column + 3u < K ? float(x[offset + 3u]) : 0.0f);
}

template <uint MATRIX_ROWS, uint K, uint K_LANES, uint INDEX_BITS,
          uint INPUT_ROWS = 1u, typename XStream, typename IndexStream,
          typename StateStream, typename AuxStream, typename ScaleStream, typename CodebookStream>
inline void mfq_nvq1_profile(
    XStream x, IndexStream indices, StateStream states, AuxStream auxiliary,
    ScaleStream scales, CodebookStream codebooks,
    thread const uint* outputs, thread const float* anchors,
    thread float* accumulators, float delta, uint x_offset,
    uint indices_offset, uint state_offset, uint auxiliary_offset,
    uint scale_offset, uint codebook_offset, uint k_lane
) {
    constexpr uint GROUPS = (K + 23u) / 24u;
    constexpr uint VECTORS = (K + 7u) / 8u;
    constexpr uint STATE_BITS = INDEX_BITS == 9u ? 4u : 3u;
    static_assert(K_LANES % 8u == 0u);
    uint4 cursors[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        uint2 index_cursor = mfq_nvq_bit_cursor(
            outputs[row] * VECTORS + k_lane * 3u, INDEX_BITS);
        uint state_index = outputs[row] * GROUPS + k_lane;
        uint2 state_cursor = mfq_nvq_bit_cursor(state_index, STATE_BITS);
        cursors[row] = uint4(indices_offset + index_cursor.x,
            state_offset + state_cursor.x, auxiliary_offset + (state_index >> 3u),
            index_cursor.y | (state_cursor.y << 3u) | ((state_index & 7u) << 6u));
    }
    #pragma clang loop unroll_count(INPUT_ROWS == 1u ? 2 : 1)
    for (uint group = k_lane; group < GROUPS; group += K_LANES) {
        uint2 records[MATRIX_ROWS];
        float weight_scales[MATRIX_ROWS];
        float signed_deltas[MATRIX_ROWS];
        uint banks[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint4 cursor = cursors[row];
            uint low = mfq_nvq_load_record4(indices, cursor.x);
            uint high = uint(indices[cursor.x + 4u]);
            uint shift = cursor.w & 7u;
            records[row] = uint2((low >> shift)
                | (shift != 0u ? high << (32u - shift) : 0u), high >> shift);
            uint state;
            if constexpr (STATE_BITS == 3u) {
                state = (mfq_nvq_load_record2(states, cursor.y)
                    >> ((cursor.w >> 3u) & 7u)) & 7u;
            } else {
                state = (uint(states[cursor.y]) >> ((cursor.w >> 3u) & 7u)) & 15u;
            }
            uint selector = (uint(auxiliary[cursor.z]) >> (cursor.w >> 6u)) & 1u;
            cursors[row] += uint4(3u * K_LANES * INDEX_BITS / 8u,
                K_LANES * STATE_BITS / 8u, K_LANES / 8u, 0u);
            weight_scales[row] = anchors[row] * scales[scale_offset + state];
            signed_deltas[row] = selector != 0u ? -delta : delta;
            banks[row] = INDEX_BITS == 9u ? selector : 0u;
        }
        #pragma clang loop unroll(full)
        for (uint local = 0u; local < 3u; ++local) {
            uint column = group * 24u + local * 8u;
            if (column >= K) break;
            float4 shared_input0 = INPUT_ROWS == 1u
                ? mfq_nvq_load_input4<K>(x, x_offset + column, column) : float4(0.0f);
            float4 shared_input1 = INPUT_ROWS == 1u && column + 4u < K
                ? mfq_nvq_load_input4<K>(x, x_offset + column + 4u, column + 4u) : float4(0.0f);
            float delta_input_sum = dot(shared_input0 + shared_input1, float4(1.0f));
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint index = INDEX_BITS == 9u ? (records[row].x >> (local * 9u)) & 511u
                    : (local == 2u ? ((records[row].x >> 22u) | (records[row].y << 10u)) & 2047u
                        : (records[row].x >> (local * 11u)) & 2047u);
                uint offset = codebook_offset + (banks[row] * 512u + index) * 8u;
                float4 code0 = mfq_nvq_load_code4(codebooks, offset);
                float4 code1 = mfq_nvq_load_code4(codebooks, offset + 4u);
                if constexpr (INPUT_ROWS > 1u) {
                    code0 += signed_deltas[row];
                    code1 += signed_deltas[row];
                }
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
                    float4 activation0 = shared_input0;
                    float4 activation1 = shared_input1;
                    if constexpr (INPUT_ROWS > 1u) {
                        auto input = mfq_nvq_input_row<K>(x, input_row);
                        activation0 = mfq_nvq_load_input4<K>(input, x_offset + column, column);
                        activation1 = column + 4u >= K ? float4(0.0f)
                            : mfq_nvq_load_input4<K>(input, x_offset + column + 4u, column + 4u);
                    }
                    float value = dot(activation0, code0) + dot(activation1, code1);
                    if constexpr (INPUT_ROWS == 1u) {
                        value = fma(signed_deltas[row], delta_input_sum, value);
                    }
                    uint target = row * INPUT_ROWS + input_row;
                    accumulators[target] = fma(weight_scales[row], value, accumulators[target]);
                }
            }
        }
    }
}

template <uint INDEX_BITS, uint MATRIX_ROWS, uint K, uint K_LANES, bool EXPANSION,
          uint VECTOR_SIZE, uint INPUT_ROWS,
          typename XStream, typename IndexStream, typename StateStream,
          typename AuxStream, typename StateBankStream, typename ScaleStream, typename CodebookStream>
inline void mfq_nvq_banked_bits(
    XStream x,
    IndexStream indices,
    StateStream states,
    AuxStream auxiliary,
    StateBankStream state_banks,
    ScaleStream scales,
    CodebookStream codebooks,
    thread const uint* outputs,
    thread const float* anchors,
    thread float* accumulators,
    uint x_offset,
    uint indices_offset,
    uint state_offset,
    uint auxiliary_offset,
    uint state_bank_offset,
    uint scale_offset,
    uint codebook_offset,
    uint k_lane,
    uint auxiliary_mode
) {
    constexpr uint GROUPS = (K + 23u) / 24u;
    constexpr uint SIGNS = (K + 7u) / 8u;
    constexpr uint VECTORS = (K + VECTOR_SIZE - 1u) / VECTOR_SIZE;
    constexpr uint index_width = INDEX_BITS;
    constexpr bool FLAT_RECORDS = EXPANSION && INPUT_ROWS > 1u;
    constexpr uint WORK = FLAT_RECORDS ? SIGNS : GROUPS;
    constexpr bool STATIC_RECORDS = !FLAT_RECORDS && VECTOR_SIZE == 4u
        && INDEX_BITS == 9u && INPUT_ROWS > 1u;
    uint state_rows[MATRIX_ROWS];
    uint4 cursors[MATRIX_ROWS];
    static_assert(K_LANES % 8u == 0u);
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        state_rows[row] = outputs[row] * GROUPS;
        uint2 index = mfq_nvq_bit_cursor(outputs[row] * VECTORS
            + k_lane * ((FLAT_RECORDS ? 8u : 24u) / VECTOR_SIZE), index_width);
        uint2 sign = mfq_nvq_bit_cursor(outputs[row] * SIGNS
            + k_lane * (FLAT_RECORDS ? 1u : 3u), 7u);
        uint state = state_rows[row] + k_lane;
        cursors[row] = uint4(indices_offset + index.x, auxiliary_offset + sign.x,
            state_offset + (state >> 1u), index.y | (sign.y << 3u) | ((state & 1u) << 6u));
    }
    #pragma clang loop unroll_count(2)
    for (uint work = k_lane; work < WORK; work += K_LANES) {
        uint group = FLAT_RECORDS ? work / 3u : work;
        float weight_scales[MATRIX_ROWS];
        uint banks[MATRIX_ROWS], signs[MATRIX_ROWS];
        uint2 packed[MATRIX_ROWS];
        uint3 static_records[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint state;
            if constexpr (FLAT_RECORDS) {
                uint state_index = state_rows[row] + group;
                state = (uint(states[state_offset + (state_index >> 1u)])
                    >> ((state_index & 1u) * 4u)) & 15u;
            } else {
                state = (uint(states[cursors[row].z]) >> ((cursors[row].w >> 6u) * 4u)) & 15u;
            }
            weight_scales[row] = scales[scale_offset + state];
            banks[row] = uint(state_banks[state_bank_offset + state]) << index_width;
            if constexpr (!FLAT_RECORDS) {
                uint4 cursor = cursors[row];
                uint offset = cursor.x;
                uint low = mfq_nvq_load_record4(indices, offset);
                uint high = mfq_nvq_load_record4(indices, offset + 4u);
                uint index_shift = cursor.w & 7u;
                uint upper = 0u;
                if constexpr (VECTOR_SIZE == 4u && INDEX_BITS == 10u && VECTORS % 2u != 0u)
                    upper = uint(indices[offset + 8u]);
                packed[row] = uint2((low >> index_shift)
                    | (index_shift != 0u ? high << (32u - index_shift) : 0u),
                    (high >> index_shift) | (index_shift != 0u ? upper << (32u - index_shift) : 0u));
                if constexpr (STATIC_RECORDS) {
                    constexpr uint PAIR_BITS = 2u * INDEX_BITS;
                    static_records[row] = uint3(packed[row].x,
                        (packed[row].x >> PAIR_BITS) | (packed[row].y << (32u - PAIR_BITS)),
                        packed[row].y >> (2u * PAIR_BITS - 32u));
                }
                signs[row] = mfq_nvq_load_record4(auxiliary, cursor.y) >> ((cursor.w >> 3u) & 7u);
                cursors[row] += uint4(K_LANES * (24u / VECTOR_SIZE) * index_width / 8u,
                    K_LANES * 21u / 8u, K_LANES / 2u, 0u);
            }
        }
        #pragma clang loop unroll_count(INPUT_ROWS == 1u ? 3 : 1)
        for (uint block = 0u; block < (FLAT_RECORDS ? 1u : 3u); ++block) {
            uint record_index = FLAT_RECORDS ? work : group * 3u + block;
            uint column = record_index * 8u;
            if (column >= K) break;
            constexpr bool REUSE_INPUTS = INPUT_ROWS == 1u;
            float4 activation0[REUSE_INPUTS ? INPUT_ROWS : 1u];
            float4 activation1[REUSE_INPUTS ? INPUT_ROWS : 1u];
            if constexpr (REUSE_INPUTS) {
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
                    auto input = mfq_nvq_input_row<K>(x, input_row);
                    activation0[input_row] = mfq_nvq_load_input4<K>(input, x_offset + column, column);
                    activation1[input_row] = column + 4u >= K ? float4(0.0f)
                        : mfq_nvq_load_input4<K>(input, x_offset + column + 4u, column + 4u);
                }
            }
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint record, sign;
                if constexpr (FLAT_RECORDS) {
                    uint4 cursor = cursors[row];
                    record = mfq_nvq_load_record4(indices, cursor.x) >> (cursor.w & 7u);
                    sign = (mfq_nvq_load_record2(auxiliary, cursor.y)
                        >> ((cursor.w >> 3u) & 7u)) & 127u;
                    cursors[row] += uint4(K_LANES * (8u / VECTOR_SIZE) * index_width / 8u,
                        K_LANES * 7u / 8u, 0u, 0u);
                } else {
                    if constexpr (STATIC_RECORDS) record = static_records[row][block];
                    else {
                        uint shift = block * (8u / VECTOR_SIZE) * index_width;
                        record = shift == 0u ? packed[row].x : shift < 32u
                            ? (packed[row].x >> shift) | (packed[row].y << (32u - shift))
                            : packed[row].y >> (shift - 32u);
                    }
                    sign = (signs[row] >> (block * 7u)) & 127u;
                }
                uint index0 = (record & ((1u << index_width) - 1u)) + banks[row];
                uint index1 = VECTOR_SIZE == 4u
                    ? ((record >> index_width) & ((1u << index_width) - 1u)) + banks[row] : index0;
                uint parity = popcount(sign) & 1u;
                if constexpr (INDEX_BITS == 8u)
                    parity ^= auxiliary_mode == 2u ? (index0 >> 7u) & 1u : 0u;
                sign |= parity << 7u;
                uint code_offset = codebook_offset + index0 * VECTOR_SIZE;
                float4 code0 = mfq_nvq_load_code4(codebooks, code_offset);
                float4 code1 = mfq_nvq_load_code4(codebooks, VECTOR_SIZE == 4u
                    ? codebook_offset + index1 * 4u : code_offset + 4u);
                code0 = select(code0, -code0, bool4((sign & 1u) != 0u,
                    (sign & 2u) != 0u, (sign & 4u) != 0u, (sign & 8u) != 0u));
                code1 = select(code1, -code1, bool4((sign & 16u) != 0u,
                    (sign & 32u) != 0u, (sign & 64u) != 0u, (sign & 128u) != 0u));
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
                    float4 input0, input1;
                    if constexpr (REUSE_INPUTS) {
                        input0 = activation0[input_row];
                        input1 = activation1[input_row];
                    } else {
                        auto input = mfq_nvq_input_row<K>(x, input_row);
                        input0 = mfq_nvq_load_input4<K>(input, x_offset + column, column);
                        input1 = column + 4u >= K ? float4(0.0f)
                            : mfq_nvq_load_input4<K>(input, x_offset + column + 4u, column + 4u);
                    }
                    float value = dot(input0, code0) + dot(input1, code1);
                    uint index = row * INPUT_ROWS + input_row;
                    accumulators[index] = fma(weight_scales[row], value, accumulators[index]);
                }
            }
        }
    }
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
            accumulators[row * INPUT_ROWS + input_row] *= anchors[row];
        }
    }
}

template <uint MATRIX_ROWS, uint K, uint K_LANES, bool EXPANSION,
          uint VECTOR_SIZE = 4u, uint INPUT_ROWS = 1u, uint INDEX_BITS = 0u,
          typename XStream, typename IndexStream, typename StateStream,
          typename AuxStream, typename StateBankStream, typename ScaleStream, typename CodebookStream>
inline void mfq_nvq_banked_profile(
    XStream x, IndexStream indices, StateStream states, AuxStream auxiliary,
    StateBankStream state_banks, ScaleStream scales, CodebookStream codebooks,
    thread const uint* outputs, thread const float* anchors, thread float* accumulators,
    uint x_offset, uint indices_offset, uint state_offset, uint auxiliary_offset,
    uint state_bank_offset, uint scale_offset, uint codebook_offset, uint k_lane, uint index_bits,
    uint auxiliary_mode = 1u
) {
    #define MFQ_NVQ_BANKED_BITS(BITS) \
        mfq_nvq_banked_bits<BITS, MATRIX_ROWS, K, K_LANES, EXPANSION, VECTOR_SIZE, INPUT_ROWS>( \
            x, indices, states, auxiliary, state_banks, scales, codebooks, outputs, anchors, accumulators, \
            x_offset, indices_offset, state_offset, auxiliary_offset, state_bank_offset, \
            scale_offset, codebook_offset, k_lane, auxiliary_mode)
    if constexpr (INDEX_BITS != 0u) { MFQ_NVQ_BANKED_BITS(INDEX_BITS); }
    else if (index_bits == 8u) { MFQ_NVQ_BANKED_BITS(8u); }
    else if constexpr (VECTOR_SIZE == 4u) {
        if (index_bits == 9u) { MFQ_NVQ_BANKED_BITS(9u); }
        else { MFQ_NVQ_BANKED_BITS(10u); }
    } else {
        if (index_bits == 10u) { MFQ_NVQ_BANKED_BITS(10u); }
        else { MFQ_NVQ_BANKED_BITS(12u); }
    }
    #undef MFQ_NVQ_BANKED_BITS
}

)METAL";

}
