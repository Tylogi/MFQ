#pragma once

namespace mfq::metal {

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

template <typename Stream>
inline uint mfq_nvq_read_bits(Stream stream, uint value_index, uint bits) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index = (value_index >> 3u) * bits + (residual_bits >> 3u);
    uint shift = residual_bits & 7u;
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
          typename ScaleStream, typename CodebookStream>
inline void mfq_nvq1_profile(
    XStream x, IndexStream indices, ScaleStream scales, CodebookStream codebooks,
    thread const uint* outputs, thread const float* anchors,
    thread float* accumulators, float delta, uint x_offset,
    uint indices_offset, uint scale_offset, uint codebook_offset, uint k_lane
) {
    constexpr uint GROUPS = (K + 23u) / 24u;
    constexpr uint RECORD_BYTES = INDEX_BITS == 9u ? 4u : 5u;
    #pragma clang loop unroll_count(INPUT_ROWS == 1u ? 2 : 1)
    for (uint group = k_lane; group < GROUPS; group += K_LANES) {
        uint2 records[MATRIX_ROWS];
        float weight_scales[MATRIX_ROWS];
        float signed_deltas[MATRIX_ROWS];
        uint banks[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint offset = indices_offset + (outputs[row] * GROUPS + group) * RECORD_BYTES;
            uint low = mfq_nvq_load_record4(indices, offset);
            uint high = INDEX_BITS == 9u ? 0u : uint(indices[offset + 4u]);
            uint state = INDEX_BITS == 9u ? (low >> 27u) & 15u : (high >> 1u) & 7u;
            uint selector = INDEX_BITS == 9u ? low >> 31u : (high >> 4u) & 1u;
            records[row] = uint2(low, high);
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

template <uint MATRIX_ROWS, uint K, uint K_LANES, bool EXPANSION,
          uint VECTOR_SIZE = 4u, uint INPUT_ROWS = 1u,
          typename XStream, typename IndexStream, typename StateStream,
          typename ScaleStream, typename CodebookStream>
inline void mfq_nvq_banked_profile(
    XStream x,
    IndexStream indices,
    StateStream states,
    ScaleStream scales,
    CodebookStream codebooks,
    thread const uint* outputs,
    thread const float* anchors,
    thread float* accumulators,
    uint x_offset,
    uint indices_offset,
    uint state_offset,
    uint scale_offset,
    uint codebook_offset,
    uint k_lane
) {
    constexpr uint GROUPS = (K + 23u) / 24u;
    constexpr uint SIGNS = (K + 7u) / 8u;
    constexpr uint RECORD_BYTES = VECTOR_SIZE == 4u ? 4u : 3u;
    constexpr bool FLAT_RECORDS = EXPANSION;
    constexpr uint WORK = FLAT_RECORDS ? SIGNS : GROUPS;
    uint state_rows[MATRIX_ROWS];
    uint state_shifts[MATRIX_ROWS];
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        state_rows[row] = outputs[row] * GROUPS;
        state_shifts[row] = ((state_rows[row] + k_lane) & 1u) * 4u;
    }
    #pragma clang loop unroll_count(2)
    for (uint work = k_lane; work < WORK; work += K_LANES) {
        uint group = FLAT_RECORDS ? work / 3u : work;
        float weight_scales[MATRIX_ROWS];
        for (uint row = 0u; row < MATRIX_ROWS; ++row) {
            uint state_index = state_rows[row] + group;
            uint shift = FLAT_RECORDS ? (state_index & 1u) * 4u : state_shifts[row];
            uint state = (uint(states[state_offset + (state_index >> 1u)])
                >> shift) & 15u;
            weight_scales[row] = scales[scale_offset + state];
        }
        for (uint block = 0u; block < (FLAT_RECORDS ? 1u : 3u); ++block) {
            uint record_index = FLAT_RECORDS ? work : group * 3u + block;
            uint column = record_index * 8u;
            if (column >= K) break;
            constexpr bool REUSE_INPUTS = INPUT_ROWS == 1u || MATRIX_ROWS <= 2u;
            float4 activation0[REUSE_INPUTS ? INPUT_ROWS : 1u];
            float4 activation1[REUSE_INPUTS ? INPUT_ROWS : 1u];
            if constexpr (REUSE_INPUTS) {
                for (uint input_row = 0u; input_row < INPUT_ROWS; ++input_row) {
                    auto input = mfq_nvq_input_row<K>(x, input_row) + x_offset + column;
                    activation0[input_row] = float4(float(input[0]), float(input[1]),
                        float(input[2]), float(input[3]));
                    activation1[input_row] = column + 4u >= K ? float4(0.0f)
                        : float4(float(input[4]), float(input[5]),
                            float(input[6]), float(input[7]));
                }
            }
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint offset = indices_offset + (outputs[row] * SIGNS + record_index) * RECORD_BYTES;
                uint record = VECTOR_SIZE == 4u ? mfq_nvq_load_record4(indices, offset)
                    : mfq_nvq_load_record2(indices, offset)
                        | (uint(indices[offset + 2u]) << 16u);
                uint index0 = record & (VECTOR_SIZE == 4u ? 4095u : 65535u);
                uint index1 = VECTOR_SIZE == 4u ? (record >> 12u) & 4095u : index0;
                uint sign = record >> (VECTOR_SIZE == 4u ? 24u : 16u);
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
                        auto input = mfq_nvq_input_row<K>(x, input_row) + x_offset + column;
                        input0 = float4(float(input[0]), float(input[1]), float(input[2]), float(input[3]));
                        input1 = column + 4u >= K ? float4(0.0f)
                            : float4(float(input[4]), float(input[5]), float(input[6]), float(input[7]));
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
)METAL";

}
