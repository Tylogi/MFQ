#pragma once

namespace mfq::metal {

inline constexpr const char* kNvq3jlHeader = R"METAL(
inline float4 mfq_nvq3jl_load_code4(device const int8_t* stream, uint offset) {
    return float4(*(device const char4*)(stream + offset));
}

inline float4 mfq_nvq3jl_load_code4(constant const int8_t* stream, uint offset) {
    return float4(*(constant const char4*)(stream + offset));
}

inline uint mfq_nvq3jl_load_record4(device const uchar* stream, uint offset) {
    return as_type<uint>(*(device const packed_uchar4*)(stream + offset));
}

inline uint mfq_nvq3jl_load_record4(constant const uchar* stream, uint offset) {
    return as_type<uint>(*(constant const packed_uchar4*)(stream + offset));
}

template <uint MATRIX_ROWS, uint K, uint K_LANES, bool EXPANSION,
          typename XStream, typename IndexStream, typename StateStream,
          typename ScaleStream, typename CodebookStream>
inline void mfq_nvq3jl_profile(
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
            uint input = x_offset + column;
            float4 activation0 = float4(float(x[input]), float(x[input + 1u]),
                float(x[input + 2u]), float(x[input + 3u]));
            float4 activation1 = column + 4u >= K ? float4(0.0f) : float4(
                float(x[input + 4u]), float(x[input + 5u]),
                float(x[input + 6u]), float(x[input + 7u]));
            for (uint row = 0u; row < MATRIX_ROWS; ++row) {
                uint offset = indices_offset + (outputs[row] * SIGNS + record_index) * 4u;
                uint record = mfq_nvq3jl_load_record4(indices, offset);
                uint index0 = record & 4095u;
                uint index1 = (record >> 12u) & 4095u;
                uint sign = record >> 24u;
                float4 code0 = mfq_nvq3jl_load_code4(codebooks, codebook_offset + index0 * 4u);
                float4 code1 = mfq_nvq3jl_load_code4(codebooks, codebook_offset + index1 * 4u);
                code0 = select(code0, -code0, bool4((sign & 1u) != 0u,
                    (sign & 2u) != 0u, (sign & 4u) != 0u, (sign & 8u) != 0u));
                code1 = select(code1, -code1, bool4((sign & 16u) != 0u,
                    (sign & 32u) != 0u, (sign & 64u) != 0u, (sign & 128u) != 0u));
                float value = dot(activation0, code0) + dot(activation1, code1);
                accumulators[row] = fma(weight_scales[row], value, accumulators[row]);
            }
        }
    }
    for (uint row = 0u; row < MATRIX_ROWS; ++row) {
        accumulators[row] *= anchors[row];
    }
}
)METAL";

}
