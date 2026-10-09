#pragma once

namespace mfq::metal::detail {

inline constexpr char kNintMetadataSource[] = R"METAL(
struct MfqNintRow {
    uint q_bits;
    uint q_shift;
    uint q_offset;
    uint sub_bits;
    uint sub_shift;
    uint sub_offset;
    float scale;
    float minimum;
};

inline MfqNintRow mfq_nint_row(uint4 metadata) {
    const float2 anchors = float2(as_type<half2>(metadata.z));
    return {
        metadata.x & 15u,
        (metadata.x >> 4u) & 7u,
        metadata.y,
        (metadata.x >> 8u) & 15u,
        (metadata.x >> 12u) & 7u,
        metadata.w,
        anchors.x,
        anchors.y
    };
}

inline MfqNintRow mfq_nint_row(device const uint* metadata, uint row) {
    return mfq_nint_row(*reinterpret_cast<device const uint4*>(metadata + row * 4u));
}

inline MfqNintRow mfq_nint_row(constant const uint* metadata, uint row) {
    return mfq_nint_row(*reinterpret_cast<constant const uint4*>(metadata + row * 4u));
}

inline uint mfq_nint_sub_word(device const uchar* stream, uint offset) {
    return uint(as_type<ushort>(
        *reinterpret_cast<device const packed_uchar2*>(stream + offset)));
}

inline uint mfq_nint_sub_word(constant const uchar* stream, uint offset) {
    return uint(as_type<ushort>(
        *reinterpret_cast<constant const packed_uchar2*>(stream + offset)));
}

template <typename Stream>
inline uint mfq_nint_sub_value(Stream stream, uint row_offset,
    uint row_shift, uint bits, uint group) {
    uint bit = row_shift + group * bits;
    uint offset = row_offset + (bit >> 3u);
    uint value = bits == 8u ? uint(stream[offset]) : mfq_nint_sub_word(stream, offset);
    return (value >> (bit & 7u)) & ((1u << bits) - 1u);
}
)METAL";

}
