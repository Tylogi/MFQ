#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace mfq::cuda::packed_nvq {
constexpr int kNvq1L = 1;
constexpr int kNvq2 = 2;
constexpr int kNvq3 = 3;
constexpr int kNvq2Exec = 4;
constexpr int kNvq2Jsc = 5;
constexpr int kNvq2JscExec = 6;
constexpr int kNpq0L = 7;
constexpr int kNvq1S = 8;
constexpr int kNpq0S = 9;
constexpr int kNvq3Jsc = 10;
constexpr int kNvq3Jsc2 = 11;
constexpr int kNvq3Jsc512 = 12;
constexpr int kNvq2JscL = 13;
constexpr int kNvq2JscXL = 14;
constexpr int kNvq3JscL = 15;
constexpr int kNvq2JscXLGroupExec = 16;
constexpr int kNvq3JscLGroupExec = 17;
constexpr int kGroupSize = 24;
constexpr int kChunksPerGroup = 6;  // six dp4a chunks of four values
constexpr int kJscMetadataBytes = 64;
constexpr int kJscLutOffset = 4;
constexpr int kJscBankMapOffset = 36;
constexpr int kJscE8CodebookBytes = 256 * 8;
constexpr int kJscE8Codebook1024Bytes = 1024 * 8;
constexpr int kJscE8Codebook4096Bytes = 4096 * 8;
constexpr int kJscE8PartialEntries = 384;
constexpr int kJscE8PartialBytes = kJscE8PartialEntries * 8;
constexpr int kJscD4CodebookBytes = 256 * 4;
constexpr int kJscD4Codebook512Bytes = 512 * 4;
constexpr int kJscD4Codebook1024Bytes = 1024 * 4;
constexpr int kNpq0LMetadataBytes = 64;
constexpr int kNpq0LLutOffset = 8;
constexpr int kNpq0LStates = 8;
constexpr int kNpq0LFirstEntries = 8;
constexpr int kNpq0LSecondEntries = 16;
constexpr int kNpq0LFirstBytesPerState = kNpq0LFirstEntries * 4;
constexpr int kNpq0LSecondBytesPerState = kNpq0LSecondEntries * 4;
constexpr int kNpq0LSecondOffset =
    kNpq0LMetadataBytes + kNpq0LStates * kNpq0LFirstBytesPerState;
constexpr int kNpq0LTableBytes =
    kNpq0LSecondOffset + kNpq0LStates * kNpq0LSecondBytesPerState;
constexpr int kNvq1SBankEntries = 512;
constexpr int kNvq1SBankBytes = kNvq1SBankEntries * 8;
constexpr int kNpq0SMetadataBytes = 64;
constexpr int kNpq0SLutOffset = 8;
constexpr int kNpq0SStates = 4;
constexpr int kNpq0SEntries = 64;
constexpr int kNpq0SCodebookBytes = kNpq0SStates * kNpq0SEntries * 8;
constexpr int kNpq0STableBytes = kNpq0SMetadataBytes + kNpq0SCodebookBytes;
constexpr int kNepq0SEntriesPerHalf = 8;
constexpr int kNepq0SBytesPerStateHalf = kNepq0SEntriesPerHalf * 4;
constexpr int kNepq0SSecondOffset =
    kNpq0SMetadataBytes + kNpq0SStates * kNepq0SBytesPerStateHalf;
constexpr int kNepq0SCompactTableBytes =
    kNepq0SSecondOffset + kNpq0SStates * kNepq0SBytesPerStateHalf;
constexpr int kNepqGroupsPerSupergroup = 4;

__device__ __forceinline__ const int8_t * nepq_active_table(
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    int row,
    int group,
    int supergroups,
    int table_stride) {
    const int selector = group / kNepqGroupsPerSupergroup;
    const uint32_t bank = bank_ids[static_cast<int64_t>(row) * supergroups + selector];
    return table_pool + static_cast<int64_t>(bank) * table_stride;
}

__host__ __device__ constexpr bool is_d4_format(int format) {
    return format == kNvq3 || format == kNvq3Jsc ||
           format == kNvq3Jsc2 || format == kNvq3Jsc512 ||
           format == kNvq3JscL || format == kNvq3JscLGroupExec;
}

__host__ __device__ constexpr bool is_e8_format(int format) {
    return format == kNvq2 || format == kNvq2Exec ||
           format == kNvq2Jsc || format == kNvq2JscExec ||
           format == kNvq2JscL || format == kNvq2JscXL ||
           format == kNvq2JscXLGroupExec;
}

__host__ __device__ constexpr int format_index_bits(int format) {
    if (format == kNvq1L) return 11;
    if (format == kNvq1S || format == kNvq3Jsc512) return 9;
    if (format == kNpq0L) return 7;
    if (format == kNpq0S) return 6;
    if (format == kNvq2JscL || format == kNvq3JscL ||
        format == kNvq3JscLGroupExec) return 10;
    if (format == kNvq2JscXL || format == kNvq2JscXLGroupExec) return 12;
    if (format == kNvq2Exec || format == kNvq2JscExec) return 16;
    return 8;
}

template <int FORMAT>
__device__ __forceinline__ const int8_t * active_codebook(
    const int8_t * metadata,
    uint32_t state) {
    if constexpr (FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscE8CodebookBytes;
    }
    if constexpr (FORMAT == kNvq2JscL) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscE8Codebook1024Bytes;
    }
    if constexpr (FORMAT == kNvq2JscXL) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscE8Codebook4096Bytes;
    }
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        const uint32_t bank = state & 3u;
        return metadata + kJscMetadataBytes + bank * kJscE8Codebook4096Bytes;
    }
    if constexpr (FORMAT == kNvq3Jsc) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4CodebookBytes;
    }
    if constexpr (FORMAT == kNvq3Jsc2) {
        const uint32_t bank = state & 1u;
        return metadata + kJscMetadataBytes + bank * kJscD4CodebookBytes;
    }
    if constexpr (FORMAT == kNvq3Jsc512) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4Codebook512Bytes;
    }
    if constexpr (FORMAT == kNvq3JscL) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4Codebook1024Bytes;
    }
    if constexpr (FORMAT == kNvq3JscLGroupExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(metadata);
        const uint32_t bank = bytes[kJscBankMapOffset + state];
        return metadata + kJscMetadataBytes + bank * kJscD4Codebook1024Bytes;
    }
    return metadata;
}

__device__ __forceinline__ uint32_t load_packed_bits(
    const uint8_t * data, int64_t bit, int bits, int64_t nbytes) {
    const int64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = data[byte];
    if (shift + bits > 8 && byte + 1 < nbytes)
        word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (shift + bits > 16 && byte + 2 < nbytes)
        word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

// One GS24 group contains three 7-bit sign masks.  Load their contiguous
// 21-bit window once so the three vector decodes can reuse it.  Four byte
// reads cover the worst seven-bit starting offset while retaining the exact
// checked tail semantics of load_packed_bits().
__device__ __forceinline__ uint32_t load_packed_sign_group3(
    const uint8_t * data, int64_t bit, int64_t nbytes) {
    const int64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = 0;
#pragma unroll
    for (int offset = 0; offset < 4; ++offset) {
        if (byte + offset < nbytes) {
            word |= static_cast<uint32_t>(data[byte + offset]) << (8 * offset);
        }
    }
    return (word >> shift) & 0x1fffffu;
}

__device__ __forceinline__ uint32_t load_packed_4(
    const uint8_t * data, int64_t linear) {
    return (data[linear >> 1] >> ((linear & 1) * 4)) & 0x0fu;
}

__device__ __forceinline__ uint64_t load_group_exec64(
    const uint8_t * data, int row, int group, int ng) {
    return reinterpret_cast<const uint64_t *>(data)[
        static_cast<int64_t>(row) * ng + group];
}

__device__ __forceinline__ void load_group_exec96_words(
    const uint8_t * data,
    int row,
    int group,
    int ng,
    uint32_t (&words)[3]) {
    const uint32_t * source = reinterpret_cast<const uint32_t *>(data) +
        (static_cast<int64_t>(row) * ng + group) * 3;
    words[0] = source[0];
    words[1] = source[1];
    words[2] = source[2];
}

__device__ __forceinline__ uint32_t load_group_exec96_bits(
    const uint8_t * data, int row, int group, int ng, int bit, int bits) {
    uint32_t words[3];
    load_group_exec96_words(data, row, group, ng, words);
    const int word = bit >> 5;
    const int shift = bit & 31;
    uint32_t value = words[word] >> shift;
    if (shift + bits > 32) value |= words[word + 1] << (32 - shift);
    return value & ((1u << bits) - 1u);
}

__device__ __forceinline__ uint32_t extract_group_exec96_bits_ptr(
    const uint32_t * words, int bit, int bits) {
    const int word = bit >> 5;
    const int shift = bit & 31;
    uint32_t value = words[word] >> shift;
    if (shift + bits > 32) value |= words[word + 1] << (32 - shift);
    return value & ((1u << bits) - 1u);
}

__device__ __forceinline__ int load_i8x4(const int8_t * p) {
    const uint8_t * u = reinterpret_cast<const uint8_t *>(p);
    return static_cast<int>(u[0]) |
           (static_cast<int>(u[1]) << 8) |
           (static_cast<int>(u[2]) << 16) |
           (static_cast<int>(u[3]) << 24);
}

__device__ __forceinline__ int parity7(uint32_t mask) {
    return __popc(mask & 0x7fu) & 1;
}

__device__ __forceinline__ int sign_bytes4(uint32_t mask8, int base) {
    int result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int negative = (mask8 >> (base + i)) & 1u;
        result |= (negative ? 0xff : 0x00) << (8 * i);
    }
    return result;
}

__device__ __forceinline__ int apply_sign4(int values, uint32_t mask8, int base) {
    const int signs = sign_bytes4(mask8, base);
    return __vsub4(values ^ signs, signs);
}

__device__ __forceinline__ int nvq1_l_scale_delta4(int values, int delta) {
    int result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = static_cast<int>(static_cast<int8_t>((values >> (8 * i)) & 0xff));
        const int scaled = 8 * v + delta;
        result |= (scaled & 0xff) << (8 * i);
    }
    return result;
}

__device__ __forceinline__ int nvq1_s_scale_delta4(int values, int delta) {
    int result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int v = static_cast<int>(static_cast<int8_t>((values >> (8 * i)) & 0xff));
        const int scaled = 32 * v + 5 * delta;
        result |= (scaled & 0xff) << (8 * i);
    }
    return result;
}

__device__ __forceinline__ int2 load_npq0_l_vec8(
    const int8_t * metadata,
    uint32_t state,
    uint32_t index) {
    const int8_t * first = metadata + kNpq0LMetadataBytes
        + state * kNpq0LFirstBytesPerState + (index & 7u) * 4;
    const int8_t * second = metadata + kNpq0LSecondOffset
        + state * kNpq0LSecondBytesPerState + (index >> 3) * 4;
    return make_int2(load_i8x4(first), load_i8x4(second));
}

__device__ __forceinline__ int2 load_npq0_s_vec8(
    const int8_t * metadata,
    uint32_t state,
    uint32_t index) {
    const int8_t * code = metadata + kNpq0SMetadataBytes
        + (state * kNpq0SEntries + index) * 8;
    return reinterpret_cast<const int2 *>(code)[0];
}

__device__ __forceinline__ int2 load_nepq0_s_vec8(
    const int8_t * metadata,
    uint32_t state,
    uint32_t index) {
    const int8_t * first = metadata + kNpq0SMetadataBytes
        + state * kNepq0SBytesPerStateHalf + (index & 7u) * 4;
    const int8_t * second = metadata + kNepq0SSecondOffset
        + state * kNepq0SBytesPerStateHalf + (index >> 3) * 4;
    return make_int2(load_i8x4(first), load_i8x4(second));
}

template <int FORMAT>
__device__ __forceinline__ int decode_chunk4(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int group,
    int chunk,
    int nvec,
    int nsign,
    int ng,
    int sign_mode,
    uint32_t state) {
    const int vector8 = group * 3 + (chunk >> 1);
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (vector8 >= nvec || vector8 >= nsign) return 0;
        const int local = vector8 - group * 3;
        const uint64_t metadata = load_group_exec64(indices, row, group, ng);
        const uint32_t segment_metadata =
            (metadata >> (local * 20)) & 0xfffffu;
        const uint32_t index = segment_metadata & 0xfffu;
        const uint32_t mask8 = segment_metadata >> 12;
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(
            load_i8x4(bank + index * 8 + (chunk & 1) * 4),
            mask8, (chunk & 1) * 4);
    } else if constexpr (FORMAT == kNvq3JscLGroupExec) {
        const int vector4 = group * 6 + chunk;
        if (vector4 >= nvec || vector8 >= nsign) return 0;
        const uint32_t index = load_group_exec96_bits(
            indices, row, group, ng, chunk * 10, 10);
        const int local = vector8 - group * 3;
        const uint32_t mask8 = load_group_exec96_bits(
            indices, row, group, ng, 60 + local * 8, 8);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(
            load_i8x4(bank + index * 4), mask8, (chunk & 1) * 4);
    } else if constexpr (
        FORMAT == kNvq3 || FORMAT == kNvq3Jsc ||
        FORMAT == kNvq3Jsc2 || FORMAT == kNvq3Jsc512 ||
        FORMAT == kNvq3JscL) {
        const int vector4 = group * 6 + chunk;
        if (vector4 >= nvec || vector8 >= nsign) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector4;
        constexpr int INDEX_BITS = format_index_bits(FORMAT);
        const uint32_t index = INDEX_BITS == 8
            ? indices[index_linear]
            : load_packed_bits(
                indices, index_linear * INDEX_BITS, INDEX_BITS, indices_nbytes);
        const int64_t sign_linear = static_cast<int64_t>(row) * nsign + vector8;
        const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
        const uint32_t mask8 = mask7 | (static_cast<uint32_t>(parity7(mask7)) << 7);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(load_i8x4(bank + index * 4), mask8, (chunk & 1) * 4);
    } else if constexpr (FORMAT == kNpq0L) {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 7, 7, indices_nbytes);
        const int2 values = load_npq0_l_vec8(codebook, state, index);
        return chunk & 1 ? values.y : values.x;
    } else if constexpr (FORMAT == kNpq0S) {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 6, 6, indices_nbytes);
        const int2 values = load_npq0_s_vec8(codebook, state, index);
        return chunk & 1 ? values.y : values.x;
    } else if constexpr (
        FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec ||
        FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) {
        if (vector8 >= nvec || vector8 >= nsign) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        uint32_t index;
        uint32_t mask8;
        if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
            const uint16_t metadata = reinterpret_cast<const uint16_t *>(indices)[index_linear];
            index = metadata & 0xffu;
            mask8 = metadata >> 8;
        } else {
            constexpr int INDEX_BITS = format_index_bits(FORMAT);
            index = INDEX_BITS == 8
                ? indices[index_linear]
                : load_packed_bits(
                    indices, index_linear * INDEX_BITS, INDEX_BITS, indices_nbytes);
            const int64_t sign_linear = static_cast<int64_t>(row) * nsign + vector8;
            const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
            const int last = parity7(mask7) ^ (sign_mode ? ((index >> 7) & 1u) : 0u);
            mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
        }
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return apply_sign4(load_i8x4(bank + index * 8 + (chunk & 1) * 4),
                           mask8, (chunk & 1) * 4);
    } else if constexpr (FORMAT == kNvq1S) {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(indices, index_linear * 9, 9, indices_nbytes);
        const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
        const int negative_delta = static_cast<int>(
            load_packed_bits(aux, delta_linear, 1, aux_nbytes));
        const int delta = negative_delta ? -1 : 1;
        const int8_t * bank = codebook + negative_delta * kNvq1SBankBytes;
        return nvq1_s_scale_delta4(
            load_i8x4(bank + index * 8 + (chunk & 1) * 4), delta);
    } else {
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(indices, index_linear * 11, 11, indices_nbytes);
        const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
        const int negative_delta = static_cast<int>(
            load_packed_bits(aux, delta_linear, 1, aux_nbytes));
        const int delta = negative_delta ? -1 : 1;
        return nvq1_l_scale_delta4(
            load_i8x4(codebook + index * 8 + (chunk & 1) * 4), delta);
    }
}

template <int FORMAT>
__device__ __forceinline__ int decode_nepq_chunk4(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int group,
    int chunk,
    int nvec,
    int nsign,
    int ng,
    int sign_mode,
    uint32_t state) {
    if constexpr (FORMAT == kNpq0S) {
        const int vector8 = group * 3 + (chunk >> 1);
        if (vector8 >= nvec) return 0;
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector8;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 6, 6, indices_nbytes);
        const uint32_t half_index = chunk & 1 ? index >> 3 : index & 7u;
        const int8_t * half = chunk & 1
            ? codebook + kNepq0SSecondOffset
            : codebook + kNpq0SMetadataBytes;
        return load_i8x4(
            half + state * kNepq0SBytesPerStateHalf + half_index * 4);
    }
    return decode_chunk4<FORMAT>(
        indices, indices_nbytes, aux, aux_nbytes, codebook,
        row, group, chunk, nvec, nsign, ng, sign_mode, state);
}

template <int FORMAT>
__device__ __forceinline__ float format_scale(
    float anchor,
    uint32_t sub_scale,
    const int8_t * codebook) {
    if constexpr (FORMAT == kNvq3Jsc2) {
        const uint32_t rank = sub_scale >> 1;
        return anchor * static_cast<float>(rank + 1);
    }
    if constexpr (
        FORMAT == kNvq3Jsc || FORMAT == kNvq3Jsc512 ||
        FORMAT == kNvq3JscL || FORMAT == kNvq3JscLGroupExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kJscLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    if constexpr (
        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec ||
        FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL ||
        FORMAT == kNvq2JscXLGroupExec) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kJscLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    if constexpr (FORMAT == kNpq0L) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kNpq0LLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    if constexpr (FORMAT == kNpq0S) {
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(codebook);
        const __half * lut = reinterpret_cast<const __half *>(bytes + kNpq0SLutOffset);
        return anchor * __half2float(lut[sub_scale]);
    }
    const float scale = anchor * static_cast<float>(sub_scale);
    if constexpr (FORMAT == kNvq1L) return scale * 0.125f;
    if constexpr (FORMAT == kNvq1S) return scale * 0.03125f;
    return scale;
}

__device__ __forceinline__ int2 apply_sign8(int2 values, uint32_t mask8) {
    const uint32_t spread0=(mask8 & 15u)*0x10204080u;
    const uint32_t spread1=((mask8 >> 4) & 15u)*0x10204080u;
    uint32_t signs0,signs1;
    asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(signs0) : "r"(spread0));
    asm("prmt.b32 %0,%1,0,0xba98;" : "=r"(signs1) : "r"(spread1));
    const uint32_t x=uint32_t(values.x)^signs0,y=uint32_t(values.y)^signs1;
    return make_int2(int(((x & 0x7f7f7f7fu)+(signs0 & 0x01010101u))^(x & 0x80808080u)),
                     int(((y & 0x7f7f7f7fu)+(signs1 & 0x01010101u))^(y & 0x80808080u)));
}

static __device__ uint32_t nvq_sign_expand4[16] = {
    0x00000000u, 0x000000ffu, 0x0000ff00u, 0x0000ffffu,
    0x00ff0000u, 0x00ff00ffu, 0x00ffff00u, 0x00ffffffu,
    0xff000000u, 0xff0000ffu, 0xff00ff00u, 0xff00ffffu,
    0xffff0000u, 0xffff00ffu, 0xffffff00u, 0xffffffffu,
};

__device__ __forceinline__ int2 apply_sign8_exec(int2 values, uint32_t mask8) {
    const int signs0 = static_cast<int>(__ldg(nvq_sign_expand4 + (mask8 & 0x0fu)));
    const int signs1 = static_cast<int>(__ldg(nvq_sign_expand4 + (mask8 >> 4)));
    return make_int2(
        __vsub4(values.x ^ signs0, signs0),
        __vsub4(values.y ^ signs1, signs1));
}

template <int FORMAT>
struct NvqVec8Values {
    int2 values;
    int delta;
    bool valid;
};

struct NvqDeviceWeight {
    const uint8_t * indices;
    int64_t indices_nbytes;
    const uint8_t * aux;
    int64_t aux_nbytes;
    const uint8_t * sub_scale;
    int64_t sub_scale_nbytes;
    const float * neuron_scale;
    const int8_t * codebook;
    int codebook_nbytes;
    int N;
    int ng;
    int nvec;
    int nsign;
    int sub_bits;
    int sign_mode;
};

template <int FORMAT>
__device__ __forceinline__ NvqVec8Values<FORMAT> load_nvq_vec8(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int segment,
    int group,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state) {
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (segment >= nsign) return {make_int2(0, 0), 0, false};
        const int local = segment - group * 3;
        const uint64_t metadata = load_group_exec64(indices, row, group, ng);
        const uint32_t segment_metadata =
            (metadata >> (local * 20)) & 0xfffffu;
        const uint32_t index = segment_metadata & 0xfffu;
        const uint32_t mask8 = segment_metadata >> 12;
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return {apply_sign8(
            reinterpret_cast<const int2 *>(bank)[index], mask8), 0, true};
    } else if constexpr (FORMAT == kNvq3JscLGroupExec) {
        const int vector4 = segment * 2;
        if (vector4 >= nvec || segment >= nsign) {
            return {make_int2(0, 0), 0, false};
        }
        const int local = segment - group * 3;
        const uint32_t index0 = load_group_exec96_bits(
            indices, row, group, ng, local * 20, 10);
        const uint32_t index1 = vector4 + 1 < nvec
            ? load_group_exec96_bits(
                indices, row, group, ng, local * 20 + 10, 10)
            : 0;
        const uint32_t mask8 = load_group_exec96_bits(
            indices, row, group, ng, 60 + local * 8, 8);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return {apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index0],
                reinterpret_cast<const int *>(bank)[index1]),
            mask8), 0, true};
    } else if constexpr (
        FORMAT == kNvq3 || FORMAT == kNvq3Jsc ||
        FORMAT == kNvq3Jsc2 || FORMAT == kNvq3Jsc512 ||
        FORMAT == kNvq3JscL) {
        const int vector4 = segment * 2;
        if (vector4 >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + vector4;
        constexpr int INDEX_BITS = format_index_bits(FORMAT);
        const uint32_t index0 = INDEX_BITS == 8
            ? indices[index_linear]
            : load_packed_bits(
                indices, index_linear * INDEX_BITS, INDEX_BITS, indices_nbytes);
        const uint32_t index1 = vector4 + 1 < nvec
            ? (INDEX_BITS != 8
                ? load_packed_bits(
                    indices, (index_linear + 1) * INDEX_BITS,
                    INDEX_BITS, indices_nbytes)
                : indices[index_linear + 1])
            : 0;
        const int64_t sign_linear = static_cast<int64_t>(row) * nsign + segment;
        const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
        const uint32_t mask8 = mask7 | (static_cast<uint32_t>(parity7(mask7)) << 7);
        const int8_t * bank = active_codebook<FORMAT>(codebook, state);
        return {apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index0],
                reinterpret_cast<const int *>(bank)[index1]),
            mask8), 0, true};
    } else {
        if (segment >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + segment;
        if constexpr (FORMAT == kNpq0L) {
            const uint32_t index = load_packed_bits(
                indices, index_linear * 7, 7, indices_nbytes);
            return {load_npq0_l_vec8(codebook, state, index), 0, true};
        } else if constexpr (FORMAT == kNpq0S) {
            const uint32_t index = load_packed_bits(
                indices, index_linear * 6, 6, indices_nbytes);
            return {load_npq0_s_vec8(codebook, state, index), 0, true};
        } else if constexpr (
            FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
            FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec ||
            FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) {
            uint32_t index;
            uint32_t mask8;
            if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
                const uint16_t metadata = reinterpret_cast<const uint16_t *>(indices)[index_linear];
                index = metadata & 0xffu;
                mask8 = metadata >> 8;
            } else {
                constexpr int INDEX_BITS = format_index_bits(FORMAT);
                index = INDEX_BITS == 8
                    ? indices[index_linear]
                    : load_packed_bits(
                        indices, index_linear * INDEX_BITS,
                        INDEX_BITS, indices_nbytes);
                const int64_t sign_linear = static_cast<int64_t>(row) * nsign + segment;
                const uint32_t mask7 = load_packed_bits(aux, sign_linear * 7, 7, aux_nbytes);
                const int last = parity7(mask7) ^ (sign_mode ? ((index >> 7) & 1u) : 0u);
                mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
            }
            const int8_t * bank = active_codebook<FORMAT>(codebook, state);
            const int2 values = reinterpret_cast<const int2 *>(bank)[index];
            if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
                return {apply_sign8_exec(values, mask8), 0, true};
            }
            return {apply_sign8(values, mask8), 0, true};
        } else if constexpr (FORMAT == kNvq1S) {
            const uint32_t index = load_packed_bits(
                indices, index_linear * 9, 9, indices_nbytes);
            const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
            const int negative_delta = static_cast<int>(
                load_packed_bits(aux, delta_linear, 1, aux_nbytes));
            const int delta = negative_delta ? -1 : 1;
            const int8_t * bank = codebook + negative_delta * kNvq1SBankBytes;
            return {reinterpret_cast<const int2 *>(bank)[index], delta, true};
        } else {
            const uint32_t index = load_packed_bits(indices, index_linear * 11, 11, indices_nbytes);
            const int64_t delta_linear = static_cast<int64_t>(row) * ng + group;
            const int delta = load_packed_bits(aux, delta_linear, 1, aux_nbytes) ? -1 : 1;
            return {reinterpret_cast<const int2 *>(codebook)[index], delta, true};
        }
    }
}

template <int FORMAT>
__device__ __forceinline__ NvqVec8Values<FORMAT> load_nepq_vec8(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    int row,
    int segment,
    int group,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state) {
    if constexpr (FORMAT == kNpq0S) {
        if (segment >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear = static_cast<int64_t>(row) * nvec + segment;
        const uint32_t index = load_packed_bits(
            indices, index_linear * 6, 6, indices_nbytes);
        return {load_nepq0_s_vec8(codebook, state, index), 0, true};
    }
    return load_nvq_vec8<FORMAT>(
        indices, indices_nbytes, aux, aux_nbytes, codebook,
        row, segment, group, ng, nvec, nsign, sign_mode, state);
}


template<int Bits>
__device__ __forceinline__ uint64_t load_packed_group_window(
        const uint8_t* data,int64_t bit,int64_t nbytes) {
    static_assert(Bits>0 && Bits<64);
    const int64_t byte=bit>>3;
    const int64_t aligned=byte-int64_t((reinterpret_cast<uintptr_t>(data)+byte)&3u);
    uint64_t value=0;
    if(aligned>=0 && aligned+12<=nbytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(data+aligned);
        const int shift=int((byte-aligned)*8+(bit&7));
        const uint32_t a=words[0],b=words[1],c=words[2];
        value=uint64_t(__funnelshift_r(a,b,shift)) |
              (uint64_t(__funnelshift_r(b,c,shift))<<32);
    }else {
        constexpr int bytes=(Bits+14)/8;
        uint32_t upper=0;
#pragma unroll
        for(int i=0;i<bytes;++i)if(byte+i<nbytes) {
            if constexpr(bytes>8) {
                if(i==8)upper=data[byte+i];
                else value|=uint64_t(data[byte+i])<<(i*8);
            }else value|=uint64_t(data[byte+i])<<(i*8);
        }
        const int shift=int(bit&7);
        if(shift)value=(value>>shift)|(uint64_t(upper)<<(64-shift));
    }
    return value&((uint64_t(1)<<Bits)-1);
}

template <int FORMAT, bool PackedIndices = false>
__device__ __forceinline__ NvqVec8Values<FORMAT> load_nvq_group_vec8(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const int8_t * codebook,
    const int8_t * bank,
    int row,
    int group,
    int segment_local,
    int ng,
    int nvec,
    int nsign,
    int sign_mode,
    uint32_t state,
    uint32_t packed_signs,
    uint64_t group_exec64,
    const uint32_t * group_exec96,
    int group_delta) {
    const int segment = group * 3 + segment_local;
    if constexpr (FORMAT == kNvq2JscXLGroupExec) {
        if (segment >= nsign) return {make_int2(0, 0), 0, false};
        const uint32_t metadata =
            (group_exec64 >> (segment_local * 20)) & 0xfffffu;
        const uint32_t index = metadata & 0xfffu;
        const uint32_t mask8 = metadata >> 12;
        return {apply_sign8(
            reinterpret_cast<const int2 *>(bank)[index], mask8), 0, true};
    } else if constexpr (FORMAT == kNvq3JscLGroupExec) {
        const int vector4 = segment * 2;
        if (vector4 >= nvec || segment >= nsign) {
            return {make_int2(0, 0), 0, false};
        }
        const uint32_t index0 = extract_group_exec96_bits_ptr(
            group_exec96, segment_local * 20, 10);
        const uint32_t index1 = vector4 + 1 < nvec
            ? extract_group_exec96_bits_ptr(
                group_exec96, segment_local * 20 + 10, 10)
            : 0;
        const uint32_t mask8 = extract_group_exec96_bits_ptr(
            group_exec96, 60 + segment_local * 8, 8);
        return {apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index0],
                reinterpret_cast<const int *>(bank)[index1]),
            mask8), 0, true};
    } else if constexpr (
        FORMAT == kNvq3 || FORMAT == kNvq3Jsc ||
        FORMAT == kNvq3Jsc2 || FORMAT == kNvq3Jsc512 ||
        FORMAT == kNvq3JscL) {
        const int vector4 = segment * 2;
        if (vector4 >= nvec || segment >= nsign) {
            return {make_int2(0, 0), 0, false};
        }
        const int64_t index_linear =
            static_cast<int64_t>(row) * nvec + vector4;
        constexpr int INDEX_BITS = format_index_bits(FORMAT);
        const uint32_t index0 = PackedIndices ? (uint32_t(group_exec64 >> (segment_local*2*INDEX_BITS)) & ((1u<<INDEX_BITS)-1)) : (INDEX_BITS == 8
            ? indices[index_linear]
            : load_packed_bits(
                indices, index_linear * INDEX_BITS,
                INDEX_BITS, indices_nbytes));
        const uint32_t index1 = PackedIndices ? (vector4 + 1 < nvec
            ? uint32_t(group_exec64 >> ((segment_local*2+1)*INDEX_BITS)) & ((1u<<INDEX_BITS)-1) : 0) : (vector4 + 1 < nvec
            ? (INDEX_BITS == 8
                ? indices[index_linear + 1]
                : load_packed_bits(
                    indices, (index_linear + 1) * INDEX_BITS,
                    INDEX_BITS, indices_nbytes))
            : 0);
        const uint32_t mask7 =
            (packed_signs >> (segment_local * 7)) & 0x7fu;
        const uint32_t mask8 = mask7 |
            (static_cast<uint32_t>(parity7(mask7)) << 7);
        return {apply_sign8(
            make_int2(
                reinterpret_cast<const int *>(bank)[index0],
                reinterpret_cast<const int *>(bank)[index1]),
            mask8), 0, true};
    } else if constexpr (
        FORMAT == kNvq2 || FORMAT == kNvq2Exec ||
        FORMAT == kNvq2Jsc || FORMAT == kNvq2JscExec ||
        FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL) {
        if (segment >= nvec || segment >= nsign) {
            return {make_int2(0, 0), 0, false};
        }
        const int64_t index_linear =
            static_cast<int64_t>(row) * nvec + segment;
        uint32_t index;
        uint32_t mask8;
        if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
            const uint16_t metadata =
                reinterpret_cast<const uint16_t *>(indices)[index_linear];
            index = metadata & 0xffu;
            mask8 = metadata >> 8;
        } else {
            constexpr int INDEX_BITS = format_index_bits(FORMAT);
            index = PackedIndices ? (uint32_t(group_exec64 >> (segment_local*INDEX_BITS)) & ((1u<<INDEX_BITS)-1)) : (INDEX_BITS == 8
                ? indices[index_linear]
                : load_packed_bits(
                    indices, index_linear * INDEX_BITS,
                    INDEX_BITS, indices_nbytes));
            const uint32_t mask7 =
                (packed_signs >> (segment_local * 7)) & 0x7fu;
            const int last = parity7(mask7) ^
                (sign_mode ? ((index >> 7) & 1u) : 0u);
            mask8 = mask7 | (static_cast<uint32_t>(last) << 7);
        }
        const int2 values = reinterpret_cast<const int2 *>(bank)[index];
        if constexpr (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec) {
            return {apply_sign8_exec(values, mask8), 0, true};
        }
        return {apply_sign8(values, mask8), 0, true};
    } else if constexpr (FORMAT == kNvq1S) {
        if (segment >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear =
            static_cast<int64_t>(row) * nvec + segment;
        const uint32_t index = PackedIndices ? (uint32_t(group_exec64 >> (segment_local*9)) & ((1u<<9)-1)) : (load_packed_bits(
            indices, index_linear * 9, 9, indices_nbytes));
        return {reinterpret_cast<const int2 *>(bank)[index], group_delta, true};
    } else if constexpr (FORMAT == kNvq1L) {
        if (segment >= nvec) return {make_int2(0, 0), 0, false};
        const int64_t index_linear =
            static_cast<int64_t>(row) * nvec + segment;
        const uint32_t index = PackedIndices ? (uint32_t(group_exec64 >> (segment_local*11)) & ((1u<<11)-1)) : (load_packed_bits(
            indices, index_linear * 11, 11, indices_nbytes));
        return {
            reinterpret_cast<const int2 *>(codebook)[index], group_delta, true};
    }
    return load_nvq_vec8<FORMAT>(
        indices, indices_nbytes, aux, aux_nbytes, codebook,
        row, segment, group, ng, nvec, nsign, sign_mode, state);
}

template <int FORMAT>
__device__ __forceinline__ float dot_nvq_vec8(
    const NvqVec8Values<FORMAT> & weight,
    const int8_t * activation) {
    if (!weight.valid) return 0.0f;
    const int2 x = *reinterpret_cast<const int2 *>(activation);
    const int dot = __dp4a(weight.values.y, x.y, __dp4a(weight.values.x, x.x, 0));
    if constexpr (FORMAT == kNvq1L) {
        const int sum = __dp4a(0x01010101, x.y, __dp4a(0x01010101, x.x, 0));
        return static_cast<float>(dot) + 0.125f * static_cast<float>(weight.delta * sum);
    }
    if constexpr (FORMAT == kNvq1S) {
        const int sum = __dp4a(0x01010101, x.y, __dp4a(0x01010101, x.x, 0));
        return static_cast<float>(dot) + 0.15625f * static_cast<float>(weight.delta * sum);
    }
    return static_cast<float>(dot);
}


}
