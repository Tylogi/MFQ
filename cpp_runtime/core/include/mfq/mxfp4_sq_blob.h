#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace mfq::sq {

// Design intent: MXFP4-SQ is a high-fidelity, fine-grained requantization
// container for weights natively QAT-trained in MXFP4.  Every reconstructed
// value remains legal MXFP4/E2M1 under native block-32 E8M0 scaling, allowing
// CUDA runtimes to retain native MXFP4 hardware acceleration.  The packed SQ
// stream is a storage representation, not a new arithmetic domain.
//
// MXFP4-SQ v2 stores one two-bit q selector per neuron.  q=1/2/3 use the
// scalar palette representation; q=4 is the lossless native-MXFP4 endpoint
// and retains the original block-32 E8M0 scales.  Legacy SQ2/SQ3 v1 payloads
// remain readable and expand to uniform row metadata.
struct Layout {
    int version = 0;
    int bits = 0; // Legacy uniform q; zero for the v2 per-neuron stream.
    int outputs = 0;
    int width = 0;
    int base = 0;
    int sq_rows = 0;
    int sq4_rows = 0;
    std::size_t q_selectors = 0;
    std::size_t symbols = 0;
    std::size_t selectors = 0;
    std::size_t scales = 0;
    std::size_t palettes = 0;
    std::size_t native_scales = 0;
    std::size_t bytes = 0;
};

struct RowMetadata {
    std::vector<std::uint8_t> q;
    std::vector<std::uint32_t> symbol_byte_offsets;
    // For q<4 this is the row rank among SQ1/2/3 rows.  For q=4 it is the
    // row rank among native MXFP4 rows.
    std::vector<std::uint32_t> auxiliary_rows;
};

inline std::uint64_t checked_weights(
    std::int64_t outputs,
    std::int64_t width) {
    if (outputs <= 0 || width <= 0 ||
        outputs > std::numeric_limits<int>::max() ||
        width > std::numeric_limits<int>::max() || width % 32 != 0) {
        throw std::runtime_error("invalid MXFP4-SQ geometry");
    }
    const auto weights = static_cast<std::uint64_t>(outputs) *
        static_cast<std::uint64_t>(width);
    if (weights > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("MXFP4-SQ payload size overflow");
    }
    return weights;
}

inline Layout layout(
    std::int64_t bits,
    std::int64_t outputs,
    std::int64_t width,
    std::int64_t base) {
    const auto weights = checked_weights(outputs, width);
    if ((bits != 2 && bits != 3) || base < 0 || base > 251 ||
        weights > (std::numeric_limits<std::uint64_t>::max() - 7) /
            static_cast<std::uint64_t>(bits)) {
        throw std::runtime_error(
            "invalid MXFP4-SQ geometry or scale base");
    }
    const auto selectors = std::uint64_t{24} +
        (weights * static_cast<std::uint64_t>(bits) + 7) / 8;
    const auto scales = selectors + (weights / 32 + 7) / 8;
    const auto palettes = scales + static_cast<std::uint64_t>(outputs) * 2;
    const auto bytes = palettes + static_cast<std::uint64_t>(outputs) * 5;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("MXFP4-SQ payload size overflow");
    }
    return {
        1,
        static_cast<int>(bits),
        static_cast<int>(outputs),
        static_cast<int>(width),
        static_cast<int>(base),
        static_cast<int>(outputs),
        0,
        0,
        24,
        static_cast<std::size_t>(selectors),
        static_cast<std::size_t>(scales),
        static_cast<std::size_t>(palettes),
        static_cast<std::size_t>(bytes),
        static_cast<std::size_t>(bytes),
    };
}

inline Layout adaptive_layout(
    std::int64_t outputs,
    std::int64_t width,
    std::int64_t base,
    std::uint64_t q_sum,
    std::int64_t sq4_rows) {
    checked_weights(outputs, width);
    if (base < 0 || base > 251 || sq4_rows < 0 || sq4_rows > outputs ||
        q_sum < static_cast<std::uint64_t>(outputs) ||
        q_sum > static_cast<std::uint64_t>(outputs) * 4) {
        throw std::runtime_error("invalid adaptive MXFP4-SQ metadata");
    }
    const auto sq_rows = outputs - sq4_rows;
    const auto blocks = static_cast<std::uint64_t>(width / 32);
    const auto used_q_selector_bytes =
        (static_cast<std::uint64_t>(outputs) * 2 + 7) / 8;
    const auto q_selector_bytes = (used_q_selector_bytes + 3) & ~std::uint64_t{3};
    const auto symbols = std::uint64_t{24} + q_selector_bytes;
    if (q_sum > std::numeric_limits<std::uint64_t>::max() /
            static_cast<std::uint64_t>(width)) {
        throw std::runtime_error("adaptive MXFP4-SQ symbol size overflow");
    }
    const auto symbol_bytes =
        q_sum * static_cast<std::uint64_t>(width) / 8;
    const auto selectors = symbols + symbol_bytes;
    const auto scales = selectors +
        (static_cast<std::uint64_t>(sq_rows) * blocks + 7) / 8;
    const auto palettes = scales + static_cast<std::uint64_t>(sq_rows) * 2;
    const auto native_scales =
        palettes + static_cast<std::uint64_t>(sq_rows) * 5;
    const auto bytes = native_scales +
        static_cast<std::uint64_t>(sq4_rows) * blocks;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("adaptive MXFP4-SQ payload size overflow");
    }
    return {
        2,
        0,
        static_cast<int>(outputs),
        static_cast<int>(width),
        static_cast<int>(base),
        static_cast<int>(sq_rows),
        static_cast<int>(sq4_rows),
        24,
        static_cast<std::size_t>(symbols),
        static_cast<std::size_t>(selectors),
        static_cast<std::size_t>(scales),
        static_cast<std::size_t>(palettes),
        static_cast<std::size_t>(native_scales),
        static_cast<std::size_t>(bytes),
    };
}

inline std::uint64_t read_u64(const std::uint8_t* data, int offset) {
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(data[offset + byte])
            << (byte * 8);
    }
    return value;
}

inline std::uint8_t read_packed_q(
    const std::uint8_t* data,
    std::size_t row) {
    const auto bit = row * 2;
    return static_cast<std::uint8_t>(
        1u + ((data[bit / 8] >> (bit & 7)) & 3u));
}

inline Layout parse(const std::uint8_t* data, std::size_t bytes) {
    if (bytes < 24 || data == nullptr) {
        throw std::runtime_error("invalid MXFP4-SQ header");
    }
    const bool sq2_magic = data[0] == 'S' && data[1] == 'Q' &&
        data[2] == '2' && data[3] == 0;
    // SQ31 was the original Metal-only SQ3 spelling. It remains readable.
    const bool sq3_magic = data[0] == 'S' && data[1] == 'Q' &&
        data[2] == '3' && (data[3] == 0 || data[3] == '1');
    const bool adaptive_magic = data[0] == 'S' && data[1] == 'Q' &&
        data[2] == 'V' && data[3] == '2';
    if (data[6] != 0 || data[7] != 0 ||
        ((!sq2_magic && !sq3_magic) && !adaptive_magic) ||
        ((sq2_magic || sq3_magic) && data[4] != 1) ||
        (adaptive_magic && data[4] != 2)) {
        throw std::runtime_error("invalid MXFP4-SQ header");
    }
    const auto outputs64 = read_u64(data, 8);
    const auto width64 = read_u64(data, 16);
    if (outputs64 > static_cast<std::uint64_t>(
            std::numeric_limits<int>::max()) ||
        width64 > static_cast<std::uint64_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("MXFP4-SQ dimension overflow");
    }
    const auto outputs = static_cast<std::int64_t>(outputs64);
    const auto width = static_cast<std::int64_t>(width64);
    if (!adaptive_magic) {
        const auto result = layout(
            data[2] - '0', outputs, width, data[5]);
        if (bytes != result.bytes) {
            throw std::runtime_error(
                "truncated or trailing MXFP4-SQ payload");
        }
        return result;
    }

    checked_weights(outputs, width);
    const auto used_selector_bytes =
        (static_cast<std::uint64_t>(outputs) * 2 + 7) / 8;
    const auto selector_bytes = (used_selector_bytes + 3) & ~std::uint64_t{3};
    if (bytes < 24 + selector_bytes) {
        throw std::runtime_error("truncated MXFP4-SQ q selectors");
    }
    std::uint64_t q_sum = 0;
    std::int64_t sq4_rows = 0;
    for (std::size_t row = 0; row < static_cast<std::size_t>(outputs); ++row) {
        const int q = read_packed_q(data + 24, row);
        q_sum += static_cast<std::uint64_t>(q);
        sq4_rows += q == 4;
    }
    if ((outputs * 2) % 8 != 0) {
        const auto used = static_cast<unsigned>((outputs * 2) % 8);
        const auto mask = static_cast<std::uint8_t>(~((1u << used) - 1u));
        if ((data[24 + used_selector_bytes - 1] & mask) != 0) {
            throw std::runtime_error(
                "non-zero MXFP4-SQ q-selector padding");
        }
    }
    for (std::size_t byte = static_cast<std::size_t>(used_selector_bytes);
         byte < static_cast<std::size_t>(selector_bytes);
         ++byte) {
        if (data[24 + byte] != 0) {
            throw std::runtime_error(
                "non-zero MXFP4-SQ q-selector alignment padding");
        }
    }
    const auto result = adaptive_layout(
        outputs, width, data[5], q_sum, sq4_rows);
    if (bytes != result.bytes) {
        throw std::runtime_error(
            "truncated or trailing adaptive MXFP4-SQ payload");
    }
    for (auto offset = result.native_scales;
         offset < result.bytes;
         ++offset) {
        if (data[offset] == 255u) {
            throw std::runtime_error(
                "MXFP4-SQ contains an E8M0 NaN native scale");
        }
    }
    return result;
}

inline RowMetadata row_metadata(
    const std::uint8_t* data,
    const Layout& layout) {
    if (data == nullptr || layout.outputs <= 0 || layout.width <= 0) {
        throw std::runtime_error("invalid MXFP4-SQ row metadata source");
    }
    RowMetadata result;
    result.q.resize(static_cast<std::size_t>(layout.outputs));
    result.symbol_byte_offsets.resize(static_cast<std::size_t>(layout.outputs));
    result.auxiliary_rows.resize(static_cast<std::size_t>(layout.outputs));
    std::size_t symbol_offset = 0;
    std::uint32_t sq_rank = 0;
    std::uint32_t native_rank = 0;
    for (std::size_t row = 0; row < result.q.size(); ++row) {
        const int q = layout.version == 1
            ? layout.bits
            : read_packed_q(data + layout.q_selectors, row);
        if (symbol_offset > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "MXFP4-SQ symbol stream exceeds uint32 row offsets");
        }
        result.q[row] = static_cast<std::uint8_t>(q);
        result.symbol_byte_offsets[row] =
            static_cast<std::uint32_t>(symbol_offset);
        result.auxiliary_rows[row] = q == 4 ? native_rank++ : sq_rank++;
        symbol_offset += static_cast<std::size_t>(layout.width) *
            static_cast<std::size_t>(q) / 8;
    }
    if (symbol_offset != layout.selectors - layout.symbols ||
        sq_rank != static_cast<std::uint32_t>(layout.sq_rows) ||
        native_rank != static_cast<std::uint32_t>(layout.sq4_rows)) {
        throw std::runtime_error("inconsistent MXFP4-SQ row metadata");
    }
    return result;
}

inline std::uint8_t read_packed_bits(
    std::span<const std::uint8_t> data,
    std::size_t offset,
    std::size_t index,
    unsigned bits) {
    const auto bit = index * bits;
    const auto shift = static_cast<unsigned>(bit & 7);
    unsigned value = data[offset + bit / 8];
    if (shift + bits > 8) {
        value |= static_cast<unsigned>(data[offset + bit / 8 + 1]) << 8;
    }
    return static_cast<std::uint8_t>(
        (value >> shift) & ((1u << bits) - 1u));
}

inline void write_packed_bits(
    std::span<std::uint8_t> data,
    std::size_t offset,
    std::size_t index,
    unsigned bits,
    std::uint8_t value) {
    const auto bit = index * bits;
    const auto shift = static_cast<unsigned>(bit & 7);
    const auto packed = static_cast<std::uint16_t>(value) << shift;
    data[offset + bit / 8] |= static_cast<std::uint8_t>(packed);
    if (shift + bits > 8) {
        data[offset + bit / 8 + 1] |=
            static_cast<std::uint8_t>(packed >> 8);
    }
}

inline std::vector<std::uint8_t> select_rows(
    std::span<const std::uint8_t> source,
    std::span<const std::int64_t> selected_rows,
    int column_begin = 0,
    int column_end = -1) {
    const auto src = parse(source.data(), source.size());
    if (column_end == -1) column_end = src.width;
    if (column_begin < 0 || column_end > src.width || column_begin >= column_end ||
        column_begin % 32 != 0 || column_end % 32 != 0) {
        throw std::runtime_error("MXFP4-SQ column selection must align to block32");
    }
    if (selected_rows.empty() ||
        selected_rows.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "MXFP4-SQ row selection must be non-empty and fit int32");
    }
    const auto source_rows = row_metadata(source.data(), src);
    std::uint64_t q_sum = 0;
    std::int64_t sq4_rows = 0;
    for (const auto row : selected_rows) {
        if (row < 0 || row >= src.outputs) {
            throw std::runtime_error(
                "MXFP4-SQ selected row is out of range");
        }
        const int q = source_rows.q[static_cast<std::size_t>(row)];
        q_sum += static_cast<std::uint64_t>(q);
        sq4_rows += q == 4;
    }
    const auto dst = adaptive_layout(
        static_cast<std::int64_t>(selected_rows.size()),
        column_end - column_begin,
        src.base,
        q_sum,
        sq4_rows);
    std::vector<std::uint8_t> result(dst.bytes, 0);
    result[0] = 'S';
    result[1] = 'Q';
    result[2] = 'V';
    result[3] = '2';
    result[4] = 2;
    result[5] = static_cast<std::uint8_t>(src.base);
    const auto write_u64 = [&result](std::size_t offset, std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            result[offset + static_cast<std::size_t>(byte)] =
                static_cast<std::uint8_t>(value >> (byte * 8));
        }
    };
    write_u64(8, selected_rows.size());
    write_u64(16, static_cast<std::uint64_t>(dst.width));

    const auto blocks = static_cast<std::size_t>(dst.width / 32);
    const auto source_blocks = static_cast<std::size_t>(src.width / 32);
    std::size_t destination_symbol = dst.symbols;
    std::size_t destination_sq_row = 0;
    std::size_t destination_native_row = 0;
    for (std::size_t destination_row = 0;
         destination_row < selected_rows.size();
         ++destination_row) {
        const auto source_row = static_cast<std::size_t>(
            selected_rows[destination_row]);
        const unsigned q = source_rows.q[source_row];
        write_packed_bits(result, dst.q_selectors, destination_row, 2,
                          static_cast<std::uint8_t>(q - 1));
        const auto row_bytes = static_cast<std::size_t>(dst.width) * q / 8;
        std::memcpy(
            result.data() + destination_symbol,
            source.data() + src.symbols +
                source_rows.symbol_byte_offsets[source_row] +
                static_cast<std::size_t>(column_begin) * q / 8,
            row_bytes);
        destination_symbol += row_bytes;

        const auto source_auxiliary =
            source_rows.auxiliary_rows[source_row];
        if (q == 4) {
            std::memcpy(
                result.data() + dst.native_scales +
                    destination_native_row * blocks,
                source.data() + src.native_scales +
                    static_cast<std::size_t>(source_auxiliary) * source_blocks + column_begin / 32,
                blocks);
            ++destination_native_row;
            continue;
        }
        for (std::size_t block = 0; block < blocks; ++block) {
            write_packed_bits(
                result,
                dst.selectors,
                destination_sq_row * blocks + block,
                1,
                read_packed_bits(
                    source,
                    src.selectors,
                    static_cast<std::size_t>(source_auxiliary) * source_blocks + column_begin / 32 + block,
                    1));
        }
        for (std::size_t state = 0; state < 8; ++state) {
            write_packed_bits(
                result,
                dst.scales,
                destination_sq_row * 8 + state,
                2,
                read_packed_bits(
                    source,
                    src.scales,
                    static_cast<std::size_t>(source_auxiliary) * 8 + state,
                    2));
            write_packed_bits(
                result,
                dst.palettes,
                destination_sq_row * 8 + state,
                5,
                read_packed_bits(
                    source,
                    src.palettes,
                    static_cast<std::size_t>(source_auxiliary) * 8 + state,
                    5));
        }
        ++destination_sq_row;
    }
    if (destination_symbol != dst.selectors ||
        destination_sq_row != static_cast<std::size_t>(dst.sq_rows) ||
        destination_native_row != static_cast<std::size_t>(dst.sq4_rows)) {
        throw std::runtime_error(
            "inconsistent MXFP4-SQ selected-row accounting");
    }
    return result;
}

} // namespace mfq::sq
