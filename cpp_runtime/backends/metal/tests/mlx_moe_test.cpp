#include "mlx_moe.h"
#include "mlx_moe_ops.h"
#include "mlx_mxfp4_sq.h"
#include "mfq_container.h"

#include "nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

constexpr int kGroupSize = 16;

template <typename T>
void append(
    std::vector<std::uint8_t>& blob,
    T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(
            &value);
    blob.insert(
        blob.end(),
        bytes,
        bytes + sizeof(value));
}

std::vector<std::uint8_t> pack_values(
    const std::vector<std::uint8_t>& values,
    int bits) {
    std::vector<std::uint8_t> packed(
        (
            values.size()
                * static_cast<std::size_t>(bits)
            + 7
        ) / 8,
        0);
    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        for (int bit = 0; bit < bits; ++bit) {
            if (((values[index] >> bit) & 1u) == 0u) {
                continue;
            }
            const auto target =
                index * static_cast<std::size_t>(bits)
                + static_cast<std::size_t>(bit);
            packed[target / 8] |=
                static_cast<std::uint8_t>(
                    1u << (target & 7));
        }
    }
    return packed;
}

void append_magic(
    std::vector<std::uint8_t>& blob,
    std::string_view magic) {
    if (magic.size() != 4) {
        throw std::runtime_error(
            "fixture magic must have four bytes");
    }
    blob.insert(
        blob.end(),
        magic.begin(),
        magic.end());
}

void append_bytes(
    std::vector<std::uint8_t>& blob,
    const std::vector<std::uint8_t>& bytes) {
    blob.insert(
        blob.end(),
        bytes.begin(),
        bytes.end());
}

std::vector<std::uint8_t> pack_vq_values(
    const std::vector<std::uint16_t>& values,
    int bits) {
    if (bits == 0) {
        if (!values.empty()) {
            throw std::runtime_error(
                "zero-bit VQ stream must be empty");
        }
        return {};
    }
    std::vector<std::uint8_t> packed(
        (
            values.size()
                * static_cast<std::size_t>(bits)
            + 7
        ) / 8,
        0);
    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        if (values[index] >= (1u << bits)) {
            throw std::runtime_error(
                "VQ fixture value exceeds bit width");
        }
        for (int bit = 0; bit < bits; ++bit) {
            if (((values[index] >> bit) & 1u) == 0u) {
                continue;
            }
            const auto destination =
                index * static_cast<std::size_t>(bits)
                + static_cast<std::size_t>(bit);
            packed[destination / 8] |=
                static_cast<std::uint8_t>(
                    1u << (destination & 7));
        }
    }
    return packed;
}

std::vector<std::uint16_t> filled_vq_values(
    std::size_t count,
    std::uint16_t value) {
    return std::vector<std::uint16_t>(
        count,
        value);
}

void append_vq_matrix_header(
    std::vector<std::uint8_t>& blob,
    std::string_view magic,
    std::uint8_t profile,
    std::uint8_t state_bits,
    std::uint16_t group_size,
    int output,
    int input) {
    append_magic(blob, magic);
    append<std::uint8_t>(blob, profile);
    append<std::uint8_t>(blob, state_bits);
    append<std::uint16_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output);
    append<std::int64_t>(blob, input);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output));
}

void append_vq_anchors(
    std::vector<std::uint8_t>& blob,
    int count) {
    for (int index = 0; index < count; ++index) {
        append<std::uint16_t>(blob, 0x3c00);
    }
}

void append_vq_matrix_streams(
    std::vector<std::uint8_t>& blob,
    int output,
    int input,
    int group_size,
    int vector_size,
    int state_bits,
    int index_bits,
    int auxiliary_bits,
    bool signs) {
    append_vq_anchors(blob, output);
    const int groups =
        (input + group_size - 1) / group_size;
    const int vectors =
        (input + vector_size - 1) / vector_size;
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output)
                    * groups,
                1),
            state_bits));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output)
                    * vectors,
                0),
            index_bits));
    if (signs) {
        const int sign_groups = (input + 7) / 8;
        append_bytes(
            blob,
            pack_vq_values(
                filled_vq_values(
                    static_cast<std::size_t>(output)
                        * sign_groups,
                    0),
                7));
    } else if (auxiliary_bits != 0) {
        append_bytes(
            blob,
            pack_vq_values(
                filled_vq_values(
                    static_cast<std::size_t>(output)
                        * groups,
                    0),
                auxiliary_bits));
    }
}

std::vector<float> repeated_vq_dense(
    int output,
    int input,
    const std::vector<float>& vector) {
    std::vector<float> dense(
        static_cast<std::size_t>(output) * input);
    for (int row = 0; row < output; ++row) {
        for (int column = 0; column < input; ++column) {
            dense[
                static_cast<std::size_t>(row)
                    * input
                + column
            ] = vector[
                static_cast<std::size_t>(column)
                    % vector.size()];
        }
    }
    return dense;
}

struct VqFixture {
    std::string dtype;
    std::vector<std::uint8_t> blob;
    std::vector<std::uint8_t> runtime;
    std::vector<float> dense;
    std::vector<std::int8_t> signs;
    int rotation_block = 0;
    std::uint64_t rotation_seed = 0;
    int output = 0;
    int input = 0;
};

struct Mxfp4Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int input = 0;
};

struct Mxfp8Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int input = 0;
};

struct Mxfp4SqFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int input = 0;
};

float decode_mxfp4_code(std::uint8_t code) {
    constexpr std::array<float, 8> table{
        0.0f, 0.5f, 1.0f, 1.5f,
        2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float value = table[code & 7u];
    return (code & 8u) == 0u ? value : -value;
}

Mxfp4Fixture make_mxfp4(int rows, int input, int seed = 0) {
    if (rows <= 0 || input <= 0 || input % 32 != 0) {
        throw std::runtime_error("invalid MXFP4 fixture geometry");
    }
    std::vector<std::uint8_t> values(
        static_cast<std::size_t>(rows) * input / 2);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(rows) * input / 32);
    std::vector<float> dense(
        static_cast<std::size_t>(rows) * input);
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < input / 32; ++group) {
            scales[
                static_cast<std::size_t>(row) * (input / 32)
                + group
            ] = static_cast<std::uint8_t>(
                125 + (row + group + seed) % 5);
        }
        for (int column = 0; column < input; column += 2) {
            const auto low = static_cast<std::uint8_t>(
                (row * 7 + column * 3 + seed * 5 + 1) & 15);
            const auto high = static_cast<std::uint8_t>(
                (row * 11 + column * 5 + seed * 3 + 6) & 15);
            values[
                static_cast<std::size_t>(row) * (input / 2)
                + column / 2
            ] = static_cast<std::uint8_t>(low | (high << 4));
            const float scale = std::ldexp(
                1.0f,
                static_cast<int>(scales[
                    static_cast<std::size_t>(row) * (input / 32)
                    + column / 32
                ]) - 127);
            dense[static_cast<std::size_t>(row) * input + column] =
                decode_mxfp4_code(low) * scale;
            dense[static_cast<std::size_t>(row) * input + column + 1] =
                decode_mxfp4_code(high) * scale;
        }
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MXT1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 4);
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input / 2);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input / 32);
    append_bytes(blob, values);
    append_bytes(blob, scales);
    return {
        std::move(blob),
        std::move(dense),
        rows,
        input,
    };
}

Mxfp4SqFixture make_mxfp4_sq2(int rows, int input) {
    if (rows <= 0 || input <= 0 || input % 32 != 0) {
        throw std::runtime_error("invalid MXFP4-SQ fixture geometry");
    }
    constexpr std::uint8_t base = 124;
    const auto weights = static_cast<std::size_t>(rows) * input;
    const auto blocks = weights / 32;
    std::vector<std::uint8_t> symbols(weights, 0);
    std::vector<std::uint8_t> selectors(blocks, 0);
    std::vector<std::uint8_t> state_scales(
        static_cast<std::size_t>(rows) * 8);
    std::vector<std::uint8_t> state_palettes(
        static_cast<std::size_t>(rows) * 8);
    std::vector<float> dense(weights);
    for (int row = 0; row < rows; ++row) {
        const auto scale_offset = static_cast<std::uint8_t>(row & 3);
        const auto palette = static_cast<std::uint8_t>((row * 7 + 3) & 31);
        for (int state = 0; state < 8; ++state) {
            const auto index = static_cast<std::size_t>(row) * 8 + state;
            state_scales[index] = scale_offset;
            state_palettes[index] = palette;
        }
        const auto nibble = mfq::metal::kMxfp4Sq2PaletteNibbles[
            static_cast<std::size_t>(palette) * 4];
        const float value = decode_mxfp4_code(nibble) *
            std::ldexp(1.0f, static_cast<int>(base + scale_offset) - 127);
        std::fill_n(
            dense.begin() + static_cast<std::ptrdiff_t>(row) * input,
            input,
            value);
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, std::string_view("SQ2\0", 4));
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, base);
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append_bytes(blob, pack_values(symbols, 2));
    append_bytes(blob, pack_values(selectors, 1));
    append_bytes(blob, pack_values(state_scales, 2));
    append_bytes(blob, pack_values(state_palettes, 5));
    return {std::move(blob), std::move(dense), rows, input};
}

float decode_mxfp8_code(std::uint8_t code) {
    const auto magnitude = static_cast<std::uint8_t>(code & 0x7fu);
    const int exponent = magnitude >> 3;
    const int mantissa = magnitude & 7;
    if (exponent == 15 && mantissa == 7) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float value = exponent == 0
        ? static_cast<float>(mantissa) / 512.0f
        : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
              exponent - 7);
    return (code & 0x80u) == 0u ? value : -value;
}

Mxfp8Fixture make_mxfp8(int rows, int input, int salt = 0) {
    if (rows <= 0 || input <= 0 || input % 128 != 0) {
        throw std::runtime_error("invalid MXFP8 fixture geometry");
    }
    const int groups = input / 128;
    const int scale_rows = (rows + 127) / 128;
    std::vector<std::uint8_t> values(
        static_cast<std::size_t>(rows) * input);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(scale_rows) * groups);
    std::vector<float> dense(
        static_cast<std::size_t>(rows) * input);
    constexpr std::array<std::uint8_t, 12> codes{
        0x00, 0x30, 0x34, 0x38, 0x3c, 0x40,
        0x80, 0xb0, 0xb4, 0xb8, 0xbc, 0xc0,
    };
    for (int scale_row = 0; scale_row < scale_rows; ++scale_row) {
        for (int group = 0; group < groups; ++group) {
            scales[static_cast<std::size_t>(scale_row) * groups + group] =
                static_cast<std::uint8_t>(126 + (scale_row + group + salt) % 3);
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < input; ++column) {
            const auto code = codes[static_cast<std::size_t>(
                row * 7 + column * 5 + salt) % codes.size()];
            values[static_cast<std::size_t>(row) * input + column] = code;
            const float scale = std::ldexp(
                1.0f,
                static_cast<int>(scales[
                    static_cast<std::size_t>(row / 128) * groups
                    + column / 128]) - 127);
            dense[static_cast<std::size_t>(row) * input + column] =
                decode_mxfp8_code(code) * scale;
        }
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MXT1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 8);
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append<std::uint64_t>(blob, scale_rows);
    append<std::uint64_t>(blob, groups);
    append_bytes(blob, values);
    append_bytes(blob, scales);
    return {
        std::move(blob),
        std::move(dense),
        rows,
        input,
    };
}

VqFixture make_plain_nvq(
    int output,
    int input,
    bool index_parity = false) {
    constexpr std::uint8_t custom_codebook = 0x40;
    constexpr std::uint8_t parity = 0x80;
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(
            1
            | custom_codebook
            | (index_parity ? parity : 0)),
        4,
        24,
        output,
        input);
    for (int entry = 0; entry < 256; ++entry) {
        append<std::uint16_t>(
            blob,
            static_cast<std::uint16_t>(entry));
    }
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        8,
        4,
        8,
        7,
        true);
    return {
        "NVQ2",
        std::move(blob),
        {},
        repeated_vq_dense(
            output,
            input,
            std::vector<float>(8, 1.0f)),
        {},
        0,
        0,
        output,
        input,
    };
}

VqFixture make_plain_nvq3(
    int output,
    int input) {
    constexpr std::uint8_t custom_codebook = 0x40;
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(2 | custom_codebook),
        4,
        24,
        output,
        input);
    for (int entry = 0; entry < 256; ++entry) {
        append<std::uint16_t>(
            blob,
            static_cast<std::uint16_t>(entry));
    }
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        4,
        4,
        8,
        7,
        true);
    return {
        "NVQ3",
        std::move(blob),
        {},
        repeated_vq_dense(
            output,
            input,
            std::vector<float>(4, 1.0f)),
        {},
        0,
        0,
        output,
        input,
    };
}

VqFixture make_jsc_nvq(
    int output,
    int input,
    std::string dtype = "NVQ2J",
    int codebook_id = 1,
    int vector_size = 8,
    int index_bits = 8,
    bool group64 = false) {
    constexpr std::uint8_t jsc = 0x20;
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(
            codebook_id | jsc),
        4,
        24,
        output,
        input);
    append<std::uint8_t>(blob, group64 ? 2 : 1);
    append<std::uint8_t>(blob, 2);
    append<std::uint8_t>(blob, 16);
    append<std::uint8_t>(blob, 0);
    for (int state = 0; state < 16; ++state) {
        append<std::uint16_t>(blob, 0x3c00);
    }
    for (int state = 0; state < 16; ++state) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(
                state & 1));
    }
    append<std::uint8_t>(blob, group64 ? 1 : 0);
    blob.insert(blob.end(), 11, 0);
    const auto code_value = [group64](
        int bank,
        int entry,
        int component) {
        return group64
            ? static_cast<std::int8_t>(
                (entry * 3 + component * 5 + bank * 7) % 15 - 7)
            : static_cast<std::int8_t>(bank + 1);
    };
    for (int bank = 0; bank < 2; ++bank) {
        for (
            int entry = 0;
            entry < (1 << index_bits);
            ++entry
        ) {
            for (int component = 0;
                 component < vector_size;
                 ++component) {
                append<std::int8_t>(
                    blob,
                    code_value(bank, entry, component));
            }
        }
    }
    std::vector<float> dense;
    if (group64) {
        const int groups = (input + 23) / 24;
        append_vq_anchors(blob, output);
        dense.resize(
            static_cast<std::size_t>(output) * input);
        for (int row = 0; row < output; ++row) {
            for (int group = 0; group < groups; ++group) {
                const auto state = static_cast<std::uint64_t>(
                    (row + group) & 15);
                std::uint64_t record = state << 60;
                for (int chunk = 0; chunk < 3; ++chunk) {
                    if (group * 3 + chunk >= (input + 7) / 8) {
                        continue;
                    }
                    const auto index = static_cast<std::uint32_t>(
                        (row * 17 + group * 11 + chunk * 3)
                        & ((1 << index_bits) - 1));
                    const auto mask7 = static_cast<std::uint32_t>(
                        (row * 13 + group * 7 + chunk * 5) & 0x7f);
                    const auto sign_bits = mask7
                        | ((std::popcount(mask7) & 1u) << 7u);
                    const auto segment = static_cast<std::uint64_t>(
                        index | (sign_bits << 12u));
                    record |= segment << (chunk * 20);
                    for (int component = 0; component < 8; ++component) {
                        const int column = group * 24
                            + chunk * 8 + component;
                        if (column >= input) {
                            continue;
                        }
                        float value = static_cast<float>(
                            code_value(
                                static_cast<int>(state & 1u),
                                static_cast<int>(index),
                                component));
                        if ((sign_bits & (1u << component)) != 0u) {
                            value = -value;
                        }
                        dense[
                            static_cast<std::size_t>(row) * input
                                + column
                        ] = value;
                    }
                }
                append<std::uint64_t>(blob, record);
            }
        }
    } else {
        append_vq_matrix_streams(
            blob,
            output,
            input,
            24,
            vector_size,
            4,
            index_bits,
            7,
            true);
    }
    return {
        std::move(dtype),
        std::move(blob),
        {},
        group64
            ? std::move(dense)
            : repeated_vq_dense(
                output,
                input,
                std::vector<float>(
                    static_cast<std::size_t>(
                        vector_size),
                    2.0f)),
        {},
        0,
        0,
        output,
        input,
    };
}

std::vector<float> decode_ternary_word(
    std::uint16_t word) {
    std::vector<float> result(8);
    for (int component = 0;
         component < 8;
         ++component) {
        const auto digit =
            (word >> (2 * component)) & 3u;
        if (digit > 2) {
            throw std::runtime_error(
                "invalid ternary fixture word");
        }
        result[component] =
            static_cast<float>(
                static_cast<int>(digit) - 1);
    }
    return result;
}

void append_nvq1_s_table(
    std::vector<std::uint8_t>& blob,
    bool reverse = false) {
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0; entry < 512; ++entry) {
            const int source =
                reverse ? (511 - entry) * 4 : entry * 4;
            append<std::uint16_t>(
                blob,
                mfq::nvq_codebooks::
                    kNvq1LCodebookPacked[source]);
        }
    }
}

VqFixture make_nvq1_l(
    int output,
    int input) {
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NQ1L",
        1,
        3,
        24,
        output,
        input);
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        8,
        3,
        11,
        1,
        false);
    auto vector = decode_ternary_word(
        mfq::nvq_codebooks::kNvq1LCodebookPacked[0]);
    for (auto& value : vector) {
        value += 0.125f;
    }
    return {
        "NVQ1-L",
        std::move(blob),
        {},
        repeated_vq_dense(output, input, vector),
        {},
        0,
        0,
        output,
        input,
    };
}

VqFixture make_nvq1_s(
    int output,
    int input) {
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NQ1S",
        1,
        4,
        24,
        output,
        input);
    append_nvq1_s_table(blob);
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        8,
        4,
        9,
        1,
        false);
    auto vector = decode_ternary_word(
        mfq::nvq_codebooks::
            kNvq1LCodebookPacked[0]);
    for (auto& value : vector) {
        value += 0.15625f;
    }
    return {
        "NVQ1-S",
        std::move(blob),
        {},
        repeated_vq_dense(output, input, vector),
        {},
        0,
        0,
        output,
        input,
    };
}

std::vector<std::uint8_t> npq_table(
    bool short_profile,
    int bank) {
    const int states = short_profile ? 4 : 8;
    const int first_entries = 8;
    const int second_entries =
        short_profile ? 8 : 16;
    const std::size_t bytes =
        short_profile ? 320 : 832;
    std::vector<std::uint8_t> table(bytes, 0);
    table[0] =
        static_cast<std::uint8_t>(
            short_profile ? 2 : 1);
    table[1] =
        static_cast<std::uint8_t>(states);
    table[2] = 3;
    table[3] =
        static_cast<std::uint8_t>(
            short_profile ? 3 : 4);
    table[4] = 24;
    table[5] = 8;
    for (int state = 0; state < states; ++state) {
        const std::uint16_t one = 0x3c00;
        std::memcpy(
            table.data() + 8 + state * 2,
            &one,
            sizeof(one));
    }
    const int first_count =
        states * first_entries * 4;
    const int second_count =
        states * second_entries * 4;
    std::fill_n(
        table.begin() + 64,
        first_count,
        static_cast<std::uint8_t>(
            1 + bank * 2));
    std::fill_n(
        table.begin() + 64 + first_count,
        second_count,
        static_cast<std::uint8_t>(
            2 + bank * 2));
    return table;
}

VqFixture make_npq(
    int output,
    int input,
    bool short_profile = false) {
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        short_profile ? "NPQS" : "NPQL",
        static_cast<std::uint8_t>(short_profile ? 2 : 1),
        static_cast<std::uint8_t>(short_profile ? 2 : 3),
        24,
        output,
        input);
    append_bytes(blob, npq_table(short_profile, 0));
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        8,
        short_profile ? 2 : 3,
        short_profile ? 6 : 7,
        0,
        false);
    return {
        short_profile ? "NPQ0-S" : "NPQ0-L",
        std::move(blob),
        {},
        repeated_vq_dense(
            output,
            input,
            {
                1.0f, 1.0f, 1.0f, 1.0f,
                2.0f, 2.0f, 2.0f, 2.0f,
            }),
        {},
        0,
        0,
        output,
        input,
    };
}

void append_nepq1_s_table(
    std::vector<std::uint8_t>& blob,
    int bank) {
    append_nvq1_s_table(blob, bank != 0);
}

std::vector<float> nepq1_s_vector(int bank) {
    auto vector = decode_ternary_word(
        mfq::nvq_codebooks::
            kNvq1LCodebookPacked[
                bank == 0 ? 0 : 511 * 4]);
    for (auto& value : vector) {
        value += 0.15625f;
    }
    return vector;
}

VqFixture make_rotated_nepq1_s(
    int output,
    int input,
    std::uint64_t seed =
        0x123456789abcdef0ull,
    int sign_shift = 0,
    int cohort_experts = 1) {
    constexpr int table_banks = 2;
    constexpr int rotation_block = 8;
    const int rows = cohort_experts * output;
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NEP1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 2);
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 1);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            cohort_experts));
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, table_banks);
    append<std::uint32_t>(blob, rotation_block);
    append<std::uint64_t>(blob, seed);
    for (int bank = 0; bank < table_banks; ++bank) {
        append_nepq1_s_table(blob, bank);
    }
    append_vq_anchors(blob, rows);
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(rows, 1),
            4));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(rows)
                    * (input / 8),
                0),
            9));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(rows, 0),
            1));
    for (int row = 0; row < rows; ++row) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(row & 1));
    }

    std::vector<float> dense(
        static_cast<std::size_t>(rows) * input);
    for (int row = 0; row < rows; ++row) {
        const auto vector =
            nepq1_s_vector(row & 1);
        for (int column = 0; column < input; ++column) {
            dense[
                static_cast<std::size_t>(row)
                    * input
                + column
            ] = vector[
                static_cast<std::size_t>(column)
                    % vector.size()];
        }
    }

    std::vector<std::uint8_t> runtime;
    append_magic(runtime, "HSG1");
    append<std::uint32_t>(runtime, input);
    append<std::uint32_t>(
        runtime,
        rotation_block);
    append<std::uint64_t>(runtime, seed);
    std::vector<std::int8_t> signs(input);
    for (int column = 0; column < input; ++column) {
        signs[column] = static_cast<std::int8_t>(
            ((column + sign_shift) % 3) == 0
                ? -1
                : 1);
        append<std::int8_t>(
            runtime,
            signs[column]);
    }
    return {
        "NEPQ1-S",
        std::move(blob),
        std::move(runtime),
        std::move(dense),
        std::move(signs),
        rotation_block,
        seed,
        rows,
        input,
    };
}

VqFixture make_rotated_nepq0_s(
    int output,
    int input,
    std::uint64_t seed = 0x1020304050607080ull) {
    constexpr int table_banks = 2;
    constexpr int rotation_block = 8;
    if (input % 8 != 0) {
        throw std::runtime_error("invalid NEPQ0-S fixture width");
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NEP1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 0);
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 1);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, table_banks);
    append<std::uint32_t>(blob, rotation_block);
    append<std::uint64_t>(blob, seed);
    for (int bank = 0; bank < table_banks; ++bank) {
        append_bytes(blob, npq_table(true, bank));
    }
    append_vq_anchors(blob, output);
    const int groups = (input + 23) / 24;
    const int vectors = input / 8;
    const int supergroups = (groups + 3) / 4;
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output) * groups,
                1),
            2));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output) * vectors,
                0),
            6));
    for (int row = 0; row < output; ++row) {
        for (int group = 0; group < supergroups; ++group) {
            append<std::uint8_t>(
                blob,
                static_cast<std::uint8_t>(row & 1));
        }
    }
    std::vector<float> dense(
        static_cast<std::size_t>(output) * input);
    for (int row = 0; row < output; ++row) {
        const float first = static_cast<float>(1 + (row & 1) * 2);
        const float second = first + 1.0f;
        for (int column = 0; column < input; ++column) {
            dense[static_cast<std::size_t>(row) * input + column] =
                (column & 7) < 4 ? first : second;
        }
    }
    std::vector<std::uint8_t> runtime;
    append_magic(runtime, "HSG1");
    append<std::uint32_t>(runtime, input);
    append<std::uint32_t>(runtime, rotation_block);
    append<std::uint64_t>(runtime, seed);
    std::vector<std::int8_t> signs(input);
    for (int column = 0; column < input; ++column) {
        signs[column] = static_cast<std::int8_t>(
            column % 3 == 0 ? -1 : 1);
        append<std::int8_t>(runtime, signs[column]);
    }
    return {
        "NEPQ0-S",
        std::move(blob),
        std::move(runtime),
        std::move(dense),
        std::move(signs),
        rotation_block,
        seed,
        output,
        input,
    };
}

VqFixture add_nepq_a_residual(
    VqFixture fixture,
    bool second_stream) {
    constexpr int dictionary_entries = 1024;
    const int position_bits = second_stream ? 4 : 5;
    const int record_bits = position_bits + 10;
    const int block_vectors = second_stream ? 16 : 24;
    const int vectors = fixture.input / 8;
    const int blocks_per_row =
        (vectors + block_vectors - 1) / block_vectors;
    const int block_count = fixture.output * blocks_per_row;
    fixture.dtype = second_stream ? "NEPQ1-A" : "NEPQ0-A";
    fixture.blob[5] = static_cast<std::uint8_t>(
        second_stream ? 5 : 4);

    append_magic(fixture.blob, "NRA1");
    append<std::uint8_t>(fixture.blob, 1);
    append<std::uint8_t>(fixture.blob, record_bits);
    append<std::uint8_t>(fixture.blob, position_bits);
    append<std::uint8_t>(fixture.blob, second_stream ? 1 : 0);
    append<std::uint32_t>(fixture.blob, dictionary_entries);
    append<std::uint32_t>(fixture.blob, block_count);
    append<std::uint32_t>(
        fixture.blob,
        second_stream ? block_count / 2 : 0);
    append<std::uint32_t>(fixture.blob, 0);
    append<std::uint64_t>(fixture.blob, 0);
    fixture.blob.insert(fixture.blob.end(), 32, 0);
    for (int dictionary = 0;
         dictionary < dictionary_entries;
         ++dictionary) {
        for (int component = 0; component < 8; ++component) {
            append<std::uint16_t>(
                fixture.blob,
                dictionary == 0
                    ? 0x3800u
                    : (dictionary == 1 ? 0xb400u : 0u));
        }
    }
    std::vector<std::uint16_t> first(block_count);
    for (int row = 0; row < fixture.output; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            const int available = std::min(
                block_vectors,
                vectors - block * block_vectors);
            const int position = row % available;
            first[row * blocks_per_row + block] =
                static_cast<std::uint16_t>(position);
            for (int component = 0; component < 8; ++component) {
                fixture.dense[
                    static_cast<std::size_t>(row) * fixture.input
                    + (block * block_vectors + position) * 8
                    + component
                ] += 0.5f;
            }
        }
    }
    append_bytes(fixture.blob, pack_vq_values(first, record_bits));
    if (second_stream) {
        std::vector<std::uint16_t> mask(block_count, 0);
        std::vector<std::uint16_t> second;
        for (int block = 1; block < block_count; block += 2) {
            const int row = block / blocks_per_row;
            const int local_block = block % blocks_per_row;
            const int available = std::min(
                block_vectors,
                vectors - local_block * block_vectors);
            const int position = (row + 1) % available;
            mask[block] = 1;
            second.push_back(static_cast<std::uint16_t>(
                (1 << position_bits) | position));
            for (int component = 0; component < 8; ++component) {
                fixture.dense[
                    static_cast<std::size_t>(row) * fixture.input
                    + (local_block * block_vectors + position) * 8
                    + component
                ] -= 0.25f;
            }
        }
        append_bytes(fixture.blob, pack_vq_values(mask, 1));
        append_bytes(fixture.blob, pack_vq_values(second, record_bits));
    }
    return fixture;
}

struct TensorFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int columns = 0;
};

TensorFixture make_dense_tensor(
    std::string_view dtype,
    int rows,
    int columns,
    int salt) {
    constexpr std::array<float, 9> values{
        0.0f, 0.25f, -0.25f, 0.5f, -0.5f,
        1.0f, -1.0f, 2.0f, -2.0f,
    };
    constexpr std::array<std::uint16_t, 9> f16_bits{
        0x0000, 0x3400, 0xb400, 0x3800, 0xb800,
        0x3c00, 0xbc00, 0x4000, 0xc000,
    };
    constexpr std::array<std::uint16_t, 9> bf16_bits{
        0x0000, 0x3e80, 0xbe80, 0x3f00, 0xbf00,
        0x3f80, 0xbf80, 0x4000, 0xc000,
    };
    const bool bf16 = dtype == "BF16";
    if ((!bf16 && dtype != "F16") || rows <= 0 || columns <= 0) {
        throw std::runtime_error("invalid dense expert fixture");
    }
    std::vector<std::uint8_t> blob;
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    std::vector<float> dense(static_cast<std::size_t>(rows) * columns);
    for (std::size_t index = 0; index < dense.size(); ++index) {
        const auto selected = (index * 5 + static_cast<std::size_t>(salt))
            % values.size();
        dense[index] = values[selected];
        append<std::uint16_t>(
            blob,
            bf16 ? bf16_bits[selected] : f16_bits[selected]);
    }
    return {
        std::move(blob),
        std::move(dense),
        rows,
        columns,
    };
}

TensorFixture make_nint_tensor(
    int bits,
    int rows,
    int columns,
    int salt,
    int group_size = kGroupSize) {
    if (group_size <= 0 || columns <= 0) {
        throw std::runtime_error("test NINT geometry is invalid");
    }
    const int groups = (columns + group_size - 1) / group_size;
    const auto metadata_count =
        static_cast<std::size_t>(rows) * groups;
    const auto value_count =
        metadata_count * group_size;
    const auto maximum = (1u << bits) - 1u;

    std::vector<std::uint8_t> sub_scale(
        metadata_count);
    std::vector<std::uint8_t> sub_min(
        metadata_count);
    for (
        std::size_t index = 0;
        index < metadata_count;
        ++index
    ) {
        sub_scale[index] =
            static_cast<std::uint8_t>(
                1 + (index + salt) % 3);
        sub_min[index] =
            static_cast<std::uint8_t>(
                (index + salt) & 1u);
    }
    std::vector<std::uint8_t> quantized(
        value_count);
    for (
        std::size_t index = 0;
        index < value_count;
        ++index
    ) {
        quantized[index] =
            static_cast<std::uint8_t>(
                (
                    index
                        * static_cast<std::size_t>(
                            bits + 3)
                    + static_cast<std::size_t>(
                        salt * 5 + 1)
                ) % (maximum + 1));
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(blob, 2);
    append<std::int32_t>(
        blob,
        group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, columns);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(rows));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(groups));

    // Exact FP16 powers of two: anchor scale=1/64, minimum=1/128.
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2400);
    }
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2000);
    }
    for (
        const auto* metadata :
        {&sub_scale, &sub_min}
    ) {
        const auto packed =
            pack_values(*metadata, 2);
        blob.insert(
            blob.end(),
            packed.begin(),
            packed.end());
    }
    const auto packed =
        pack_values(quantized, bits);
    blob.insert(
        blob.end(),
        packed.begin(),
        packed.end());

    std::vector<float> dense(
        static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (
            int column = 0;
            column < columns;
            ++column
        ) {
            const int group =
                column / group_size;
            const auto metadata =
                static_cast<std::size_t>(row)
                    * groups
                + group;
            const auto value =
                metadata * group_size
                + column % group_size;
            dense[
                static_cast<std::size_t>(row)
                    * columns
                + column
            ] =
                static_cast<float>(
                    sub_scale[metadata])
                    * (1.0f / 64.0f)
                    * static_cast<float>(
                        quantized[value])
                - static_cast<float>(
                    sub_min[metadata])
                    * (1.0f / 128.0f);
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        rows,
        columns,
    };
}

TensorFixture make_nint_v2_tensor(
    int rows,
    int columns,
    int salt,
    int group_size = kGroupSize) {
    constexpr int nominal_bits = 4;
    constexpr int nominal_sub_bits = 6;
    const int groups = (columns + group_size - 1) / group_size;
    const int values_per_row = groups * group_size;
    std::vector<std::uint8_t> row_q_bits(rows);
    std::vector<std::uint8_t> row_sub_bits(rows);
    std::vector<std::uint8_t> q_selectors(rows);
    std::vector<std::uint8_t> sub_selectors(rows);
    std::vector<std::uint8_t> quantized(
        static_cast<std::size_t>(rows) * values_per_row);
    std::vector<std::uint8_t> sub_scale(
        static_cast<std::size_t>(rows) * groups);
    std::vector<std::uint8_t> sub_min(sub_scale.size());
    for (int row = 0; row < rows; ++row) {
        const int q_bits = 2 + (row + salt) % 5;
        const int sub_bits = 5 + (row + salt) % 4;
        row_q_bits[row] = static_cast<std::uint8_t>(q_bits);
        row_sub_bits[row] = static_cast<std::uint8_t>(sub_bits);
        q_selectors[row] = static_cast<std::uint8_t>(q_bits - 1);
        sub_selectors[row] = static_cast<std::uint8_t>(sub_bits - 5);
        const int q_maximum = (1 << q_bits) - 1;
        for (int group = 0; group < groups; ++group) {
            const auto metadata = static_cast<std::size_t>(row) * groups + group;
            sub_scale[metadata] = static_cast<std::uint8_t>(
                1 + (row * 3 + group + salt) % 7);
            sub_min[metadata] = static_cast<std::uint8_t>(
                (row + group + salt) & 1);
            for (int element = 0; element < group_size; ++element) {
                const auto value = static_cast<std::size_t>(row) * values_per_row
                    + group * group_size + element;
                quantized[value] = static_cast<std::uint8_t>(
                    (row * 11 + group * 5 + element * 3 + salt) & q_maximum);
            }
        }
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, 0x80u | nominal_bits);
    append<std::uint8_t>(blob, nominal_sub_bits);
    append<std::int32_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, columns);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    append<std::uint32_t>(blob, static_cast<std::uint32_t>(rows));
    append<std::uint32_t>(blob, static_cast<std::uint32_t>(groups));
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2400);
    }
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2000);
    }
    append_bytes(blob, pack_values(sub_selectors, 2));
    for (int selector = 0; selector < 4; ++selector) {
        std::vector<std::uint8_t> scales;
        std::vector<std::uint8_t> minima;
        for (int row = 0; row < rows; ++row) {
            if (sub_selectors[row] != selector) continue;
            const auto begin = static_cast<std::size_t>(row) * groups;
            scales.insert(scales.end(), sub_scale.begin() + begin,
                sub_scale.begin() + begin + groups);
            minima.insert(minima.end(), sub_min.begin() + begin,
                sub_min.begin() + begin + groups);
        }
        append_bytes(blob, pack_values(scales, nominal_sub_bits - 1 + selector));
        append_bytes(blob, pack_values(minima, nominal_sub_bits - 1 + selector));
    }
    append_bytes(blob, pack_values(q_selectors, 3));
    for (int selector = 0; selector < 8; ++selector) {
        std::vector<std::uint8_t> cohort;
        for (int row = 0; row < rows; ++row) {
            if (q_selectors[row] != selector) continue;
            const auto begin = static_cast<std::size_t>(row) * values_per_row;
            cohort.insert(cohort.end(), quantized.begin() + begin,
                quantized.begin() + begin + values_per_row);
        }
        append_bytes(blob, pack_values(cohort, selector + 1));
    }

    std::vector<float> dense(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const int group = column / group_size;
            const auto metadata = static_cast<std::size_t>(row) * groups + group;
            const auto value = static_cast<std::size_t>(row) * values_per_row
                + group * group_size + column % group_size;
            dense[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(sub_scale[metadata]) * (1.0f / 64.0f)
                    * static_cast<float>(quantized[value])
                - static_cast<float>(sub_min[metadata]) * (1.0f / 128.0f);
        }
    }
    return {std::move(blob), std::move(dense), rows, columns};
}

TensorFixture make_q8_tensor(
    int rows,
    int columns,
    int salt) {
    if (columns % 32 != 0) {
        throw std::runtime_error(
            "test Q8 width must be a multiple of 32");
    }
    constexpr std::uint16_t scale_bits[] = {
        0x2000,
        0x2400,
        0x2800,
        0x2c00,
    };
    constexpr float scale_values[] = {
        1.0f / 128.0f,
        1.0f / 64.0f,
        1.0f / 32.0f,
        1.0f / 16.0f,
    };
    const int groups = columns / 32;
    std::vector<std::int8_t> quantized(
        static_cast<std::size_t>(rows) * columns);
    for (
        std::size_t index = 0;
        index < quantized.size();
        ++index
    ) {
        quantized[index] =
            static_cast<std::int8_t>(
                static_cast<int>(
                    (
                        index * 11
                        + static_cast<std::size_t>(
                            salt * 7)
                    ) % 25)
                - 12);
    }

    std::vector<std::uint8_t> blob{
        'N', 'I', '8', '0',
    };
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, columns);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(rows));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(groups));
    for (int row = 0; row < rows; ++row) {
        for (
            int group = 0;
            group < groups;
            ++group
        ) {
            const int scale =
                (row + group + salt) & 3;
            append<std::uint16_t>(
                blob,
                scale_bits[scale]);
            const auto offset =
                static_cast<std::size_t>(row)
                    * columns
                + group * 32;
            const auto* bytes =
                reinterpret_cast<
                    const std::uint8_t*>(
                    quantized.data() + offset);
            blob.insert(
                blob.end(),
                bytes,
                bytes + 32);
        }
    }

    std::vector<float> dense(
        static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (
            int column = 0;
            column < columns;
            ++column
        ) {
            const int group = column / 32;
            dense[
                static_cast<std::size_t>(row)
                    * columns
                + column
            ] =
                scale_values[
                    (row + group + salt) & 3]
                * static_cast<float>(
                    quantized[
                        static_cast<std::size_t>(row)
                            * columns
                        + column]);
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        rows,
        columns,
    };
}

struct PoolFixture {
    std::vector<std::int32_t> expert_ids;
    std::string dtype;
    TensorFixture tensor;
    std::vector<std::uint8_t> runtime;
};

struct MoeFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int experts = 0;
    int output = 0;
    int input = 0;
};

MoeFixture make_moe_fixture(
    const std::vector<std::string>& profiles,
    int output,
    int input,
    int salt,
    int nint_group_size = kGroupSize) {
    const int experts =
        static_cast<int>(profiles.size());
    std::vector<PoolFixture> pools;
    pools.reserve(profiles.size());

    // Deliberately reverse the pool order.  Global expert order must not be
    // confused with cohort-local row order.
    for (
        int expert = experts - 1;
        expert >= 0;
        --expert
    ) {
        const auto& profile = profiles[expert];
        TensorFixture tensor = [&] {
            if (profile == "NINT8-0") {
                return make_q8_tensor(output, input, salt + expert);
            }
            if (profile == "F16" || profile == "BF16") {
                return make_dense_tensor(
                    profile, output, input, salt + expert);
            }
            if (profile == "NINTv2") {
                return make_nint_v2_tensor(
                    output,
                    input,
                    salt + expert,
                    nint_group_size);
            }
            return make_nint_tensor(
                std::stoi(profile.substr(4)),
                output,
                input,
                salt + expert,
                nint_group_size);
        }();
        pools.push_back({
            {expert},
            profile,
            std::move(tensor),
            {},
        });
    }

    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(experts));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(input));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            pools.size()));
    std::vector<float> dense(
        static_cast<std::size_t>(experts)
            * output * input);
    for (const auto& pool : pools) {
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.expert_ids.size()));
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.dtype.size()));
        append<std::uint64_t>(
            blob,
            static_cast<std::uint64_t>(
                pool.tensor.blob.size()));
        append<std::uint64_t>(
            blob,
            static_cast<std::uint64_t>(
                pool.runtime.size()));
        for (const auto expert : pool.expert_ids) {
            append<std::int32_t>(blob, expert);
        }
        blob.insert(
            blob.end(),
            pool.dtype.begin(),
            pool.dtype.end());
        blob.insert(
            blob.end(),
            pool.runtime.begin(),
            pool.runtime.end());
        blob.insert(
            blob.end(),
            pool.tensor.blob.begin(),
            pool.tensor.blob.end());

        for (
            std::size_t local = 0;
            local < pool.expert_ids.size();
            ++local
        ) {
            const int expert =
                pool.expert_ids[local];
            const auto source =
                local
                * static_cast<std::size_t>(output)
                * input;
            const auto target =
                static_cast<std::size_t>(expert)
                * output * input;
            std::copy_n(
                pool.tensor.dense.begin()
                    + static_cast<std::ptrdiff_t>(
                        source),
                static_cast<std::size_t>(output)
                    * input,
                dense.begin()
                    + static_cast<std::ptrdiff_t>(
                        target));
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        experts,
        output,
        input,
    };
}

std::vector<std::uint8_t> make_raw_nim2(
    int experts,
    int output,
    int input,
    const std::vector<PoolFixture>& pools) {
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            pools.size()));
    for (const auto& pool : pools) {
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.expert_ids.size()));
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.dtype.size()));
        append<std::uint64_t>(
            blob,
            pool.tensor.blob.size());
        append<std::uint64_t>(
            blob,
            pool.runtime.size());
        for (const auto expert : pool.expert_ids) {
            append<std::int32_t>(blob, expert);
        }
        blob.insert(
            blob.end(),
            pool.dtype.begin(),
            pool.dtype.end());
        blob.insert(
            blob.end(),
            pool.runtime.begin(),
            pool.runtime.end());
        blob.insert(
            blob.end(),
            pool.tensor.blob.begin(),
            pool.tensor.blob.end());
    }
    return blob;
}

std::vector<std::uint8_t> make_nim1(
    int experts,
    int output,
    int input,
    const TensorFixture& tensor) {
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '1',
    };
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    append<std::uint64_t>(
        blob,
        tensor.blob.size());
    for (int expert = 0; expert < experts; ++expert) {
        append<std::int32_t>(blob, expert);
    }
    blob.insert(
        blob.end(),
        tensor.blob.begin(),
        tensor.blob.end());
    return blob;
}

struct RoutedVqFixture {
    std::vector<std::uint8_t> blob;
    std::vector<VqFixture> expert_weights;
    int experts = 0;
    int output = 0;
    int input = 0;
};

RoutedVqFixture make_vq_moe_fixture(
    std::vector<VqFixture> weights) {
    if (weights.empty()) {
        throw std::runtime_error(
            "VQ MoE fixture cannot be empty");
    }
    const int output = weights.front().output;
    const int input = weights.front().input;
    for (const auto& weight : weights) {
        if (
            weight.output != output
            || weight.input != input
        ) {
            throw std::runtime_error(
                "VQ MoE fixture shape mismatch");
        }
    }
    const int experts =
        static_cast<int>(weights.size());
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, experts);
    for (int expert = experts - 1;
         expert >= 0;
         --expert) {
        const auto& weight = weights[expert];
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                weight.dtype.size()));
        append<std::uint64_t>(
            blob,
            weight.blob.size());
        append<std::uint64_t>(
            blob,
            weight.runtime.size());
        append<std::int32_t>(blob, expert);
        blob.insert(
            blob.end(),
            weight.dtype.begin(),
            weight.dtype.end());
        blob.insert(
            blob.end(),
            weight.runtime.begin(),
            weight.runtime.end());
        blob.insert(
            blob.end(),
            weight.blob.begin(),
            weight.blob.end());
    }
    return {
        std::move(blob),
        std::move(weights),
        experts,
        output,
        input,
    };
}

std::vector<float> rotate_vq_input(
    const std::vector<float>& source,
    const VqFixture& weight) {
    auto values = source;
    if (weight.rotation_block == 0) {
        return values;
    }
    for (int start = 0;
         start < weight.input;
         start += weight.rotation_block) {
        for (int index = 0;
             index < weight.rotation_block;
             ++index) {
            values[start + index] *=
                weight.signs[start + index];
        }
        for (int stride = 1;
             stride < weight.rotation_block;
             stride <<= 1) {
            for (int base = 0;
                 base < weight.rotation_block;
                 base += stride * 2) {
                for (int offset = 0;
                     offset < stride;
                     ++offset) {
                    const int first =
                        start + base + offset;
                    const int second = first + stride;
                    const float first_value =
                        values[first];
                    const float second_value =
                        values[second];
                    values[first] =
                        first_value + second_value;
                    values[second] =
                        first_value - second_value;
                }
            }
        }
        const float inverse =
            1.0f / std::sqrt(
                static_cast<float>(
                    weight.rotation_block));
        for (int index = 0;
             index < weight.rotation_block;
             ++index) {
            values[start + index] *= inverse;
        }
    }
    return values;
}

float routed_vq_dot(
    const std::vector<float>& source,
    const RoutedVqFixture& fixture,
    int expert,
    int output) {
    const auto rotated = rotate_vq_input(
        source,
        fixture.expert_weights[expert]);
    float result = 0.0f;
    const auto& weight =
        fixture.expert_weights[expert];
    for (int column = 0;
         column < fixture.input;
         ++column) {
        result +=
            rotated[column]
            * weight.dense[
                static_cast<std::size_t>(output)
                    * fixture.input
                + column];
    }
    return result;
}

void require(
    bool condition,
    const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance = 8e-4f) {
    if (
        !std::isfinite(actual)
        || std::fabs(actual - expected) > tolerance
    ) {
        throw std::runtime_error(
            "MFE Metal result mismatch: actual="
            + std::to_string(actual)
            + " expected="
            + std::to_string(expected));
    }
}

float dot(
    const std::vector<float>& input,
    std::size_t input_offset,
    const MoeFixture& weight,
    int expert,
    int output) {
    float result = 0.0f;
    const auto weight_offset = (
        static_cast<std::size_t>(expert)
            * weight.output
        + output
    ) * weight.input;
    for (
        int column = 0;
        column < weight.input;
        ++column
    ) {
        result +=
            input[input_offset + column]
            * weight.dense[
                weight_offset + column];
    }
    return result;
}

std::vector<float> evaluated_floats(
    mlx::core::array value) {
    value = mlx::core::astype(
        value,
        mlx::core::float32);
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
    };
}

void append_mfq_string(
    std::vector<std::uint8_t>& output,
    std::string_view value) {
    append<std::uint32_t>(
        output,
        static_cast<std::uint32_t>(
            value.size()));
    output.insert(
        output.end(),
        value.begin(),
        value.end());
}

class TemporaryMfq {
public:
    explicit TemporaryMfq(
        const std::vector<MappedRecordFixture>&
            records)
        : path_(
              std::filesystem::
                  temp_directory_path()
              / "mfq-metal-streamed-mfe-test.mfq") {
        std::vector<std::uint8_t> file;
        append_magic(file, "MFQ1");
        append<std::uint32_t>(file, 1);
        append_mfq_string(
            file,
            "streamed-mfe-test");
        append<std::uint32_t>(
            file,
            static_cast<std::uint32_t>(
                records.size()));
        for (const auto& record : records) {
            append_mfq_string(
                file,
                record.name);
            append_mfq_string(
                file,
                record.dtype);
            append<std::uint64_t>(
                file,
                record.payload.size());
        }
        for (const auto& record : records) {
            append_bytes(
                file,
                record.payload);
        }
        std::ofstream stream(
            path_,
            std::ios::binary
                | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error(
                "cannot create streamed MFE "
                "test container");
        }
        stream.write(
            reinterpret_cast<const char*>(
                file.data()),
            static_cast<std::streamsize>(
                file.size()));
        if (!stream) {
            throw std::runtime_error(
                "cannot write streamed MFE "
                "test container");
        }
    }

    ~TemporaryMfq() {
        std::error_code ignored;
        std::filesystem::remove(
            path_,
            ignored);
    }

    const std::filesystem::path& path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

template <typename Function>
void test_all_families_and_projections() {
    constexpr int tokens = 3;
    constexpr int routes = 4;
    constexpr int input_width = 64;
    constexpr int output_width = 5;
    const std::vector<std::string> profiles{
        "NINT2",
        "NINT7",
        "NINT4",
        "NINT6",
        "NINT8-0",
        "NINT3",
        "NINT8",
        "NINT1",
        "NINT5",
        "NINTv2",
    };
    const auto first = make_moe_fixture(
        profiles,
        output_width,
        input_width,
        1);
    const auto second = make_moe_fixture(
        profiles,
        output_width,
        input_width,
        17);
    const auto first_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            first.blob);
    const auto second_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            second.blob);
    require(
        first_weight.experts()
                == static_cast<int>(
                    profiles.size())
            && first_weight.out_per_expert()
                == output_width
            && first_weight.neuron_len()
                == input_width
            && first_weight.projections() == 1
            && first_weight.packed_nbytes() > 0,
        "MFE shape metadata mismatch");

    const std::vector<std::int32_t> ids{
        0, 1, 4, 8,
        7, 5, -1, 10,
        2, 3, 6, 9,
    };
    std::vector<float> shared_input(
        tokens * input_width);
    for (
        std::size_t index = 0;
        index < shared_input.size();
        ++index
    ) {
        shared_input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 7 + 3) % 23)
                - 11)
            / 64.0f;
    }
    auto shared = first_weight.routed_matmul(
        mlx::core::array(
            shared_input.begin(),
            mlx::core::Shape{
                tokens,
                input_width,
            }),
        mlx::core::array(
            ids.begin(),
            mlx::core::Shape{
                tokens,
                routes,
            }));
    const auto shared_values =
        evaluated_floats(std::move(shared));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            for (
                int output = 0;
                output < output_width;
                ++output
            ) {
                const float expected =
                    expert < 0
                            || expert >= first.experts
                    ? 0.0f
                    : dot(
                          shared_input,
                          static_cast<std::size_t>(
                              token)
                              * input_width,
                          first,
                          expert,
                          output);
                require_close(
                    shared_values[
                        (
                            token * routes + route
                        ) * output_width
                        + output],
                    expected);
            }
        }
    }

    // Exercise the separate float16 Metal specialization as used by normal
    // model inference.  The reference rounds both the source and output to
    // FP16 at the same boundaries as the kernel.
    const auto half_values = evaluated_floats(
        first_weight.routed_matmul(
            mlx::core::astype(
                mlx::core::array(
                    shared_input.begin(),
                    mlx::core::Shape{
                        tokens,
                        input_width,
                    }),
                mlx::core::float16),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> rounded_input(
            input_width);
        for (
            int column = 0;
            column < input_width;
            ++column
        ) {
            rounded_input[column] =
                static_cast<float>(
                    static_cast<
                        mlx::core::float16_t>(
                        shared_input[
                            token * input_width
                            + column]));
        }
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            for (
                int output = 0;
                output < output_width;
                ++output
            ) {
                const float reference =
                    expert < 0
                            || expert >= first.experts
                    ? 0.0f
                    : dot(
                          rounded_input,
                          0,
                          first,
                          expert,
                          output);
                const float expected =
                    static_cast<float>(
                        static_cast<
                            mlx::core::float16_t>(
                            reference));
                require_close(
                    half_values[
                        (
                            token * routes + route
                        ) * output_width
                        + output],
                    expected,
                    4e-3f);
            }
        }
    }

    std::vector<float> routed_input(
        tokens * routes * input_width);
    for (
        std::size_t index = 0;
        index < routed_input.size();
        ++index
    ) {
        routed_input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 5 + 9) % 29)
                - 14)
            / 96.0f;
    }
    const auto grouped =
        mfq::metal::MlxMoeWeight::
            concatenate_projections(
                {first_weight, second_weight});
    require(
        grouped.projections() == 2,
        "MFE grouped projection count mismatch");
    require(
        grouped.packed_nbytes()
            == first_weight.packed_nbytes()
                + second_weight.packed_nbytes(),
        "split shared-kernel projection byte accounting mismatch");
    const auto grouped_values = evaluated_floats(
        grouped.routed_matmul(
            mlx::core::array(
                routed_input.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                    input_width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            const auto source_offset =
                static_cast<std::size_t>(
                    token * routes + route)
                * input_width;
            for (
                int projection = 0;
                projection < 2;
                ++projection
            ) {
                const auto& fixture =
                    projection == 0 ? first : second;
                for (
                    int output = 0;
                    output < output_width;
                    ++output
                ) {
                    const float expected =
                        expert < 0
                                || expert
                                    >= fixture.experts
                        ? 0.0f
                        : dot(
                              routed_input,
                              source_offset,
                              fixture,
                              expert,
                              output);
                    require_close(
                        grouped_values[
                            (
                                (
                                    token * routes
                                    + route
                                ) * 2
                                + projection
                            ) * output_width
                            + output],
                        expected);
                }
            }
        }
    }
}

void test_swiglu_ffn() {
    constexpr int tokens = 2;
    constexpr int routes = 2;
    constexpr int hidden = 32;
    constexpr int intermediate = 32;
    const std::vector<std::string> profiles{
        "NINT2",
        "NINT7",
        "NINT8-0",
    };
    const auto gate = make_moe_fixture(
        profiles,
        intermediate,
        hidden,
        2);
    const auto up = make_moe_fixture(
        profiles,
        intermediate,
        hidden,
        9);
    const auto down = make_moe_fixture(
        profiles,
        hidden,
        intermediate,
        15);
    const auto ffn =
        mfq::metal::MlxRoutedSwiGluFfn::
            from_blobs(
                gate.blob,
                up.blob,
                down.blob);
    require(
        ffn.gate_up_weight().projections() == 2,
        "SwiGLU gate/up did not group projections");

    const std::vector<std::int32_t> ids{
        0, 2,
        1, 0,
    };
    const std::vector<float> route_weights{
        0.65f, 0.35f,
        0.25f, 0.75f,
    };
    std::vector<float> input(tokens * hidden);
    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 3 + 1) % 17)
                - 8)
            / 256.0f;
    }
    const auto input_array = mlx::core::array(
        input.begin(),
        mlx::core::Shape{tokens, hidden});
    const auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto activated_actual = evaluated_floats(
        ffn.gate_up_weight().routed_swiglu(
            input_array,
            ids_array));
    const std::vector<float> single_input(
        input.begin(),
        input.begin() + hidden);
    const std::vector<std::int32_t> single_ids(
        ids.begin(),
        ids.begin() + routes);
    const auto single_activated_actual = evaluated_floats(
        ffn.gate_up_weight().routed_swiglu(
            mlx::core::array(
                single_input.begin(),
                mlx::core::Shape{1, hidden}),
            mlx::core::array(
                single_ids.begin(),
                mlx::core::Shape{1, routes})));
    for (
        std::size_t index = 0;
        index < single_activated_actual.size();
        ++index
    ) {
        require_close(
            single_activated_actual[index],
            activated_actual[index],
            3e-3f);
    }
    const auto actual = evaluated_floats(
        ffn.forward(
            input_array,
            ids_array,
            mlx::core::array(
                route_weights.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));

    std::vector<float> expected(
        tokens * hidden,
        0.0f);
    std::vector<float> activated(intermediate);
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            for (
                int column = 0;
                column < intermediate;
                ++column
            ) {
                const float gate_value = dot(
                    input,
                    static_cast<std::size_t>(token)
                        * hidden,
                    gate,
                    expert,
                    column);
                const float up_value = dot(
                    input,
                    static_cast<std::size_t>(token)
                        * hidden,
                    up,
                    expert,
                    column);
                activated[column] =
                    gate_value
                    / (
                        1.0f
                        + std::exp(-gate_value)
                    )
                    * up_value;
                require_close(
                    activated_actual[
                        (token * routes + route)
                            * intermediate
                        + column],
                    activated[column],
                    3e-3f);
            }
            const auto weight_offset =
                static_cast<std::size_t>(expert)
                * hidden * intermediate;
            for (int output = 0; output < hidden; ++output) {
                float value = 0.0f;
                for (
                    int column = 0;
                    column < intermediate;
                    ++column
                ) {
                    value +=
                        activated[column]
                        * down.dense[
                            weight_offset
                            + static_cast<std::size_t>(
                                output)
                                * intermediate
                            + column];
                }
                expected[
                    token * hidden + output
                ] +=
                    route_weights[
                        token * routes + route]
                    * value;
            }
        }
    }
    for (
        std::size_t index = 0;
        index < expected.size();
        ++index
    ) {
        require_close(
            actual[index],
            expected[index],
            3e-3f);
    }
}

void test_mxfp4_mfe_and_projection_offsets() {
    constexpr int experts = 2;
    constexpr int output = 7;
    constexpr int input = 64;
    constexpr int tokens = 2;
    constexpr int routes = 2;
    const auto fixture = make_mxfp4(experts * output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, 5);
    append<std::uint64_t>(blob, fixture.blob.size());
    append<std::uint64_t>(blob, 0);
    // Reverse global IDs so descriptor local-row mapping is independently
    // checked rather than accidentally matching global expert order.
    append<std::int32_t>(blob, 1);
    append<std::int32_t>(blob, 0);
    blob.insert(blob.end(), {'M', 'X', 'F', 'P', '4'});
    append_bytes(blob, fixture.blob);

    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.experts() == experts
            && weight.out_per_expert() == output
            && weight.neuron_len() == input
            && weight.packed_nbytes() >= fixture.blob.size(),
        "MXFP4 MFE metadata mismatch");
    std::vector<float> source(tokens * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 13 + 5) % 29) - 14) / 128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 1, 0};
    const auto input_array = mlx::core::array(
        source.begin(),
        mlx::core::Shape{tokens, input});
    const auto id_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));
    require(
        weight.supports_grouped_mmq(),
        "MXFP4 MFE must support grouped matrix prefill");
    const std::vector<std::int32_t> order{0, 3, 1, 2};
    const auto order_array = mlx::core::array(
        order.begin(),
        mlx::core::Shape{tokens * routes});
    const auto plan = weight.build_grouped_mmq_plan(
        id_array,
        order_array);
    const auto matrix_actual = evaluated_floats(
        mlx::core::take(
            weight.routed_matmul_sorted(
                input_array,
                id_array,
                order_array,
                false,
                false,
                0.0f,
                &plan),
            mlx::core::argsort(order_array),
            0));
    const auto concatenated =
        mfq::metal::MlxMoeWeight::concatenate_projections(
            {weight, weight});
    const auto projected = evaluated_floats(
        concatenated.routed_matmul(input_array, id_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const int local_expert = expert == 1 ? 0 : 1;
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                const auto dense_offset = (
                    static_cast<std::size_t>(local_expert) * output + row
                ) * input;
                for (int column = 0; column < input; ++column) {
                    expected += source[token * input + column]
                        * fixture.dense[dense_offset + column];
                }
                const auto route_offset =
                    static_cast<std::size_t>(token * routes + route);
                require_close(
                    actual[route_offset * output + row],
                    expected,
                    2e-4f);
                require_close(
                    matrix_actual[route_offset * output + row],
                    actual[route_offset * output + row],
                    4e-3f);
                require_close(
                    projected[route_offset * (2 * output) + row],
                    expected,
                    2e-4f);
                require_close(
                    projected[
                        route_offset * (2 * output) + output + row
                    ],
                    expected,
                    2e-4f);
            }
        }
    }
}

void test_mxfp4_multi_pool_native_slots() {
    constexpr int experts = 2;
    constexpr int output = 7;
    constexpr int input = 64;
    constexpr int tokens = 2;
    const std::array<Mxfp4Fixture, experts> fixtures{
        make_mxfp4(output, input, 0),
        make_mxfp4(output, input, 7),
    };
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MFE1");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, experts);
    for (int expert = 0; expert < experts; ++expert) {
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(blob, 5);
        append<std::uint64_t>(blob, fixtures[expert].blob.size());
        append<std::uint64_t>(blob, 0);
        append<std::int32_t>(blob, expert);
        blob.insert(blob.end(), {'M', 'X', 'F', 'P', '4'});
        append_bytes(blob, fixtures[expert].blob);
    }

    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);
    std::vector<float> source(tokens * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 13 + 5) % 29) - 14) / 128.0f;
    }
    const auto source_array = mlx::core::astype(
        mlx::core::array(source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::bfloat16);
    const std::vector<std::int32_t> ids{1, 0};
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, 1});
    const auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(mlx::core::reshape(
                ids_array,
                mlx::core::Shape{tokens})),
            mlx::core::int32));
    auto sorted = weight.routed_matmul_sorted(
        source_array,
        ids_array,
        route_order,
        false,
        false,
        0.0f,
        nullptr,
        true);
    require(
        sorted.dtype() == mlx::core::bfloat16,
        "multi-pool MXFP4 gather-QMM did not preserve BF16");
    const auto actual = evaluated_floats(mlx::core::take(
        std::move(sorted),
        mlx::core::argsort(route_order),
        0));
    const auto rounded_source = evaluated_floats(source_array);
    for (int token = 0; token < tokens; ++token) {
        const int expert = ids[static_cast<std::size_t>(token)];
        for (int row = 0; row < output; ++row) {
            float expected = 0.0f;
            for (int column = 0; column < input; ++column) {
                expected += rounded_source[
                    static_cast<std::size_t>(token) * input + column]
                    * fixtures[expert].dense[
                        static_cast<std::size_t>(row) * input + column];
            }
            require_close(
                actual[static_cast<std::size_t>(token) * output + row],
                expected,
                2.5e-2f);
        }
    }
}

void test_mxfp4_pair_blocks_matches_native_projections() {
    constexpr int experts = 4;
    constexpr int slots = 4;
    constexpr int output = 32;
    constexpr int input = 64;
    constexpr int tokens = 45;
    constexpr int routes = 1;
    constexpr int route_count = tokens * routes;
    constexpr std::size_t packed_stride =
        static_cast<std::size_t>(output) * input / 2;
    constexpr std::size_t scale_stride =
        static_cast<std::size_t>(output) * input / 32;

    std::vector<std::uint8_t> gate_values(slots * packed_stride);
    std::vector<std::uint8_t> up_values(slots * packed_stride);
    std::vector<std::uint8_t> gate_scales(slots * scale_stride);
    std::vector<std::uint8_t> up_scales(slots * scale_stride);
    for (std::size_t index = 0; index < gate_values.size(); ++index) {
        gate_values[index] = static_cast<std::uint8_t>(
            ((index * 13 + 0x31) & 15u)
            | (((index * 7 + 0x0b) & 15u) << 4));
        up_values[index] = static_cast<std::uint8_t>(
            ((index * 5 + 0x07) & 15u)
            | (((index * 11 + 0x03) & 15u) << 4));
    }
    for (std::size_t index = 0; index < gate_scales.size(); ++index) {
        gate_scales[index] = static_cast<std::uint8_t>(126 + index % 3);
        up_scales[index] = static_cast<std::uint8_t>(125 + index % 4);
    }
    const std::vector<std::int32_t> gate_slots{2, 0, 3, 1};
    const std::vector<std::int32_t> up_slots{1, 3, 0, 2};
    const auto gate = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        gate_slots,
        mlx::core::array(
            gate_values.begin(),
            mlx::core::Shape{slots, static_cast<int>(packed_stride)}),
        mlx::core::array(
            gate_scales.begin(),
            mlx::core::Shape{slots, static_cast<int>(scale_stride)}));
    const auto up = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        up_slots,
        mlx::core::array(
            up_values.begin(),
            mlx::core::Shape{slots, static_cast<int>(packed_stride)}),
        mlx::core::array(
            up_scales.begin(),
            mlx::core::Shape{slots, static_cast<int>(scale_stride)}));
    require(
        gate.supports_mxfp4_pair_blocks(up),
        "compatible MXFP4 slot views rejected pair blocks");

    std::vector<float> input_values(tokens * input);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 41) - 20) / 256.0f;
    }
    std::vector<std::int32_t> sorted_ids(route_count);
    std::fill(sorted_ids.begin(), sorted_ids.begin() + 35, 0);
    std::fill(sorted_ids.begin() + 35, sorted_ids.begin() + 40, 1);
    std::fill(sorted_ids.begin() + 40, sorted_ids.begin() + 43, 2);
    std::fill(sorted_ids.begin() + 43, sorted_ids.end(), 3);
    std::vector<std::int32_t> ids(route_count);
    for (int index = 0; index < route_count; ++index) {
        ids[static_cast<std::size_t>((index * 17) % route_count)] =
            sorted_ids[static_cast<std::size_t>(index)];
    }
    const auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(mlx::core::reshape(
                ids_array,
                mlx::core::Shape{route_count})),
            mlx::core::int32));
    const auto plan = gate.build_grouped_mmq_plan(
        ids_array,
        route_order,
        32);
    require(
        plan.block_rows == 32 && plan.route_count == route_count,
        "MXFP4 pair block plan geometry mismatch");

    const auto check_dtype = [&](mlx::core::Dtype dtype, float tolerance) {
        auto source = mlx::core::astype(
            mlx::core::array(
                input_values.begin(),
                mlx::core::Shape{tokens, input}),
            dtype);
        auto sorted_source = mlx::core::take(source, route_order, 0);
        const auto single_reference = evaluated_floats(
            gate.routed_matmul_sorted(
                sorted_source,
                ids_array,
                route_order,
                true,
                false,
                0.0f,
                &plan,
                true));
        auto single = gate.mxfp4_block_matmul_sorted(
            sorted_source,
            plan);
        require(
            single.dtype() == mlx::core::float16,
            "MXFP4 single block output dtype mismatch");
        const auto single_actual = evaluated_floats(std::move(single));
        require(
            single_actual.size() == single_reference.size(),
            "MXFP4 single block output size mismatch");
        for (std::size_t index = 0; index < single_actual.size(); ++index) {
            require_close(
                single_actual[index],
                single_reference[index],
                tolerance);
        }
        auto gate_output = gate.routed_matmul_sorted(
            sorted_source,
            ids_array,
            route_order,
            true,
            false,
            0.0f,
            &plan,
            true);
        auto up_output = up.routed_matmul_sorted(
            sorted_source,
            ids_array,
            route_order,
            true,
            false,
            0.0f,
            &plan,
            true);
        constexpr float swiglu_limit = 10.0f;
        const auto reference = evaluated_floats(
            mfq::metal::moe_limited_swiglu_pair(
                std::move(gate_output),
                std::move(up_output),
                swiglu_limit));
        auto paired = gate.mxfp4_pair_swiglu_sorted(
            up,
            sorted_source,
            plan,
            swiglu_limit);
        require(
            paired.dtype() == mlx::core::float16,
            "MXFP4 pair block output dtype mismatch");
        const auto actual = evaluated_floats(std::move(paired));
        require(
            actual.size() == reference.size(),
            "MXFP4 pair block output size mismatch");
        for (std::size_t index = 0; index < actual.size(); ++index) {
            require_close(actual[index], reference[index], tolerance);
        }

        for (const int small_tokens : {1, 2, 4, 6}) {
            auto small_source = mlx::core::slice(
                source,
                mlx::core::Shape{0, 0},
                mlx::core::Shape{small_tokens, input});
            const auto small_ids = mlx::core::array(
                ids.begin(),
                mlx::core::Shape{small_tokens, routes});
            const auto small_reference = evaluated_floats(
                mfq::metal::moe_limited_swiglu_pair(
                    gate.routed_matmul(small_source, small_ids),
                    up.routed_matmul(small_source, small_ids),
                    swiglu_limit));
            auto small_paired = gate.routed_swiglu_pair(
                up,
                small_source,
                small_ids,
                swiglu_limit);
            require(
                small_paired.dtype() == mlx::core::float16,
                "MXFP4 pair small-M output dtype mismatch");
            const auto small_actual = evaluated_floats(
                std::move(small_paired));
            require(
                small_actual.size() == small_reference.size(),
                "MXFP4 pair small-M output size mismatch");
            for (std::size_t index = 0;
                 index < small_actual.size();
                 ++index) {
                require_close(
                    small_actual[index],
                    small_reference[index],
                    tolerance);
            }
        }
    };
    check_dtype(mlx::core::float16, 8e-3f);
    check_dtype(mlx::core::bfloat16, 3e-2f);
}

void test_mxfp4_sq_mfe_reuses_linear_kernel() {
    constexpr int experts = 3;
    constexpr int output = 5;
    constexpr int input = 64;
    constexpr int tokens = 2;
    constexpr int routes = 3;
    const auto fixture = make_mxfp4_sq2(experts * output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MFE1");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    const std::string dtype = "MXFP4-SQ";
    append<std::uint32_t>(blob, dtype.size());
    append<std::uint64_t>(blob, fixture.blob.size());
    append<std::uint64_t>(blob, 0);
    // Cohort-local rows intentionally use a non-global expert order.
    const std::array<std::int32_t, experts> local_to_global{2, 0, 1};
    for (const auto expert : local_to_global) {
        append<std::int32_t>(blob, expert);
    }
    blob.insert(blob.end(), dtype.begin(), dtype.end());
    append_bytes(blob, fixture.blob);

    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.experts() == experts &&
            weight.out_per_expert() == output &&
            weight.neuron_len() == input &&
            weight.packed_nbytes() >= fixture.blob.size(),
        "MXFP4-SQ MFE metadata mismatch");
    require(
        !weight.supports_grouped_mmq(),
        "MXFP4-SQ must not create a second heterogeneous grouped kernel");

    std::vector<float> source(static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] =
            static_cast<float>(static_cast<int>((index * 5 + 1) % 23) - 11) /
            128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 2, 2, 0, 1};
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto id_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const auto found = std::find(
                local_to_global.begin(), local_to_global.end(), expert);
            const int local = static_cast<int>(
                found - local_to_global.begin());
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input; ++column) {
                    expected +=
                        source[static_cast<std::size_t>(token) * input + column]
                        * fixture.dense[
                            (static_cast<std::size_t>(local) * output + row)
                                * input
                            + column];
                }
                const auto index =
                    (static_cast<std::size_t>(token) * routes + route)
                        * output
                    + row;
                require_close(actual[index], expected, 3e-3f);
            }
        }
    }

    const auto doubled = mfq::metal::MlxMoeWeight::concatenate_projections(
        {weight, weight});
    const auto projected = evaluated_floats(
        doubled.routed_matmul(input_array, id_array));
    for (std::size_t route = 0;
         route < static_cast<std::size_t>(tokens * routes);
         ++route) {
        for (int row = 0; row < output; ++row) {
            require_close(
                projected[route * 2 * output + row],
                actual[route * output + row],
                1e-6f);
            require_close(
                projected[route * 2 * output + output + row],
                actual[route * output + row],
                1e-6f);
        }
    }
}

void test_mxfp4_sq_mixed_mfe_routes_without_sq_heterogeneous_kernel() {
    constexpr int experts = 3;
    constexpr int output = 5;
    constexpr int input = 64;
    constexpr int tokens = 2;
    constexpr int routes = 3;
    auto sq = make_mxfp4_sq2(2 * output, input);
    auto nint = make_nint_tensor(4, output, input, 17, kGroupSize);
    std::vector<PoolFixture> pools;
    pools.push_back({
        {2, 0},
        "MXFP4-SQ2",
        TensorFixture{
            std::move(sq.blob),
            std::move(sq.dense),
            2 * output,
            input,
        },
        {},
    });
    pools.push_back({{1}, "NINT4", std::move(nint), {}});
    const auto blob = make_raw_nim2(experts, output, input, pools);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);

    std::vector<float> source(static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] =
            static_cast<float>(static_cast<int>((index * 11 + 7) % 37) - 18)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 2, 2, 1, 0};
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto id_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));

    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const PoolFixture* pool = expert == 1 ? &pools[1] : &pools[0];
            const auto found = std::find(
                pool->expert_ids.begin(), pool->expert_ids.end(), expert);
            const int local = static_cast<int>(found - pool->expert_ids.begin());
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input; ++column) {
                    expected +=
                        source[static_cast<std::size_t>(token) * input + column]
                        * pool->tensor.dense[
                            (static_cast<std::size_t>(local) * output + row)
                                * input
                            + column];
                }
                const auto index =
                    (static_cast<std::size_t>(token) * routes + route)
                        * output
                    + row;
                require_close(actual[index], expected, 8e-3f);
            }
        }
    }
}

void test_mxfp4_smallm_nax_policy() {
    constexpr int experts = 24;
    constexpr int output = 8;
    constexpr int input = 64;
    constexpr int routes = 6;
    std::vector<std::int32_t> slot_for_expert(experts);
    std::iota(slot_for_expert.begin(), slot_for_expert.end(), 0);
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(experts) * output * input / 2,
        0x22);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(experts) * output * input / 32,
        127);
    const auto weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        slot_for_expert,
        mlx::core::array(
            packed.begin(),
            mlx::core::Shape{experts, output * input / 2}),
        mlx::core::array(
            scales.begin(),
            mlx::core::Shape{experts, output * input / 32}));

    std::vector<std::int32_t> repeated(4 * routes);
    for (int row = 0; row < 4; ++row) {
        for (int route = 0; route < routes; ++route) {
            repeated[static_cast<std::size_t>(row) * routes + route] = route;
        }
    }
    std::vector<std::int32_t> unique(4 * routes);
    std::iota(unique.begin(), unique.end(), 0);
    const mlx::core::array repeated_ids(
        repeated.begin(),
        mlx::core::Shape{4, routes});
    const mlx::core::array unique_ids(
        unique.begin(),
        mlx::core::Shape{4, routes});
    const mlx::core::array short_ids(
        repeated.begin(),
        mlx::core::Shape{3, routes});

    const char* prior = std::getenv("MFQ_METAL_MFE_SMALLM_NAX");
    const std::optional<std::string> saved = prior == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior);
    setenv("MFQ_METAL_MFE_SMALLM_NAX", "1", 1);
    require(
        weight.prefers_mxfp4_smallm_nax(repeated_ids),
        "small-M MXFP4 NAX must accept repeated M=4 routes");
    require(
        !weight.prefers_mxfp4_smallm_nax(unique_ids),
        "small-M MXFP4 NAX must preserve the unique-route kernel");
    require(
        !weight.prefers_mxfp4_smallm_nax(short_ids),
        "small-M MXFP4 NAX must preserve the M<4 kernel");
    constexpr int arena_slots = 300;
    std::vector<std::int32_t> arena_slot_map(arena_slots);
    std::iota(arena_slot_map.begin(), arena_slot_map.end(), 0);
    std::vector<std::uint8_t> arena_packed(
        static_cast<std::size_t>(arena_slots) * output * input / 2,
        0x22);
    std::vector<std::uint8_t> arena_scales(
        static_cast<std::size_t>(arena_slots) * output * input / 32,
        127);
    const auto arena_weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        arena_slots,
        output,
        input,
        arena_slot_map,
        mlx::core::array(
            arena_packed.begin(),
            mlx::core::Shape{arena_slots, output * input / 2}),
        mlx::core::array(
            arena_scales.begin(),
            mlx::core::Shape{arena_slots, output * input / 32}),
        256);
    std::vector<std::int32_t> high_physical_ids(4 * routes, 299);
    require(
        arena_weight.prefers_mxfp4_smallm_nax(
            mlx::core::array(
                high_physical_ids.begin(),
                mlx::core::Shape{4, routes})),
        "logical expert geometry rejected a valid high SSD arena slot");
    setenv("MFQ_METAL_MFE_SMALLM_NAX", "0", 1);
    require(
        !weight.prefers_mxfp4_smallm_nax(repeated_ids),
        "small-M MXFP4 NAX disable override was ignored");

    const char* prior_prefill = std::getenv("MFQ_METAL_MFE_PREFILL_NAX");
    const std::optional<std::string> saved_prefill = prior_prefill == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior_prefill);
    setenv("MFQ_METAL_MFE_PREFILL_NAX", "1", 1);
    require(
        weight.prefers_mxfp4_nax_prefill(4 * routes),
        "MXFP4 NAX prefill query ignored an explicit enable");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(routes) == 5440,
        "MXFP4 NAX top-k=6 prefill recommendation mismatch");
    const auto automatic_nax_disabled =
        weight.with_automatic_mxfp4_nax_prefill(false);
    require(
        automatic_nax_disabled.recommended_mxfp4_nax_prefill_tokens(routes)
            == 5440,
        "explicit MXFP4 NAX force must override the source safety policy");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(1) == 32768,
        "MXFP4 NAX top-k=1 prefill recommendation mismatch");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(0) == 0,
        "MXFP4 NAX accepted an invalid route count");
    const auto paired = mfq::metal::MlxMoeWeight::concatenate_projections(
        {weight, weight});
    require(
        paired.recommended_mxfp4_nax_prefill_tokens(routes) == 5440,
        "split MXFP4 projections lost their prefill recommendation");
    setenv("MFQ_METAL_MFE_PREFILL_NAX", "0", 1);
    require(
        !weight.prefers_mxfp4_nax_prefill(4 * routes),
        "MXFP4 NAX prefill query ignored an explicit disable");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(routes) == 0,
        "MXFP4 NAX prefill disable override was ignored");
    if (saved_prefill.has_value()) {
        setenv(
            "MFQ_METAL_MFE_PREFILL_NAX",
            saved_prefill->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_PREFILL_NAX");
    }

    std::vector<float> input_values(4 * input);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(
            static_cast<int>((index * 11 + 3) % 31) - 15) / 256.0f;
    }
    const mlx::core::array input_array(
        input_values.begin(),
        mlx::core::Shape{4, input});
    const auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    repeated_ids,
                    mlx::core::Shape{4 * routes})),
            mlx::core::int32));
    const auto reference = evaluated_floats(
        weight.routed_matmul(input_array, repeated_ids));
    const auto forced_nax = evaluated_floats(
        mlx::core::take(
            weight.routed_matmul_sorted(
                input_array,
                repeated_ids,
                route_order,
                false,
                false,
                0.0f,
                nullptr,
                true),
            mlx::core::argsort(route_order),
            0));
    require(
        reference.size() == forced_nax.size(),
        "small-M MXFP4 NAX output size mismatch");
    for (std::size_t index = 0; index < reference.size(); ++index) {
        require_close(reference[index], forced_nax[index], 4e-3f);
    }

    const auto bf16_input = mlx::core::astype(
        input_array,
        mlx::core::bfloat16);
    auto bf16_sorted = weight.routed_matmul_sorted(
        bf16_input,
        repeated_ids,
        route_order,
        false,
        false,
        0.0f,
        nullptr,
        true);
    require(
        bf16_sorted.dtype() == mlx::core::bfloat16,
        "native MXFP4 gather-QMM did not preserve BF16 activations");
    const auto bf16_nax = evaluated_floats(
        mlx::core::take(
            std::move(bf16_sorted),
            mlx::core::argsort(route_order),
            0));
    require(
        reference.size() == bf16_nax.size(),
        "BF16 MXFP4 NAX output size mismatch");
    for (std::size_t index = 0; index < reference.size(); ++index) {
        require_close(reference[index], bf16_nax[index], 1.5e-2f);
    }

    if (saved.has_value()) {
        setenv(
            "MFQ_METAL_MFE_SMALLM_NAX",
            saved->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_SMALLM_NAX");
    }
}

void test_mxfp4_decode_down_reduce() {
    constexpr int experts = 8;
    constexpr int output = 13;
    constexpr int input = 64;
    constexpr int routes = 6;
    const auto fixture = make_mxfp4(experts * output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, 5);
    append<std::uint64_t>(blob, fixture.blob.size());
    append<std::uint64_t>(blob, 0);
    for (int expert = 0; expert < experts; ++expert) {
        append<std::int32_t>(blob, experts - expert - 1);
    }
    blob.insert(blob.end(), {'M', 'X', 'F', 'P', '4'});
    append_bytes(blob, fixture.blob);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);

    std::vector<float> source_values(routes * input);
    for (std::size_t index = 0; index < source_values.size(); ++index) {
        source_values[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 37) - 18) / 128.0f;
    }
    const std::vector<std::int32_t> ids{7, 1, 6, 0, 3, 2};
    const std::vector<float> route_weights{
        0.07f, 0.13f, 0.19f, 0.23f, 0.17f, 0.21f};
    const auto source = mlx::core::astype(
        mlx::core::array(
            source_values.begin(),
            mlx::core::Shape{1, routes, input}),
        mlx::core::float16);
    const mlx::core::array expert_ids(
        ids.begin(),
        mlx::core::Shape{1, routes});
    const mlx::core::array weights(
        route_weights.begin(),
        mlx::core::Shape{1, routes});

    const char* prior = std::getenv(
        "MFQ_METAL_MFE_DECODE_DOWN_REDUCE");
    const std::optional<std::string> saved = prior == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior);
    const char* prior_rows = std::getenv(
        "MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS");
    const std::optional<std::string> saved_rows = prior_rows == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior_rows);
    const char* prior_pack = std::getenv(
        "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    const std::optional<std::string> saved_pack = prior_pack == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior_pack);
    setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", "1", 1);
    const auto pair_reference = evaluated_floats(
        weight.routed_matmul(source, expert_ids));
    for (const char* rows : {"2", "4"}) {
        setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", rows, 1);
        const auto packed_rows = evaluated_floats(
            weight.routed_matmul(source, expert_ids));
        require(pair_reference.size() == packed_rows.size(),
                "MXFP4 decode row packing output shape mismatch");
        for (std::size_t index = 0; index < pair_reference.size(); ++index) {
            require_close(pair_reference[index], packed_rows[index], 1e-4f);
        }
    }
    setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", "1", 1);
    setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "0", 1);
    const auto reference = evaluated_floats(
        weight.routed_matmul_reduce(source, expert_ids, weights));
    setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "1", 1);
    for (const char* rows : {"1", "2", "4"}) {
        setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS", rows, 1);
        const auto fused = evaluated_floats(
            weight.routed_matmul_reduce(source, expert_ids, weights));
        require(
            reference.size() == static_cast<std::size_t>(output)
                && fused.size() == reference.size(),
            "MXFP4 decode down-reduce output shape mismatch");
        for (std::size_t index = 0; index < reference.size(); ++index) {
            require_close(reference[index], fused[index], 1e-4f);
        }
    }
    std::vector<std::int32_t> packed_ids;
    packed_ids.reserve(ids.size());
    for (const auto expert : ids) {
        packed_ids.push_back(((expert + 1) << 8) | expert);
    }
    const auto packed_fused = evaluated_floats(
        weight.routed_matmul_reduce_packed(
            source,
            mlx::core::array(
                packed_ids.begin(),
                mlx::core::Shape{1, routes}),
            weights));
    require(
        packed_fused.size() == reference.size(),
        "packed MXFP4 decode down-reduce output shape mismatch");
    for (std::size_t index = 0; index < reference.size(); ++index) {
        require_close(reference[index], packed_fused[index], 1e-4f);
    }

    for (const int tokens : {2, 6}) {
        std::vector<float> multi_source_values(
            static_cast<std::size_t>(tokens) * routes * input);
        for (std::size_t index = 0;
             index < multi_source_values.size();
             ++index) {
            multi_source_values[index] = static_cast<float>(
                static_cast<int>((index * 11 + 5) % 41) - 20) / 128.0f;
        }
        std::vector<std::int32_t> multi_ids(
            static_cast<std::size_t>(tokens) * routes);
        std::vector<float> multi_weights(
            static_cast<std::size_t>(tokens) * routes);
        for (int token = 0; token < tokens; ++token) {
            for (int route = 0; route < routes; ++route) {
                const auto index = static_cast<std::size_t>(
                    token * routes + route);
                multi_ids[index] = ids[
                    static_cast<std::size_t>((route + token) % routes)];
                multi_weights[index] = route_weights[
                    static_cast<std::size_t>(route)];
            }
        }
        const auto multi_source = mlx::core::astype(
            mlx::core::array(
                multi_source_values.begin(),
                mlx::core::Shape{tokens, routes, input}),
            mlx::core::float16);
        const auto multi_expert_ids = mlx::core::array(
            multi_ids.begin(),
            mlx::core::Shape{tokens, routes});
        const auto multi_route_weights = mlx::core::array(
            multi_weights.begin(),
            mlx::core::Shape{tokens, routes});
        setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "0", 1);
        const auto multi_reference = evaluated_floats(
            weight.routed_matmul_reduce(
                multi_source,
                multi_expert_ids,
                multi_route_weights));
        setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "1", 1);
        const auto multi_fused = evaluated_floats(
            weight.routed_matmul_reduce(
                multi_source,
                multi_expert_ids,
                multi_route_weights));
        require(
            multi_reference.size()
                    == static_cast<std::size_t>(tokens * output)
                && multi_fused.size() == multi_reference.size(),
            "MXFP4 small-M down-reduce output shape mismatch");
        for (std::size_t index = 0;
             index < multi_reference.size();
             ++index) {
            require_close(
                multi_reference[index],
                multi_fused[index],
                1e-4f);
        }
    }

    if (saved.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_DOWN_REDUCE",
            saved->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE");
    }
    if (saved_rows.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS",
            saved_rows->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS");
    }
    if (saved_pack.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD",
            saved_pack->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    }
}

void test_mxfp4_decode_swiglu_row_packing() {
    constexpr int experts = 8;
    constexpr int matrix_output = 16;
    constexpr int input = 64;
    constexpr int routes = 6;
    std::vector<std::int32_t> slots(experts);
    for (int expert = 0; expert < experts; ++expert) {
        slots[static_cast<std::size_t>(expert)] = experts - expert - 1;
    }
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(experts) * matrix_output * input / 2,
        0x32);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(experts) * matrix_output * input / 32,
        127);
    const auto weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        matrix_output,
        input,
        slots,
        mlx::core::array(
            packed.begin(),
            mlx::core::Shape{
                experts,
                matrix_output * input / 2,
            }),
        mlx::core::array(
            scales.begin(),
            mlx::core::Shape{
                experts,
                matrix_output * input / 32,
            }));
    std::vector<float> source_values(input);
    for (int column = 0; column < input; ++column) {
        source_values[static_cast<std::size_t>(column)] =
            static_cast<float>((column * 7 + 3) % 23 - 11) / 128.0f;
    }
    const auto source = mlx::core::astype(
        mlx::core::array(
            source_values.begin(),
            mlx::core::Shape{1, input}),
        mlx::core::float16);
    const std::vector<std::int32_t> id_values{7, 1, 6, 0, 3, 2};
    const mlx::core::array expert_ids(
        id_values.begin(),
        mlx::core::Shape{1, routes});
    const char* prior = std::getenv(
        "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    const std::optional<std::string> saved = prior == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior);
    setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", "1", 1);
    const auto reference = evaluated_floats(
        weight.routed_swiglu(source, expert_ids, 0.0f));
    for (const char* rows : {"2", "4"}) {
        setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", rows, 1);
        const auto candidate = evaluated_floats(
            weight.routed_swiglu(source, expert_ids, 0.0f));
        require(reference.size() == candidate.size(),
                "MXFP4 decode SwiGLU row packing shape mismatch");
        for (std::size_t index = 0; index < reference.size(); ++index) {
            require_close(reference[index], candidate[index], 1e-4f);
        }
    }
    if (saved.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD",
            saved->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    }
}

void test_large_mxfp4_arena_avoids_single_group_builder() {
    constexpr int experts = 1025;
    constexpr int output = 1;
    constexpr int input = 32;
    std::vector<std::int32_t> slots(experts);
    std::iota(slots.begin(), slots.end(), 0);
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(experts) * output * input / 2,
        0x22);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(experts) * output * input / 32,
        127);
    const auto weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        slots,
        mlx::core::array(
            packed.begin(),
            mlx::core::Shape{experts, output * input / 2}),
        mlx::core::array(
            scales.begin(),
            mlx::core::Shape{experts, output * input / 32}));
    require(
        !weight.supports_grouped_mmq(),
        "large SSD arena must not use a single-threadgroup block builder");
    std::vector<float> input_values(32 * input, 0.0f);
    std::vector<std::int32_t> expert_ids(32, 0);
    const auto output_values = evaluated_floats(weight.routed_matmul(
        mlx::core::astype(
            mlx::core::array(
                input_values.begin(),
                mlx::core::Shape{32, input}),
            mlx::core::float16),
        mlx::core::array(
            expert_ids.begin(),
            mlx::core::Shape{32, 1})));
    require(
        output_values.size() == 32,
        "large SSD arena mapped fallback output shape mismatch");

    std::vector<float> small_input_values(4 * input, 0.0f);
    std::vector<std::int32_t> small_expert_ids(4, 0);
    std::vector<std::int32_t> small_order{0, 1, 2, 3};
    const auto direct_values = evaluated_floats(
        weight.routed_matmul_sorted(
            mlx::core::array(
                small_input_values.begin(),
                mlx::core::Shape{4, input}),
            mlx::core::array(
                small_expert_ids.begin(),
                mlx::core::Shape{4, 1}),
            mlx::core::array(
                small_order.begin(),
                mlx::core::Shape{4}),
            false,
            false,
            0.0f,
            nullptr,
            true));
    require(
        direct_values.size() == 4,
        "large SSD arena direct MXFP4 output shape mismatch");
}

void test_vq_cohorts_and_ffn() {
    constexpr int tokens = 3;
    constexpr int routes = 3;
    constexpr int width = 24;
    auto fixture = make_vq_moe_fixture({
        make_plain_nvq(width, width),
        make_plain_nvq(width, width, true),
        make_jsc_nvq(width, width),
        make_jsc_nvq(
            width,
            width,
            "NVQ3J",
            2,
            4,
            8),
        make_jsc_nvq(
            width,
            width,
            "NVQ3J-512",
            3,
            4,
            9),
        make_jsc_nvq(
            width,
            width,
            "NVQ2J-XL",
            5,
            8,
            12,
            true),
        make_nvq1_s(width, width),
        make_npq(width, width),
        make_rotated_nepq1_s(width, width),
    });
    const char* previous_exec = std::getenv(
        "MFQ_METAL_MFE_JSC_EXEC");
    const bool had_previous_exec =
        previous_exec != nullptr;
    const std::string previous_exec_value =
        had_previous_exec ? previous_exec : "";
    setenv("MFQ_METAL_MFE_JSC_EXEC", "1", 1);
    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(
            fixture.blob);
    setenv("MFQ_METAL_MFE_JSC_EXEC", "0", 1);
    const auto legacy_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            fixture.blob);
    if (had_previous_exec) {
        setenv(
            "MFQ_METAL_MFE_JSC_EXEC",
            previous_exec_value.c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_JSC_EXEC");
    }
    require(
        weight.experts() == fixture.experts
            && weight.out_per_expert() == width
            && weight.neuron_len() == width,
        "VQ MFE metadata mismatch");

    const std::vector<std::int32_t> ids{
        0, 8, 2,
        3, 1, 7,
        5, 6, 4,
    };
    std::vector<float> input(tokens * width);
    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 5 + 3) % 19)
                - 9)
            / 256.0f;
    }
    const auto actual = evaluated_floats(
        weight.routed_matmul(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{
                    tokens,
                    width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    const auto legacy_actual = evaluated_floats(
        legacy_weight.routed_matmul(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{
                    tokens,
                    width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    require(
        legacy_actual.size() == actual.size(),
        "JSC execution layout result size mismatch");
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        require_close(
            actual[index],
            legacy_actual[index],
            1e-6f);
    }
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0;
             route < routes;
             ++route) {
            const int expert =
                ids[token * routes + route];
            for (int output = 0;
                 output < width;
                 ++output) {
                require_close(
                    actual[
                        (
                            token * routes + route
                        ) * width + output],
                    routed_vq_dot(
                        source,
                        fixture,
                        expert,
                        output),
                    2e-3f);
            }
        }
    }

    // NEPQ payloads are cross-expert tensors.  One cohort owns two global
    // experts here, with deliberately reversed global IDs, so descriptor
    // local-expert row mapping is tested independently of pool ordering.
    auto cohort = make_rotated_nepq1_s(
        width,
        width,
        0x8899aabbccddeeffull,
        0,
        2);
    std::vector<std::uint8_t> multi_blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(multi_blob, 2);
    append<std::uint32_t>(multi_blob, width);
    append<std::uint32_t>(multi_blob, width);
    append<std::uint32_t>(multi_blob, 1);
    append<std::uint32_t>(multi_blob, 2);
    append<std::uint32_t>(
        multi_blob,
        cohort.dtype.size());
    append<std::uint64_t>(
        multi_blob,
        cohort.blob.size());
    append<std::uint64_t>(
        multi_blob,
        cohort.runtime.size());
    append<std::int32_t>(multi_blob, 1);
    append<std::int32_t>(multi_blob, 0);
    multi_blob.insert(
        multi_blob.end(),
        cohort.dtype.begin(),
        cohort.dtype.end());
    multi_blob.insert(
        multi_blob.end(),
        cohort.runtime.begin(),
        cohort.runtime.end());
    multi_blob.insert(
        multi_blob.end(),
        cohort.blob.begin(),
        cohort.blob.end());

    const std::vector<std::int32_t> multi_ids{
        0, 1,
    };
    const auto multi_actual = evaluated_floats(
        mfq::metal::MlxMoeWeight::from_blob(
            multi_blob).routed_matmul(
                mlx::core::array(
                    input.begin(),
                    mlx::core::Shape{1, width}),
                mlx::core::array(
                    multi_ids.begin(),
                    mlx::core::Shape{1, 2})));
    std::vector<float> multi_source(
        input.begin(),
        input.begin() + width);
    const auto rotated_source =
        rotate_vq_input(multi_source, cohort);
    for (int route = 0; route < 2; ++route) {
        const int global_expert = multi_ids[route];
        const int local_expert =
            global_expert == 0 ? 1 : 0;
        for (int output = 0;
             output < width;
             ++output) {
            float expected_value = 0.0f;
            const auto weight_offset = (
                static_cast<std::size_t>(
                    local_expert)
                    * width
                + output
            ) * width;
            for (int column = 0;
                 column < width;
                 ++column) {
                expected_value +=
                    rotated_source[column]
                    * cohort.dense[
                        weight_offset + column];
            }
            require_close(
                multi_actual[
                    route * width + output],
                expected_value,
                2e-3f);
        }
    }

    // Using the same projection fixture three times intentionally exercises
    // projection-buffer offsets and HSG1 variant de-duplication in gate/up.
    const auto ffn =
        mfq::metal::MlxRoutedSwiGluFfn::
            from_blobs(
                fixture.blob,
                fixture.blob,
                fixture.blob);
    const std::vector<float> route_weights{
        0.50f, 0.30f, 0.20f,
        0.15f, 0.55f, 0.30f,
        0.25f, 0.35f, 0.40f,
    };
    const auto ffn_actual = evaluated_floats(
        ffn.forward(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{
                    tokens,
                    width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                }),
            mlx::core::array(
                route_weights.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));

    std::vector<float> expected(
        tokens * width,
        0.0f);
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0;
             route < routes;
             ++route) {
            const int expert =
                ids[token * routes + route];
            std::vector<float> hidden(width);
            for (int column = 0;
                 column < width;
                 ++column) {
                const float projected =
                    routed_vq_dot(
                        source,
                        fixture,
                        expert,
                        column);
                hidden[column] =
                    projected
                    / (
                        1.0f
                        + std::exp(-projected)
                    )
                    * projected;
            }
            for (int output = 0;
                 output < width;
                 ++output) {
                expected[token * width + output] +=
                    route_weights[
                        token * routes + route]
                    * routed_vq_dot(
                        hidden,
                        fixture,
                        expert,
                        output);
            }
        }
    }
    for (
        std::size_t index = 0;
        index < expected.size();
        ++index
    ) {
        require_close(
            ffn_actual[index],
            expected[index],
            8e-3f);
    }
}

void test_nepq_a_routed_and_fused_swiglu() {
    constexpr int tokens = 2;
    constexpr int routes = 2;
    constexpr int width = 24;
    auto fixture = make_vq_moe_fixture({
        add_nepq_a_residual(
            make_rotated_nepq0_s(width, width),
            false),
        add_nepq_a_residual(
            make_rotated_nepq1_s(width, width),
            true),
    });
    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NEPQ-A must support residual-aware grouped prefill");
    std::vector<float> input(tokens * width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 7 + 3) % 17) - 8)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 1, 0};
    const mlx::core::array input_array(
        input.begin(),
        mlx::core::Shape{tokens, width});
    const mlx::core::array id_array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));
    const auto projected_pairs = evaluated_floats(
        mfq::metal::MlxMoeWeight::concatenate_projections(
            {weight, weight}).routed_matmul(
                input_array,
                id_array));
    auto fused_fixture = make_vq_moe_fixture({
        add_nepq_a_residual(
            make_rotated_nepq0_s(2 * width, width),
            false),
        add_nepq_a_residual(
            make_rotated_nepq1_s(2 * width, width),
            true),
    });
    const auto fused_weight =
        mfq::metal::MlxMoeWeight::from_blob(fused_fixture.blob);
    const auto fused = evaluated_floats(
        fused_weight.routed_swiglu(input_array, id_array));
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int output = 0; output < width; ++output) {
                const float projected_value = routed_vq_dot(
                    source,
                    fixture,
                    expert,
                    output);
                const auto offset = static_cast<std::size_t>(
                    (token * routes + route) * width + output);
                require_close(actual[offset], projected_value, 3e-3f);
                require_close(
                    projected_pairs[
                        static_cast<std::size_t>(
                            token * routes + route)
                            * (2 * width)
                        + output],
                    projected_value,
                    3e-3f);
                require_close(
                    projected_pairs[
                        static_cast<std::size_t>(
                            token * routes + route)
                            * (2 * width)
                        + width + output],
                    projected_value,
                    3e-3f);
                const float gate = routed_vq_dot(
                    source,
                    fused_fixture,
                    expert,
                    output);
                const float up = routed_vq_dot(
                    source,
                    fused_fixture,
                    expert,
                    width + output);
                const float expected_fused = gate
                    / (1.0f + std::exp(-gate))
                    * up;
                require_close(fused[offset], expected_fused, 8e-3f);
            }
        }
    }

    constexpr int prefill_tokens = 49;
    std::vector<float> prefill_input(
        prefill_tokens * width);
    for (std::size_t index = 0;
         index < prefill_input.size();
         ++index) {
        prefill_input[index] = static_cast<float>(
            static_cast<int>((index * 11 + 1) % 23) - 11)
            / 128.0f;
    }
    std::vector<std::int32_t> prefill_ids(
        prefill_tokens * routes);
    for (int token = 0; token < prefill_tokens; ++token) {
        prefill_ids[token * routes] = token & 1;
        prefill_ids[token * routes + 1] = 1 - (token & 1);
    }
    const auto prefill_actual = evaluated_floats(
        weight.routed_matmul(
            mlx::core::astype(
                mlx::core::array(
                    prefill_input.begin(),
                    mlx::core::Shape{
                        prefill_tokens,
                        width,
                    }),
                mlx::core::float16),
            mlx::core::array(
                prefill_ids.begin(),
                mlx::core::Shape{
                    prefill_tokens,
                    routes,
                })));
    for (int token = 0; token < prefill_tokens; ++token) {
        const std::vector<float> token_input(
            prefill_input.begin() + token * width,
            prefill_input.begin() + (token + 1) * width);
        for (int route = 0; route < routes; ++route) {
            const int expert = prefill_ids[token * routes + route];
            for (int output = 0; output < width; ++output) {
                const auto offset = static_cast<std::size_t>(
                    (token * routes + route) * width + output);
                require_close(
                    prefill_actual[offset],
                    routed_vq_dot(
                        token_input,
                        fixture,
                        expert,
                        output),
                    4e-3f);
            }
        }
    }
}

mfq::metal::MlxMoeWeight make_v4f_nepq_benchmark_weight(
    bool residual) {
    constexpr int experts = 6;
    constexpr int output = 2048;
    constexpr int input = 4096;
    std::vector<VqFixture> weights;
    weights.reserve(experts);
    for (int expert = 0; expert < experts; ++expert) {
        auto weight = make_rotated_nepq0_s(
            output,
            input,
            0x1020304050607080ull);
        if (residual) {
            weight = add_nepq_a_residual(
                std::move(weight),
                false);
        }
        weights.push_back(std::move(weight));
    }
    auto fixture = make_vq_moe_fixture(
        std::move(weights));
    return mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
}

double benchmark_v4f_nepq_weight(
    const mfq::metal::MlxMoeWeight& weight,
    const mlx::core::array& input,
    const mlx::core::array& expert_ids,
    int repetitions) {
    const auto started =
        std::chrono::steady_clock::now();
    for (int repetition = 0;
         repetition < repetitions;
         ++repetition) {
        auto output = weight.routed_matmul(
            input,
            expert_ids);
        output.eval();
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()
            - started).count();
    return elapsed / repetitions;
}

void benchmark_v4f_nepq_a() {
    constexpr int input_width = 4096;
    constexpr int routes = 6;
    constexpr int repetitions = 40;
    std::vector<float> source(input_width);
    for (int column = 0;
         column < input_width;
         ++column) {
        source[static_cast<std::size_t>(column)] =
            static_cast<float>((column * 7) % 29 - 14)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids{
        0, 1, 2, 3, 4, 5,
    };
    const auto input = mlx::core::astype(
        mlx::core::array(
            source.begin(),
            mlx::core::Shape{1, input_width}),
        mlx::core::float16);
    const mlx::core::array expert_ids(
        ids.begin(),
        mlx::core::Shape{1, routes});

    const auto base =
        make_v4f_nepq_benchmark_weight(false);
    const auto residual =
        make_v4f_nepq_benchmark_weight(true);
    for (int warmup = 0; warmup < 8; ++warmup) {
        auto base_output = base.routed_matmul(input, expert_ids);
        auto residual_output = residual.routed_matmul(input, expert_ids);
        base_output.eval();
        residual_output.eval();
    }
    std::vector<double> base_samples;
    std::vector<double> residual_samples;
    for (int round = 0; round < 9; ++round) {
        auto measure_base = [&] {
            base_samples.push_back(
                benchmark_v4f_nepq_weight(
                    base,
                    input,
                    expert_ids,
                    repetitions));
        };
        auto measure_residual = [&] {
            residual_samples.push_back(
                benchmark_v4f_nepq_weight(
                    residual,
                    input,
                    expert_ids,
                    repetitions));
        };
        if ((round & 1) == 0) {
            measure_base();
            measure_residual();
        } else {
            measure_residual();
            measure_base();
        }
    }
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const double base_ms = median(base_samples);
    const double residual_ms = median(residual_samples);
    constexpr int prefill_tokens = 128;
    constexpr int prefill_repetitions = 8;
    std::vector<float> prefill_source(
        prefill_tokens * input_width);
    for (std::size_t index = 0;
         index < prefill_source.size();
         ++index) {
        prefill_source[index] = static_cast<float>(
            static_cast<int>((index * 5 + 3) % 31) - 15)
            / 128.0f;
    }
    std::vector<std::int32_t> prefill_id_values(
        prefill_tokens * routes);
    for (int token = 0; token < prefill_tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            prefill_id_values[token * routes + route] =
                (token + route) % routes;
        }
    }
    const auto prefill_input = mlx::core::astype(
        mlx::core::array(
            prefill_source.begin(),
            mlx::core::Shape{
                prefill_tokens,
                input_width,
            }),
        mlx::core::float16);
    const mlx::core::array prefill_ids(
        prefill_id_values.begin(),
        mlx::core::Shape{
            prefill_tokens,
            routes,
        });
    for (int warmup = 0; warmup < 4; ++warmup) {
        auto base_output = base.routed_matmul(
            prefill_input,
            prefill_ids);
        auto residual_output = residual.routed_matmul(
            prefill_input,
            prefill_ids);
        base_output.eval();
        residual_output.eval();
    }
    base_samples.clear();
    residual_samples.clear();
    for (int round = 0; round < 7; ++round) {
        auto measure_base = [&] {
            base_samples.push_back(
                benchmark_v4f_nepq_weight(
                    base,
                    prefill_input,
                    prefill_ids,
                    prefill_repetitions));
        };
        auto measure_residual = [&] {
            residual_samples.push_back(
                benchmark_v4f_nepq_weight(
                    residual,
                    prefill_input,
                    prefill_ids,
                    prefill_repetitions));
        };
        if ((round & 1) == 0) {
            measure_base();
            measure_residual();
        } else {
            measure_residual();
            measure_base();
        }
    }
    const double base_prefill_ms = median(base_samples);
    const double residual_prefill_ms = median(residual_samples);
    const auto bandwidth = [](const auto& weight, double ms) {
        return static_cast<double>(weight.packed_nbytes())
            / (ms * 1.0e6);
    };
    std::cout
        << "V4F routed MoE decode benchmark"
        << " in=4096 out=2048 top_k=6 reps="
        << repetitions << "\n"
        << "NEPQ0-S " << base_ms << " ms/dispatch "
        << bandwidth(base, base_ms) << " effective-GB/s\n"
        << "NEPQ0-A " << residual_ms << " ms/dispatch "
        << bandwidth(residual, residual_ms)
        << " effective-GB/s overhead="
        << (residual_ms / base_ms - 1.0) * 100.0
        << "%\n"
        << "grouped-M128 NEPQ0-S " << base_prefill_ms
        << " ms/dispatch\n"
        << "grouped-M128 NEPQ0-A " << residual_prefill_ms
        << " ms/dispatch overhead="
        << (residual_prefill_ms / base_prefill_ms - 1.0) * 100.0
        << "%\n";
}

void benchmark_shared_nint_mfe() {
    constexpr int width = 4096;
    constexpr int repetitions = 30;
    const auto fixture = make_moe_fixture(
        {"NINTv2"}, width, width, 43, 32);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NINTv2 benchmark must retain grouped-prefill support");
    for (int rows = 2; rows <= 6; ++rows) {
        std::vector<float> values(
            static_cast<std::size_t>(rows) * width);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(
                static_cast<int>((index * 13 + 5) % 31) - 15) / 128.0f;
        }
        const auto input = mlx::core::astype(
            mlx::core::array(
                values.begin(),
                mlx::core::Shape{rows, width}),
            mlx::core::float16);
        const std::vector<std::int32_t> ids(
            static_cast<std::size_t>(rows), 0);
        const auto expert_ids = mlx::core::array(
            ids.begin(), mlx::core::Shape{rows, 1});
        for (int warmup = 0; warmup < 4; ++warmup) {
            auto output = weight.routed_matmul(input, expert_ids);
            output.eval();
        }
        const double mean_ms = benchmark_v4f_nepq_weight(
            weight,
            input,
            expert_ids,
            repetitions);
        std::cout << "NINTv2 shared-kernel M=" << rows
                  << " " << mean_ms << " ms/dispatch\n";
    }
}

void test_grouped_mmq_prefill() {
    constexpr int tokens = 49;
    constexpr int routes = 2;
    constexpr int width = 24;
    auto fixture = make_vq_moe_fixture({
        make_jsc_nvq(
            width,
            width,
            "NVQ3J",
            2,
            4,
            8),
        make_jsc_nvq(width, width),
    });
    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(
            fixture.blob);
    std::vector<float> input(tokens * width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 7 + 5) % 23) - 11)
            / 128.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int row = 0; row < tokens * routes; ++row) {
        ids[row] = row < 31 ? 0 : 1;
    }
    auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, width}),
        mlx::core::float16);
    const auto actual = evaluated_floats(
        weight.routed_matmul(
            input_array,
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{tokens, routes})));
    auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    ids_array,
                    mlx::core::Shape{tokens * routes})),
            mlx::core::int32));
    auto block_plan = weight.build_grouped_mmq_plan(
        ids_array,
        route_order);
    require(block_plan.block_rows == 32, "unexpected block row size");
    require(
        block_plan.route_count == tokens * routes,
        "unexpected block plan route count");
    auto plain_sorted = weight.routed_matmul_sorted(
        input_array,
        ids_array,
        route_order,
        false,
        false,
        0.0f,
        &block_plan);
    const auto plain_actual = evaluated_floats(
        mlx::core::take(
            std::move(plain_sorted),
            mlx::core::argsort(route_order),
            0));
    auto fused_sorted = weight.routed_matmul_sorted(
        input_array,
        ids_array,
        route_order,
        false,
        true,
        0.0f,
        &block_plan);
    const auto fused_actual = evaluated_floats(
        mlx::core::reshape(
            mlx::core::take(
                std::move(fused_sorted),
                mlx::core::argsort(route_order),
                0),
            mlx::core::Shape{tokens, routes, width / 2}));
    const auto fused_automatic = evaluated_floats(
        weight.routed_swiglu(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0; route < routes; ++route) {
            int expert = ids[token * routes + route];
            for (int output = 0; output < width; ++output) {
                require_close(
                    plain_actual[
                        (token * routes + route) * width + output],
                    actual[(token * routes + route) * width + output],
                    4e-3f);
                require_close(
                    actual[(token * routes + route) * width + output],
                    routed_vq_dot(
                        source,
                        fixture,
                        expert,
                        output),
                    4e-3f);
            }
            for (int output = 0; output < width / 2; ++output) {
                const float gate = routed_vq_dot(
                    source,
                    fixture,
                    expert,
                    output);
                const float up = routed_vq_dot(
                    source,
                    fixture,
                    expert,
                    output + width / 2);
                require_close(
                    fused_actual[
                        (token * routes + route) * (width / 2)
                        + output],
                    gate / (1.0f + std::exp(-gate)) * up,
                    6e-3f);
                require_close(
                    fused_automatic[
                        (token * routes + route) * (width / 2)
                        + output],
                    fused_actual[
                        (token * routes + route) * (width / 2)
                        + output],
                    1e-6f);
            }
        }
    }
}

void test_grouped_vq_decoder_tail_prefill() {
    constexpr int output = 16;
    constexpr int input_width = 640;
    auto fixture = make_vq_moe_fixture({
        make_npq(output, input_width, true),
        make_npq(output, input_width, false),
        make_nvq1_s(output, input_width),
        make_nvq1_l(output, input_width),
        make_plain_nvq(output, input_width),
        make_plain_nvq(output, input_width, true),
        make_plain_nvq3(output, input_width),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ3J-512",
            3,
            4,
            9),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ2J-L",
            4,
            8,
            10),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ2J-XL",
            5,
            8,
            12),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ2J-XL",
            5,
            8,
            12,
            true),
    });
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed VQ decoder tail fixture must support grouped prefill");
    const auto exercise = [&](int tokens) {
        std::vector<float> input(tokens * input_width);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(
                static_cast<int>((index * 13 + 7) % 29) - 14)
                / 1024.0f;
        }
        std::vector<std::int32_t> ids(tokens);
        for (int token = 0; token < tokens; ++token) {
            ids[token] = token % fixture.experts;
        }
        const auto actual = evaluated_floats(
            weight.routed_matmul(
                mlx::core::astype(
                    mlx::core::array(
                        input.begin(),
                        mlx::core::Shape{tokens, input_width}),
                    mlx::core::float16),
                mlx::core::array(
                    ids.begin(), mlx::core::Shape{tokens, 1})));
        for (int token = 0; token < tokens; ++token) {
            const std::vector<float> source(
                input.begin() + token * input_width,
                input.begin() + (token + 1) * input_width);
            for (int row = 0; row < output; ++row) {
                require_close(
                    actual[token * output + row],
                    routed_vq_dot(source, fixture, ids[token], row),
                    6e-2f);
            }
        }
    };
    exercise(49);
    exercise(1025);
}

void test_grouped_nint_mmq_prefill() {
    // Cross the heterogeneous NAX threshold on supported Apple GPUs while
    // retaining the same reference coverage on compatibility-only devices.
    constexpr int tokens = 513;
    constexpr int routes = 2;
    constexpr int output = 48;
    constexpr int input_width = 96;
    const std::vector<std::string> profiles{
        "NINT1", "NINT2", "NINT3", "NINT4", "NINT5",
        "NINT6", "NINT7", "NINT8", "NINT8-0"};
    auto fixture = make_moe_fixture(
        profiles, output, input_width, 9, 24);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "metadata-driven NINT must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 11 + 3) % 29) - 14)
            / 128.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int row = 0; row < tokens * routes; ++row) {
        ids[row] = row % static_cast<int>(profiles.size());
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto plain = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    const auto fused = evaluated_floats(
        weight.routed_swiglu(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * fixture.dense[
                            (expert * output + row) * input_width + column];
                }
                require_close(
                    plain[(token * routes + route) * output + row],
                    expected,
                    2e-2f);
            }
            for (int row = 0; row < output / 2; ++row) {
                const float gate = plain[
                    (token * routes + route) * output + row];
                const float up = plain[
                    (token * routes + route) * output + output / 2 + row];
                require_close(
                    fused[(token * routes + route) * (output / 2) + row],
                    gate / (1.0f + std::exp(-gate)) * up,
                    2e-2f);
            }
        }
    }
}

void test_grouped_split_nint_swiglu_prefill() {
    constexpr int tokens = 49;
    constexpr int routes = 2;
    constexpr int output = 24;
    constexpr int input_width = 96;
    const std::vector<std::string> profiles{
        "NINTv2", "NINT2", "NINT5", "NINT8",
    };
    const auto gate = make_moe_fixture(
        profiles, output, input_width, 13, 24);
    const auto up = make_moe_fixture(
        profiles, output, input_width, 37, 24);
    const auto gate_weight = mfq::metal::MlxMoeWeight::from_blob(
        gate.blob);
    const auto up_weight = mfq::metal::MlxMoeWeight::from_blob(
        up.blob);
    const auto gate_up =
        mfq::metal::MlxMoeWeight::concatenate_projections(
            {gate_weight, up_weight});
    require(
        gate_up.projections() == 2
            && gate_up.supports_grouped_mmq(),
        "split NINT Gate/Up must retain one grouped MFE dispatch");

    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 31) - 15)
            / 256.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int row = 0; row < tokens * routes; ++row) {
        ids[row] = row % static_cast<int>(profiles.size());
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        gate_up.routed_swiglu(input_array, ids_array));
    const auto gate_values = evaluated_floats(
        gate_weight.routed_matmul(input_array, ids_array));
    const auto up_values = evaluated_floats(
        up_weight.routed_matmul(input_array, ids_array));

    auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    ids_array,
                    mlx::core::Shape{tokens * routes})),
            mlx::core::int32));
    const auto plan = gate_up.build_grouped_mmq_plan(
        ids_array, route_order);
    const auto sorted = gate_up.routed_matmul_sorted(
        input_array,
        ids_array,
        route_order,
        false,
        true,
        0.0f,
        &plan);
    const auto planned = evaluated_floats(
        mlx::core::reshape(
            mlx::core::take(
                sorted,
                mlx::core::argsort(route_order),
                0),
            mlx::core::Shape{tokens, routes, output}));
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float gate_value = gate_values[index];
        const float expected = gate_value
            / (1.0f + std::exp(-gate_value))
            * up_values[index];
        require_close(actual[index], expected, 2e-2f);
        require_close(planned[index], actual[index], 1e-6f);
    }
}

void test_shared_nint2_prefill() {
    constexpr int tokens = 1024;
    constexpr int output = 24;
    constexpr int input_width = 96;
    const auto fixture = make_moe_fixture(
        {"NINT2"}, output, input_width, 23);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NINT2-16 must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 17 + 7) % 31) - 15)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids(tokens, 0);
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, 1});
    const auto plain = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    const auto fused = evaluated_floats(
        weight.routed_swiglu(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < output; ++row) {
            float expected = 0.0f;
            for (int column = 0; column < input_width; ++column) {
                expected += input[token * input_width + column]
                    * fixture.dense[row * input_width + column];
            }
            require_close(
                plain[token * output + row], expected, 2e-2f);
        }
        for (int row = 0; row < output / 2; ++row) {
            const float gate = plain[token * output + row];
            const float up = plain[
                token * output + output / 2 + row];
            require_close(
                fused[token * (output / 2) + row],
                gate / (1.0f + std::exp(-gate)) * up,
                2e-2f);
        }
    }
}

void test_shared_nint5_group28_prefill() {
    constexpr int output = 16;
    constexpr int input_width = 640;
    const auto fixture = make_moe_fixture(
        {"NINT5"}, output, input_width, 37, 28);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NINT5-28 must support grouped prefill");
    const auto exercise = [&](int tokens) {
        std::vector<float> input(tokens * input_width);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(
                static_cast<int>((index * 19 + 11) % 37) - 18)
                / 256.0f;
        }
        const std::vector<std::int32_t> ids(tokens, 0);
        const auto actual = evaluated_floats(
            weight.routed_matmul(
                mlx::core::astype(
                    mlx::core::array(
                        input.begin(),
                        mlx::core::Shape{tokens, input_width}),
                    mlx::core::float16),
                mlx::core::array(
                    ids.begin(), mlx::core::Shape{tokens, 1})));
        for (int token = 0; token < tokens; ++token) {
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * fixture.dense[row * input_width + column];
                }
                require_close(
                    actual[token * output + row], expected, 8e-2f);
            }
        }
    };
    exercise(49);
    exercise(1025);
}

void test_shared_nint8_mixed_prefill() {
    constexpr int tokens = 49;
    constexpr int routes = 2;
    constexpr int output = 24;
    constexpr int input_width = 640;
    const auto fixture = make_moe_fixture(
        {"NINT8", "F16"}, output, input_width, 31, 48);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed NINT8-48/F16 MFE must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 13 + 5) % 31) - 15)
            / 256.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int index = 0; index < tokens * routes; ++index) {
        ids[index] = index % 2;
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * fixture.dense[
                            (expert * output + row) * input_width + column];
                }
                require_close(
                    actual[(token * routes + route) * output + row],
                    expected,
                    8e-2f);
            }
        }
    }
}

void test_shared_nint_mapped_and_packed_routes() {
    constexpr int tokens = 3;
    constexpr int routes = 2;
    constexpr int output = 16;
    constexpr int input_width = 96;
    const auto fixture = make_moe_fixture(
        {"NINT3", "NINT6"}, output, input_width, 43, 24);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mapped NINT routes must retain grouped-prefill support");

    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 31) - 15)
            / 128.0f;
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);

    const std::vector<std::int32_t> logical_ids{
        2, 1,
        1, 2,
        2, 1,
    };
    const std::vector<std::int32_t> page_table{-1, 1, 0};
    const std::vector<std::int32_t> mapped_experts{
        0, 1,
        1, 0,
        0, 1,
    };
    const auto mapped = evaluated_floats(
        weight.routed_matmul_mapped(
            input_array,
            mlx::core::array(
                logical_ids.begin(),
                mlx::core::Shape{tokens, routes}),
            mlx::core::array(
                page_table.begin(),
                mlx::core::Shape{static_cast<int>(page_table.size())})));

    const std::vector<std::int32_t> packed_ids{
        263, 519,
        519, 263,
        263, 519,
    };
    const auto packed = evaluated_floats(
        weight.routed_matmul_packed(
            input_array,
            mlx::core::array(
                packed_ids.begin(),
                mlx::core::Shape{tokens, routes})));

    const auto validate = [&](const std::vector<float>& actual) {
        for (int token = 0; token < tokens; ++token) {
            for (int route = 0; route < routes; ++route) {
                const int expert = mapped_experts[token * routes + route];
                for (int row = 0; row < output; ++row) {
                    float expected = 0.0f;
                    for (int column = 0; column < input_width; ++column) {
                        expected += input[token * input_width + column]
                            * fixture.dense[
                                (expert * output + row) * input_width
                                + column];
                    }
                    require_close(
                        actual[(token * routes + route) * output + row],
                        expected,
                        6e-2f);
                }
            }
        }
    };
    validate(mapped);
    validate(packed);
}

void test_grouped_dense_quad_tail_prefill() {
    constexpr int tokens = 1025;
    constexpr int routes = 2;
    constexpr int output = 24;
    constexpr int input_width = 642;
    constexpr int experts = 2;
    const auto fixture = make_moe_fixture(
        {"F16", "BF16"}, output, input_width, 37, 48);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed F16/BF16 tail fixture must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 7 + 3) % 23) - 11)
            / 256.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int index = 0; index < tokens * routes; ++index) {
        ids[index] = index % experts;
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * fixture.dense[
                            (expert * output + row) * input_width + column];
                }
                require_close(
                    actual[(token * routes + route) * output + row],
                    expected,
                    1e-1f);
            }
        }
    }
}

void test_mixed_mfe_native_and_grouped_dispatch() {
    constexpr int experts = 4;
    constexpr int routes = 2;
    constexpr int output = 48;
    constexpr int input_width = 128;
    const auto nint = make_nint_tensor(4, output, input_width, 3, 32);
    const auto mxfp8 = make_mxfp8(output, input_width, 5);
    const auto bf16 = make_dense_tensor("BF16", output, input_width, 7);
    const auto f16 = make_dense_tensor("F16", output, input_width, 11);

    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input_width);
    append<std::uint32_t>(blob, experts);
    auto append_pool = [&blob](
        int expert,
        std::string_view dtype,
        const std::vector<std::uint8_t>& payload) {
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(dtype.size()));
        append<std::uint64_t>(blob, payload.size());
        append<std::uint64_t>(blob, 0);
        append<std::int32_t>(blob, expert);
        blob.insert(blob.end(), dtype.begin(), dtype.end());
        append_bytes(blob, payload);
    };
    // Reverse and mix pool order so descriptor-local and global expert rows
    // cannot accidentally coincide.
    append_pool(3, "F16", f16.blob);
    append_pool(1, "MXFP8", mxfp8.blob);
    append_pool(0, "NINT4", nint.blob);
    append_pool(2, "BF16", bf16.blob);

    std::vector<float> dense(
        static_cast<std::size_t>(experts) * output * input_width);
    const auto copy_expert = [&](int expert, const std::vector<float>& source) {
        std::copy(
            source.begin(),
            source.end(),
            dense.begin() + static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(expert) * output * input_width));
    };
    copy_expert(0, nint.dense);
    copy_expert(1, mxfp8.dense);
    copy_expert(2, bf16.dense);
    copy_expert(3, f16.dense);

    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.supports_grouped_mmq(),
        "metadata-driven NINT must support mixed grouped prefill");
    const auto exercise = [&](int tokens) {
        std::vector<float> input(
            static_cast<std::size_t>(tokens) * input_width);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(
                static_cast<int>((index * 13 + 5) % 31) - 15)
                / 1024.0f;
        }
        std::vector<std::int32_t> ids(tokens * routes);
        for (int row = 0; row < tokens * routes; ++row) {
            ids[row] = (row * 3 + row / 5) % experts;
        }
        const auto input_array = mlx::core::astype(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{tokens, input_width}),
            mlx::core::float16);
        const auto ids_array = mlx::core::array(
            ids.begin(),
            mlx::core::Shape{tokens, routes});
        const auto plain = evaluated_floats(
            weight.routed_matmul(input_array, ids_array));
        const auto fused = evaluated_floats(
            weight.routed_swiglu(input_array, ids_array));
        for (int token = 0; token < tokens; ++token) {
            for (int route = 0; route < routes; ++route) {
                const int expert = ids[token * routes + route];
                for (int row = 0; row < output; ++row) {
                    float expected = 0.0f;
                    for (int column = 0; column < input_width; ++column) {
                        expected += input[token * input_width + column]
                            * dense[
                                (expert * output + row) * input_width
                                + column];
                    }
                    require_close(
                        plain[(token * routes + route) * output + row],
                        expected,
                        8e-2f);
                }
                for (int row = 0; row < output / 2; ++row) {
                    const float gate = plain[
                        (token * routes + route) * output + row];
                    const float up = plain[
                        (token * routes + route) * output + output / 2 + row];
                    require_close(
                        fused[(token * routes + route) * (output / 2) + row],
                        gate / (1.0f + std::exp(-gate)) * up,
                        8e-2f);
                }
            }
        }
    };
    exercise(3);
    exercise(49);
}

void test_multiple_nint_pools_share_cpp_dispatch() {
    constexpr int experts = 4;
    constexpr int tokens = 3;
    constexpr int routes = 2;
    constexpr int output = 9;
    constexpr int input = 48;
    std::vector<PoolFixture> pools;
    pools.push_back({
        {2, 0},
        "NINT",
        make_nint_tensor(2, 2 * output, input, 17, 24),
        {},
    });
    pools.push_back({
        {3, 1},
        "NINT",
        make_nint_tensor(6, 2 * output, input, 29, 24),
        {},
    });
    auto blob = make_raw_nim2(experts, output, input, pools);
    std::copy_n("MFE1", 4, blob.begin());

    const auto weight = mfq::metal::MlxMfeWeight::from_blob(blob);
    std::vector<float> source(static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 7 + 3) % 29) - 14) / 256.0f;
    }
    const std::vector<std::int32_t> ids{0, 3, 2, 1, 1, 2};
    const auto input_array = mlx::core::astype(
        mlx::core::array(source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));

    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const PoolFixture* owner = nullptr;
            std::size_t local = 0;
            for (const auto& pool : pools) {
                const auto found = std::find(
                    pool.expert_ids.begin(), pool.expert_ids.end(), expert);
                if (found != pool.expert_ids.end()) {
                    owner = &pool;
                    local = static_cast<std::size_t>(
                        found - pool.expert_ids.begin());
                    break;
                }
            }
            require(owner != nullptr, "test expert has no NINT pool");
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                const auto weight_row =
                    (local * output + static_cast<std::size_t>(row)) * input;
                for (int column = 0; column < input; ++column) {
                    expected += source[token * input + column]
                        * owner->tensor.dense[weight_row + column];
                }
                require_close(
                    actual[(token * routes + route) * output + row],
                    expected,
                    3e-3f);
            }
        }
    }
}

void test_grouped_mxfp4_vq_mmq_prefill() {
    constexpr int experts = 2;
    constexpr int tokens = 37;
    constexpr int routes = 2;
    constexpr int output = 17;
    constexpr int input = 96;
    const auto mx_weight = make_mxfp4(output, input);
    const auto vq_weight = make_jsc_nvq(output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, experts);
    auto append_pool = [&blob](
        int expert,
        std::string_view dtype,
        const std::vector<std::uint8_t>& payload,
        const std::vector<std::uint8_t>& runtime) {
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(dtype.size()));
        append<std::uint64_t>(blob, payload.size());
        append<std::uint64_t>(blob, runtime.size());
        append<std::int32_t>(blob, expert);
        blob.insert(blob.end(), dtype.begin(), dtype.end());
        append_bytes(blob, runtime);
        append_bytes(blob, payload);
    };
    append_pool(0, "MXFP4", mx_weight.blob, {});
    append_pool(
        1,
        vq_weight.dtype,
        vq_weight.blob,
        vq_weight.runtime);

    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed MXFP4/VQ MFE must support grouped matrix prefill");
    std::vector<float> source(tokens * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 11 + 3) % 31) - 15) / 128.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        ids[index] = static_cast<std::int32_t>((index / 5) & 1u);
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(),
            mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto direct = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    const auto order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    ids_array,
                    mlx::core::Shape{tokens * routes})),
            mlx::core::int32));
    const auto plan = weight.build_grouped_mmq_plan(
        ids_array,
        order);
    const auto matrix = evaluated_floats(
        mlx::core::take(
            weight.routed_matmul_sorted(
                input_array,
                ids_array,
                order,
                false,
                false,
                0.0f,
                &plan),
            mlx::core::argsort(order),
            0));
    require(
        matrix.size() == direct.size(),
        "mixed MXFP4/VQ grouped prefill output size mismatch");
    for (std::size_t index = 0; index < direct.size(); ++index) {
        require_close(matrix[index], direct[index], 1e-2f);
    }
}

void expect_invalid(
    const std::vector<std::uint8_t>& blob,
    const std::string& label) {
    bool rejected = false;
    try {
        (void)mfq::metal::MlxMoeWeight::
            from_blob(blob);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(
        rejected,
        "malformed MFE was accepted: " + label);
}

void test_concatenate_resident_mfe_experts() {
    constexpr int experts = 3;
    constexpr int tokens = 17;
    constexpr int routes = 2;
    constexpr int output = 11;
    constexpr int input = 96;
    const auto nint = make_nint_tensor(5, output, input, 17, 24);
    const auto mx = make_mxfp4(output, input);
    const auto vq = make_jsc_nvq(output, input);
    const TensorFixture mx_tensor{
        mx.blob, mx.dense, output, input};
    const TensorFixture vq_tensor{
        vq.blob, vq.dense, output, input};

    const std::vector<PoolFixture> full_pools{
        {{0}, "NINT5", nint, {}},
        {{1}, "MXFP4", mx_tensor, {}},
        {{2}, vq.dtype, vq_tensor, vq.runtime},
    };
    const auto full = mfq::metal::MlxMoeWeight::from_blob(
        make_raw_nim2(experts, output, input, full_pools));
    const auto combined =
        mfq::metal::MlxMoeWeight::concatenate_experts({
            mfq::metal::MlxMoeWeight::from_blob(
                make_raw_nim2(
                    1, output, input,
                    {{{0}, "NINT5", nint, {}}})),
            mfq::metal::MlxMoeWeight::from_blob(
                make_raw_nim2(
                    1, output, input,
                    {{{0}, "MXFP4", mx_tensor, {}}})),
            mfq::metal::MlxMoeWeight::from_blob(
                make_raw_nim2(
                    1, output, input,
                    {{{0}, vq.dtype, vq_tensor, vq.runtime}})),
        });
    require(
        combined.experts() == experts
            && combined.projections() == 1
            && combined.supports_grouped_mmq(),
        "resident MFE expert concatenation metadata mismatch");

    std::vector<float> source(
        static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 19 + 7) % 37) - 18) / 256.0f;
    }
    std::vector<std::int32_t> ids(
        static_cast<std::size_t>(tokens) * routes);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        ids[index] = static_cast<std::int32_t>((index * 5 + 1) % experts);
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto id_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto expected = evaluated_floats(
        full.routed_matmul(input_array, id_array));
    const auto actual = evaluated_floats(
        combined.routed_matmul(input_array, id_array));
    require(
        actual.size() == expected.size(),
        "resident MFE expert concatenation output shape mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-6f) {
            const auto route = index / output;
            throw std::runtime_error(
                "resident MFE expert concatenation mismatch: expert="
                + std::to_string(ids[route])
                + " actual=" + std::to_string(actual[index])
                + " expected=" + std::to_string(expected[index]));
        }
    }
}

void test_streamed_mixed_mfe_residency() {
    constexpr int experts = 8;
    constexpr int output = 13;
    constexpr int input = 96;
    constexpr int tokens = 5;
    constexpr int routes = 3;
    const auto nint = make_nint_v2_tensor(
        2 * output, input, 43, 24);
    const auto legacy_nint = make_nint_tensor(
        5, 2 * output, input, 47, 24);
    const auto mx = make_mxfp4(2 * output, input);
    const auto vq = make_jsc_nvq(2 * output, input);
    const TensorFixture mx_tensor{
        mx.blob, mx.dense, 2 * output, input};
    const TensorFixture vq_tensor{
        vq.blob, vq.dense, 2 * output, input};
    const std::vector<PoolFixture> pools{
        {{4, 1}, "NINTv2", nint, {}},
        {{6, 2}, "NINT5", legacy_nint, {}},
        {{3, 0}, "MXFP4", mx_tensor, {}},
        {{7, 5}, vq.dtype, vq_tensor, vq.runtime},
    };
    const auto blob = make_raw_nim2(
        experts, output, input, pools);
    const auto alternate_blob = make_raw_nim2(
        4,
        output,
        input,
        {
            {
                {3, 1, 0, 2},
                "NINTv2",
                make_nint_v2_tensor(4 * output, input, 59, 24),
                {},
            },
        });
    const TemporaryMfq file({
        {"mixed", "MFE", blob},
        {"alternate", "MFE", alternate_blob},
    });
    const mfq::metal::MfqContainer model(file.path());
    mfq::metal::MlxMfeOffloadCache residency(
        model, 1u << 24);
    const auto info = residency.projection_info("mixed");
    require(
        residency.can_stream("mixed"),
        "mixed NINTv1/NINTv2/NVQ/MX MFE record was not streamable");
    require(
        info.experts == experts
            && info.out_per_expert == output
            && info.neuron_len == input
            && info.available_experts.size() == experts,
        "mixed streamed MFE projection metadata mismatch");
    const auto alternate_info = residency.projection_info("alternate");
    require(
        alternate_info.experts == 4
            && residency.grouped_mfe("alternate", {3, 0}).experts() == 2,
        "dynamic streamed MFE expert geometry mismatch");
    residency.discard_record("alternate");

    const auto full = mfq::metal::MlxMfeWeight::from_blob(blob);
    const auto exercise = [&](const std::vector<std::int32_t>& active) {
        const auto paged = residency.grouped_mfe("mixed", active);
        require(
            paged.experts() == static_cast<int>(active.size())
                && paged.out_per_expert() == output
                && paged.neuron_len() == input
                && paged.supports_grouped_mmq(),
            "mixed streamed MFE active weight metadata mismatch");
        std::vector<float> source(
            static_cast<std::size_t>(tokens) * input);
        for (std::size_t index = 0; index < source.size(); ++index) {
            source[index] = static_cast<float>(
                static_cast<int>((index * 23 + 11) % 41) - 20) / 256.0f;
        }
        std::vector<std::int32_t> global_ids(
            static_cast<std::size_t>(tokens) * routes);
        std::vector<std::int32_t> local_ids(global_ids.size());
        for (std::size_t index = 0; index < global_ids.size(); ++index) {
            const auto local = static_cast<std::int32_t>(index % active.size());
            local_ids[index] = local;
            global_ids[index] = active[static_cast<std::size_t>(local)];
        }
        const auto input_array = mlx::core::astype(
            mlx::core::array(
                source.begin(), mlx::core::Shape{tokens, input}),
            mlx::core::float16);
        const auto expected = evaluated_floats(full.routed_matmul(
            input_array,
            mlx::core::array(
                global_ids.begin(), mlx::core::Shape{tokens, routes})));
        const auto actual = evaluated_floats(paged.routed_matmul(
            input_array,
            mlx::core::array(
                local_ids.begin(), mlx::core::Shape{tokens, routes})));
        require(
            actual.size() == expected.size(),
            "mixed streamed MFE output shape mismatch");
        for (std::size_t index = 0; index < actual.size(); ++index) {
            require_close(actual[index], expected[index], 1e-6f);
        }
    };
    exercise({1, 2, 3, 5});
    exercise({4, 6, 0, 7});
    require(
        residency.cached_expert_count() == experts
            && residency.resident_packed_bytes() > 0,
        "mixed streamed MFE cache accounting mismatch");
    residency.discard_record("mixed");
    require(
        residency.cached_expert_count() == 0
            && residency.resident_packed_bytes() == 0,
        "mixed streamed MFE record discard mismatch");
}

void test_container_validation() {
    const auto one =
        make_nint_tensor(4, 1, 32, 1);
    const auto two =
        make_nint_tensor(4, 2, 32, 2);

    // Legacy NIM1 remains a valid all-NINT container.
    const auto legacy = make_nim1(
        2,
        1,
        32,
        two);
    const auto legacy_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            legacy);
    require(
        legacy_weight.experts() == 2
            && legacy_weight.out_per_expert() == 1,
        "valid NIM1 container was decoded incorrectly");
    const std::vector<float> legacy_input(32, 0.125f);
    const std::vector<std::int32_t> legacy_ids{0, 1};
    const auto legacy_output = evaluated_floats(
        legacy_weight.routed_matmul(
            mlx::core::array(
                legacy_input.begin(),
                mlx::core::Shape{1, 32}),
            mlx::core::array(
                legacy_ids.begin(),
                mlx::core::Shape{1, 2})));
    for (int expert = 0; expert < 2; ++expert) {
        float expected = 0.0f;
        for (int column = 0; column < 32; ++column) {
            expected += legacy_input[column]
                * two.dense[expert * 32 + column];
        }
        require_close(
            legacy_output[expert], expected, 2e-2f);
    }

    expect_invalid(
        std::vector<std::uint8_t>(10, 0),
        "truncated header");

    std::vector<std::uint8_t> impossible_size{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(
        impossible_size,
        std::numeric_limits<std::int32_t>::max());
    append<std::uint32_t>(impossible_size, 1);
    append<std::uint32_t>(impossible_size, 32);
    append<std::uint32_t>(impossible_size, 1);
    expect_invalid(
        impossible_size,
        "expert count larger than payload");

    auto bad_magic = legacy;
    bad_magic[0] = 'X';
    expect_invalid(bad_magic, "bad magic");

    auto zero_pools = legacy;
    std::fill(
        zero_pools.begin() + 16,
        zero_pools.begin() + 20,
        0);
    expect_invalid(zero_pools, "zero pools");

    expect_invalid(
        make_raw_nim2(
            2,
            1,
            32,
            {
                {{0}, "NINT4", one, {}},
                {{0}, "NINT4", one, {}},
            }),
        "duplicate expert ownership");
    expect_invalid(
        make_raw_nim2(
            3,
            1,
            32,
            {
                {{0}, "NINT4", one, {}},
                {{1}, "NINT4", one, {}},
            }),
        "missing expert");
    expect_invalid(
        make_raw_nim2(
            2,
            1,
            32,
            {
                {{0}, "NINT4", one, {}},
                {{2}, "NINT4", one, {}},
            }),
        "out-of-range expert");
    expect_invalid(
        make_raw_nim2(
            1,
            1,
            32,
            {
                {{0}, "NVQ2", one, {}},
            }),
        "unsupported cohort dtype");
    expect_invalid(
        make_raw_nim2(
            1,
            1,
            32,
            {
                {
                    {0},
                    "NINT4",
                    one,
                    {1},
                },
            }),
        "unexpected runtime payload");
    expect_invalid(
        make_raw_nim2(
            1,
            2,
            32,
            {
                {{0}, "NINT4", one, {}},
            }),
        "nested row mismatch");

    auto trailing = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, "NINT4", one, {}},
        });
    trailing.push_back(0);
    expect_invalid(trailing, "trailing byte");

    auto truncated = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, "NINT4", one, {}},
        });
    truncated.pop_back();
    expect_invalid(truncated, "truncated payload");

    auto non_ascii = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, std::string(1, '\xff'), one, {}},
        });
    expect_invalid(non_ascii, "non-ASCII dtype");

    auto wrong_nested_type = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, "NINT8-0", one, {}},
        });
    expect_invalid(
        wrong_nested_type,
        "dtype/payload mismatch");

    auto unexpected_runtime =
        make_plain_nvq(24, 24);
    unexpected_runtime.runtime = {1};
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(unexpected_runtime)}).blob,
        "matrix VQ runtime metadata");

    auto missing_hsg1 =
        make_rotated_nepq1_s(24, 24);
    missing_hsg1.runtime.clear();
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(missing_hsg1)}).blob,
        "missing HSG1 metadata");

    auto bad_hsg1_magic =
        make_rotated_nepq1_s(24, 24);
    bad_hsg1_magic.runtime[0] = 'X';
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(bad_hsg1_magic)}).blob,
        "bad HSG1 magic");

    auto mismatched_hsg1 =
        make_rotated_nepq1_s(24, 24);
    const std::uint32_t wrong_block = 16;
    std::memcpy(
        mismatched_hsg1.runtime.data() + 8,
        &wrong_block,
        sizeof(wrong_block));
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(mismatched_hsg1)}).blob,
        "mismatched HSG1 block");

    auto invalid_hsg1_sign =
        make_rotated_nepq1_s(24, 24);
    invalid_hsg1_sign.runtime[20] = 0;
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(invalid_hsg1_sign)}).blob,
        "invalid HSG1 sign");

    auto truncated_vq =
        make_npq(24, 24);
    truncated_vq.blob.pop_back();
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(truncated_vq)}).blob,
        "truncated VQ cohort");

    auto trailing_vq =
        make_nvq1_s(24, 24);
    trailing_vq.blob.push_back(0);
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(trailing_vq)}).blob,
        "trailing VQ cohort byte");

    auto wrong_vq_dtype =
        make_plain_nvq(24, 24);
    wrong_vq_dtype.dtype = "NPQ0-L";
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(wrong_vq_dtype)}).blob,
        "VQ dtype/blob mismatch");

    auto wrong_vq_shape =
        make_vq_moe_fixture({
            make_plain_nvq(24, 24),
        }).blob;
    const std::uint32_t wrong_output = 23;
    std::memcpy(
        wrong_vq_shape.data() + 8,
        &wrong_output,
        sizeof(wrong_output));
    expect_invalid(
        wrong_vq_shape,
        "VQ outer/nested shape mismatch");

    expect_invalid(
        make_vq_moe_fixture({
            make_rotated_nepq1_s(24, 24),
            make_rotated_nepq1_s(
                24,
                24,
                0x123456789abcdef0ull,
                1),
        }).blob,
        "conflicting HSG1 signs");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (
            argc == 2
            && std::string_view(argv[1])
                == "--benchmark-nepq-a"
        ) {
            benchmark_v4f_nepq_a();
            return 0;
        }
        if (
            argc == 2
            && std::string_view(argv[1])
                == "--benchmark-nint-shared"
        ) {
            benchmark_shared_nint_mfe();
            return 0;
        }
        if (argc != 1) {
            throw std::runtime_error(
                "usage: mfq-metal-moe-test "
                "[--benchmark-nepq-a|--benchmark-nint-shared]");
        }
        test_all_families_and_projections();
        test_swiglu_ffn();
        test_mxfp4_mfe_and_projection_offsets();
        test_mxfp4_multi_pool_native_slots();
        test_mxfp4_pair_blocks_matches_native_projections();
        test_mxfp4_sq_mfe_reuses_linear_kernel();
        test_mxfp4_sq_mixed_mfe_routes_without_sq_heterogeneous_kernel();
        test_mxfp4_smallm_nax_policy();
        test_mxfp4_decode_down_reduce();
        test_mxfp4_decode_swiglu_row_packing();
        test_large_mxfp4_arena_avoids_single_group_builder();
        test_vq_cohorts_and_ffn();
        test_nepq_a_routed_and_fused_swiglu();
        test_grouped_mmq_prefill();
        test_grouped_vq_decoder_tail_prefill();
        test_grouped_nint_mmq_prefill();
        test_grouped_split_nint_swiglu_prefill();
        test_shared_nint2_prefill();
        test_shared_nint5_group28_prefill();
        test_shared_nint8_mixed_prefill();
        test_shared_nint_mapped_and_packed_routes();
        test_grouped_dense_quad_tail_prefill();
        test_mixed_mfe_native_and_grouped_dispatch();
        test_multiple_nint_pools_share_cpp_dispatch();
        test_grouped_mxfp4_vq_mmq_prefill();
        test_concatenate_resident_mfe_experts();
        test_streamed_mixed_mfe_residency();
        test_container_validation();
        std::cout
            << "MFQ native heterogeneous MFE/streamed "
               "routed Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
