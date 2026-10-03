#include "mlx_fp8_sq.h"
#include "mlx_grouped_linear.h"
#include "mlx_moe.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void append(std::vector<std::uint8_t>& output, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(T));
}

std::vector<std::uint8_t> pack_bits(
    const std::vector<std::uint8_t>& values,
    unsigned bits) {
    std::vector<std::uint8_t> output((values.size() * bits + 7) / 8, 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto bit = index * bits;
        const auto value = static_cast<std::uint16_t>(values[index]) << (bit & 7);
        output[bit / 8] |= static_cast<std::uint8_t>(value);
        if ((bit & 7) + bits > 8) {
            output[bit / 8 + 1] |= static_cast<std::uint8_t>(value >> 8);
        }
    }
    return output;
}

float e4m3(std::uint8_t raw) {
    const auto magnitude = static_cast<unsigned>(raw & 0x7f);
    require(magnitude != 0x7f, "fixture contains E4M3 NaN");
    const auto exponent = magnitude >> 3;
    const auto mantissa = magnitude & 7;
    const float value = exponent == 0
        ? static_cast<float>(mantissa) * std::ldexp(1.0f, -9)
        : (1.0f + static_cast<float>(mantissa) / 8.0f) *
            std::ldexp(1.0f, static_cast<int>(exponent) - 7);
    return (raw & 0x80) == 0 ? value : -value;
}

std::vector<std::uint8_t> legal_codes() {
    std::vector<std::uint8_t> result;
    result.reserve(254);
    for (int code = 0; code < 256; ++code) {
        if ((code & 0x7f) != 0x7f) {
            result.push_back(static_cast<std::uint8_t>(code));
        }
    }
    return result;
}

struct Fixture {
    std::string dtype;
    int rows = 0;
    int columns = 0;
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
};

Fixture make_fixture(
    std::string dtype,
    int mx_block_rows = 128,
    int mx_block_columns = 128,
    int fp_scale_kind = 4,
    int rows = 8,
    int columns = 128) {
    const bool mxfp8 = dtype == "MXFP8-SQ";
    const int block_rows = mxfp8 ? mx_block_rows : 128;
    const int block_columns = mxfp8 ? mx_block_columns : 128;
    const int scale_rows = (rows + block_rows - 1) / block_rows;
    const int scale_columns = (columns + block_columns - 1) / block_columns;
    const auto legal = legal_codes();
    std::array<std::uint8_t, 256> palettes{};
    for (int q = 1; q <= 7; ++q) {
        const auto begin = (1u << q) - 2u;
        for (unsigned index = 0; index < (1u << q); ++index) {
            palettes[begin + index] = legal[index];
        }
    }

    std::vector<std::uint8_t> q;
    std::vector<std::vector<std::uint8_t>> streams;
    std::vector<float> dense(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        const int bits = row % 8 + 1;
        q.push_back(static_cast<std::uint8_t>(bits - 1));
        std::vector<std::uint8_t> values(columns);
        for (int column = 0; column < columns; ++column) {
            std::uint8_t code = 0;
            if (bits == 8) {
                code = legal[static_cast<std::size_t>(column * 13 + 151) % legal.size()];
                values[column] = code;
            } else {
                const auto symbol = static_cast<std::uint8_t>(
                    (row * 11 + column * 5) & ((1 << bits) - 1));
                values[column] = symbol;
                code = palettes[(1u << bits) - 2u + symbol];
            }
            const int scale_index =
                (row / block_rows) * scale_columns + column / block_columns;
            const float scale = mxfp8
                ? std::ldexp(1.0f, scale_index % 3)
                : 1.5f;
            dense[static_cast<std::size_t>(row) * columns + column] =
                e4m3(code) * scale;
        }
        streams.push_back(bits == 8 ? values : pack_bits(values, bits));
    }

    std::vector<std::uint8_t> blob = mxfp8
        ? std::vector<std::uint8_t>{'M', '8', 'S', 'Q'}
        : std::vector<std::uint8_t>{'F', '8', 'S', 'Q'};
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, mxfp8 ? 1 : fp_scale_kind);
    append<std::uint8_t>(blob, 0);
    append<std::uint8_t>(blob, 0);
    append<std::uint16_t>(blob, static_cast<std::uint16_t>(block_rows));
    append<std::uint16_t>(blob, static_cast<std::uint16_t>(block_columns));
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, columns);
    append<std::uint64_t>(blob, scale_rows);
    append<std::uint64_t>(blob, scale_columns);
    auto packed_q = pack_bits(q, 3);
    blob.insert(blob.end(), packed_q.begin(), packed_q.end());
    while (blob.size() % 4 != 0) {
        blob.push_back(0);
    }
    blob.insert(blob.end(), palettes.begin(), palettes.end());
    for (const auto& stream : streams) {
        blob.insert(blob.end(), stream.begin(), stream.end());
    }
    if (mxfp8) {
        for (int scale = 0; scale < scale_rows * scale_columns; ++scale) {
            append<std::uint8_t>(
                blob, static_cast<std::uint8_t>(127 + scale % 3));
        }
    } else if (fp_scale_kind == 2) {
        for (int scale = 0; scale < scale_rows * scale_columns; ++scale) {
            append<std::uint16_t>(blob, 0x3fc0);
        }
    } else if (fp_scale_kind == 3) {
        for (int scale = 0; scale < scale_rows * scale_columns; ++scale) {
            append<std::uint16_t>(blob, 0x3e00);
        }
    } else {
        for (int scale = 0; scale < scale_rows * scale_columns; ++scale) {
            append<float>(blob, 1.5f);
        }
    }
    return {std::move(dtype), rows, columns, std::move(blob), std::move(dense)};
}

void test_format(
    const std::string& dtype,
    int mx_block_rows = 128,
    int mx_block_columns = 128,
    int fp_scale_kind = 4) {
    using namespace mlx::core;
    const auto fixture = make_fixture(
        dtype, mx_block_rows, mx_block_columns, fp_scale_kind);
    const auto weight = mfq::metal::MlxFp8SqWeight::from_blob(
        dtype, fixture.blob);
    require(weight.dtype() == dtype, "FP8-SQ dtype mismatch");
    require(weight.input_size() == fixture.columns, "FP8-SQ width mismatch");
    require(weight.output_size() == fixture.rows, "FP8-SQ output mismatch");
    require(weight.descriptor().distribution_entropy == 3.0,
            "FP8-SQ mixed-q entropy mismatch");

    auto decoded = contiguous(weight.dequantize(float32));
    eval(decoded);
    for (std::size_t index = 0; index < fixture.dense.size(); ++index) {
        require(decoded.data<float>()[index] == fixture.dense[index],
                dtype + " Metal dequantization mismatch");
    }

    std::vector<float> source(static_cast<std::size_t>(fixture.columns));
    std::vector<float> expected(static_cast<std::size_t>(fixture.rows), 0.0f);
    for (int column = 0; column < fixture.columns; ++column) {
        source[column] = static_cast<float>((column * 7) % 23 - 11) / 128.0f;
    }
    for (int row = 0; row < fixture.rows; ++row) {
        for (int column = 0; column < fixture.columns; ++column) {
            expected[row] += source[column] *
                fixture.dense[static_cast<std::size_t>(row) * fixture.columns + column];
        }
    }
    auto input = astype(array(source.begin(), Shape{1, fixture.columns}), float16);
    auto output = contiguous(astype(weight.matmul(input), float32));
    eval(output);
    require(output.shape() == Shape{1, fixture.rows}, "FP8-SQ matmul shape mismatch");
    for (int row = 0; row < fixture.rows; ++row) {
        const float tolerance = std::max(0.02f, std::fabs(expected[row]) * 0.002f);
        require(std::fabs(output.data<float>()[row] - expected[row]) < tolerance,
                dtype + " Metal packed matmul mismatch");
    }
}

void test_projection_group(const std::string& dtype) {
    using namespace mlx::core;
    const auto first_fixture = make_fixture(
        dtype, 32, 32, dtype == "MXFP8-SQ" ? 1 : 4, 8, 128);
    const auto second_fixture = make_fixture(
        dtype, 32, 32, dtype == "MXFP8-SQ" ? 1 : 4, 6, 128);
    const auto first = mfq::metal::MlxFp8SqWeight::from_blob(
        dtype, first_fixture.blob);
    const auto second = mfq::metal::MlxFp8SqWeight::from_blob(
        dtype, second_fixture.blob);
    const mfq::metal::MlxGroupedLinear grouped({&first, &second});
    require(grouped.uses_zero_copy_storage(),
            dtype + " projection group copied packed storage");
    require(grouped.copied_packed_nbytes() == 0,
            dtype + " projection group reported a packed copy");
    require(grouped.supports_single_row_projection_fusion(),
            dtype + " projection group missed decode fusion");

    for (int rows = 1; rows <= 6; ++rows) {
        std::vector<float> values(static_cast<std::size_t>(rows) * 128);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(
                static_cast<int>((index * 13 + 9) % 37) - 18) / 256.0f;
        }
        auto input = astype(array(values.begin(), Shape{rows, 128}), float16);
        auto actual = grouped(input);
        std::array<array, 2> expected{
            first.matmul(input),
            second.matmul(input),
        };
        require(actual.size() == expected.size(),
                dtype + " projection group output count mismatch");
        for (std::size_t projection = 0;
             projection < actual.size();
             ++projection) {
            auto difference = contiguous(astype(
                abs(astype(actual[projection], float32) -
                    astype(expected[projection], float32)),
                float32));
            eval(difference);
            float maximum = 0.0f;
            for (std::size_t index = 0; index < difference.size(); ++index) {
                maximum = std::max(maximum, difference.data<float>()[index]);
            }
            require(maximum < 2.0e-3f,
                    dtype + " projection group M=" + std::to_string(rows) +
                        " mismatch: max_abs=" + std::to_string(maximum));
        }
    }
}

void test_packed_backward(
    const std::string& dtype,
    int block_rows,
    int block_columns,
    int scale_kind,
    int columns = 128) {
    using namespace mlx::core;
    constexpr int output_rows = 68;
    const auto fixture = make_fixture(
        dtype, block_rows, block_columns, scale_kind, output_rows, columns);
    const auto weight = mfq::metal::MlxFp8SqWeight::from_blob(
        dtype, fixture.blob);
    for (const int rows : {1, 2, 4, 6, 8}) {
        std::vector<float> values(
            static_cast<std::size_t>(rows) * output_rows);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(
                static_cast<int>((index * 19 + 7) % 47) - 23) / 2048.0f;
        }
        auto gradient = astype(
            array(values.begin(), Shape{1, rows, output_rows}), float16);
        auto actual = contiguous(astype(
            weight.backward_input(gradient), float32));
        auto expected = contiguous(astype(matmul(
            reshape(gradient, Shape{rows, output_rows}),
            weight.dequantize(float16)), float32));
        eval(actual, expected);
        require(actual.shape() == Shape{1, rows, columns},
                dtype + " packed backward shape mismatch");
        float maximum_difference = 0.0f;
        float maximum_reference = 0.0f;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            maximum_difference = std::max(
                maximum_difference,
                std::fabs(actual.data<float>()[index]
                    - expected.data<float>()[index]));
            maximum_reference = std::max(
                maximum_reference,
                std::fabs(expected.data<float>()[index]));
        }
        require(
            maximum_difference < std::max(0.05f, maximum_reference * 0.004f),
            dtype + " packed backward M=" + std::to_string(rows)
                + " mismatch: max_abs="
                + std::to_string(maximum_difference));
    }
}

void test_mfe_format(const std::string& dtype) {
    using namespace mlx::core;
    constexpr int experts = 2;
    constexpr int output = 4;
    constexpr int columns = 128;
    constexpr int tokens = 2;
    constexpr int routes = 2;
    const auto fixture = make_fixture(dtype);
    require(fixture.rows == experts * output, "invalid FP8-SQ MFE fixture rows");

    std::vector<std::uint8_t> blob{'M', 'F', 'E', '1'};
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, columns);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, static_cast<std::uint32_t>(dtype.size()));
    append<std::uint64_t>(blob, fixture.blob.size());
    append<std::uint64_t>(blob, 0);
    const std::array<std::int32_t, experts> local_to_global{1, 0};
    for (const auto expert : local_to_global) {
        append<std::int32_t>(blob, expert);
    }
    blob.insert(blob.end(), dtype.begin(), dtype.end());
    blob.insert(blob.end(), fixture.blob.begin(), fixture.blob.end());

    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);
    require(weight.experts() == experts, dtype + " MFE expert count mismatch");
    require(weight.out_per_expert() == output, dtype + " MFE output mismatch");
    require(!weight.supports_grouped_mmq(), dtype + " created a duplicate MFE kernel");

    std::vector<float> source(static_cast<std::size_t>(tokens) * columns);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 128.0f;
    }
    const std::array<std::int32_t, tokens * routes> ids{0, 1, 1, 0};
    const auto input = astype(
        array(source.begin(), Shape{tokens, columns}),
        float16);
    const auto expert_ids = array(ids.begin(), Shape{tokens, routes});
    auto actual = contiguous(astype(weight.routed_matmul(input, expert_ids), float32));
    eval(actual);
    require(actual.shape() == Shape{tokens, routes, output}, dtype + " MFE shape mismatch");
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[static_cast<std::size_t>(token) * routes + route];
            const int local = expert == local_to_global[0] ? 0 : 1;
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < columns; ++column) {
                    expected += source[static_cast<std::size_t>(token) * columns + column] *
                        fixture.dense[(static_cast<std::size_t>(local) * output + row) * columns + column];
                }
                const auto index =
                    (static_cast<std::size_t>(token) * routes + route) * output + row;
                const float tolerance = std::max(0.03f, std::fabs(expected) * 0.003f);
                require(
                    std::fabs(actual.data<float>()[index] - expected) < tolerance,
                    dtype + " MFE routed result mismatch");
            }
        }
    }
}

} // namespace

int main() {
    try {
        test_format("MXFP8-SQ");
        test_format("MXFP8-SQ", 1, 32);
        test_format("MXFP8-SQ", 32, 32);
        test_format("FP8-128SQ", 128, 128, 2);
        test_format("FP8-128SQ", 128, 128, 3);
        test_format("FP8-128SQ", 128, 128, 4);
        test_projection_group("MXFP8-SQ");
        test_projection_group("FP8-128SQ");
        test_packed_backward("MXFP8-SQ", 1, 32, 1);
        test_packed_backward("MXFP8-SQ", 32, 32, 1);
        test_packed_backward("MXFP8-SQ", 128, 128, 1);
        test_packed_backward("FP8-128SQ", 128, 128, 2);
        test_packed_backward("FP8-128SQ", 128, 128, 3);
        test_packed_backward("FP8-128SQ", 128, 128, 4);
        test_packed_backward("FP8-128SQ", 128, 128, 4, 160);
        test_mfe_format("MXFP8-SQ");
        test_mfe_format("FP8-128SQ");
        std::cout << "MFQ MXFP8-SQ/FP8-128SQ Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
