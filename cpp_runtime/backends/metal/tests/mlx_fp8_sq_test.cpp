#include "mlx_fp8_sq.h"
#include "mlx_grouped_linear.h"
#include "mlx_moe.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
    int columns = 128,
    int uniform_q = 0,
    int maximum_q = 8) {
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
            palettes[begin + index] = legal[(index * 137 + q * 31) % legal.size()];
        }
    }

    std::vector<std::uint8_t> q;
    std::vector<std::vector<std::uint8_t>> streams;
    std::vector<float> dense(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        const int bits = uniform_q != 0 ? uniform_q : row % maximum_q + 1;
        q.push_back(static_cast<std::uint8_t>(bits - 1));
        std::vector<std::uint8_t> values(columns);
        std::uint32_t symbol_state = 0x9e3779b9u ^ static_cast<std::uint32_t>(row + 1);
        for (int column = 0; column < columns; ++column) {
            symbol_state ^= symbol_state << 13;
            symbol_state ^= symbol_state >> 17;
            symbol_state ^= symbol_state << 5;
            std::uint8_t code = 0;
            if (bits == 8) {
                code = legal[static_cast<std::size_t>(column * 13 + 151) % legal.size()];
                values[column] = code;
            } else {
                const auto symbol = static_cast<std::uint8_t>(symbol_state & ((1u << bits) - 1u));
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
    int fp_scale_kind = 4,
    int columns = 128) {
    using namespace mlx::core;
    const auto fixture = make_fixture(
        dtype, mx_block_rows, mx_block_columns, fp_scale_kind, 8, columns);
    const auto weight = mfq::metal::MlxFp8SqWeight::from_blob(
        dtype, fixture.blob);
    require(weight.dtype() == dtype, "FP8-SQ dtype mismatch");
    require(weight.input_size() == fixture.columns, "FP8-SQ width mismatch");
    require(weight.output_size() == fixture.rows, "FP8-SQ output mismatch");
    require(weight.descriptor().distribution_entropy == 3.0,
            "FP8-SQ mixed-q entropy mismatch");

    for (const auto decoded_dtype : {float16, float32}) {
        auto decoded = contiguous(astype(weight.dequantize(decoded_dtype), float32));
        eval(decoded);
        for (std::size_t index = 0; index < fixture.dense.size(); ++index) {
            require(decoded.data<float>()[index] == fixture.dense[index],
                    dtype + " Metal dequantization mismatch K=" + std::to_string(columns)
                        + " index=" + std::to_string(index) + " actual="
                        + std::to_string(decoded.data<float>()[index]) + " expected="
                        + std::to_string(fixture.dense[index]));
        }
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
    for (int tokens = 1; tokens <= 6; ++tokens) {
        std::vector<float> values(tokens * fixture.columns);
        for (int token = 0; token < tokens; ++token) {
            for (int column = 0; column < fixture.columns; ++column) {
                values[token * fixture.columns + column] = source[column] * ((token + 1) / 8.0f);
            }
        }
        auto input = astype(array(values.begin(), Shape{tokens, fixture.columns}), float16);
        auto output = contiguous(astype(weight.matmul(input), float32));
        eval(output);
        require(output.shape() == Shape{tokens, fixture.rows}, "FP8-SQ matmul shape mismatch");
        for (int token = 0; token < tokens; ++token) {
            for (int row = 0; row < fixture.rows; ++row) {
                const float reference = expected[row] * ((token + 1) / 8.0f);
                const float tolerance = std::max(0.02f, std::fabs(reference) * 0.002f);
                require(std::fabs(output.data<float>()[token * fixture.rows + row] - reference) < tolerance,
                    dtype + " Metal packed matmul M=" + std::to_string(tokens) + " mismatch");
            }
        }
    }
}

void test_varied_scale_dequantize(const std::string& dtype, int kind, int columns,
    int maximum_q = 8) {
    using namespace mlx::core;
    auto fixture = make_fixture(dtype, 32, 32, kind, 129, columns, 0, maximum_q);
    const auto original = mfq::metal::MlxFp8SqWeight::from_blob(dtype, fixture.blob);
    const auto layout = original.wire_layout();
    constexpr std::array<std::uint16_t, 6> bf16{0x3ec0, 0x3f81, 0x4001, 0x3881, 0x3f35, 0x4049};
    constexpr std::array<std::uint16_t, 6> fp16{0x3601, 0x3c01, 0x4001, 0x0401, 0x39a9, 0x4249};
    constexpr std::array<float, 6> fp32{0.12345679f, 1.000001f, 2.000001f, 0.000061097f, 0.7083333f, 3.1415927f};
    std::vector<float> scales(layout.scale_rows * layout.scale_columns);
    for (std::size_t index = 0; index < scales.size(); ++index) {
        auto* destination = fixture.blob.data() + layout.scales;
        if (dtype == "MXFP8-SQ") {
            destination[index] = 123 + index % 7;
            scales[index] = std::ldexp(1.0f, int(destination[index]) - 127);
        } else if (kind == 2 || kind == 3) {
            const auto raw = kind == 2 ? bf16[index % bf16.size()] : fp16[index % fp16.size()];
            std::memcpy(destination + index * 2, &raw, 2);
            scales[index] = kind == 2 ? std::bit_cast<float>(std::uint32_t(raw) << 16)
                : std::ldexp(1.0f + float(raw & 1023u) / 1024.0f, int((raw >> 10) & 31u) - 15);
        } else {
            scales[index] = fp32[index % fp32.size()];
            std::memcpy(destination + index * 4, &scales[index], 4);
        }
    }
    for (int row = 0; row < fixture.rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const auto scale = row / layout.block_rows * layout.scale_columns + column / layout.block_columns;
            fixture.dense[row * columns + column] = fixture.dense[row * columns + column]
                / (dtype == "MXFP8-SQ" ? std::ldexp(1.0f, scale % 3) : 1.5f) * scales[scale];
        }
    }
    const auto weight = mfq::metal::MlxFp8SqWeight::from_blob(dtype, fixture.blob);
    for (const auto dtype_out : {float16, float32}) {
        auto actual = contiguous(astype(weight.dequantize(dtype_out), float32));
        auto expected = contiguous(astype(astype(array(fixture.dense.begin(), Shape{fixture.rows, columns}), dtype_out), float32));
        eval(actual, expected);
        for (std::size_t index = 0; index < fixture.dense.size(); ++index) {
            require(actual.data<float>()[index] == expected.data<float>()[index],
                dtype + " varied scale kind=" + std::to_string(kind) + " K=" + std::to_string(columns)
                    + " index=" + std::to_string(index));
        }
    }
    if (maximum_q < 8) {
        std::vector<float> values(columns);
        for (int column = 0; column < columns; ++column) {
            values[column] = float(column % 19 - 9) / 128.0f;
        }
        auto input = astype(array(values.begin(), Shape{1, columns}), float16);
        auto actual = astype(weight.matmul(input), float32);
        eval(actual);
        for (int row = 0; row < fixture.rows; ++row) {
            double expected = 0.0;
            for (int column = 0; column < columns; ++column) {
                expected += double(values[column]) * fixture.dense[row * columns + column];
            }
            require(std::fabs(actual.data<float>()[row] - expected)
                    < std::max(0.003, std::fabs(expected) * 0.001),
                dtype + " varied-scale matmul mismatch");
        }
    }
}

void test_uniform_forward(const std::string& dtype, int bits,
    int output_rows, int columns, int maximum_q = 8) {
    using namespace mlx::core;
    const auto fixture = make_fixture(dtype, 32, 32, 4, output_rows, columns, bits, maximum_q);
    const auto weight = mfq::metal::MlxFp8SqWeight::from_blob(dtype, fixture.blob);
    require(bits == 0 || weight.descriptor().distribution_entropy == 0.0,
            dtype + " uniform-q entropy mismatch");
    require(weight.descriptor().q_mask == (bits == 0 ? (1u << maximum_q) - 1u : 1u << (bits - 1)),
            dtype + " q mask mismatch");
    const mfq::metal::MlxGroupedLinear grouped({&weight, &weight});
    for (int tokens = 1; tokens <= 6; ++tokens) {
        std::vector<float> values(static_cast<std::size_t>(tokens) * columns);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(
                static_cast<int>((index * 19 + 11) % 41) - 20) / 2048.0f;
        }
        auto input = astype(array(values.begin(), Shape{tokens, columns}), float16);
        auto results = grouped(input);
        results.push_back(weight.matmul(input));
        for (auto& result : results) {
            result = contiguous(astype(result, float32));
        }
        eval(results);
        for (int token = 0; token < tokens; ++token) {
            for (int row = 0; row < output_rows; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < columns; ++column) {
                    expected += values[token * columns + column]
                        * fixture.dense[static_cast<std::size_t>(row) * columns + column];
                }
                const float tolerance = std::max(0.002f, std::fabs(expected) * 0.001f);
                for (const auto& result : results) {
                    require(std::fabs(result.data<float>()[token * output_rows + row]
                        - expected) < tolerance, dtype + " uniform q=" + std::to_string(bits)
                        + " M=" + std::to_string(tokens) + " forward mismatch");
                }
            }
        }
    }
}

void test_projection_group(const std::string& dtype,
    mlx::core::Dtype input_dtype = mlx::core::float16, int columns = 128,
    int first_maximum_q = 8, int second_maximum_q = 8) {
    using namespace mlx::core;
    const auto first_fixture = make_fixture(
        dtype, 32, 32, dtype == "MXFP8-SQ" ? 1 : 4, 8, columns, 0, first_maximum_q);
    const auto second_fixture = make_fixture(
        dtype, 32, 32, dtype == "MXFP8-SQ" ? 1 : 4, 6, columns, 0, second_maximum_q);
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

    for (const int rows : {1, 2, 3, 4, 5, 6, 7, 9, 16}) {
        std::vector<float> values(static_cast<std::size_t>(rows) * columns);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(
                static_cast<int>((index * 13 + 9) % 37) - 18) / 256.0f;
        }
        auto input = astype(array(values.begin(), Shape{rows, columns}), input_dtype);
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
    auto decoded = contiguous(astype(weight.dequantize(float16), float32));
    eval(decoded);
    for (std::size_t index = 0; index < fixture.dense.size(); ++index) {
        require(decoded.data<float>()[index] == fixture.dense[index],
            dtype + " backward reference dequantization mismatch K=" + std::to_string(columns)
                + " index=" + std::to_string(index) + " actual="
                + std::to_string(decoded.data<float>()[index]) + " expected="
                + std::to_string(fixture.dense[index]));
    }
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

void test_mfe_format(const std::string& dtype, int maximum_q = 8,
    int uniform_q = 0, int output = 4) {
    using namespace mlx::core;
    constexpr int experts = 2;
    constexpr int columns = 128;
    constexpr int routes = 2;
    const auto fixture = make_fixture(dtype, 128, 128, 4, experts * output, columns, uniform_q, maximum_q);
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

    const std::array<std::span<const std::uint8_t>, 2> projection_blobs{blob, blob};
    const auto direct = mfq::metal::MlxMoeWeight::from_projection_blobs(projection_blobs);
    const auto projected = mfq::metal::MlxMoeWeight::concatenate_projections({weight, weight});
    require(!direct.supports_grouped_mmq(), dtype + " direct loader changed SQ dispatch");
    for (int tokens = 1; tokens <= 6; ++tokens) {
        for (const bool shared_input : {true, false}) {
            std::vector<float> source(static_cast<std::size_t>(tokens)
                * (shared_input ? 1 : routes) * columns);
            for (std::size_t index = 0; index < source.size(); ++index) {
                source[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 128.0f;
            }
            std::vector<std::int32_t> ids(tokens * routes);
            for (std::size_t index = 0; index < ids.size(); ++index) {
                ids[index] = static_cast<std::int32_t>((index + index / routes) % experts);
            }
            const auto input = astype(array(source.begin(), shared_input
                ? Shape{tokens, columns} : Shape{tokens, routes, columns}), float16);
            const auto expert_ids = array(ids.begin(), Shape{tokens, routes});
            auto actual = contiguous(astype(weight.routed_matmul(input, expert_ids), float32));
            eval(actual);
            require(actual.shape() == Shape{tokens, routes, output}, dtype + " MFE shape mismatch");
            require(all(equal(direct.routed_matmul(input, expert_ids),
                              projected.routed_matmul(input, expert_ids))).item<bool>(),
                    dtype + " direct projection matmul mismatch");
            require(all(equal(direct.routed_swiglu(input, expert_ids),
                              projected.routed_swiglu(input, expert_ids))).item<bool>(),
                    dtype + " direct projection SwiGLU mismatch");
            for (int token = 0; token < tokens; ++token) {
                for (int route = 0; route < routes; ++route) {
                    const int expert = ids[static_cast<std::size_t>(token) * routes + route];
                    const int local = expert == local_to_global[0] ? 0 : 1;
                    const auto input_row = shared_input ? token : token * routes + route;
                    for (int row = 0; row < output; ++row) {
                        float expected = 0.0f;
                        for (int column = 0; column < columns; ++column) {
                            expected += source[static_cast<std::size_t>(input_row) * columns + column] *
                                fixture.dense[(static_cast<std::size_t>(local) * output + row) * columns + column];
                        }
                        const auto index =
                            (static_cast<std::size_t>(token) * routes + route) * output + row;
                        const float tolerance = std::max(0.03f, std::fabs(expected) * 0.003f);
                        require(std::fabs(actual.data<float>()[index] - expected) < tolerance,
                            dtype + " MFE M=" + std::to_string(tokens) + " routed result mismatch");
                    }
                }
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
        test_format("FP8-128SQ", 128, 128, 4, 129);
        test_format("FP8-128SQ", 128, 128, 4, 130);
        for (const int columns : {4, 8, 12, 131, 132}) {
            test_format("FP8-128SQ", 128, 128, 4, columns);
        }
        for (int bits = 1; bits <= 8; ++bits) {
            test_uniform_forward("MXFP8-SQ", bits, 129, 64);
            test_uniform_forward("MXFP8-SQ", bits, 33, 256);
            test_uniform_forward("FP8-128SQ", bits, 33, 72);
            test_uniform_forward("FP8-128SQ", bits, 129, 65);
            test_uniform_forward("FP8-128SQ", bits, 33, 257);
        }
        for (const int columns : {640, 2560, 6144}) {
            for (int bits = 1; bits <= 8; ++bits) {
                test_uniform_forward("MXFP8-SQ", bits, 33, columns);
                test_uniform_forward("FP8-128SQ", bits, 33, columns);
            }
        }
        for (const int columns : {256, 257}) {
            if (columns % 32 == 0) test_varied_scale_dequantize("MXFP8-SQ", 1, columns);
            for (const int kind : {2, 3, 4}) test_varied_scale_dequantize("FP8-128SQ", kind, columns);
        }
        test_projection_group("MXFP8-SQ");
        test_projection_group("FP8-128SQ");
        test_projection_group("MXFP8-SQ", mlx::core::float32);
        test_projection_group("FP8-128SQ", mlx::core::float32);
        test_projection_group("FP8-128SQ", mlx::core::float16, 129);
        test_projection_group("FP8-128SQ", mlx::core::float32, 129);
        test_packed_backward("MXFP8-SQ", 1, 32, 1);
        test_packed_backward("MXFP8-SQ", 32, 32, 1);
        test_packed_backward("MXFP8-SQ", 128, 128, 1);
        test_packed_backward("FP8-128SQ", 128, 128, 2);
        test_packed_backward("FP8-128SQ", 128, 128, 3);
        test_packed_backward("FP8-128SQ", 128, 128, 4);
        test_packed_backward("FP8-128SQ", 128, 128, 4, 160);
        test_mfe_format("MXFP8-SQ");
        test_mfe_format("FP8-128SQ");
        for (const std::string dtype : {"MXFP8-SQ", "FP8-128SQ"}) {
            for (const int maximum_q : {2, 3}) {
                test_uniform_forward(dtype, 0, 33, 640, maximum_q);
                test_mfe_format(dtype, maximum_q);
            }
            test_projection_group(dtype, mlx::core::float16, 128, 1, 2);
            test_projection_group(dtype, mlx::core::float16, 128, 2, 3);
            test_projection_group(dtype, mlx::core::float32, 128, 2, 3);
            test_projection_group(dtype, mlx::core::float16, 128, 3, 8);
            for (const int bits : {1, 2, 3, 4, 5, 6, 7}) {
                test_mfe_format(dtype, 8, bits, 5);
            }
        }
        test_uniform_forward("FP8-128SQ", 0, 33, 257, 3);
        test_varied_scale_dequantize("MXFP8-SQ", 1, 256, 3);
        test_varied_scale_dequantize("MXFP8-SQ", 1, 256, 7);
        for (const int kind : {2, 3, 4}) {
            test_varied_scale_dequantize("FP8-128SQ", kind, 256, 3);
            test_varied_scale_dequantize("FP8-128SQ", kind, 256, 7);
        }
        std::cout << "MFQ MXFP8-SQ/FP8-128SQ Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
