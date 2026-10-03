#include "mlx_mxfp4_sq.h"
#include "mlx_grouped_linear.h"

#include <algorithm>
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

template <typename T> void append(std::vector<std::uint8_t> &target, T value) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(&value);
  target.insert(target.end(), bytes, bytes + sizeof(T));
}

void append_bytes(std::vector<std::uint8_t> &target,
                  const std::vector<std::uint8_t> &values) {
  target.insert(target.end(), values.begin(), values.end());
}

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::uint8_t> pack_bits(const std::vector<std::uint8_t> &values,
                                    unsigned bits) {
  require(bits > 0 && bits <= 8, "invalid test bit width");
  std::vector<std::uint8_t> packed((values.size() * bits + 7) / 8, 0);
  const std::uint16_t mask = static_cast<std::uint16_t>((1u << bits) - 1u);
  for (std::size_t index = 0; index < values.size(); ++index) {
    require(values[index] <= mask, "test value does not fit bit width");
    const std::size_t bit_offset = index * bits;
    const std::size_t byte_index = bit_offset / 8;
    const unsigned shift = static_cast<unsigned>(bit_offset % 8);
    const std::uint16_t value = static_cast<std::uint16_t>(values[index])
                                << shift;
    packed[byte_index] |= static_cast<std::uint8_t>(value);
    if (shift + bits > 8) {
      packed[byte_index + 1] |= static_cast<std::uint8_t>(value >> 8);
    }
  }
  return packed;
}

float fp4(std::uint8_t nibble) {
  constexpr float magnitudes[]{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = magnitudes[nibble & 7u];
  return (nibble & 8u) == 0 ? magnitude : -magnitude;
}

struct Fixture {
  int rows = 0;
  int columns = 0;
  std::uint8_t matrix_scale_base = 0;
  std::size_t payload_nbytes = 0;
  std::vector<std::uint8_t> blob;
  std::vector<float> dense;
};

Fixture make_fixture(int rows, int columns) {
  require(rows > 0 && columns > 0 && columns % 32 == 0,
          "invalid MXFP4-SQ2 test fixture geometry");
  constexpr std::uint8_t matrix_scale_base = 124;
  const int blocks_per_row = columns / 32;
  const std::size_t weights = static_cast<std::size_t>(rows) * columns;
  const std::size_t blocks = static_cast<std::size_t>(rows) * blocks_per_row;

  std::vector<std::uint8_t> symbols(weights);
  std::vector<std::uint8_t> selectors(blocks);
  std::vector<std::uint8_t> state_scales(static_cast<std::size_t>(rows) * 8);
  std::vector<std::uint8_t> state_palettes(static_cast<std::size_t>(rows) * 8);
  std::vector<float> dense(weights);
  std::array<bool, 8> seen_tags{};

  for (int row = 0; row < rows; ++row) {
    for (int state = 0; state < 8; ++state) {
      const std::size_t index = static_cast<std::size_t>(row) * 8 + state;
      state_scales[index] = static_cast<std::uint8_t>((row + state * 3) & 3);
      state_palettes[index] =
          static_cast<std::uint8_t>((row * 9 + state * 5) & 31);
    }
    for (int block = 0; block < blocks_per_row; ++block) {
      const std::size_t block_index =
          static_cast<std::size_t>(row) * blocks_per_row + block;
      selectors[block_index] =
          static_cast<std::uint8_t>((row + (block >> 2)) & 1);
      std::uint8_t low_tag = 0;
      for (int lane = 0; lane < 31; ++lane) {
        const int column = block * 32 + lane;
        const std::size_t index =
            static_cast<std::size_t>(row) * columns + column;
        const auto symbol = static_cast<std::uint8_t>(
            (row * 11 + block * 5 + lane * 3 + (lane >> 2)) & 3);
        symbols[index] = symbol;
        low_tag ^= symbol;
      }
      const auto required_low_tag =
          static_cast<std::uint8_t>((row + block) & 3);
      const std::size_t last_index =
          static_cast<std::size_t>(row) * columns + block * 32 + 31;
      symbols[last_index] = low_tag ^ required_low_tag;
      low_tag ^= symbols[last_index];
      require(low_tag == required_low_tag,
              "failed to construct MXFP4-SQ2 implicit test tag");
      const std::uint8_t tag =
          static_cast<std::uint8_t>(low_tag | (selectors[block_index] << 2u));
      seen_tags[tag] = true;
      const std::size_t state_index = static_cast<std::size_t>(row) * 8 + tag;
      const auto scale_offset = state_scales[state_index];
      const auto palette = state_palettes[state_index];
      const float scale = std::ldexp(1.0f, static_cast<int>(matrix_scale_base) +
                                               scale_offset - 127);
      for (int lane = 0; lane < 32; ++lane) {
        const int column = block * 32 + lane;
        const std::size_t index =
            static_cast<std::size_t>(row) * columns + column;
        const auto nibble = mfq::metal::kMxfp4Sq2PaletteNibbles
            [static_cast<std::size_t>(palette) * 4 + symbols[index]];
        dense[index] = fp4(nibble) * scale;
      }
    }
  }
  if (blocks >= seen_tags.size()) {
    for (bool seen : seen_tags) {
      require(seen, "MXFP4-SQ2 fixture missed a block tag");
    }
  }

  const auto packed_symbols = pack_bits(symbols, 2);
  const auto packed_selectors = pack_bits(selectors, 1);
  const auto packed_state_scales = pack_bits(state_scales, 2);
  const auto packed_state_palettes = pack_bits(state_palettes, 5);

  std::vector<std::uint8_t> blob{'S', 'Q', '2', 0};
  append<std::uint8_t>(blob, 1);
  append<std::uint8_t>(blob, matrix_scale_base);
  append<std::uint16_t>(blob, 0);
  append<std::uint64_t>(blob, rows);
  append<std::uint64_t>(blob, columns);
  const std::size_t header_nbytes = blob.size();
  append_bytes(blob, packed_symbols);
  append_bytes(blob, packed_selectors);
  append_bytes(blob, packed_state_scales);
  append_bytes(blob, packed_state_palettes);
  return {
      rows,
      columns,
      matrix_scale_base,
      blob.size(),
      std::move(blob),
      std::move(dense),
  };
}

Fixture make_adaptive_fixture(int rows, int columns) {
  require(rows >= 4 && columns > 0 && columns % 32 == 0,
          "invalid adaptive MXFP4-SQ test fixture geometry");
  constexpr std::uint8_t matrix_scale_base = 124;
  const int blocks_per_row = columns / 32;

  std::vector<std::uint8_t> row_q(static_cast<std::size_t>(rows));
  std::vector<std::vector<std::uint8_t>> symbol_rows;
  std::vector<std::uint8_t> selectors;
  std::vector<std::uint8_t> state_scales;
  std::vector<std::uint8_t> state_palettes;
  std::vector<std::uint8_t> native_scales;
  std::vector<float> dense(static_cast<std::size_t>(rows) * columns);
  symbol_rows.reserve(static_cast<std::size_t>(rows));

  int sq_row = 0;
  int native_row = 0;
  for (int row = 0; row < rows; ++row) {
    const unsigned bits = static_cast<unsigned>(row % 4 + 1);
    row_q[static_cast<std::size_t>(row)] =
        static_cast<std::uint8_t>(bits);
    std::vector<std::uint8_t> symbols(static_cast<std::size_t>(columns));
    const auto symbol_mask = static_cast<std::uint8_t>((1u << bits) - 1u);
    for (int column = 0; column < columns; ++column) {
      symbols[static_cast<std::size_t>(column)] =
          static_cast<std::uint8_t>(
              (row * 13 + column * 5 + (column >> 2)) & symbol_mask);
    }

    if (bits == 4) {
      for (int block = 0; block < blocks_per_row; ++block) {
        const auto exponent = static_cast<std::uint8_t>(
            123 + ((native_row * 3 + block) & 3));
        native_scales.push_back(exponent);
        const float scale = std::ldexp(1.0f, static_cast<int>(exponent) - 127);
        for (int lane = 0; lane < 32; ++lane) {
          const int column = block * 32 + lane;
          const auto index = static_cast<std::size_t>(row) * columns + column;
          dense[index] = fp4(symbols[static_cast<std::size_t>(column)]) * scale;
        }
      }
      ++native_row;
    } else {
      for (int state = 0; state < 8; ++state) {
        state_scales.push_back(static_cast<std::uint8_t>(
            (sq_row + state * 3) & 3));
        state_palettes.push_back(static_cast<std::uint8_t>(
            (sq_row * 7 + state * 5) & 31));
      }
      for (int block = 0; block < blocks_per_row; ++block) {
        const auto selector = static_cast<std::uint8_t>(
            (row + block * 3) & 1);
        selectors.push_back(selector);
        std::uint8_t low_tag = 0;
        for (int lane = 0; lane < 32; ++lane) {
          const auto symbol = symbols[
              static_cast<std::size_t>(block * 32 + lane)];
          if (bits == 1) {
            low_tag ^= static_cast<std::uint8_t>(
                symbol << static_cast<unsigned>(lane & 1));
          } else {
            low_tag ^= symbol;
          }
        }
        low_tag &= 3u;
        const auto tag = static_cast<std::uint8_t>(
            low_tag | (selector << 2u));
        const auto scale_offset = state_scales[
            static_cast<std::size_t>(sq_row) * 8 + tag];
        const auto palette = state_palettes[
            static_cast<std::size_t>(sq_row) * 8 + tag];
        const float scale = std::ldexp(
            1.0f,
            static_cast<int>(matrix_scale_base) + scale_offset - 127);
        for (int lane = 0; lane < 32; ++lane) {
          const int column = block * 32 + lane;
          const auto symbol = symbols[static_cast<std::size_t>(column)];
          std::uint8_t nibble = 0;
          if (bits == 1) {
            nibble = mfq::metal::kMxfp4Sq1PaletteNibbles[
                static_cast<std::size_t>(palette) * 2 + symbol];
          } else if (bits == 2) {
            nibble = mfq::metal::kMxfp4Sq2PaletteNibbles[
                static_cast<std::size_t>(palette) * 4 + symbol];
          } else {
            nibble = mfq::metal::kMxfp4Sq3PaletteNibbles[
                static_cast<std::size_t>(palette) * 8 + symbol];
          }
          dense[static_cast<std::size_t>(row) * columns + column] =
              fp4(nibble) * scale;
        }
      }
      ++sq_row;
    }
    symbol_rows.push_back(pack_bits(symbols, bits));
  }

  std::vector<std::uint8_t> q_descriptors(row_q.size());
  std::transform(
      row_q.begin(), row_q.end(), q_descriptors.begin(),
      [](std::uint8_t q) { return static_cast<std::uint8_t>(q - 1); });
  auto packed_q = pack_bits(q_descriptors, 2);
  packed_q.resize((packed_q.size() + 3) & ~std::size_t{3}, 0);

  std::vector<std::uint8_t> blob{'S', 'Q', 'V', '2'};
  append<std::uint8_t>(blob, 2);
  append<std::uint8_t>(blob, matrix_scale_base);
  append<std::uint16_t>(blob, 0);
  append<std::uint64_t>(blob, rows);
  append<std::uint64_t>(blob, columns);
  append_bytes(blob, packed_q);
  for (const auto &symbols : symbol_rows) {
    append_bytes(blob, symbols);
  }
  append_bytes(blob, pack_bits(selectors, 1));
  append_bytes(blob, pack_bits(state_scales, 2));
  append_bytes(blob, pack_bits(state_palettes, 5));
  append_bytes(blob, native_scales);
  return {
      rows,
      columns,
      matrix_scale_base,
      blob.size(),
      std::move(blob),
      std::move(dense),
  };
}

void test_adaptive_q1234_dequant_and_small_m() {
  using namespace mlx::core;
  // More than 16 block-32 groups exercises the full 32-lane reduction tree
  // selected for heterogeneous q.
  const auto fixture = make_adaptive_fixture(12, 640);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  require(weight.format_version() == 2,
          "adaptive MXFP4-SQ format version mismatch");
  require(weight.descriptor().format_version == 2,
          "adaptive MXFP4-SQ descriptor version mismatch");
  require(weight.bits() == 0,
          "adaptive MXFP4-SQ must not expose one uniform q");
  require(std::fabs(weight.descriptor().distribution_entropy - 2.0) < 1e-12,
          "adaptive MXFP4-SQ q entropy mismatch");

  auto decoded = contiguous(weight.dequantize(float32));
  eval(decoded);
  for (std::size_t index = 0; index < fixture.dense.size(); ++index) {
    require(decoded.data<float>()[index] == fixture.dense[index],
            "adaptive MXFP4-SQ dequantization mismatch");
  }

  for (int rows = 1; rows <= 6; ++rows) {
    std::vector<float> input_values(
        static_cast<std::size_t>(rows) * fixture.columns);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
      input_values[index] = static_cast<float>(
          static_cast<int>((index * 11 + 7) % 37) - 18) / 128.0f;
    }
    auto input = astype(
        array(input_values.begin(), Shape{rows, fixture.columns}), float16);
    auto actual = contiguous(astype(weight.matmul(input), float32));
    eval(actual);
    float maximum_difference = 0.0f;
    for (int input_row = 0; input_row < rows; ++input_row) {
      for (int output_row = 0; output_row < fixture.rows; ++output_row) {
        float expected = 0.0f;
        for (int column = 0; column < fixture.columns; ++column) {
          expected += input_values[
                          static_cast<std::size_t>(input_row) * fixture.columns
                          + column]
              * fixture.dense[
                  static_cast<std::size_t>(output_row) * fixture.columns
                  + column];
        }
        maximum_difference = std::max(
            maximum_difference,
            std::fabs(actual.data<float>()[
                          static_cast<std::size_t>(input_row) * fixture.rows
                          + output_row]
                      - expected));
      }
    }
    require(maximum_difference < 1.5e-2f,
            "adaptive MXFP4-SQ M=" + std::to_string(rows)
                + " mismatch: max_abs="
                + std::to_string(maximum_difference));
  }

  const std::vector<std::int64_t> selected_rows{3, 0, 2, 1, 7};
  const auto selected_blob = mfq::sq::select_rows(
      fixture.blob, selected_rows);
  const auto selected_layout = mfq::sq::parse(
      selected_blob.data(), selected_blob.size());
  const auto selected_metadata = mfq::sq::row_metadata(
      selected_blob.data(), selected_layout);
  require(selected_layout.version == 2,
          "selected MXFP4-SQ rows must use the adaptive wire format");
  require(selected_metadata.q ==
              std::vector<std::uint8_t>({4, 1, 3, 2, 4}),
          "selected MXFP4-SQ q descriptors changed order");
  const auto selected_weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(selected_blob);
  auto selected_dense = contiguous(selected_weight.dequantize(float32));
  eval(selected_dense);
  for (std::size_t destination = 0;
       destination < selected_rows.size();
       ++destination) {
    const auto source_row = static_cast<std::size_t>(
        selected_rows[destination]);
    for (int column = 0; column < fixture.columns; ++column) {
      require(
          selected_dense.data<float>()[
              destination * static_cast<std::size_t>(fixture.columns)
              + static_cast<std::size_t>(column)] ==
              fixture.dense[
                  source_row * static_cast<std::size_t>(fixture.columns)
                  + static_cast<std::size_t>(column)],
          "MXFP4-SQ selected-row payload changed decoded values");
    }
  }
}

void test_projection_group_m1_through_m6() {
  using namespace mlx::core;
  const auto first_fixture = make_adaptive_fixture(8, 128);
  const auto second_fixture = make_fixture(7, 128);
  const auto first =
      mfq::metal::MlxMxfp4SqWeight::from_blob(first_fixture.blob);
  const auto second =
      mfq::metal::MlxMxfp4SqWeight::from_blob(second_fixture.blob);
  const mfq::metal::MlxGroupedLinear grouped({&first, &second});
  require(grouped.uses_zero_copy_storage(),
          "MXFP4-SQ projection group copied packed storage");
  require(grouped.copied_packed_nbytes() == 0,
          "MXFP4-SQ projection group reported a packed copy");
  require(grouped.supports_single_row_projection_fusion(),
          "MXFP4-SQ projection group missed decode fusion");

  for (int rows = 1; rows <= 6; ++rows) {
    std::vector<float> values(static_cast<std::size_t>(rows) * 128);
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<float>(
          static_cast<int>((index * 17 + 5) % 43) - 21) / 256.0f;
    }
    auto input = astype(array(values.begin(), Shape{rows, 128}), float16);
    auto actual = grouped(input);
    std::array<array, 2> expected{
        first.matmul(input),
        second.matmul(input),
    };
    require(actual.size() == expected.size(),
            "MXFP4-SQ projection group output count mismatch");
    for (std::size_t projection = 0; projection < actual.size(); ++projection) {
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
              "MXFP4-SQ projection group M=" + std::to_string(rows) +
                  " mismatch: max_abs=" + std::to_string(maximum));
    }
  }
}

void test_dequantize() {
  using namespace mlx::core;
  const auto fixture = make_fixture(7, 160);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  require(weight.input_size() == fixture.columns,
          "MXFP4-SQ2 input size mismatch");
  require(weight.output_size() == fixture.rows,
          "MXFP4-SQ2 output size mismatch");
  require(weight.matrix_scale_base() == fixture.matrix_scale_base,
          "MXFP4-SQ2 scale base mismatch");
  require(weight.packed_nbytes() == fixture.payload_nbytes,
          "MXFP4-SQ2 payload byte count mismatch");
  require(weight.descriptor().format_version == 1,
          "legacy MXFP4-SQ descriptor version mismatch");

  auto fp32 = contiguous(weight.dequantize(float32));
  auto fp16 = contiguous(astype(weight.dequantize(float16), float32));
  eval(fp32, fp16);
  for (std::size_t index = 0; index < fixture.dense.size(); ++index) {
    require(fp32.data<float>()[index] == fixture.dense[index],
            "MXFP4-SQ2 FP32 dequantization mismatch");
    require(fp16.data<float>()[index] == fixture.dense[index],
            "MXFP4-SQ2 FP16 dequantization mismatch");
  }
}

void test_fused_gemv() {
  using namespace mlx::core;
  const auto fixture = make_fixture(19, 160);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  std::vector<float> input_values(static_cast<std::size_t>(fixture.columns));
  for (int column = 0; column < fixture.columns; ++column) {
    input_values[static_cast<std::size_t>(column)] =
        static_cast<float>((column % 17) - 8) / 64.0f;
  }
  std::vector<float> expected_values(static_cast<std::size_t>(fixture.rows));
  for (int row = 0; row < fixture.rows; ++row) {
    for (int column = 0; column < fixture.columns; ++column) {
      expected_values[static_cast<std::size_t>(row)] +=
          input_values[static_cast<std::size_t>(column)] *
          fixture
              .dense[static_cast<std::size_t>(row) * fixture.columns + column];
    }
  }

  auto input_fp32 = array(input_values.begin(), Shape{1, fixture.columns});
  auto input_fp16 = astype(input_fp32, float16);
  auto output_fp16 = contiguous(astype(weight.matmul(input_fp16), float32));
  auto output_fp32 = contiguous(weight.matmul(input_fp32));
  eval(output_fp16, output_fp32);
  require(output_fp16.shape() == Shape{1, fixture.rows},
          "MXFP4-SQ2 fused GEMV shape mismatch");
  require(output_fp32.dtype() == float32,
          "MXFP4-SQ2 FP32 GEMV dtype mismatch");
  float fp16_maximum_difference = 0.0f;
  float fp32_maximum_difference = 0.0f;
  for (int row = 0; row < fixture.rows; ++row) {
    const float expected = expected_values[static_cast<std::size_t>(row)];
    fp16_maximum_difference =
        std::max(fp16_maximum_difference,
                 std::fabs(output_fp16.data<float>()[row] - expected));
    fp32_maximum_difference =
        std::max(fp32_maximum_difference,
                 std::fabs(output_fp32.data<float>()[row] - expected));
  }
  require(fp16_maximum_difference < 2e-3f,
          "MXFP4-SQ2 FP16 fused GEMV mismatch: max_abs=" +
              std::to_string(fp16_maximum_difference));
  require(fp32_maximum_difference < 2e-3f,
          "MXFP4-SQ2 FP32 fused GEMV mismatch: max_abs=" +
              std::to_string(fp32_maximum_difference));
}

void test_multirow_buckets() {
  using namespace mlx::core;
  const auto fixture = make_fixture(19, 96);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  constexpr std::array<int, 9> row_counts{2, 6, 7, 16, 17, 32, 33, 64, 65};
  for (const int rows : row_counts) {
    std::vector<float> input_values(static_cast<std::size_t>(rows) *
                                    fixture.columns);
    for (int input_row = 0; input_row < rows; ++input_row) {
      for (int column = 0; column < fixture.columns; ++column) {
        const auto index =
            static_cast<std::size_t>(input_row) * fixture.columns + column;
        input_values[index] =
            static_cast<float>((input_row * 7 + column * 3 + 5) % 29 - 14) /
            128.0f;
      }
    }
    const Shape input_shape =
        rows == 6 ? Shape{2, 3, fixture.columns} : Shape{rows, fixture.columns};
    const Shape expected_shape =
        rows == 6 ? Shape{2, 3, fixture.rows} : Shape{rows, fixture.rows};
    auto input = astype(array(input_values.begin(), input_shape), float16);
    auto output = contiguous(astype(weight.matmul(input), float32));
    eval(output);
    require(output.shape() == expected_shape,
            "MXFP4-SQ2 multirow shape mismatch at M=" +
                std::to_string(rows));
    float maximum_difference = 0.0f;
    for (int input_row = 0; input_row < rows; ++input_row) {
      for (int output_row = 0; output_row < fixture.rows; ++output_row) {
        float expected = 0.0f;
        for (int column = 0; column < fixture.columns; ++column) {
          expected += input_values[static_cast<std::size_t>(input_row) *
                                       fixture.columns +
                                   column] *
                      fixture.dense[static_cast<std::size_t>(output_row) *
                                        fixture.columns +
                                    column];
        }
        maximum_difference = std::max(
            maximum_difference,
            std::fabs(
                output.data<float>()[input_row * fixture.rows + output_row] -
                expected));
      }
    }
    require(maximum_difference < 8e-3f,
            "MXFP4-SQ2 multirow mismatch at M=" + std::to_string(rows) +
                ": max_abs=" + std::to_string(maximum_difference));
  }
}

void test_fp32_multirow_contract() {
  using namespace mlx::core;
  const auto fixture = make_fixture(9, 96);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  constexpr int rows = 7;
  std::vector<float> values(static_cast<std::size_t>(rows) * fixture.columns);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] =
        static_cast<float>(static_cast<int>(index % 23) - 11) / 128.0f;
  }
  auto input = array(values.begin(), Shape{rows, fixture.columns});
  auto actual = contiguous(weight.matmul(input));
  auto expected =
      contiguous(matmul(input, transpose(weight.dequantize(float32))));
  eval(actual, expected);
  require(actual.dtype() == float32,
          "MXFP4-SQ2 FP32 multirow dtype mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require(actual.data<float>()[index] == expected.data<float>()[index],
            "MXFP4-SQ2 FP32 multirow contract mismatch");
  }
}

void test_backward_input() {
  using namespace mlx::core;
  const auto fixture = make_fixture(19, 96);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  constexpr int rows = 6;
  std::vector<float> values(static_cast<std::size_t>(rows) * fixture.rows);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] =
        static_cast<float>(static_cast<int>(index % 31) - 15) / 128.0f;
  }
  auto gradient = array(values.begin(), Shape{2, 3, fixture.rows});
  for (const auto dtype : {float16, float32}) {
    auto source = astype(gradient, dtype);
    auto actual = contiguous(astype(weight.backward_input(source), float32));
    auto expected = contiguous(astype(matmul(
        reshape(source, Shape{rows, fixture.rows}),
        weight.dequantize(dtype)), float32));
    eval(actual, expected);
    require(actual.shape() == Shape{2, 3, fixture.columns},
            "MXFP4-SQ2 backward-input shape mismatch");
    float maximum_difference = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      maximum_difference = std::max(
          maximum_difference,
          std::fabs(actual.data<float>()[index] - expected.data<float>()[index]));
    }
    require(maximum_difference < 8e-3f,
            "MXFP4-SQ2 backward-input mismatch: max_abs=" +
                std::to_string(maximum_difference));
  }
}

void test_adaptive_q1234_backward_input() {
  using namespace mlx::core;
  const auto fixture = make_adaptive_fixture(68, 640);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  for (const int rows : {1, 2, 4, 6, 8}) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * fixture.rows);
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<float>(
          static_cast<int>((index * 17 + 5) % 43) - 21) / 256.0f;
    }
    auto source = astype(
        array(values.begin(), Shape{rows, fixture.rows}), float16);
    auto actual = contiguous(astype(weight.backward_input(source), float32));
    auto expected = contiguous(astype(matmul(
        source,
        weight.dequantize(float16)), float32));
    eval(actual, expected);
    require(actual.shape() == Shape{rows, fixture.columns},
            "adaptive MXFP4-SQ backward-input shape mismatch");
    float maximum_difference = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      maximum_difference = std::max(
          maximum_difference,
          std::fabs(
              actual.data<float>()[index] - expected.data<float>()[index]));
    }
    require(maximum_difference < 2.5e-2f,
            "adaptive MXFP4-SQ backward M=" + std::to_string(rows)
                + " mismatch: max_abs="
                + std::to_string(maximum_difference));
  }
}

void test_routed_cohort_uses_shared_kernel() {
  using namespace mlx::core;
  constexpr int experts = 3;
  constexpr int output = 5;
  constexpr int tokens = 2;
  constexpr int routes = 3;
  const auto fixture = make_fixture(experts * output, 96);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  std::vector<float> input_values(
      static_cast<std::size_t>(tokens) * fixture.columns);
  for (std::size_t index = 0; index < input_values.size(); ++index) {
    input_values[index] =
        static_cast<float>(static_cast<int>((index * 7 + 3) % 31) - 15) /
        128.0f;
  }
  // Global 0/1/2 map to cohort-local 2/0/1.  Global 3 is not in this
  // cohort and must produce zeros without another dispatch implementation.
  const std::vector<std::int32_t> ids{0, 1, 3, 2, 0, -1};
  const std::vector<std::int32_t> map{2, 0, 1, -1};
  auto input = astype(
      array(input_values.begin(), Shape{tokens, fixture.columns}), float16);
  auto id_array = array(ids.begin(), Shape{tokens, routes});
  auto map_array = array(map.begin(), Shape{static_cast<int>(map.size())});
  auto actual = contiguous(astype(
      weight.routed_matmul(input, id_array, map_array, output), float32));
  eval(actual);
  require(actual.shape() == Shape{tokens, routes, output},
          "MXFP4-SQ2 routed output shape mismatch");
  float maximum_difference = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    for (int route = 0; route < routes; ++route) {
      const int global = ids[token * routes + route];
      const int local = global >= 0 && global < static_cast<int>(map.size())
                            ? map[static_cast<std::size_t>(global)]
                            : -1;
      for (int row = 0; row < output; ++row) {
        float expected = 0.0f;
        if (local >= 0) {
          for (int column = 0; column < fixture.columns; ++column) {
            expected += input_values[
                            static_cast<std::size_t>(token) * fixture.columns +
                            column] *
                        fixture.dense[
                            (static_cast<std::size_t>(local) * output + row) *
                                fixture.columns +
                            column];
          }
        }
        const auto index =
            (static_cast<std::size_t>(token) * routes + route) * output + row;
        maximum_difference = std::max(
            maximum_difference,
            std::fabs(actual.data<float>()[index] - expected));
      }
    }
  }
  require(maximum_difference < 8e-3f,
          "MXFP4-SQ2 routed shared-kernel mismatch: max_abs=" +
              std::to_string(maximum_difference));
}

void test_adaptive_routed_cohort_uses_shared_kernel() {
  using namespace mlx::core;
  constexpr int experts = 3;
  constexpr int output = 4;
  constexpr int tokens = 2;
  constexpr int routes = 3;
  const auto fixture = make_adaptive_fixture(experts * output, 96);
  const auto weight =
      mfq::metal::MlxMxfp4SqWeight::from_blob(fixture.blob);
  std::vector<float> input_values(
      static_cast<std::size_t>(tokens) * fixture.columns);
  for (std::size_t index = 0; index < input_values.size(); ++index) {
    input_values[index] = static_cast<float>(
        static_cast<int>((index * 7 + 3) % 31) - 15) / 128.0f;
  }
  const std::vector<std::int32_t> ids{0, 1, 3, 2, 0, -1};
  const std::vector<std::int32_t> map{2, 0, 1, -1};
  auto input = astype(
      array(input_values.begin(), Shape{tokens, fixture.columns}), float16);
  auto actual = contiguous(astype(
      weight.routed_matmul(
          input,
          array(ids.begin(), Shape{tokens, routes}),
          array(map.begin(), Shape{static_cast<int>(map.size())}),
          output),
      float32));
  eval(actual);
  float maximum_difference = 0.0f;
  for (int token = 0; token < tokens; ++token) {
    for (int route = 0; route < routes; ++route) {
      const int expert = ids[token * routes + route];
      const int local = expert >= 0 && expert < static_cast<int>(map.size())
          ? map[static_cast<std::size_t>(expert)]
          : -1;
      for (int row = 0; row < output; ++row) {
        float expected = 0.0f;
        if (local >= 0) {
          for (int column = 0; column < fixture.columns; ++column) {
            expected += input_values[
                            static_cast<std::size_t>(token) * fixture.columns
                            + column]
                * fixture.dense[
                    (static_cast<std::size_t>(local) * output + row)
                        * fixture.columns
                    + column];
          }
        }
        const auto index =
            (static_cast<std::size_t>(token) * routes + route) * output + row;
        maximum_difference = std::max(
            maximum_difference,
            std::fabs(actual.data<float>()[index] - expected));
      }
    }
  }
  require(maximum_difference < 1.5e-2f,
          "adaptive MXFP4-SQ routed mismatch: max_abs=" +
              std::to_string(maximum_difference));
}

void test_blob_validation() {
  auto fixture = make_fixture(3, 64);
  const auto require_rejected = [](const std::vector<std::uint8_t> &blob,
                                   const std::string &message) {
    bool rejected = false;
    try {
      (void)mfq::metal::MlxMxfp4SqWeight::from_blob(blob);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, message);
  };

  auto truncated = fixture.blob;
  truncated.pop_back();
  require_rejected(truncated, "MXFP4-SQ2 truncated payload was accepted");

  auto invalid_magic = fixture.blob;
  invalid_magic[0] = 'X';
  require_rejected(invalid_magic, "MXFP4-SQ2 invalid magic was accepted");

  auto invalid_version = fixture.blob;
  invalid_version[4] = 2;
  require_rejected(invalid_version,
                   "MXFP4-SQ2 invalid version was accepted");

  auto invalid_reserved = fixture.blob;
  invalid_reserved[6] = 1;
  require_rejected(invalid_reserved,
                   "MXFP4-SQ2 nonzero reserved field was accepted");

  auto zero_rows = fixture.blob;
  std::fill(zero_rows.begin() + 8, zero_rows.begin() + 16, 0);
  require_rejected(zero_rows, "MXFP4-SQ2 zero row count was accepted");

  auto invalid_width = fixture.blob;
  std::fill(invalid_width.begin() + 16, invalid_width.begin() + 24, 0);
  invalid_width[16] = 33;
  require_rejected(invalid_width,
                   "MXFP4-SQ2 non-block-aligned width was accepted");

  auto trailing = fixture.blob;
  trailing.push_back(0);
  require_rejected(trailing, "MXFP4-SQ2 trailing payload was accepted");

  auto invalid_scale = fixture.blob;
  invalid_scale[5] = 252;
  require_rejected(invalid_scale,
                   "MXFP4-SQ2 invalid E8M0 base was accepted");

  auto adaptive = make_adaptive_fixture(4, 64);
  const auto adaptive_layout = mfq::sq::parse(
      adaptive.blob.data(), adaptive.blob.size());
  adaptive.blob[adaptive_layout.native_scales] = 255;
  require_rejected(
      adaptive.blob,
      "adaptive MXFP4-SQ E8M0 NaN native scale was accepted");
}

} // namespace

int main() {
  try {
    test_dequantize();
    test_adaptive_q1234_dequant_and_small_m();
    test_projection_group_m1_through_m6();
    test_fused_gemv();
    test_multirow_buckets();
    test_fp32_multirow_contract();
    test_backward_input();
    test_adaptive_q1234_backward_input();
    test_routed_cohort_uses_shared_kernel();
    test_adaptive_routed_cohort_uses_shared_kernel();
    test_blob_validation();
    std::cout << "MFQ native-MXFP4 SQ2 Metal dequant/GEMV/MMQ passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
