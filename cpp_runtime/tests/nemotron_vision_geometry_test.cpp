// RADIO cropped-position-embedding geometry: no GPU, no weights.
//
// The table is the four-tap bilinear resampling of the square position table
// that `RadioPatchEmbeddings._interpolate_position_embedding` performs by
// upsampling to max(rows, columns) and cropping, so it is checked against
// analytic weights rather than against a second copy of the formula.
#include "models/nemotron_h/vision.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using mfq::models::nemotron_h::RadioPositionInterpolation;
using mfq::models::nemotron_h::make_radio_position_interpolation;

constexpr std::int32_t kSide = 8;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

RadioPositionInterpolation grid(std::int32_t rows, std::int32_t columns) {
    const auto table = make_radio_position_interpolation(rows, columns, kSide);
    require(table.patch_count == static_cast<std::int64_t>(rows) * columns,
            "position table patch count mismatch");
    require(table.indices.size() == static_cast<std::size_t>(table.patch_count) * 4 &&
                table.weights.size() == table.indices.size(),
            "position table tap geometry mismatch");
    return table;
}

// Assert the four taps of one patch against explicit indices and weights.
void require_patch(const RadioPositionInterpolation& table, std::int32_t columns,
                   std::int32_t row, std::int32_t column,
                   const std::vector<std::int32_t>& indices,
                   const std::vector<double>& weights) {
    require(indices.size() == 4 && weights.size() == 4, "patch expectation must have 4 taps");
    const auto offset = (static_cast<std::size_t>(row) * columns + column) * 4;
    double sum = 0.0;
    for (std::size_t tap = 0; tap < 4; ++tap) {
        require(table.indices[offset + tap] == indices[tap],
                "cropped-position interpolation picked an unexpected tap");
        const auto actual = static_cast<double>(table.weights[offset + tap]);
        require(std::abs(actual - weights[tap]) < 1e-6,
                "cropped-position interpolation weight is wrong");
        require(actual >= -1e-6, "cropped-position interpolation weight is negative");
        sum += actual;
    }
    require(std::abs(sum - 1.0) < 1e-5, "cropped-position weights must sum to one");
}

void require_taps_in_range(const RadioPositionInterpolation& table) {
    for (const auto index : table.indices) {
        require(index >= 0 && index < kSide * kSide,
                "cropped-position interpolation left the position table");
    }
}

} // namespace

int main() {
    try {
        // A grid that matches the table is the identity: one tap of weight one.
        const auto identity = grid(kSide, kSide);
        require_taps_in_range(identity);
        for (std::int32_t row = 0; row < kSide; ++row) {
            for (std::int32_t column = 0; column < kSide; ++column) {
                // An exact sample keeps the four-tap layout (the neighbour tap
                // points at the next entry with zero weight) and reads the
                // table entry itself.
                const auto next_row = std::min(row + 1, kSide - 1);
                const auto next_column = std::min(column + 1, kSide - 1);
                require_patch(
                    identity, kSide, row, column,
                    {row * kSide + column, row * kSide + next_column,
                     next_row * kSide + column, next_row * kSide + next_column},
                    {1.0, 0.0, 0.0, 0.0});
            }
        }

        // 4x6 patches: the reference upsamples the 8x8 table to 6x6 and crops,
        // i.e. scale 8/6.  Patch (0,0) samples source coordinate 1/6, patch
        // (3,5) samples (4+1/6, 6+5/6).
        const auto tall = grid(4, 6);
        require_taps_in_range(tall);
        require_patch(tall, 6, 0, 0, {0, 1, kSide, kSide + 1},
                      {25.0 / 36.0, 5.0 / 36.0, 5.0 / 36.0, 1.0 / 36.0});
        require_patch(tall, 6, 3, 5, {4 * kSide + 6, 4 * kSide + 7, 5 * kSide + 6, 5 * kSide + 7},
                      {5.0 / 36.0, 25.0 / 36.0, 1.0 / 36.0, 5.0 / 36.0});

        // Transposing the grid transposes the table: rows and columns are not
        // interchangeable (the square upsample is, the crop is not).
        const auto wide = grid(6, 4);
        require_taps_in_range(wide);
        require_patch(wide, 4, 0, 0, {0, 1, kSide, kSide + 1},
                      {25.0 / 36.0, 5.0 / 36.0, 5.0 / 36.0, 1.0 / 36.0});
        require_patch(wide, 4, 5, 3, {6 * kSide + 4, 6 * kSide + 5, 7 * kSide + 4, 7 * kSide + 5},
                      {5.0 / 36.0, 1.0 / 36.0, 25.0 / 36.0, 5.0 / 36.0});

        // A single-row grid: the row axis is a pure upsample of the table.
        const auto strip = grid(1, 5);
        require_taps_in_range(strip);
        require_patch(strip, 5, 0, 0, {0, 1, kSide, kSide + 1},
                      {0.49, 0.21, 0.21, 0.09});
        require_patch(strip, 5, 0, 4, {6, 7, kSide + 6, kSide + 7},
                      {0.21, 0.49, 0.09, 0.21});

        std::cout << "RADIO cropped-position interpolation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RADIO cropped-position interpolation test failed: " << error.what()
                  << '\n';
        return 1;
    }
}
