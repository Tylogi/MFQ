#pragma once
#include "mfq/nint_rows.h"
#include <cmath>
#include <cstdint>
#include <vector>
#include <utility>

namespace mfq::test {

template <typename T>
void append(std::vector<std::uint8_t>& bytes, T value) {
    const auto* start = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), start, start + sizeof(value));
}

inline void pack(std::vector<std::uint8_t>& blob, const std::vector<std::uint8_t>& values, int bits) {
    const auto start = blob.size();
    blob.resize(start + (values.size() * bits + 7) / 8, 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        for (int bit = 0; bit < bits; ++bit) {
            const auto position = index * bits + bit;
            blob[start + position / 8] |= ((values[index] >> bit) & 1u) << (position & 7);
        }
    }
}

struct Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> reference;
    int width;
};

inline Fixture fixture(int rows, int width, int gs, int nominal_k, bool adaptive = true) {
    const int groups = (width + gs - 1) / gs;
    const int padded = groups * gs;
    std::vector<std::uint8_t> qs(rows), ks(rows);
    std::vector<std::uint8_t> q(rows * padded), s(rows * groups), m(rows * groups);
    std::vector<float> reference(rows * width);
    for (int row = 0; row < rows; ++row) {
        qs[row] = adaptive ? (row / 4 + row * 3) % 8 : 3;
        ks[row] = adaptive ? (row * 5 + 1) % 4 : 1;
        const int qb = qs[row] + 1;
        const int kb = nominal_k - 1 + ks[row];
        for (int group = 0; group < groups; ++group) {
            s[row * groups + group] = (row * 3 + group * 5 + 1) % (1 << kb);
            m[row * groups + group] = (row + group * 7) % (1 << kb);
        }
        for (int column = 0; column < padded; ++column) {
            q[row * padded + column] = (row * 17 + column * 11 + 1) % (1 << qb);
            if (column < width) {
                reference[row * width + column] = (row % 2 ? 0.5f : 1.0f) *
                    s[row * groups + column / gs] * q[row * padded + column] -
                    0.25f * m[row * groups + column / gs];
            }
        }
    }
    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, adaptive ? 0x84 : 4);
    append<std::uint8_t>(blob, nominal_k);
    append<std::int32_t>(blob, gs);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, width);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, width);
    append<std::uint32_t>(blob, rows);
    append<std::uint32_t>(blob, groups);
    for (int row = 0; row < rows; ++row) append<std::uint16_t>(blob, row % 2 ? 0x3800 : 0x3c00);
    for (int row = 0; row < rows; ++row) append<std::uint16_t>(blob, 0x3400);
    if (adaptive) {
        pack(blob, ks, 2);
        for (int cohort = 0; cohort < 4; ++cohort) {
            for (const auto& values : {s, m}) {
                std::vector<std::uint8_t> selected;
                for (int row = 0; row < rows; ++row) {
                    if (ks[row] == cohort) selected.insert(selected.end(),
                        values.begin() + row * groups, values.begin() + (row + 1) * groups);
                }
                pack(blob, selected, nominal_k - 1 + cohort);
            }
        }
        pack(blob, qs, 3);
        for (int cohort = 0; cohort < 8; ++cohort) {
            std::vector<std::uint8_t> selected;
            for (int row = 0; row < rows; ++row) {
                if (qs[row] == cohort) selected.insert(selected.end(),
                    q.begin() + row * padded, q.begin() + (row + 1) * padded);
            }
            pack(blob, selected, cohort + 1);
        }
    } else {
        pack(blob, s, nominal_k);
        pack(blob, m, nominal_k);
        pack(blob, q, 4);
    }

    return {std::move(blob), std::move(reference), width};
}

inline float half_value(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    const float magnitude = exponent == 0
        ? std::ldexp(float(fraction), -24)
        : std::ldexp(float(1024 + fraction), exponent - 25);
    return bits & 0x8000u ? -magnitude : magnitude;
}

inline float row_value(const NintRowBatch& batch, std::size_t row, int column) {
    const auto* d = batch.descriptors().data() + row * 6;
    const auto bits = [&](int offset, int shift, int index, int width) {
        unsigned value = 0;
        for (int bit = 0; bit < width; ++bit) {
            const auto p = shift + index * width + bit;
            value |= ((batch.packed().at(offset + p / 8) >> (p & 7)) & 1u) << bit;
        }
        return value;
    };
    const auto layout = d[3];
    const auto group = column / d[4];
    const float q = bits(d[0], (layout >> 8) & 7, column, layout & 15);
    const float s = bits(d[1], (layout >> 12) & 7, group, (layout >> 4) & 15);
    const float m = bits(d[2], (layout >> 16) & 7, group, (layout >> 4) & 15);
    const volatile float scale = half_value(d[5] & 65535u) * s;
    const volatile float scaled = scale * q;
    const volatile float minimum = half_value(d[5] >> 16) * m;
    return scaled - minimum;
}
} // namespace mfq::test
