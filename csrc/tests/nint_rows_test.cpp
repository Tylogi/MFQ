#include "nint_row_fixture.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejected(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    require(failed, "invalid range input accepted");
}
void check(int width, int gs, int nominal_k, bool adaptive) {
    const auto f = mfq::test::fixture(529, width, gs, nominal_k, adaptive);
    mfq::NintRows memory(f.blob.data(), f.blob.size());
    std::size_t bytes = 0;
    mfq::NintRows ranges(f.blob.size(), [&](std::size_t off, std::uint8_t* out, std::size_t count) {
        require(off <= f.blob.size() && count <= f.blob.size() - off, "range exceeded source");
        bytes += count;
        std::memcpy(out, f.blob.data() + off, count);
    });
    require(bytes == 42 + (adaptive ? (529 * 2 + 7) / 8 + (529 * 3 + 7) / 8 : 0),
        "initialization read payload instead of selectors");
    bytes = 0;
    mfq::NintRowBatch a, b;
    std::vector<int> ids{528, 257, 255, 256, 0, 3, 528};
    for (int row = 0; row < 32; ++row) ids.push_back(row);
    for (const auto id : ids) { memory.append_row(id, a); ranges.append_row(id, b); }
    mfq::NintRowBatch merged;
    for (const auto id : ids) {
        mfq::NintRowBatch row;
        memory.append_row(id, row);
        merged.append_batch(row);
    }
    a.validate(); b.validate();
    require(merged.packed() == a.packed() && merged.descriptors() == a.descriptors() &&
        merged.source_bytes_read() == a.source_bytes_read(), "merged row protocol differs");
    rejected([&] { merged.append_batch(merged); });
    rejected([&] { merged.append_batch(mfq::NintRowBatch{}); });
    const auto other_fixture = mfq::test::fixture(1, width + 1, gs, nominal_k, adaptive);
    mfq::NintRows other_table(other_fixture.blob.data(), other_fixture.blob.size());
    mfq::NintRowBatch other_row;
    other_table.append_row(0, other_row);
    rejected([&] { merged.append_batch(other_row); });
    require(a.packed() == b.packed() && a.descriptors() == b.descriptors(), "memory/range protocol differs");
    require(bytes == b.source_bytes_read() && bytes < f.blob.size(), "range access copied full payload");
    for (std::size_t row = 0; row < ids.size(); ++row)
        for (int col = 0; col < width; ++col)
            require(mfq::test::row_value(b, row, col) == f.reference[ids[row] * width + col], "range oracle differs");
    rejected([&] { ranges.append_row(-1, b); });
    rejected([&] { ranges.append_row(529, b); });
}
void production_ranges() {
    constexpr int rows = 2500012, width = 160, groups = 7, gs = 24;
    const auto seed = mfq::test::fixture(1, width, gs, 6);
    auto header = std::vector<std::uint8_t>(seed.blob.begin(), seed.blob.begin() + 42);
    const std::int64_t shape_rows = rows;
    const std::uint32_t nrows = rows;
    std::memcpy(header.data() + 18, &shape_rows, 8);
    std::memcpy(header.data() + 34, &nrows, 4);
    const std::size_t kb = (std::size_t(rows) * 2 + 7) / 8, qb = (std::size_t(rows) * 3 + 7) / 8;
    const std::size_t meta = (std::size_t(rows) * groups * 5 + 7) / 8;
    const std::size_t koffset = 42 + std::size_t(rows) * 4;
    const std::size_t qoffset = koffset + kb + 2 * meta;
    const std::size_t size = qoffset + qb + std::size_t(rows) * groups * gs / 8;
    bool lookup = false;
    std::size_t bytes = 0;
    mfq::NintRows table(size, [&](std::size_t off, std::uint8_t* out, std::size_t count) {
        if (!lookup) require((off < 42 && count <= 42 - off) ||
            (off == koffset && count == kb) || (off == qoffset && count == qb),
            "production initialization touched payload");
        else require(count < 128, "production lookup read bulk payload");
        std::memset(out, 0, count);
        if (off < 42) std::memcpy(out, header.data() + off, count);
        bytes += count;
    });
    require(table.index_nbytes() < std::size_t(rows), "range index exceeded one byte per row");
    lookup = true; bytes = 0;
    mfq::NintRowBatch batch;
    table.append_row(rows - 1, batch);
    batch.validate();
    require(bytes < 128 && bytes == batch.source_bytes_read(), "production demand access too large");
}
}
int main() {
    try {
        for (const auto width : {53, 160})
            for (const auto gs : {5, 24})
                for (const auto k : {2, 6}) {
                    check(width, gs, k, true);
                    check(width, gs, k, false);
                }
        production_ranges();
        std::cout << "NINT shared memory/range protocol and production demand access passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
