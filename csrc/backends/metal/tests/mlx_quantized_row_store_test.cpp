#include "storage/mlx_quantized_row_store.h"
#include "ops/mlx_nint.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void append_little(std::string& output, T value) {
    using Unsigned = std::make_unsigned_t<T>;
    const auto bits = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        output.push_back(static_cast<char>(bits >> (index * 8)));
    }
}

void append_string(std::string& output, const std::string& value) {
    append_little<std::uint32_t>(
        output, static_cast<std::uint32_t>(value.size()));
    output.append(value);
}

std::string row_table_payload() {
    constexpr std::uint64_t rows = 4;
    constexpr std::uint64_t width = 32;
    std::string result{"MXT1", 4};
    result.push_back(1);
    result.push_back(8);
    result.push_back(0);
    result.push_back(0);
    for (const auto value : {rows, width, rows, width, rows, width / 32}) {
        append_little<std::uint64_t>(result, value);
    }
    for (std::uint64_t row = 0; row < rows; ++row) {
        result.append(
            static_cast<std::size_t>(width),
            static_cast<char>((row & 1u) == 0 ? 0x38 : 0x40));
    }
    result.append(static_cast<std::size_t>(rows), static_cast<char>(127));
    return result;
}

void write_fixture(const std::filesystem::path& path,
                   const std::string& dtype = "MXFP8",
                   const std::string& payload = row_table_payload()) {
    const std::string name = "table.weight";
    std::string bytes{"MFQ1", 4};
    append_little<std::uint32_t>(bytes, 2);
    append_string(bytes, "unit-test");
    append_little<std::uint32_t>(bytes, 0);
    append_little<std::uint32_t>(bytes, 1);
    append_string(bytes, name);
    append_string(bytes, dtype);
    append_little<std::uint64_t>(bytes, payload.size());
    bytes.append(payload);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(stream), "cannot write MXFP8 row fixture");
}

std::string nint_payload(bool adaptive) {
    constexpr int rows = 4, width = 32, group = 16, groups = 2;
    const std::array<int, rows> qs{1, 2, 4, 8}, ks{1, 2, 3, 4};
    std::string result;
    result.push_back(static_cast<char>(4 | (adaptive ? 128 : 0)));
    result.push_back(static_cast<char>(adaptive ? 2 : 4));
    append_little<std::int32_t>(result, group);
    append_little<std::int32_t>(result, 0);
    append_little<std::int32_t>(result, width);
    append_little<std::uint32_t>(result, 2);
    append_little<std::int64_t>(result, rows);
    append_little<std::int64_t>(result, width);
    append_little<std::uint32_t>(result, rows);
    append_little<std::uint32_t>(result, groups);
    for (int row = 0; row < rows; ++row) append_little<std::uint16_t>(result, 0x3800);
    for (int row = 0; row < rows; ++row) append_little<std::uint16_t>(result, 0x3400);
    const auto pack = [&](const std::vector<unsigned>& values, int bits) {
        std::string bytes((values.size() * bits + 7) / 8, '\0');
        for (std::size_t i = 0; i < values.size(); ++i) {
            for (int bit = 0; bit < bits; ++bit) {
                if (values[i] & (1u << bit)) {
                    const auto offset = i * bits + bit;
                    bytes[offset / 8] = static_cast<char>(
                        static_cast<unsigned char>(bytes[offset / 8]) | (1u << (offset % 8)));
                }
            }
        }
        result += bytes;
    };
    if (adaptive) pack({0, 1, 2, 3}, 2);
    for (int bits = 1; bits <= 8; ++bits) {
        if (!adaptive && bits != 4) continue;
        for (int field = 0; field < 2; ++field) {
            std::vector<unsigned> values;
            for (int row = 0; row < rows; ++row) {
                if (adaptive && ks[row] != bits) continue;
                for (int g = 0; g < groups; ++g)
                    values.push_back(field ? g : ((g + 1) & ((1u << bits) - 1)));
            }
            pack(values, bits);
        }
    }
    if (adaptive) pack({0, 1, 3, 7}, 3);
    for (int bits = 1; bits <= 8; ++bits) {
        if (!adaptive && bits != 4) continue;
        std::vector<unsigned> values;
        for (int row = 0; row < rows; ++row) {
            if (adaptive && qs[row] != bits) continue;
            for (int column = 0; column < width; ++column)
                values.push_back((row + column) & ((1u << bits) - 1));
        }
        pack(values, bits);
    }
    return result;
}

std::vector<float> values(mlx::core::array value);

void test_nint_rows(const std::filesystem::path& path, bool adaptive) {
    const auto payload = nint_payload(adaptive);
    write_fixture(path, "NINT", payload);
    mfq::metal::MfqContainer model(path);
    const auto blob = std::span(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
    auto reference = values(mfq::metal::MlxNintWeight::from_blob(blob)
        .dequantize(mlx::core::float16));
    mfq::metal::MlxQuantizedRowStore store(model, "table.weight", 4, 32, 2, 2);
    const std::array<std::int64_t, 5> request{3, 0, 3, 1, 2};
    const auto actual = values(store.gather(request));
    for (std::size_t row = 0; row < request.size(); ++row)
        for (int column = 0; column < 32; ++column)
            require(actual[row * 32 + column] == reference[request[row] * 32 + column],
                "streamed NINT row differs from resident decoder");
    const auto stats = store.stats();
    require(stats.row_requests == 5 && stats.rows_loaded == 4 && stats.cache_hits == 1
        && stats.read_calls == 20 && stats.resident_rows == 2
        && stats.resident_payload_bytes <= stats.cache_limit_bytes,
        "NINT row-store accounting mismatch");
    const std::array<std::int64_t, 1> first{3}, second{0};
    auto a = std::async(std::launch::async, [&] { return store.prepare_rows(first); });
    auto b = std::async(std::launch::async, [&] { return store.prepare_rows(second); });
    require(values(a.get().decode()) == std::vector<float>(reference.begin() + 96, reference.end()),
        "concurrent NINT first row changed");
    require(values(b.get().decode()) == std::vector<float>(reference.begin(), reference.begin() + 32),
        "concurrent NINT second row changed");
    store.clear();
    require(store.cached_rows() == 0, "NINT row cache was not cleared");
    bool rejected = false;
    try { const std::array<std::int64_t, 1> invalid{4}; (void)store.gather(invalid); }
    catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "NINT row store accepted an invalid row");
    rejected = false;
    try { mfq::metal::MlxQuantizedRowStore wrong(model, "table.weight", 5, 32, 2, 2); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "NINT row store accepted mismatched table geometry");
}

std::vector<float> values(mlx::core::array value) {
    value = mlx::core::astype(value, mlx::core::float32);
    value.eval();
    return {value.data<float>(), value.data<float>() + value.size()};
}

void require_row(
    const std::vector<float>& source,
    std::size_t row,
    float expected) {
    const auto begin = source.begin() + static_cast<std::ptrdiff_t>(row * 32);
    require(
        std::all_of(begin, begin + 32, [expected](float value) {
            return std::fabs(value - expected) < 1e-6f;
        }),
        "decoded MXFP8 row mismatch");
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("mfq-mxfp8-row-store-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(root);
        const auto path = root / "fixture.mfq";
        write_fixture(path);
        mfq::metal::MfqContainer model(path);
        mfq::metal::MlxQuantizedRowStore store(
            model, "table.weight", 4, 32, 2, 2);

        const std::array<std::int64_t, 3> request{0, 1, 0};
        auto decoded = store.gather(request);
        require(
            decoded.shape() == mlx::core::Shape{3, 32},
            "MXFP8 gather shape mismatch");
        const auto output = values(std::move(decoded));
        require_row(output, 0, 1.0f);
        require_row(output, 1, 2.0f);
        require_row(output, 2, 1.0f);
        auto stats = store.stats();
        require(
            stats.row_requests == 3 && stats.cache_hits == 1 &&
            stats.cache_misses == 2 && stats.rows_loaded == 2 &&
            stats.bytes_read == 66 && stats.read_calls == 4 &&
            stats.resident_rows == 2 &&
            stats.resident_payload_bytes == 128 &&
            stats.cache_limit_bytes == 128,
            "MXFP8 row cache statistics mismatch");

        (void)store.gather(std::span<const std::int64_t>(request.data(), 1));
        stats = store.stats();
        require(
            stats.rows_loaded == 2 && stats.cache_hits == 2,
            "resident MXFP8 row did not hit the cache");

        store.clear();
        const std::array<std::int64_t, 1> even_request{2};
        const std::array<std::int64_t, 1> odd_request{3};
        auto even_future = std::async(std::launch::async, [&] {
            return store.prepare_rows(even_request);
        });
        auto odd_future = std::async(std::launch::async, [&] {
            return store.prepare_rows(odd_request);
        });
        require_row(values(even_future.get().decode()), 0, 1.0f);
        require_row(values(odd_future.get().decode()), 0, 2.0f);

        const auto previous_stream = mlx::core::default_stream(mlx::core::Device::gpu);
        const auto runtime_stream = mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu);
        mlx::core::set_default_stream(runtime_stream);
        test_nint_rows(root / "nint.mfq", false);
        test_nint_rows(root / "adaptive-nint.mfq", true);
        mlx::core::synchronize();
        mlx::core::set_default_stream(previous_stream);

        std::filesystem::remove_all(root);
        std::cout << "MXFP8/NINT row store passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        std::cerr << "MXFP8 row store failed: " << error.what() << '\n';
        return 1;
    }
}
