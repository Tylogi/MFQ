#include "mlx_nint.h"
#include "mlx_nint_rows.h"
#include "mlx_qwen4_causal_lm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

template <typename T>
void append(std::vector<std::uint8_t>& bytes, T value) {
    const auto* start = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), start, start + sizeof(value));
}

void pack(std::vector<std::uint8_t>& blob, const std::vector<std::uint8_t>& values, int bits) {
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

Fixture fixture(int rows, int width, int gs, int nominal_k, bool adaptive = true) {
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

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename F> void require_rejected(F function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid NINT row input was accepted");
}

void test_mixed_rows(int width, int gs, int nominal_k, bool adaptive) {
    const auto data = fixture(529, width, gs, nominal_k, adaptive);
    const mfq::metal::MlxMappedNintRows table(data.blob);
    std::vector<std::int64_t> ids{528, 257, 255, 256, 0, 3, 528};
    for (int row = 0; row < 32; ++row) ids.push_back(row);
    mfq::metal::MlxNintRowBatch batch;
    for (auto row : ids) table.append_row(row, batch);
    require(batch.source_bytes_read() < data.blob.size(), "row lookup read a full table");
    for (auto dtype : {mlx::core::float16, mlx::core::float32}) {
        auto actual = mlx::core::astype(batch.decode(dtype), mlx::core::float32);
        actual.eval();
        for (std::size_t index = 0; index < ids.size(); ++index) {
            for (int column = 0; column < width; ++column) {
                auto expected = data.reference[ids[index] * width + column];
                if (dtype == mlx::core::float16) {
                    expected = static_cast<float>(mlx::core::float16_t(expected));
                }
                require(actual.data<float>()[index * width + column] == expected,
                    "mapped row kernel differs from independent scalar reference");
            }
        }
    }
    // Compare with the existing fully resident native decoder as a second oracle.
    auto resident = mfq::metal::MlxNintWeight::from_blob(data.blob).dequantize(mlx::core::float32);
    resident.eval();
    for (std::size_t index = 0; index < data.reference.size(); ++index) {
        require(resident.data<float>()[index] == data.reference[index], "reference disagrees with native decoder");
    }
    require_rejected([&] { table.append_row(-1, batch); });
    require_rejected([&] { table.append_row(529, batch); });
}

void test_cross_table_batch() {
    const auto first = fixture(513, 160, 24, 2);
    const auto second = fixture(513, 160, 7, 6);
    const mfq::metal::MlxMappedNintRows a(first.blob), b(second.blob);
    mfq::metal::MlxNintRowBatch batch;
    a.append_row(512, batch);
    b.append_row(257, batch);
    a.append_row(0, batch);
    auto values = batch.decode(mlx::core::float32);
    values.eval();
    for (int column = 0; column < 160; ++column) {
        require(values.data<float>()[column] == first.reference[512 * 160 + column], "cross-table first row");
        require(values.data<float>()[160 + column] == second.reference[257 * 160 + column], "cross-table second row");
        require(values.data<float>()[320 + column] == first.reference[column], "cross-table last row");
    }
    const auto wrong = fixture(10, 53, 5, 6);
    const mfq::metal::MlxMappedNintRows wrong_width(wrong.blob);
    require_rejected([&] { wrong_width.append_row(0, batch); });
}

void test_fractional_and_subnormal_anchors() {
    auto data = fixture(32, 160, 24, 6);
    const std::uint16_t scales[]{0x0001, 0x0355, 0x2355, 0x3555, 0xb855};
    const std::uint16_t minima[]{0x0001, 0x0255, 0x2155, 0x3555, 0xb455};
    const auto half_value = [](std::uint16_t bits) {
        mlx::core::float16_t half;
        std::memcpy(&half, &bits, sizeof(bits));
        return static_cast<float>(half);
    };
    for (int row = 0; row < 32; ++row) {
        std::memcpy(data.blob.data() + 42 + row * 2, &scales[row % 5], 2);
        std::memcpy(data.blob.data() + 42 + 64 + row * 2, &minima[row % 5], 2);
    }
    const mfq::metal::MlxMappedNintRows table(data.blob);
    mfq::metal::MlxNintRowBatch batch;
    for (int row = 0; row < 32; ++row) table.append_row(row, batch);
    auto values = batch.decode(mlx::core::float32);
    values.eval();
    for (int row = 0; row < 32; ++row) {
        const int qbits = (row / 4 + row * 3) % 8 + 1;
        const int kbits = 5 + (row * 5 + 1) % 4;
        for (int column = 0; column < 160; ++column) {
            const int group = column / 24;
            const int q = (row * 17 + column * 11 + 1) % (1 << qbits);
            const int s = (row * 3 + group * 5 + 1) % (1 << kbits);
            const int m = (row + group * 7) % (1 << kbits);
            const float scale = half_value(scales[row % 5]) * s;
            const float minimum = half_value(minima[row % 5]) * m;
            const float expected = scale * q - minimum;
            const float actual = values.data<float>()[row * 160 + column];
            require(std::abs(expected - actual) <= 2e-7f * std::max(1.0f, std::abs(expected)),
                "fractional/subnormal NINT anchors changed decode arithmetic");
        }
    }
}

void test_invalid_blobs() {
    const auto data = fixture(9, 53, 5, 6);
    for (auto length : {std::size_t(0), std::size_t(17), data.blob.size() - 1}) {
        require_rejected([&] { mfq::metal::MlxMappedNintRows table(std::span(data.blob).first(length)); });
    }
    auto bad = data.blob;
    bad.push_back(0);
    require_rejected([&] { mfq::metal::MlxMappedNintRows table(bad); });
    bad = data.blob;
    bad[1] = 8; // Existing nonempty k=9 cohort is invalid.
    require_rejected([&] { mfq::metal::MlxMappedNintRows table(bad); });
    bad = data.blob;
    bad[6] = 1; // axis must be zero.
    require_rejected([&] { mfq::metal::MlxMappedNintRows table(bad); });
    bad = data.blob;
    bad[42] = 0; bad[43] = 0x7c; // Nonfinite row-0 scale; checked on row access.
    const mfq::metal::MlxMappedNintRows table(bad);
    mfq::metal::MlxNintRowBatch batch;
    require_rejected([&] { table.append_row(0, batch); });
    require_rejected([&] { batch.decode(); });
}

void test_production_geometry_page_guards() {
    constexpr int rows = 2500012, width = 160, gs = 24, groups = 7;
    const auto kb = (std::size_t(rows) * 2 + 7) / 8;
    const auto qb = (std::size_t(rows) * 3 + 7) / 8;
    const auto metadata = (std::size_t(rows) * groups * 5 + 7) / 8;
    const auto qbytes = std::size_t(rows) * groups * gs / 8;
    std::vector<std::uint8_t> header;
    append<std::uint8_t>(header, 0x84); append<std::uint8_t>(header, 6);
    append<std::int32_t>(header, gs); append<std::int32_t>(header, 0);
    append<std::int32_t>(header, width); append<std::uint32_t>(header, 2);
    append<std::int64_t>(header, rows); append<std::int64_t>(header, width);
    append<std::uint32_t>(header, rows); append<std::uint32_t>(header, groups);
    const auto scale = header.size() + std::size_t(rows) * 4 + kb;
    const auto minimum = scale + metadata;
    const auto quantized = minimum + metadata + qb;
    const auto size = quantized + qbytes;
    auto* mapping = static_cast<std::uint8_t*>(mmap(nullptr, size,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    require(mapping != MAP_FAILED, "production mmap failed");
    std::memcpy(mapping, header.data(), header.size());
    // All q=1/k=5 selectors are zero. Entire packed streams are inaccessible,
    // proving construction cannot copy/decode the table. Enable only the pages
    // touched by the final row before requesting it.
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const auto protect = [&](std::size_t start, std::size_t end, int mode) {
        start = (start + page - 1) / page * page;
        end = end / page * page;
        if (end > start) require(mprotect(mapping + start, end - start, mode) == 0, "mprotect failed");
    };
    protect(scale, minimum + metadata, PROT_NONE);
    protect(quantized, size, PROT_NONE);
    {
        const mfq::metal::MlxMappedNintRows table({mapping, size});
        require(table.index_nbytes() < std::size_t(rows) / 4, "rank index is not compact");
        for (const auto offset : {scale + metadata - 8, minimum + metadata - 8, quantized + qbytes - 32}) {
            const auto start = offset / page * page;
            const auto length = std::min(page * 2, (size + page - 1) / page * page - start);
            require(mprotect(mapping + start, length, PROT_READ | PROT_WRITE) == 0, "row page enable failed");
        }
        mfq::metal::MlxNintRowBatch batch;
        table.append_row(rows - 1, batch);
        require(batch.source_bytes_read() < 128, "selected production row copied too much");
        auto values = batch.decode(mlx::core::float32);
        values.eval();
        for (int column = 0; column < width; ++column) require(values.data<float>()[column] == 0, "guarded row result");
    }
    require(munmap(mapping, size) == 0, "production munmap failed");
}

void test_qwen_ple_graph(const char* fp8_path, const char* nint_path) {
    const mfq::metal::MfqContainer fp8_container(fp8_path), nint_container(nint_path);
    auto fp8 = mfq::metal::MlxQwen4CausalLm::load(fp8_container, 64);
    auto nint = mfq::metal::MlxQwen4CausalLm::load(nint_container, 64);
    require(nint.kv_cache_bytes() == 0 && nint.kv_cache_contexts() == 0,
        "new Qwen model must not report live KV before reset/forward");
    require(nint.ssd_ple_payload_bytes() > 0, "PLE backing payload not reported");
    require(nint.ssd_expert_payload_bytes() == 0, "full resident experts reported as SSD streamed");
    const auto compare = [&](const std::vector<std::int32_t>& tokens, int batch, bool cache) {
        const mlx::core::array ids(tokens.begin(),
            mlx::core::Shape{batch, static_cast<int>(tokens.size()) / batch});
        auto expected = mlx::core::astype(fp8.forward(ids, cache), mlx::core::float32);
        auto actual = mlx::core::astype(nint.forward(ids, cache), mlx::core::float32);
        mlx::core::eval(expected, actual);
        require(expected.shape() == actual.shape(), "PLE graph output shape");
        for (std::size_t index = 0; index < expected.size(); ++index) {
            require(std::isfinite(actual.data<float>()[index]), "PLE graph returned nonfinite logits");
            require(expected.data<float>()[index] == actual.data<float>()[index], "NINTv2/FP8 matched PLE graph logits differ");
        }
    };
    compare({1, 2, 7, 3, 4, 1}, 1, false);
    fp8.reset_cache(); nint.reset_cache();
    require(nint.kv_cache_bytes() > 0 && nint.kv_cache_contexts() == 1,
        "Qwen context cache allocations not reported");
    compare({1, 2, 7}, 1, true);
    compare({3}, 1, true);
    compare({4, 1}, 1, true);
    fp8.reset_cache(2); nint.reset_cache(2);
    compare({1, 7, 3, 2, 4, 1}, 2, true);
    compare({4, 5}, 2, true);
    require(nint.kv_cache_contexts() == 2, "Qwen batched contexts not reported");
    const auto active_before = mlx::core::get_active_memory();
    const auto cache_bytes = nint.kv_cache_bytes();
    require(cache_bytes > 0 && mlx::core::get_active_memory() == active_before,
        "resource telemetry must not allocate/evaluate device arrays");
    nint.clear_cache();
    require(nint.kv_cache_bytes() == 0 && nint.kv_cache_contexts() == 0,
        "cleared Qwen context resources not reported as zero");
    std::cout << "matched FP8/NINTv2 PLE graph prefill, decode, EOS and batch logits passed\n";
}

void test_qwen_prefix_graph(const char* fp8_path, const char* nint_path) {
    using namespace mfq::metal;
    const MfqContainer container(nint_path), baseline_container(fp8_path);
    auto model = MlxQwen4CausalLm::load(container, 64);
    MlxSamplingParams sampling;
    sampling.temperature = 0;
    std::vector<std::int64_t> tokens{1, 2, 3};
    std::vector<MlxQwen4TextSessionState> checkpoints;
    MlxPrefixCacheHooks hooks{4, tokens.size(), true, [&](std::size_t position) {
        require(position <= tokens.size(), "Flash checkpoint includes rejected drafts");
        checkpoints.push_back(model.capture_text_session_state(
            {tokens.begin(), tokens.begin() + position}));
    }};
    model.generate({1, 2, 3}, sampling, 8, [&](std::int64_t token) {
        tokens.push_back(token);
        return true;
    }, {}, {}, 3, 2, hooks);
    require(checkpoints.size() >= 3 && checkpoints.front().tokens.size() == 3 &&
        checkpoints.back().tokens.size() == tokens.size() - 1,
        "Flash input/output/tail checkpoints missing");
    for (const auto& state : checkpoints) {
        auto resumed = MlxQwen4CausalLm::load(container, 64);
        resumed.restore_text_session_state(state);
        auto prompt = state.tokens;
        prompt.push_back(5);
        std::vector<std::int64_t> actual, expected;
        std::size_t prefilled = 0;
        auto resume_hooks = hooks;
        resume_hooks.input_tokens = prompt.size();
        resume_hooks.capture = [](std::size_t) {};
        resumed.generate(prompt, sampling, 4, [&](std::int64_t token) {
            actual.push_back(token);
            return token != resumed.config().eos_token_id;
        }, [&](std::size_t count, double) { prefilled = count; }, {}, state.tokens.size(), 2, resume_hooks);
        auto cold = MlxQwen4CausalLm::load(baseline_container, 64);
        auto cold_sampling = sampling;
        cold_sampling.enable_mtp = false;
        cold.generate(prompt, cold_sampling, 4, [&](std::int64_t token) {
            expected.push_back(token);
            return token != cold.config().eos_token_id;
        });
        if (prefilled != 1 || actual != expected) {
            std::cerr << "checkpoint=" << state.tokens.size() << " prefilled=" << prefilled << " actual=";
            for (const auto token : actual) std::cerr << token << ',';
            std::cerr << " expected=";
            for (const auto token : expected) std::cerr << token << ',';
            std::cerr << '\n';
            throw std::runtime_error("Flash restored checkpoint changed target tokens");
        }
    }
    std::cout << "Flash-Next GDN/QSA/PLE/MTP prefix checkpoints passed\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--qwen-prefix") {
            test_qwen_prefix_graph(argv[2], argv[3]);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--qwen-ple") {
            test_qwen_ple_graph(argv[2], argv[3]);
            return 0;
        }
        for (int k : {2, 6}) {
            test_mixed_rows(53, 5, k, true);
            test_mixed_rows(160, 24, k, true);
        }
        test_mixed_rows(53, 5, 6, false);
        test_cross_table_batch();
        test_fractional_and_subnormal_anchors();
        test_invalid_blobs();
        test_production_geometry_page_guards();
        std::cout << "mixed q/k mapped NINT rows, page guards and negative tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
