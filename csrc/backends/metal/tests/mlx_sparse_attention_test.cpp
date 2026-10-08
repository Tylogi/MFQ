#include "mlx_sparse_attention.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

mlx::core::array patterned_half(
    std::size_t count,
    const mlx::core::Shape& shape,
    int multiplier,
    float scale) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>(
            static_cast<int>((index * multiplier) % 251) - 125) * scale;
    }
    return mlx::core::astype(
        mlx::core::array(
            values.begin(),
            shape,
            mlx::core::float32),
        mlx::core::float16);
}

void test_block_gqa_matches_expanded_reference(
    float query_scale, float key_scale, int batch = 1, int queries = 3,
    int keys = 20, mlx::core::Dtype dtype = mlx::core::float16) {
    using namespace mlx::core;
    constexpr int heads = 24;
    constexpr int kv_heads = 2;
    constexpr int dimension = 256;
    const int query_offset = keys - queries;
    constexpr int block_size = 4;
    constexpr int selected_count = 3;

    auto query = astype(patterned_half(
        batch * heads * queries * dimension,
        Shape{batch, heads, queries, dimension},
        37,
        query_scale), dtype);
    auto key = astype(patterned_half(
        batch * kv_heads * keys * dimension,
        Shape{batch, kv_heads, keys, dimension},
        53,
        key_scale), dtype);
    auto value = astype(patterned_half(
        batch * kv_heads * keys * dimension,
        Shape{batch, kv_heads, keys, dimension},
        71,
        1.0f / 463.0f), dtype);

    std::vector<std::int32_t> block_values(batch * queries * selected_count);
    for (int row = 0; row < batch * queries; ++row) {
        block_values[row * 3] = 0;
        block_values[row * 3 + 1] = queries >= 32 && row % 3 == 0 ? 0 : 2;
        block_values[row * 3 + 2] = queries < 32 ? 3 :
            (row % 3 == 0 ? -1 : (row % 3 == 1 ? keys : (keys - 1) / 4));
    }
    const array blocks(
        block_values.begin(),
        Shape{batch, queries, selected_count},
        int32);

    constexpr int expanded = selected_count * block_size + block_size - 1;
    std::vector<std::int32_t> indices(
        batch * queries * expanded,
        -1);
    for (int token = 0; token < batch * queries; ++token) {
        const int absolute = query_offset + token % queries;
        const int complete = (absolute + 1) / block_size;
        const int valid_blocks = std::min(selected_count, complete);
        int output = token * expanded;
        for (int selected = 0; selected < valid_blocks; ++selected) {
            const int block = block_values[token * selected_count + selected];
            for (int offset = 0; offset < block_size; ++offset) {
                const int position = block * block_size + offset;
                indices[output++] = block >= 0 && position < keys && position <= absolute
                    ? position : -1;
            }
        }
        output = token * expanded + selected_count * block_size;
        for (int position = complete * block_size;
             position <= absolute && output < (token + 1) * expanded;
             ++position) {
            indices[output++] = position;
        }
    }
    const array expanded_indices(
        indices.begin(),
        Shape{batch, queries, expanded},
        int32);

    auto actual = mfq::metal::mlx_sparse_block_gqa_attention(
        query,
        key,
        value,
        blocks,
        query_offset,
        block_size);
    auto normalized = mfq::metal::mlx_sparse_block_gqa_attention(
        astype(query, float32),
        astype(key, float32),
        astype(value, float32),
        astype(blocks, int64),
        query_offset,
        block_size);
    actual = astype(actual, float32);
    normalized = astype(normalized, float32);
    auto query32 = astype(query, float32);
    auto key32 = astype(key, float32);
    auto value32 = astype(value, float32);
    eval(actual, normalized, query32, key32, value32, expanded_indices);

    std::vector<float> expected(actual.size(), 0.0f);
    const auto* q = query32.data<float>();
    const auto* k = key32.data<float>();
    const auto* v = value32.data<float>();
    for (int token = 0; token < batch * queries; ++token) {
        const int batch_index = token / queries;
        const int query_index = token % queries;
        for (int head = 0; head < heads; ++head) {
            const int kv_head = head / (heads / kv_heads);
            std::vector<float> scores(expanded, -INFINITY);
            float maximum_score = -INFINITY;
            for (int selected = 0; selected < expanded; ++selected) {
                const int row = indices[token * expanded + selected];
                if (row < 0 || row >= keys) continue;
                float dot = 0.0f;
                for (int d = 0; d < dimension; ++d) {
                    dot += q[((batch_index * heads + head) * queries + query_index) * dimension + d]
                        * k[((batch_index * kv_heads + kv_head) * keys + row) * dimension + d];
                }
                scores[selected] = dot / std::sqrt(float(dimension));
                maximum_score = std::max(maximum_score, scores[selected]);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                if (!std::isfinite(score)) continue;
                score = std::exp(score - maximum_score);
                denominator += score;
            }
            for (int selected = 0; selected < expanded; ++selected) {
                const int row = indices[token * expanded + selected];
                if (row < 0 || row >= keys || scores[selected] <= 0.0f) continue;
                const float probability = scores[selected] / denominator;
                for (int d = 0; d < dimension; ++d) {
                    expected[(token * heads + head) * dimension + d] +=
                        probability
                        * v[((batch_index * kv_heads + kv_head) * keys + row) * dimension + d];
                }
            }
        }
    }

    float maximum = 0.0f;
    float normalization_maximum = 0.0f;
    double squared = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual.data<float>()[index])) {
            throw std::runtime_error("selected-block sparse GQA produced nonfinite output");
        }
        const float error = std::fabs(
            actual.data<float>()[index] - expected[index]);
        maximum = std::max(maximum, error);
        squared += static_cast<double>(error) * error;
        normalization_maximum = std::max(
            normalization_maximum,
            std::fabs(
                actual.data<float>()[index]
                - normalized.data<float>()[index]));
    }
    const float rms = static_cast<float>(
        std::sqrt(squared / static_cast<double>(actual.size())));
    if (maximum > 8e-3f || rms > 1e-3f ||
        (dtype == float16 && normalization_maximum != 0.0f)) {
        throw std::runtime_error(
            "selected-block sparse GQA mismatch: max="
            + std::to_string(maximum)
            + " rms=" + std::to_string(rms)
            + " normalized=" + std::to_string(normalization_maximum));
    }
    std::cout << "selected-block sparse GQA max=" << maximum
              << " rms=" << rms << "\n";
}

void test_block_gqa_decode(int block_size) {
    using namespace mlx::core;
    for (const auto dtype : {float16, bfloat16}) {
        for (const int keys : {1, 2, 3, 4, 16, 17, 18, 19, 2048, 2051, 4096}) {
            for (const bool invalid_blocks : {false, true}) {
                const int count = keys < 2048 ? 3 : 512;
                auto query = astype(patterned_half(
                    24 * 256, Shape{1, 24, 1, 256}, 37, 1.0f / 511.0f), dtype);
                auto key = astype(patterned_half(
                    2 * keys * 256, Shape{1, 2, keys, 256}, 53, 1.0f / 487.0f), dtype);
                auto value = astype(patterned_half(
                    2 * keys * 256, Shape{1, 2, keys, 256}, 71, 1.0f / 463.0f), dtype);
                std::vector<std::int32_t> ids(count);
                for (int i = 0; i < count; ++i) ids[i] = i * (keys / block_size) / count;
                if (invalid_blocks) {
                    ids[0] = -1;
                    ids[count - 1] = keys / block_size + 1;
                }
                array blocks(ids.begin(), Shape{1, 1, count});
                setenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER", "0", 1);
                auto expected = mfq::metal::mlx_sparse_block_gqa_attention(
                    query, key, value, blocks, keys - 1, block_size);
                setenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER", "1", 1);
                auto actual = mfq::metal::mlx_sparse_block_gqa_attention(
                    query, key, value, blocks, keys - 1, block_size);
                if (actual.shape() != expected.shape() || actual.dtype() != dtype) {
                    throw std::runtime_error("sparse GQA decode shape/dtype mismatch");
                }
                actual = astype(contiguous(actual), float32);
                expected = astype(contiguous(expected), float32);
                eval(actual, expected);
                const auto* a = actual.data<float>();
                const auto* e = expected.data<float>();
                float maximum = 0.0f;
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    maximum = std::max(maximum, std::fabs(a[i] - e[i]));
                    if (!std::isfinite(a[i]) || maximum > 2e-3f) {
                        throw std::runtime_error(
                            "sparse GQA decode mismatch: " + std::to_string(maximum));
                    }
                }
            }
        }
    }
    unsetenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER");
}

void test_block_gqa_small_m(int block_size, int inner_stride) {
    using namespace mlx::core;
    for (const auto dtype : {float16, bfloat16}) {
        for (const int batch : {1, 2}) {
            for (int rows = 1; rows <= 6; ++rows) {
                for (const int keys : {3, 4, 17, 18, 19, 2048, 2051, 4096}) {
                    if (rows > keys) continue;
                    for (const int offset : {0, keys - rows}) {
                        const int count = keys < 2048 ? 3 : 512;
                        auto query = astype(patterned_half(
                            batch * 24 * rows * 256, Shape{batch, 24, rows, 256},
                            37, 1.0f / 511.0f), dtype);
                        auto key = slice(astype(patterned_half(
                            batch * 2 * (keys + 3) * 256 * inner_stride,
                            Shape{batch, 2, keys + 3, 256 * inner_stride},
                            53, 1.0f / 487.0f), dtype),
                            Shape{0, 0, 0, 0}, Shape{batch, 2, keys, 256 * inner_stride},
                            Shape{1, 1, 1, inner_stride});
                        auto value = slice(transpose(astype(patterned_half(
                            batch * (keys + 5) * 2 * 256,
                            Shape{batch, keys + 5, 2, 256},
                            71, 1.0f / 463.0f), dtype), {0, 2, 1, 3}),
                            Shape{0, 0, 0, 0}, Shape{batch, 2, keys, 256});
                        eval(key, value);
                        std::vector<std::int32_t> ids(batch * rows * count);
                        for (int row = 0; row < batch * rows; ++row) {
                            for (int i = 0; i < count; ++i) {
                                ids[row * count + i] =
                                    ((i + row) % count) * (keys / block_size) / count;
                            }
                            ids[row * count] = 0;
                            ids[row * count + count / 2] = -1;
                            ids[(row + 1) * count - 1] = keys / block_size + 1;
                        }
                        array blocks(ids.begin(), Shape{batch, rows, count});
                        setenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER", "0", 1);
                        auto expected = mfq::metal::mlx_sparse_block_gqa_attention(
                            query, key, value, blocks, offset, block_size);
                        setenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER", "1", 1);
                        auto actual = mfq::metal::mlx_sparse_block_gqa_attention(
                            query, key, value, blocks, offset, block_size);
                        if (actual.shape() != Shape{batch, rows, 24, 256} ||
                            actual.dtype() != dtype) {
                            throw std::runtime_error("sparse GQA small-M shape/dtype mismatch");
                        }
                        actual = astype(contiguous(actual), float32);
                        expected = astype(contiguous(expected), float32);
                        eval(actual, expected);
                        float maximum = 0.0f;
                        for (std::size_t i = 0; i < actual.size(); ++i) {
                            const float error = std::fabs(
                                actual.data<float>()[i] - expected.data<float>()[i]);
                            maximum = std::max(maximum, error);
                            if (!std::isfinite(actual.data<float>()[i]) ||
                                !std::isfinite(expected.data<float>()[i]) || maximum > 2e-3f) {
                                throw std::runtime_error("sparse GQA small-M mismatch: "
                                    + std::to_string(batch) + "/" + std::to_string(rows)
                                    + "/" + std::to_string(keys) + "/" + std::to_string(offset)
                                    + " max=" + std::to_string(maximum));
                            }
                        }
                    }
                }
            }
        }
    }
    unsetenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER");
}

void benchmark_block_gqa_decode() {
    using namespace mlx::core;
    for (const int keys : {2051, 4096, 16387}) {
        constexpr int count = 512;
        auto query = patterned_half(
            24 * 256, Shape{1, 24, 1, 256}, 37, 1.0f / 511.0f);
        auto key = patterned_half(
            2 * keys * 256, Shape{1, 2, keys, 256}, 53, 1.0f / 487.0f);
        auto value = patterned_half(
            2 * keys * 256, Shape{1, 2, keys, 256}, 71, 1.0f / 463.0f);
        std::vector<std::int32_t> ids(count);
        for (int i = 0; i < count; ++i) ids[i] = i * (keys / 4) / count;
        array blocks(ids.begin(), Shape{1, 1, count});
        eval(query, key, value, blocks);
        auto execute = [&] {
            auto output = mfq::metal::mlx_sparse_block_gqa_attention(
                query, key, value, blocks, keys - 1, 4);
            output.eval();
        };
        for (int warm = 0; warm < 24; ++warm) {
            setenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER", warm % 2 ? "1" : "0", 1);
            execute();
        }
        for (int pair = 0; pair < 8; ++pair) {
            double times[2]{};
            for (int arm = 0; arm < 2; ++arm) {
                const int gather = (pair + arm) % 2;
                setenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER", gather ? "1" : "0", 1);
                const auto start = std::chrono::steady_clock::now();
                for (int rep = 0; rep < 50; ++rep) execute();
                times[gather] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count() / 50.0;
            }
            std::cout << "sparse_gqa keys=" << keys << " pair=" << pair
                      << " direct_ms=" << times[0] << " gather_ms=" << times[1] << '\n';
        }
    }
    unsetenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER");
}

void test_indexer_topk(int width, int rows, int heads, int batch, int offset, bool tied) {
    using namespace mlx::core;
    std::vector<float> values(batch * rows * heads * width);
    std::uint32_t seed = 123456789;
    for (auto& value : values) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        value = tied ? 0.0f : float(int(seed & 65535u) - 32768) / 32768.0f;
    }
    auto input = array(values.begin(), Shape{batch, rows, heads, width});
    if (batch == 2) {
        input = contiguous(transpose(input, {0, 2, 1, 3}));
        input = transpose(input, {0, 2, 1, 3});
    }
    auto actual = mfq::metal::mlx_sparse_indexer_topk512(input, offset, 4);
    eval(actual);
    const auto* indices = actual.data<std::int32_t>();
    for (int row = 0; row < batch * rows; ++row) {
        const int active = std::min(width, (offset + row % rows + 1) / 4);
        std::vector<int> expected(512);
        if (active <= 512) {
            std::iota(expected.begin(), expected.end(), 0);
        } else {
            std::vector<float> scores(active, 0.0f);
            for (int index = 0; index < active; ++index)
                for (int head = 0; head < heads; ++head)
                    scores[index] += std::max(values[(row * heads + head) * width + index], 0.0f);
            std::vector<int> ranked(active);
            std::iota(ranked.begin(), ranked.end(), 0);
            std::partial_sort(ranked.begin(), ranked.begin() + 512, ranked.end(),
                [&](int left, int right) {
                    return scores[left] == scores[right]
                        ? left > right : scores[left] > scores[right];
                });
            std::copy_n(ranked.begin(), 512, expected.begin());
            std::sort(expected.begin(), expected.end());
        }
        for (int index = 0; index < 512; ++index) {
            if (indices[row * 512 + index] != expected[index])
                throw std::runtime_error("sparse indexer top512 disagrees with CPU oracle");
        }
    }
}

void test_indexer_topk_rejects_invalid_geometry() {
    using namespace mlx::core;
    for (const Shape& shape : {Shape{1, 0, 4, 750}, Shape{1, 1, 4, 511},
            Shape{1, 1, 4, 32769}, Shape{1, 4, 750}}) {
        bool rejected = false;
        try {
            mfq::metal::mlx_sparse_indexer_topk512(zeros(shape), 2998, 4);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("sparse indexer accepted invalid shape");
    }
    for (auto params : {std::pair{-1, 4}, std::pair{2998, 0},
            std::pair{std::numeric_limits<int>::max(), 4}}) {
        bool rejected = false;
        try {
            mfq::metal::mlx_sparse_indexer_topk512(
                zeros(Shape{1, 2, 4, 750}), params.first, params.second);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("sparse indexer accepted invalid visibility");
    }
}

void test_qsa_prefill(int tokens, int keys, mlx::core::Dtype dtype, bool tied, int view = 0) {
    using namespace mlx::core;
    const int offset = keys - tokens;
    auto query = astype(patterned_half(24ULL * tokens * 256,
        Shape{1, 24, tokens, 256}, 37, 1.0f / 511.0f), dtype);
    if (view == 1) {
        query = concatenate({zeros(Shape{1, 24, 1, 256}, dtype), query}, 2);
        query = slice(query, Shape{0, 0, 1, 0}, Shape{1, 24, tokens + 1, 256});
    }
    if (view == 2) {
        query = reshape(stack({query, zeros(query.shape(), dtype)}, -1),
            Shape{1, 24, tokens, 512});
        query = slice(query, Shape{0, 0, 0, 0},
            Shape{1, 24, tokens, 512}, Shape{1, 1, 1, 2});
    }
    auto key = astype(patterned_half(2ULL * keys * 256,
        Shape{1, 2, keys, 256}, 53, 1.0f / 487.0f), dtype);
    auto value = astype(patterned_half(2ULL * keys * 256,
        Shape{1, 2, keys, 256}, 71, 1.0f / 463.0f), dtype);
    auto index = tied ? zeros(Shape{1, tokens, 4, 128}, dtype)
        : astype(patterned_half(4ULL * tokens * 128,
            Shape{1, tokens, 4, 128}, 31, 1.0f / 512.0f), dtype);
    auto pooled = astype(patterned_half((keys / 4ULL) * 128,
        Shape{1, keys / 4, 128}, 29, 1.0f / 512.0f), dtype);
    auto actual = mfq::metal::mlx_qsa_prefill_attention(
        query, key, value, index, astype(pooled, float32), offset);
    if (view != 0) {
        auto copied = mfq::metal::mlx_qsa_prefill_attention(
            contiguous(query), key, value, index, astype(pooled, float32), offset);
        auto matches = all(equal(actual, copied));
        auto difference = max(abs(astype(actual, float32) - astype(copied, float32)));
        eval(matches, difference);
        if (!matches.item<bool>()) throw std::runtime_error("QSA prefill view/copy differs: view=" +
            std::to_string(view) + " max=" + std::to_string(difference.item<float>()));
    }
    int dense_rows = std::min(tokens, std::max(0, 2051 - offset));
    if (tokens - dense_rows == 1) --dense_rows;
    std::vector<array> parts;
    if (dense_rows > 0) {
        const int end = offset + dense_rows;
        parts.push_back(transpose(fast::scaled_dot_product_attention(
            slice(query, Shape{0, 0, 0, 0}, Shape{1, 24, dense_rows, 256}),
            slice(key, Shape{0, 0, 0, 0}, Shape{1, 2, end, 256}),
            slice(value, Shape{0, 0, 0, 0}, Shape{1, 2, end, 256}),
            1.0f / std::sqrt(256.0f), "causal"), {0, 2, 1, 3}));
    }
    if (dense_rows < tokens) {
        const int rows = tokens - dense_rows;
        auto iq = slice(index, Shape{0, dense_rows, 0, 0}, index.shape());
        auto scores = reshape(matmul(reshape(astype(iq, float32),
            Shape{1, rows * 4, 128}), transpose(astype(pooled, float32), {0, 2, 1})),
            Shape{1, rows, 4, keys / 4});
        auto blocks = mfq::metal::mlx_sparse_indexer_topk512(scores, offset + dense_rows, 4);
        parts.push_back(mfq::metal::mlx_sparse_block_gqa_attention(
            slice(query, Shape{0, 0, dense_rows, 0}, Shape{1, 24, tokens, 256}),
            key, value, blocks, offset + dense_rows, 4));
    }
    auto reference = parts.size() == 1 ? parts.front() : concatenate(parts, 1);
    auto error = abs(astype(actual, float32) - astype(reference, float32));
    auto maximum = max(error);
    auto rms = sqrt(mean(square(error)));
    auto finite = all(isfinite(actual));
    eval(maximum, rms, finite);
    const float tolerance = dtype == bfloat16 ? 0.002f : 0.00025f;
    if (actual.shape() != Shape{1, tokens, 24, 256} ||
        !finite.item<bool>() || maximum.item<float>() > tolerance ||
        rms.item<float>() > tolerance / 8.0f) {
        throw std::runtime_error("QSA prefill disagrees with unfused reference: " +
            std::to_string(tokens) + "/" + std::to_string(keys) +
            " view=" + std::to_string(view) +
            " max=" + std::to_string(maximum.item<float>()) +
            " rms=" + std::to_string(rms.item<float>()));
    }
}

void test_qsa_prefill_rejects_invalid_geometry() {
    using namespace mlx::core;
    auto query = zeros(Shape{1, 24, 32, 256}, float16);
    auto key = zeros(Shape{1, 2, 2083, 256}, float16);
    auto index = zeros(Shape{1, 32, 4, 128}, float16);
    auto pooled = zeros(Shape{1, 520, 128}, float32);
    for (int offset : {-1, 2050, std::numeric_limits<int>::max()}) {
        bool rejected = false;
        try {
            mfq::metal::mlx_qsa_prefill_attention(query, key, key, index, pooled, offset);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("QSA prefill accepted an invalid cache position");
    }
    for (const Shape& shape : {Shape{1, 32, 3, 128}, Shape{1, 32, 4, 127}}) {
        bool rejected = false;
        try {
            mfq::metal::mlx_qsa_prefill_attention(
                query, key, key, zeros(shape, float16), pooled, 2051);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("QSA prefill accepted invalid index geometry");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        mlx::core::set_default_device(mlx::core::Device::gpu);
        for (const auto dtype : {mlx::core::float16, mlx::core::bfloat16}) {
            test_qsa_prefill(64, 64, dtype, false);
            test_qsa_prefill(64, 2051, dtype, false);
            test_qsa_prefill(64, 2052, dtype, false);
            test_qsa_prefill(64, 2059, dtype, false);
            test_qsa_prefill(64, 2082, dtype, false);
            test_qsa_prefill(65, 3073, dtype, false);
            test_qsa_prefill(65, 3073, dtype, true);
            test_qsa_prefill(65, 3073, dtype, false, 1);
            test_qsa_prefill(65, 3073, dtype, false, 2);
            test_qsa_prefill(4096, 4096, dtype, false);
            test_qsa_prefill(8192, 8192, dtype, false);
        }
        test_qsa_prefill(2049, 32769, mlx::core::float16, false);
        test_qsa_prefill(1025, 65537, mlx::core::float16, false);
        test_qsa_prefill_rejects_invalid_geometry();
        if (argc == 2 && std::string(argv[1]) == "--prefill-only") return 0;
        for (int rows = 1; rows <= 6; ++rows) {
            for (int width : {513, 750, 2048, 8192, 16384, 32768}) {
                test_indexer_topk(width, rows, 4, 1, width * 4 - rows, false);
            }
            test_indexer_topk(750, rows, 7, 2, 750 * 4 - rows, false);
            test_indexer_topk(750, rows, 16, 1, 2998, true);
            test_indexer_topk(513, rows, 4, 1, 511 * 4 - 1, false);
        }
        test_indexer_topk(1024, 2048, 4, 1, 2048, false);
        test_indexer_topk(2051, 65, 7, 2, 8192 - 65, false);
        test_indexer_topk(750, 33, 4, 1, 2998, true);
        test_indexer_topk_rejects_invalid_geometry();
        test_block_gqa_matches_expanded_reference(1.0f / 511.0f, 1.0f / 487.0f);
        test_block_gqa_matches_expanded_reference(1.0f / 64.0f, 1.0f / 64.0f);
        test_block_gqa_matches_expanded_reference(1.0f / 128.0f, 1.0f / 256.0f);
        for (const auto dtype : {mlx::core::float16, mlx::core::bfloat16}) {
            test_block_gqa_matches_expanded_reference(1.0f / 64.0f, 1.0f / 64.0f,
                1, 32, 33, dtype);
            test_block_gqa_matches_expanded_reference(1.0f / 511.0f, 1.0f / 487.0f,
                2, 65, 67, dtype);
        }
        for (int block_size : {2, 4, 8}) {
            test_block_gqa_decode(block_size);
            test_block_gqa_small_m(block_size, 1);
            test_block_gqa_small_m(block_size, 2);
        }
        if (argc == 2 && std::string(argv[1]) == "--benchmark-decode") {
            benchmark_block_gqa_decode();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
