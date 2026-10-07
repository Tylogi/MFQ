#include "qwen4_ops.h"
#include "mlx_transformer.h"
#include "mlx_sparse_attention.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/primitives.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;

array patterned_bfloat(
    std::size_t count,
    const Shape& shape,
    int multiplier,
    float scale) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int centered =
            static_cast<int>((index * multiplier) % 257) - 128;
        values[index] = static_cast<float>(centered) * scale;
    }
    return mlx::core::astype(
        array(values.begin(), shape, mlx::core::float32),
        mlx::core::bfloat16);
}

void require_bit_exact(
    array actual,
    array expected,
    const char* name) {
    if (actual.shape() != expected.shape()) {
        throw std::runtime_error(
            std::string(name) + " shape mismatch");
    }
    if (actual.dtype() != expected.dtype() ||
        (actual.dtype() != mlx::core::bfloat16 &&
         actual.dtype() != mlx::core::float16)) {
        throw std::runtime_error(
            std::string(name) + " dtype mismatch: actual=" +
            std::to_string(static_cast<int>(actual.dtype().val())) +
            " expected=" +
            std::to_string(static_cast<int>(expected.dtype().val())));
    }
    actual = mlx::core::contiguous(actual);
    expected = mlx::core::contiguous(expected);
    mlx::core::eval(actual, expected);
    const auto* actual_bits = actual.data<std::uint16_t>();
    const auto* expected_bits = expected.data<std::uint16_t>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual_bits[index] != expected_bits[index]) {
            auto actual_float = mlx::core::astype(
                actual, mlx::core::float32);
            auto expected_float = mlx::core::astype(
                expected, mlx::core::float32);
            mlx::core::eval(actual_float, expected_float);
            throw std::runtime_error(
                std::string(name) + " changed element " +
                std::to_string(index) + ": actual=" +
                std::to_string(actual_float.data<float>()[index]) +
                " expected=" +
                std::to_string(expected_float.data<float>()[index]));
        }
    }
}

void require_close(
    array actual,
    array expected,
    float tolerance,
    const char* name) {
    if (actual.shape() != expected.shape() ||
        actual.dtype() != expected.dtype()) {
        throw std::runtime_error(
            std::string(name) + " shape/dtype mismatch");
    }
    actual = mlx::core::contiguous(mlx::core::astype(actual, mlx::core::float32));
    expected = mlx::core::contiguous(mlx::core::astype(expected, mlx::core::float32));
    mlx::core::eval(actual, expected);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float av = actual.data<float>()[index];
        const float ev = expected.data<float>()[index];
        if (!std::isfinite(av) ||
            std::fabs(av - ev) > tolerance * (1.0f + std::fabs(ev))) {
            throw std::runtime_error(
                std::string(name) + " changed element " +
                std::to_string(index) + ": actual=" +
                std::to_string(av) + " expected=" +
                std::to_string(ev));
        }
    }
}

void test_qsa_prefill_output_gate() {
    using namespace mlx::core;
    for (auto dtype : {float16, bfloat16, float32}) {
        for (int rows : {1, 32, 65}) {
            for (int layout = 0; layout < 4; ++layout) {
                constexpr int batch = 2, heads = 3, dimension = 7;
                auto attended = astype(patterned_bfloat(batch * rows * heads * dimension,
                    Shape{batch, heads, rows, dimension}, 37, 1.0f / 47.0f), dtype);
                attended = transpose(attended, {0, 2, 1, 3});
                auto packed = astype(patterned_bfloat(batch * rows * heads * 2 * dimension,
                    Shape{batch, rows, heads, 2 * dimension}, 53, 1.0f / 7.0f), dtype);
                auto gate = split(packed, 2, -1).at(1);
                if (layout == 1) {
                    gate = slice(packed, Shape{0, 0, 0, 0}, packed.shape(), Shape{1, 1, 1, 2});
                } else if (layout == 2) {
                    gate = broadcast_to(slice(gate, Shape{0, 0, 0, 0},
                        Shape{1, 1, heads, dimension}), attended.shape());
                } else if (layout == 3) {
                    gate = contiguous(gate);
                    attended = contiguous(attended);
                    eval(gate, attended);
                }
                const Shape shape{batch, rows, heads * dimension};
                auto expected = astype(reshape(astype(attended, float32), shape) *
                    sigmoid(reshape(astype(gate, float32), shape)), dtype);
                auto actual = mfq::metal::qwen4_qsa_prefill_output_gate(attended, gate);
                if (dtype == float32)
                    require_close(actual, expected, 1e-7f, "QSA FP32 prefill gate");
                else
                    require_bit_exact(actual, expected, "QSA strided prefill gate");
            }
        }
    }
    for (auto dtype : {float16, bfloat16}) {
        constexpr int heads = 24, dimension = 256;
        for (int rows : {4096, 8192}) {
            auto attended = astype(patterned_bfloat(rows * heads * dimension,
                Shape{1, rows, heads, dimension}, 37, 1.0f / 47.0f), dtype);
            auto packed = astype(patterned_bfloat(rows * heads * 2 * dimension,
                Shape{1, rows, heads, 2 * dimension}, 53, 1.0f / 7.0f), dtype);
            auto gate = split(packed, 2, -1).at(1);
            const Shape shape{1, rows, heads * dimension};
            auto expected = astype(reshape(astype(attended, float32), shape) *
                sigmoid(reshape(astype(gate, float32), shape)), dtype);
            require_bit_exact(mfq::metal::qwen4_qsa_prefill_output_gate(attended, gate),
                expected, "QSA production-width prefill gate");
        }
        const std::vector<float> gates{-INFINITY, -100.0f, -20.0f, -0.0f,
            0.0f, 1e-7f, 20.0f, 100.0f, INFINITY};
        auto gate = astype(array(gates.begin(), Shape{1, 1, 1, 9}), dtype);
        auto attended = astype(array({-2.0f, -1.0f, 0.0f, 2.0f, -2.0f,
            0.125f, 0.0f, 1.0f, -0.0f}, Shape{1, 1, 1, 9}), dtype);
        auto expected = astype(reshape(astype(attended, float32) *
            sigmoid(astype(gate, float32)), Shape{1, 1, 9}), dtype);
        require_bit_exact(mfq::metal::qwen4_qsa_prefill_output_gate(attended, gate),
            expected, "QSA saturated prefill gate");
    }
    auto valid = zeros(Shape{1, 32, 3, 7}, float16);
    for (const auto& pair : std::vector<std::pair<array, array>>{
        {valid, zeros(Shape{1, 31, 3, 7}, float16)},
        {valid, astype(valid, bfloat16)},
        {astype(valid, int32), astype(valid, int32)},
        {reshape(valid, Shape{32, 21}), reshape(valid, Shape{32, 21})},
        {zeros(Shape{1, 0, 3, 7}, float16), zeros(Shape{1, 0, 3, 7}, float16)}}) {
        bool rejected = false;
        try { mfq::metal::qwen4_qsa_prefill_output_gate(pair.first, pair.second); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("QSA prefill gate accepted invalid inputs");
    }
}

void test_qsa_block_scores() {
    using namespace mlx::core;
    for (auto dtype : {float16, bfloat16, float32}) {
        for (int batch : {1, 2}) {
            for (int rows : {1, 2, 3, 6, 32}) {
                for (int blocks : {749, 2048}) {
                    auto query = astype(patterned_bfloat(batch * rows * 8 * 128,
                        Shape{batch, rows, 8, 128}, 37, 1.0f / 511.0f), dtype);
                    auto keys = astype(patterned_bfloat(batch * blocks * 128,
                        Shape{batch, blocks, 128}, 53, 1.0f / 487.0f), dtype);
                    auto products = matmul(astype(query, float32),
                        expand_dims(transpose(astype(keys, float32), {0, 2, 1}), 1));
                    auto expected = sum(maximum(products, array(0.0f)), -2) / std::sqrt(128.0f);
                    require_close(mfq::metal::qwen4_qsa_block_scores(query, keys),
                        expected, 2e-5f, "QSA flattened indexer scores");
                }
            }
        }
    }
}

void test_qsa_dense_attention_causal_boundaries() {
    using namespace mlx::core;
    for (auto dtype : {float16, bfloat16, float32}) {
        for (int batch : {1, 2}) {
            for (int rows : {1, 2, 3, 4, 5, 6, 9}) {
                for (int offset : {0, 17, 512}) {
                    for (int future : {0, 3}) {
                        constexpr int query_heads = 24, key_heads = 2, dimension = 256;
                        const int keys = offset + rows + future;
                        const auto context = "QSA dtype=" + std::to_string(static_cast<int>(dtype.val())) +
                            " batch=" + std::to_string(batch) + " rows=" + std::to_string(rows) +
                            " offset=" + std::to_string(offset) + " future=" + std::to_string(future);
                        auto query = astype(patterned_bfloat(
                            static_cast<std::size_t>(batch) * query_heads * rows * dimension,
                            Shape{batch, rows, query_heads, dimension}, 37, 1.0f / 257.0f), dtype);
                        query = transpose(query, {0, 2, 1, 3});
                        auto key = astype(patterned_bfloat(
                            static_cast<std::size_t>(batch) * key_heads * keys * dimension,
                            Shape{batch, keys, key_heads, dimension}, 43, 1.0f / 257.0f), dtype);
                        key = transpose(key, {0, 2, 1, 3});
                        auto value = astype(patterned_bfloat(
                            static_cast<std::size_t>(batch) * key_heads * keys * dimension,
                            Shape{batch, key_heads, keys, dimension}, 53, 1.0f / 257.0f), dtype);
                        auto mask = expand_dims(expand_dims(
                            reshape(arange(0, keys, 1, int32), Shape{1, keys}) <=
                            reshape(arange(offset, offset + rows, 1, int32), Shape{rows, 1}), 0), 0);
                        auto expected = transpose(mfq::metal::scaled_dot_product_attention(
                            query, key, value, false, 1.0f / std::sqrt(float(dimension)), mask), {0, 2, 1, 3});
                        auto actual = mfq::metal::qwen4_dense_gqa_attention(query, key, value, offset);
                        require_close(actual, expected, 0.0f,
                            (context + " causal attention").c_str());
                        if (future == 0 && rows > 1) {
                            std::vector<array> serial;
                            for (int row = 0; row < rows; ++row) {
                                auto one_query = slice(query, Shape{0, 0, row, 0}, Shape{batch, query_heads, row + 1, dimension});
                                auto one_key = slice(key, Shape{0, 0, 0, 0}, Shape{batch, key_heads, offset + row + 1, dimension});
                                auto one_value = slice(value, Shape{0, 0, 0, 0}, Shape{batch, key_heads, offset + row + 1, dimension});
                                serial.push_back(mfq::metal::qwen4_dense_gqa_attention(one_query, one_key, one_value, offset + row));
                            }
                            auto serial_output = concatenate(serial, 1);
                            require_close(expected, serial_output, 5e-3f,
                                (context + " explicit-mask batch versus serial").c_str());
                            require_close(actual, serial_output, 5e-3f,
                                (context + " causal batch versus serial").c_str());
                        }
                    }
                }
            }
        }
    }
}

void test_qsa_canonical_prefill_prefix() {
    using namespace mlx::core;
    constexpr int heads = 24, kv_heads = 2, dimension = 256;
    constexpr int rows = 65, budget = 64, block_size = 4, selected_count = 16;
    for (auto dtype : {float16, bfloat16}) {
        for (int batch : {1, 2}) {
            for (int offset : {0, 17, 61}) {
                const int keys = offset + rows;
                const int dense_rows = budget - offset;
                auto query = astype(patterned_bfloat(
                    static_cast<std::size_t>(batch) * heads * rows * dimension,
                    Shape{batch, heads, rows, dimension}, 37, 1.0f / 257.0f), dtype);
                auto key = astype(patterned_bfloat(
                    static_cast<std::size_t>(batch) * kv_heads * keys * dimension,
                    Shape{batch, kv_heads, keys, dimension}, 43, 1.0f / 257.0f), dtype);
                auto value = astype(patterned_bfloat(
                    static_cast<std::size_t>(batch) * kv_heads * keys * dimension,
                    Shape{batch, kv_heads, keys, dimension}, 53, 1.0f / 257.0f), dtype);
                std::vector<std::int32_t> selected(batch * rows * selected_count);
                for (int item = 0; item < batch; ++item) {
                    for (int row = 0; row < rows; ++row) {
                        const int begin = std::max(0,
                            (offset + row + 1) / block_size - selected_count);
                        for (int block = 0; block < selected_count; ++block)
                            selected[(item * rows + row) * selected_count + block] = begin + block;
                    }
                }
                array blocks(selected.begin(), Shape{batch, rows, selected_count});
                auto expected = mfq::metal::mlx_sparse_block_gqa_attention(
                    query, key, value, blocks, offset, block_size);
                auto prefix = mfq::metal::qwen4_dense_gqa_attention(
                    slice(query, Shape{0, 0, 0, 0}, Shape{batch, heads, dense_rows, dimension}),
                    slice(key, Shape{0, 0, 0, 0}, Shape{batch, kv_heads, budget, dimension}),
                    slice(value, Shape{0, 0, 0, 0}, Shape{batch, kv_heads, budget, dimension}),
                    offset);
                auto tail = mfq::metal::mlx_sparse_block_gqa_attention(
                    slice(query, Shape{0, 0, dense_rows, 0}, query.shape()), key, value,
                    slice(blocks, Shape{0, dense_rows, 0}, blocks.shape()), budget, block_size);
                require_close(concatenate({prefix, tail}, 1), expected,
                    dtype == float16 ? 1e-3f : 1e-2f, "QSA canonical prefill prefix");
            }
        }
    }
}

void test_qsa_decode_prologue_matches_reference(
    mlx::core::Dtype dtype, int position, float theta, float eps, int rows) {
    using namespace mlx::core;
    constexpr int query_heads = 24;
    constexpr int key_heads = 2;
    constexpr int index_heads = 4;
    constexpr int head_dimension = 256;
    constexpr int index_dimension = 128;
    constexpr int rotary_dimension = 64;
    const std::vector<std::int64_t> sections{11, 11, 10};

    const auto query_gate = astype(patterned_bfloat(
        static_cast<std::size_t>(rows) * query_heads * 2 * head_dimension,
        Shape{1, rows, query_heads * 2 * head_dimension},
        37,
        1.0f / 257.0f), dtype);
    const auto key_input = astype(patterned_bfloat(
        static_cast<std::size_t>(rows) * key_heads * head_dimension,
        Shape{1, rows, key_heads * head_dimension},
        29,
        1.0f / 193.0f), dtype);
    const auto index_input = astype(patterned_bfloat(
        static_cast<std::size_t>(rows) * (index_heads + 1) * index_dimension,
        Shape{1, rows, (index_heads + 1) * index_dimension},
        43,
        1.0f / 211.0f), dtype);
    const auto query_weight = astype(patterned_bfloat(
        head_dimension, Shape{head_dimension}, 17, 1.0f / 4096.0f),
        float32);
    const auto key_weight = astype(patterned_bfloat(
        head_dimension, Shape{head_dimension}, 19, 1.0f / 4096.0f),
        float32);
    const auto index_weight = astype(patterned_bfloat(
        index_dimension, Shape{index_dimension}, 23, 1.0f / 4096.0f),
        float32);
    const mfq::metal::MlxRmsNorm query_norm(query_weight, eps, 1.0f);
    const mfq::metal::MlxRmsNorm key_norm(key_weight, eps, 1.0f);
    const mfq::metal::MlxRmsNorm index_norm(index_weight, eps, 1.0f);
    const auto positions = arange(position, position + rows, 1, int32);

    auto query_parts = split(reshape(
        query_gate,
        Shape{1, rows, query_heads, 2 * head_dimension}), 2, -1);
    auto reference_query = mfq::metal::apply_rope(
        transpose(query_norm(query_parts.at(0)), {0, 2, 1, 3}),
        positions,
        rotary_dimension,
        theta,
        sections,
        true);
    auto reference_key = mfq::metal::apply_rope(
        transpose(key_norm(reshape(
            key_input,
            Shape{1, rows, key_heads, head_dimension})), {0, 2, 1, 3}),
        positions,
        rotary_dimension,
        theta,
        sections,
        true);
    auto index_parts = split(
        index_input,
        Shape{index_heads * index_dimension},
        -1);
    auto reference_index = transpose(
        mfq::metal::apply_rope(
            transpose(index_norm(reshape(
                index_parts.at(0),
                Shape{1, rows, index_heads, index_dimension})),
                {0, 2, 1, 3}),
            positions,
            rotary_dimension,
            theta,
            sections,
            true),
        {0, 2, 1, 3});
    auto actual = mfq::metal::qwen4_qsa_decode_prologue(
        query_gate,
        key_input,
        index_input,
        query_norm.weight(),
        key_norm.weight(),
        index_norm.weight(),
        positions,
        query_heads,
        key_heads,
        index_heads,
        head_dimension,
        index_dimension,
        rotary_dimension,
        theta,
        eps);
    const auto check = [dtype](array a, array b, const char* name) {
        if (dtype == float32) {
            require_close(std::move(a), std::move(b), 1e-6f, name);
        } else {
            require_bit_exact(std::move(a), std::move(b), name);
        }
    };
    check(
        std::move(actual.query),
        std::move(reference_query),
        "Qwen QSA fused query");
    check(
        std::move(actual.output_gate),
        reshape(query_parts.at(1), Shape{1, rows, query_heads * head_dimension}),
        "Qwen QSA fused output gate");
    check(
        std::move(actual.key),
        std::move(reference_key),
        "Qwen QSA fused key");
    check(
        std::move(actual.index_query),
        std::move(reference_index),
        "Qwen QSA fused index query");
}

void benchmark_qsa_decode_prologue() {
    using namespace mlx::core;
    constexpr int query_heads = 24;
    constexpr int key_heads = 2;
    constexpr int index_heads = 4;
    constexpr int head_dimension = 256;
    constexpr int index_dimension = 128;
    constexpr int rotary_dimension = 64;
    constexpr float theta = 1.0e7f;
    constexpr float eps = 1.0e-6f;
    const std::vector<std::int64_t> sections{11, 11, 10};

    const auto query_gate = astype(patterned_bfloat(
        static_cast<std::size_t>(query_heads) * 2 * head_dimension,
        Shape{1, 1, query_heads * 2 * head_dimension},
        37,
        1.0f / 257.0f), float16);
    const auto key_input = astype(patterned_bfloat(
        static_cast<std::size_t>(key_heads) * head_dimension,
        Shape{1, 1, key_heads * head_dimension},
        29,
        1.0f / 193.0f), float16);
    const auto index_input = astype(patterned_bfloat(
        static_cast<std::size_t>(index_heads + 1) * index_dimension,
        Shape{1, 1, (index_heads + 1) * index_dimension},
        43,
        1.0f / 211.0f), float16);
    const auto query_weight = astype(patterned_bfloat(
        head_dimension, Shape{head_dimension}, 17, 1.0f / 4096.0f),
        float32);
    const auto key_weight = astype(patterned_bfloat(
        head_dimension, Shape{head_dimension}, 19, 1.0f / 4096.0f),
        float32);
    const auto index_weight = astype(patterned_bfloat(
        index_dimension, Shape{index_dimension}, 23, 1.0f / 4096.0f),
        float32);
    const mfq::metal::MlxRmsNorm query_norm(query_weight, eps, 1.0f);
    const mfq::metal::MlxRmsNorm key_norm(key_weight, eps, 1.0f);
    const mfq::metal::MlxRmsNorm index_norm(index_weight, eps, 1.0f);
    const array positions({17}, Shape{1}, int32);
    eval(query_gate, key_input, index_input, query_weight, key_weight,
         index_weight, positions);

    const auto run = [&](bool fused) {
        if (fused) {
            auto result = mfq::metal::qwen4_qsa_decode_prologue(
                query_gate,
                key_input,
                index_input,
                query_norm.weight(),
                key_norm.weight(),
                index_norm.weight(),
                positions,
                query_heads,
                key_heads,
                index_heads,
                head_dimension,
                index_dimension,
                rotary_dimension,
                theta,
                eps);
            eval(result.query, result.output_gate, result.key,
                 result.index_query);
        } else {
            auto query_parts = split(reshape(
                query_gate,
                Shape{1, 1, query_heads, 2 * head_dimension}), 2, -1);
            auto query = mfq::metal::apply_rope(
                transpose(query_norm(query_parts.at(0)), {0, 2, 1, 3}),
                positions,
                rotary_dimension,
                theta,
                sections,
                true);
            auto key = mfq::metal::apply_rope(
                transpose(key_norm(reshape(
                    key_input,
                    Shape{1, 1, key_heads, head_dimension})),
                    {0, 2, 1, 3}),
                positions,
                rotary_dimension,
                theta,
                sections,
                true);
            auto index_parts = split(
                index_input,
                Shape{index_heads * index_dimension},
                -1);
            auto index = transpose(
                mfq::metal::apply_rope(
                    transpose(index_norm(reshape(
                        index_parts.at(0),
                        Shape{1, 1, index_heads, index_dimension})),
                        {0, 2, 1, 3}),
                    positions,
                    rotary_dimension,
                    theta,
                    sections,
                    true),
                {0, 2, 1, 3});
            auto output_gate = reshape(
                query_parts.at(1),
                Shape{1, 1, query_heads * head_dimension});
            eval(query, output_gate, key, index);
        }
    };

    for (int warmup = 0; warmup < 32; ++warmup)
        run(warmup % 2 != 0);
    constexpr int iterations = 100;
    for (int sample = 0; sample < 12; ++sample) {
        double timings[2];
        for (int order = 0; order < 2; ++order) {
            const int fused = (sample + order) % 2;
            const auto started = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < iterations; ++iteration)
                run(fused != 0);
            timings[fused] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count() /
                iterations;
        }
        std::cout << "qsa_prologue_pair sample=" << sample
                  << " first=" << (sample % 2 == 0 ? "reference" : "fused")
                  << " reference_ms=" << timings[0]
                  << " fused_ms=" << timings[1] << '\n';
    }
}

array reference_grouped_rms_norm(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    auto shape = value.shape();
    const int width = shape.back();
    shape.pop_back();
    shape.push_back(width / group_size);
    shape.push_back(group_size);
    auto normalized = mlx::core::fast::rms_norm(
        mlx::core::reshape(
            mlx::core::astype(value, mlx::core::float32), shape),
        std::nullopt,
        eps);
    normalized = mlx::core::reshape(normalized, value.shape()) *
        (array(1.0f) +
         mlx::core::astype(weight, mlx::core::float32));
    return mlx::core::astype(normalized, value.dtype());
}

void test_prefill_mixing_epilogue() {
    using namespace mlx::core;
    constexpr int batch = 2, tokens = 65, hidden = 137, low_rank = 19;
    for (int streams : {3, 4}) {
        const int width = hidden * streams;
        for (auto input_dtype : {float16, bfloat16, float32}) {
            for (auto weight_dtype : {float16, bfloat16, float32}) {
                auto input = astype(patterned_bfloat(batch * tokens * width,
                    Shape{batch, tokens, width}, 37, 1.0f / 257.0f), input_dtype);
                auto norm = astype(patterned_bfloat(width, Shape{width}, 17, 1.0f / 1024.0f), float32);
                auto down = astype(patterned_bfloat(low_rank * width,
                    Shape{low_rank, width}, 29, 1.0f / 4096.0f), weight_dtype);
                auto up = astype(patterned_bfloat(width * low_rank,
                    Shape{width, low_rank}, 43, 1.0f / 4096.0f), weight_dtype);
                auto normalized = mfq::metal::qwen4_grouped_rms_norm(input, norm, hidden);
                auto low = matmul(astype(normalized, weight_dtype), transpose(down)) /
                    array(float(streams), normalized.dtype());
                low = low * sigmoid(low);
                auto mixing = sigmoid(matmul(astype(low, weight_dtype), transpose(up)));
                auto expected = mean(reshape(mixing, Shape{batch, tokens, streams, hidden}) *
                    reshape(normalized, Shape{batch, tokens, streams, hidden}), -2);
                auto actual = mfq::metal::qwen4_gated_residual_pre(input, norm, down, up,
                    std::nullopt, hidden, streams);
                require_close(actual.branch, expected,
                    expected.dtype() == bfloat16 ? 1e-2f : expected.dtype() == float16 ? 1e-3f : 2e-5f,
                    "prefill MHC mixing epilogue");
            }
        }
    }
}

void test_decode_fast_path_matches_reference() {
    // Exercise the real Flash-Next/Qwen decode geometry so the combined
    // 10240->320 and 10240->4 projection kernel is covered as well.
    constexpr int hidden = 2560;
    constexpr int streams = 4;
    constexpr int low_rank = 320;
    constexpr int width = hidden * streams;
    constexpr float eps = 1e-6f;

    const auto input = patterned_bfloat(
        width, Shape{1, 1, width}, 37, 1.0f / 53.0f);
    const auto norm = patterned_bfloat(
        width, Shape{width}, 17, 1.0f / 1024.0f);
    const auto down = patterned_bfloat(
        static_cast<std::size_t>(low_rank) * width,
        Shape{low_rank, width}, 29, 1.0f / 4096.0f);
    const auto up = patterned_bfloat(
        static_cast<std::size_t>(width) * low_rank,
        Shape{width, low_rank}, 43, 1.0f / 4096.0f);
    const auto injection = patterned_bfloat(
        static_cast<std::size_t>(streams) * width,
        Shape{streams, width}, 61, 1.0f / 4096.0f);

    auto normalized = reference_grouped_rms_norm(
        input, norm, hidden, eps);
    const array connection_count(
        static_cast<float>(streams), normalized.dtype());
    auto low = mlx::core::matmul(
        normalized, mlx::core::transpose(down)) /
        connection_count;
    low = low * mlx::core::sigmoid(low);
    auto mixing = mlx::core::sigmoid(mlx::core::matmul(
        low, mlx::core::transpose(up)));
    auto reference_branch = mlx::core::mean(
        mlx::core::reshape(
            mixing, Shape{1, 1, streams, hidden}) *
            mlx::core::reshape(
                normalized, Shape{1, 1, streams, hidden}),
        -2);
    auto reference_injection =
        array(2.0f, normalized.dtype()) * mlx::core::sigmoid(
        mlx::core::matmul(
            normalized, mlx::core::transpose(injection)) /
        connection_count);

    auto actual = mfq::metal::qwen4_gated_residual_pre(
        input,
        norm,
        down,
        up,
        std::optional<array>(injection),
        hidden,
        streams,
        eps);
    require_bit_exact(
        mfq::metal::qwen4_grouped_rms_norm(
            input, norm, hidden, eps),
        normalized,
        "Qwen grouped RMSNorm");
    if (!actual.injection) {
        throw std::runtime_error(
            "Qwen gated-HC decode path omitted injection");
    }
    require_close(
        std::move(actual.branch),
        std::move(reference_branch),
        2e-2f,
        "Qwen gated-HC branch");
    require_bit_exact(
        std::move(*actual.injection),
        std::move(reference_injection),
        "Qwen gated-HC injection");
}

void benchmark_decode_path() {
    constexpr int hidden = 2560;
    constexpr int streams = 4;
    constexpr int low_rank = 320;
    constexpr int width = hidden * streams;
    constexpr int warmups = 24;
    constexpr int samples = 7;
    constexpr int iterations = 200;
    constexpr float eps = 1e-6f;

    const auto input = patterned_bfloat(
        width, Shape{1, 1, width}, 37, 1.0f / 53.0f);
    const auto norm = patterned_bfloat(
        width, Shape{width}, 17, 1.0f / 1024.0f);
    const auto down = patterned_bfloat(
        static_cast<std::size_t>(low_rank) * width,
        Shape{low_rank, width}, 29, 1.0f / 4096.0f);
    const auto up = patterned_bfloat(
        static_cast<std::size_t>(width) * low_rank,
        Shape{width, low_rank}, 43, 1.0f / 4096.0f);
    const auto injection = patterned_bfloat(
        static_cast<std::size_t>(streams) * width,
        Shape{streams, width}, 61, 1.0f / 4096.0f);
    mlx::core::eval({input, norm, down, up, injection});

    auto run = [&] {
        auto result = mfq::metal::qwen4_gated_residual_pre(
            input,
            norm,
            down,
            up,
            std::optional<array>(injection),
            hidden,
            streams,
            eps);
        mlx::core::eval({result.branch, *result.injection});
    };
    for (int iteration = 0; iteration < warmups; ++iteration)
        run();
    std::vector<double> timings;
    timings.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration)
            run();
        timings.push_back(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started)
                .count() /
            iterations);
    }
    std::sort(timings.begin(), timings.end());
    const char* setting = std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST");
    std::cout
        << "Qwen gated-HC decode microbenchmark: path="
        << ((setting == nullptr || std::atoi(setting) != 0)
                ? "fast" : "reference")
        << " median_ms=" << timings[samples / 2]
        << " min_ms=" << timings.front()
        << " max_ms=" << timings.back() << '\n';
}

void benchmark_residual_write_norm_chain() {
    constexpr int hidden = 2560;
    constexpr int streams = 4;
    constexpr int low_rank = 320;
    constexpr int width = hidden * streams;
    const auto previous_branch = patterned_bfloat(
        hidden, Shape{1, 1, hidden}, 23, 1.0f / 53.0f);
    const auto previous_residual = mlx::core::astype(patterned_bfloat(
        width, Shape{1, 1, width}, 37, 1.0f / 71.0f),
        mlx::core::float32);
    const auto previous_injection = mlx::core::astype(patterned_bfloat(
        streams, Shape{1, 1, streams}, 17, 1.0f / 97.0f),
        mlx::core::float32);
    const auto norm = mlx::core::astype(patterned_bfloat(
        width, Shape{width}, 19, 1.0f / 1024.0f),
        mlx::core::float32);
    const auto down = patterned_bfloat(
        static_cast<std::size_t>(low_rank) * width,
        Shape{low_rank, width}, 29, 1.0f / 4096.0f);
    const auto up = patterned_bfloat(
        static_cast<std::size_t>(width) * low_rank,
        Shape{width, low_rank}, 31, 1.0f / 4096.0f);
    const auto injection = patterned_bfloat(
        static_cast<std::size_t>(streams) * width,
        Shape{streams, width}, 61, 1.0f / 4096.0f);
    mlx::core::eval(
        previous_branch,
        previous_residual,
        previous_injection,
        norm,
        down,
        up,
        injection);
    const auto run = [&](bool fused) {
        auto result = [&] {
            if (fused) {
                return mfq::metal::qwen4_gated_residual_pre_after(
                    previous_branch,
                    previous_residual,
                    previous_injection,
                    norm,
                    down,
                    up,
                    injection,
                    hidden,
                    streams,
                    1e-6f);
            }
            auto residual = mfq::metal::qwen4_gated_residual_post(
                previous_branch,
                previous_residual,
                previous_injection,
                streams);
            return mfq::metal::qwen4_gated_residual_pre(
                residual,
                norm,
                down,
                up,
                injection,
                hidden,
                streams,
                1e-6f);
        }();
        mlx::core::eval(
            result.branch,
            result.residual,
            *result.injection);
    };
    for (int warmup = 0; warmup < 24; ++warmup)
        run(warmup % 2 != 0);
    constexpr int iterations = 100;
    for (int sample = 0; sample < 10; ++sample) {
        double timings[2];
        for (int order = 0; order < 2; ++order) {
            const int fused = (sample + order) % 2;
            const auto started = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < iterations; ++iteration)
                run(fused != 0);
            timings[fused] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count() /
                iterations;
        }
        std::cout << "hc_write_norm_pair sample=" << sample
                  << " first=" << (sample % 2 == 0 ? "composed" : "fused")
                  << " composed_ms=" << timings[0]
                  << " fused_ms=" << timings[1] << '\n';
    }
}

void test_mixed_storage_decode_matches_promoted_reference() {
    constexpr int hidden = 2560, streams = 4, low_rank = 320;
    constexpr int width = hidden * streams;
    const mlx::core::Dtype dtypes[] = {
        mlx::core::float16, mlx::core::bfloat16, mlx::core::float32};
    // Cover every activation/down/up promotion, including the NINT-decoded
    // FP16 down + dense BF16 up used by EWQ, without rounding FP32 norms.
    for (int combination = 0; combination < 27; ++combination) {
        const auto dtype = dtypes[combination / 9];
        const auto down_dtype = dtypes[(combination / 3) % 3];
        const auto up_dtype = dtypes[combination % 3];
        const auto input = mlx::core::astype(patterned_bfloat(
            width, Shape{1, 1, width}, 37, 1.0f / 53.0f), dtype);
        const auto norm = mlx::core::astype(patterned_bfloat(
            width, Shape{width}, 17, 1.0f / 1024.0f), mlx::core::float32) + array(0.000013f);
        const auto down = mlx::core::astype(patterned_bfloat(
            low_rank * width, Shape{low_rank, width}, 29, 1.0f / 4096.0f), down_dtype);
        const auto up = mlx::core::astype(patterned_bfloat(width * low_rank,
            Shape{width, low_rank}, 43, 1.0f / 4096.0f), up_dtype);
        const auto inject = patterned_bfloat(streams * width,
            Shape{streams, width}, 61, 1.0f / 4096.0f);
        auto normalized = reference_grouped_rms_norm(input, norm, hidden, 1e-6f);
        const array count(4.0f, dtype);
        auto low = mlx::core::matmul(normalized, mlx::core::transpose(down)) / count;
        low = low * mlx::core::sigmoid(low);
        auto mixing = mlx::core::sigmoid(mlx::core::matmul(low, mlx::core::transpose(up)));
        auto expected = mlx::core::mean(
            mlx::core::reshape(mixing, Shape{1, 1, streams, hidden}) *
            mlx::core::reshape(normalized, Shape{1, 1, streams, hidden}), -2);
        auto expected_inject = array(2.0f, dtype) * mlx::core::sigmoid(
            mlx::core::matmul(normalized, mlx::core::transpose(inject)) / count);
        auto actual = mfq::metal::qwen4_gated_residual_pre(
            input, norm, down, up, inject, hidden, streams, 1e-6f);
        const auto check = [](array actual, array reference, const char* name) {
            if (actual.dtype() != reference.dtype() || actual.shape() != reference.shape())
                throw std::runtime_error(std::string(name) + " changed promoted dtype/shape");
            auto a = mlx::core::astype(actual, mlx::core::float32);
            auto b = mlx::core::astype(reference, mlx::core::float32);
            mlx::core::eval(a, b);
            for (std::size_t i = 0; i < a.size(); ++i) {
                const float av = a.data<float>()[i], bv = b.data<float>()[i];
                if (!std::isfinite(av) || std::fabs(av - bv) > 5e-3f * (1.0f + std::fabs(bv)))
                    throw std::runtime_error(std::string(name) + " numerical mismatch: " +
                        std::to_string(av) + " vs " + std::to_string(bv));
            }
        };
        check(mfq::metal::qwen4_grouped_rms_norm(input, norm, hidden, 1e-6f),
            normalized, "mixed norm");
        check(actual.branch, expected, "mixed branch");
        check(*actual.injection, expected_inject, "mixed injection");
    }
}

void benchmark_mixed_decode_path() {
    constexpr int hidden = 2560, streams = 4, low_rank = 320;
    constexpr int width = hidden * streams;
    const auto input = mlx::core::astype(patterned_bfloat(
        width, Shape{1, 1, width}, 37, 1.0f / 53.0f), mlx::core::float32);
    const auto norm = mlx::core::astype(patterned_bfloat(
        width, Shape{width}, 17, 1.0f / 1024.0f), mlx::core::float32);
    const auto down = mlx::core::astype(patterned_bfloat(
        low_rank * width, Shape{low_rank, width}, 29, 1.0f / 4096.0f), mlx::core::float16);
    const auto up = patterned_bfloat(
        width * low_rank, Shape{width, low_rank}, 43, 1.0f / 4096.0f);
    const auto inject = patterned_bfloat(
        streams * width, Shape{streams, width}, 61, 1.0f / 4096.0f);
    mlx::core::eval(input, norm, down, up, inject);
    auto run = [&](bool fused) {
        if (fused) {
            auto result = mfq::metal::qwen4_gated_residual_pre(
                input, norm, down, up, inject, hidden, streams, 1e-6f);
            mlx::core::eval(result.branch, *result.injection);
        } else {
            auto normalized = reference_grouped_rms_norm(input, norm, hidden, 1e-6f);
            auto low = mlx::core::matmul(normalized, mlx::core::transpose(down)) / array(4.0f);
            low = low * mlx::core::sigmoid(low);
            auto mixing = mlx::core::sigmoid(mlx::core::matmul(low, mlx::core::transpose(up)));
            auto branch = mlx::core::mean(
                mlx::core::reshape(mixing, Shape{1, 1, streams, hidden}) *
                mlx::core::reshape(normalized, Shape{1, 1, streams, hidden}), -2);
            auto injection = array(2.0f) * mlx::core::sigmoid(
                mlx::core::matmul(normalized, mlx::core::transpose(inject)) / array(4.0f));
            mlx::core::eval(branch, injection);
        }
    };
    for (int warmup = 0; warmup < 24; ++warmup)
        run(warmup % 2 != 0);
    // Alternate both order and path in the same process to reduce thermal
    // and clock drift. This is an operator test, not a whole-model estimate.
    constexpr int iterations = 50;
    for (int sample = 0; sample < 12; ++sample) {
        double timings[2];
        for (int order = 0; order < 2; ++order) {
            const int fused = (sample + order) % 2;
            const auto started = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < iterations; ++iteration)
                run(fused != 0);
            timings[fused] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count() / iterations;
        }
        std::cout << "mixed_hc_pair sample=" << sample
                  << " first=" << (sample % 2 == 0 ? "reference" : "fused")
                  << " reference_ms=" << timings[0]
                  << " fused_ms=" << timings[1] << '\n';
    }
}

void test_residual_post_preserves_rounding() {
    const mlx::core::Dtype dtypes[] = {
        mlx::core::float16, mlx::core::bfloat16, mlx::core::float32};
    // Include non-aligned hidden sizes and batched prefill, not only decode.
    constexpr int hidden = 137, streams = 4;
    for (const int tokens : {1, 3, 30, 54}) {
        for (int combination = 0; combination < 27; ++combination) {
            auto branch = mlx::core::astype(patterned_bfloat(2 * tokens * hidden,
                Shape{2, tokens, hidden}, 37, 1.0f / 53.0f), dtypes[combination / 9]);
            auto residual = mlx::core::astype(patterned_bfloat(2 * tokens * hidden * streams,
                Shape{2, tokens, hidden * streams}, 43, 1.0f / 71.0f), dtypes[(combination / 3) % 3]);
            auto injection = mlx::core::astype(patterned_bfloat(2 * tokens * streams,
                Shape{2, tokens, streams}, 17, 1.0f / 97.0f), dtypes[combination % 3]);
            // Ensure FP32 operands are not accidentally exactly BF16 values.
            if (branch.dtype() == mlx::core::float32) branch = branch + array(0.000123f);
            if (injection.dtype() == mlx::core::float32) injection = injection + array(0.000137f);
            auto update = mlx::core::expand_dims(branch, -2) *
                mlx::core::expand_dims(injection, -1);
            auto expected = residual + mlx::core::reshape(update, residual.shape());
            auto actual = mfq::metal::qwen4_gated_residual_post(branch, residual, injection, streams);
            if (actual.dtype() != expected.dtype() || actual.shape() != expected.shape() ||
                !mlx::core::all(mlx::core::equal(actual, expected)).item<bool>()) {
                throw std::runtime_error("Qwen gated-HC post changed multiply/add rounding");
            }
        }
    }
}

void test_residual_write_norm_matches_composition() {
    constexpr int hidden = 128;
    constexpr int streams = 4;
    constexpr int width = hidden * streams;
    constexpr int low_rank = 64;
    constexpr float eps = 1e-6f;
    const auto previous_branch = patterned_bfloat(
        hidden, Shape{1, 1, hidden}, 37, 1.0f / 53.0f);
    const auto previous_residual = patterned_bfloat(
        width, Shape{1, 1, width}, 43, 1.0f / 71.0f);
    const auto previous_injection = patterned_bfloat(
        streams, Shape{1, 1, streams}, 17, 1.0f / 97.0f);
    const auto norm = mlx::core::astype(patterned_bfloat(
        width, Shape{width}, 19, 1.0f / 1024.0f),
        mlx::core::float32);
    const auto down = patterned_bfloat(
        static_cast<std::size_t>(low_rank) * width,
        Shape{low_rank, width}, 29, 1.0f / 4096.0f);
    const auto up = patterned_bfloat(
        static_cast<std::size_t>(width) * low_rank,
        Shape{width, low_rank}, 31, 1.0f / 4096.0f);
    const auto injection = patterned_bfloat(
        static_cast<std::size_t>(streams) * width,
        Shape{streams, width}, 61, 1.0f / 4096.0f);

    auto expected_residual = mfq::metal::qwen4_gated_residual_post(
        previous_branch,
        previous_residual,
        previous_injection,
        streams);
    auto expected = mfq::metal::qwen4_gated_residual_pre(
        expected_residual,
        norm,
        down,
        up,
        injection,
        hidden,
        streams,
        eps);
    auto actual = mfq::metal::qwen4_gated_residual_pre_after(
        previous_branch,
        previous_residual,
        previous_injection,
        norm,
        down,
        up,
        injection,
        hidden,
        streams,
        eps);
    if (!expected.injection || !actual.injection) {
        throw std::runtime_error(
            "Qwen chained gated-HC path omitted injection");
    }
    require_bit_exact(
        std::move(actual.residual),
        std::move(expected_residual),
        "Qwen fused residual write");
    require_bit_exact(
        std::move(actual.branch),
        std::move(expected.branch),
        "Qwen fused write/RMS branch");
    require_bit_exact(
        std::move(*actual.injection),
        std::move(*expected.injection),
        "Qwen fused write/RMS injection");
}

void test_packed_mhc(const char* path) {
    using namespace mlx::core;
    using mfq::metal::MlxLinear;
    const mfq::metal::MfqContainer model(path);
    const std::string root = model.contains("test.mhc.pre.down.weight")
        ? "test.mhc" : "model.block.0.attention.mhc";
    const auto down = MlxLinear::load(model, root + ".pre.down.weight");
    const auto up = MlxLinear::load(model, root + ".pre.up.weight");
    const std::optional<MlxLinear> injection = MlxLinear::load(model, root + ".post.inject.weight");
    const auto mapped_norm = model.map_record(root + ".pre.norm.weight");
    const auto norm = mfq::metal::load_dense_array(
        model.record(root + ".pre.norm.weight").dtype, mapped_norm.view());
    const auto reference = [](const MlxLinear& weight) {
        if (const auto* nint = weight.nint_weight_ref()) return nint->dequantize();
        return *weight.dense_weight_ref();
    };
    const auto dense_down = reference(down);
    const auto dense_up = reference(up);
    const std::optional<array> dense_injection = reference(*injection);
    std::size_t packed_bytes = 0, expanded_bytes = 0;
    for (const auto* weight : {&down, &up, &*injection}) {
        if (const auto* nint = weight->nint_weight_ref()) {
            if (weight->dense_weight_ref()) throw std::runtime_error("MHC retained a dense weight copy");
            packed_bytes += nint->packed_nbytes() + nint->row_metadata().nbytes();
            expanded_bytes += static_cast<std::size_t>(weight->input_size()) * weight->output_size() * 2;
        }
    }
    if (packed_bytes == 0 || packed_bytes >= expanded_bytes)
        throw std::runtime_error("MHC packed storage did not reduce resident bytes");
    constexpr int hidden = 2560, streams = 4, width = hidden * streams;
    const auto compare = [&](Dtype dtype, int rows) {
        auto input = astype(patterned_bfloat(
            static_cast<std::size_t>(rows) * width, Shape{1, rows, width},
            37, 1.0f / 257.0f), dtype);
        auto expected = mfq::metal::qwen4_gated_residual_pre(
            input, norm, dense_down, dense_up, dense_injection, hidden, streams);
        auto actual = mfq::metal::qwen4_gated_residual_pre(
            input, norm, down, up, injection, hidden, streams);
        if (rows <= 6 && down.nint_weight_ref() && up.nint_weight_ref()
            && injection->nint_weight_ref() && std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST") == nullptr
            && std::string(actual.branch.primitive().name()) != "PackedHcPrimitive")
            throw std::runtime_error("packed MHC did not dispatch its native two-stage primitive");
        require_close(actual.branch, expected.branch, 5e-3f, "packed MHC branch");
        require_close(*actual.injection, *expected.injection, 5e-3f, "packed MHC injection");
        if (rows >= 64) {
            auto expected_after = mfq::metal::qwen4_gated_residual_pre_after(
                expected.branch, input, *expected.injection, norm,
                dense_down, dense_up, dense_injection, hidden, streams);
            auto actual_after = mfq::metal::qwen4_gated_residual_pre_after(
                expected.branch, input, *expected.injection, norm,
                down, up, injection, hidden, streams);
            require_close(actual_after.residual, expected_after.residual, 0.0f,
                          "prefill MHC residual");
            require_close(actual_after.branch, expected_after.branch, 5e-3f,
                          "prefill MHC chained branch");
            require_close(*actual_after.injection, *expected_after.injection, 5e-3f,
                          "prefill MHC chained injection");
        }
        if (rows <= 6) {
            std::vector<array> branches, gates;
            for (int row = 0; row < rows; ++row) {
                auto one = mfq::metal::qwen4_gated_residual_pre(
                    slice(input, Shape{0, row, 0}, Shape{1, row + 1, width}),
                    norm, down, up, injection, hidden, streams);
                branches.push_back(one.branch);
                gates.push_back(*one.injection);
            }
            require_close(actual.branch, concatenate(branches, 1), 0.0f,
                          "packed MHC batch/row branch");
            require_close(*actual.injection, concatenate(gates, 1), 0.0f,
                          "packed MHC batch/row injection");
            auto expected_after = mfq::metal::qwen4_gated_residual_pre_after(
                expected.branch, input, *expected.injection, norm,
                dense_down, dense_up, dense_injection, hidden, streams);
            auto actual_after = mfq::metal::qwen4_gated_residual_pre_after(
                expected.branch, input, *expected.injection, norm,
                down, up, injection, hidden, streams);
            if (std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST") == nullptr &&
                std::string(actual_after.branch.primitive().name()) != "PackedHcPrimitive")
                throw std::runtime_error("packed MHC batch chain missed native two-stage dispatch");
            branches.clear();
            gates.clear();
            std::vector<array> residuals;
            for (int row = 0; row < rows; ++row) {
                auto one = mfq::metal::qwen4_gated_residual_pre_after(
                    slice(expected.branch, Shape{0, row, 0}, Shape{1, row + 1, hidden}),
                    slice(input, Shape{0, row, 0}, Shape{1, row + 1, width}),
                    slice(*expected.injection, Shape{0, row, 0}, Shape{1, row + 1, streams}),
                    norm, down, up, injection, hidden, streams);
                branches.push_back(one.branch);
                gates.push_back(*one.injection);
                residuals.push_back(one.residual);
            }
            require_close(actual_after.branch, concatenate(branches, 1), 0.0f,
                          "packed MHC batch/row chained branch");
            require_close(*actual_after.injection, concatenate(gates, 1), 0.0f,
                          "packed MHC batch/row chained injection");
            require_close(actual_after.residual, concatenate(residuals, 1), 0.0f,
                          "packed MHC batch/row residual");
            require_close(actual_after.residual, expected_after.residual, 0.0f, "packed MHC residual");
            require_close(actual_after.branch, expected_after.branch, 5e-3f, "packed MHC chained branch");
            require_close(*actual_after.injection, *expected_after.injection, 5e-3f, "packed MHC chained injection");
            auto rounded_branch = astype(expected.branch, dtype);
            auto rounded_gate = astype(*expected.injection, dtype);
            auto rounded_expected = mfq::metal::qwen4_gated_residual_pre_after(
                rounded_branch, input, rounded_gate, norm,
                dense_down, dense_up, dense_injection, hidden, streams);
            auto rounded_actual = mfq::metal::qwen4_gated_residual_pre_after(
                rounded_branch, input, rounded_gate, norm,
                down, up, injection, hidden, streams);
            require_close(rounded_actual.residual, rounded_expected.residual, 0.0f,
                          "packed MHC low-precision residual");
            require_close(rounded_actual.branch, rounded_expected.branch, 5e-3f,
                          "packed MHC low-precision chained branch");
            auto expected_mix = mfq::metal::qwen4_gated_residual_pre(
                input, norm, dense_down, dense_up, std::nullopt, hidden, streams);
            auto actual_mix = mfq::metal::qwen4_gated_residual_pre(
                input, norm, down, up, std::nullopt, hidden, streams);
            if (actual_mix.injection) throw std::runtime_error("packed final MHC mixer added injection");
            require_close(actual_mix.branch, expected_mix.branch, 5e-3f, "packed final MHC mixer");
        }
    };
    for (auto dtype : {float16, bfloat16, float32}) {
        for (int rows : {1, 2, 3, 4, 5, 6, 7, 64, 2048, 4096}) {
            try { compare(dtype, rows); }
            catch (const std::exception& error) {
                throw std::runtime_error("dtype=" + std::to_string(static_cast<int>(dtype.val())) + " rows=" +
                    std::to_string(rows) + ": " + error.what());
            }
        }
    }
    std::cout << "packed MHC decode, chained residual, final mixer and 4096-row prefill passed: "
              << packed_bytes << " packed bytes / " << expanded_bytes << " expanded bytes\n";
}

void benchmark_packed_mhc(const char* path) {
    using namespace mlx::core;
    using mfq::metal::MlxLinear;
    const mfq::metal::MfqContainer model(path);
    struct Weights {
        array norm;
        MlxLinear down;
        MlxLinear up;
        std::optional<MlxLinear> injection;
        array dense_down;
        array dense_up;
        std::optional<array> dense_injection;
    };
    std::vector<std::string> names;
    for (const auto& [name, record] : model.records()) {
        if (name.find(".mhc.") != std::string::npos &&
            name.ends_with(".pre.down.weight") && mfq::metal::is_nint_dtype(record.dtype))
            names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    std::vector<Weights> weights;
    for (const auto& name : names) {
        const auto root = name.substr(0, name.size() - std::string(".pre.down.weight").size());
        auto down = MlxLinear::load(model, name);
        auto up = MlxLinear::load(model, root + ".pre.up.weight");
        const auto mapped_norm = model.map_record(root + ".pre.norm.weight");
        auto norm = mfq::metal::load_dense_array(
            model.record(root + ".pre.norm.weight").dtype, mapped_norm.view());
        std::optional<MlxLinear> injection;
        if (model.contains(root + ".post.inject.weight"))
            injection = MlxLinear::load(model, root + ".post.inject.weight");
        const auto reference = [](const MlxLinear& weight) {
            if (const auto* nint = weight.nint_weight_ref()) return nint->dequantize();
            return *weight.dense_weight_ref();
        };
        auto dense_down = reference(down), dense_up = reference(up);
        std::optional<array> dense_injection;
        if (injection) dense_injection = reference(*injection);
        weights.push_back({std::move(norm), std::move(down), std::move(up), std::move(injection),
            std::move(dense_down), std::move(dense_up), std::move(dense_injection)});
    }
    if (weights.empty()) throw std::runtime_error("no quantized MHC projections to benchmark");
    const auto input = astype(patterned_bfloat(10240, Shape{1, 1, 10240}, 37, 1.0f / 257.0f), float32);
    const auto execute = [&](const Weights& w, bool packed) {
        auto result = packed
            ? mfq::metal::qwen4_gated_residual_pre(input, w.norm, w.down, w.up, w.injection, 2560, 4)
            : mfq::metal::qwen4_gated_residual_pre(input, w.norm, w.dense_down, w.dense_up, w.dense_injection, 2560, 4);
        std::vector<array> outputs{result.branch};
        if (result.injection) outputs.push_back(*result.injection);
        eval(outputs);
        return result;
    };
    for (const auto& w : weights) {
        auto packed = execute(w, true), dense = execute(w, false);
        require_close(packed.branch, dense.branch, 5e-3f, "rotating real MHC branch");
        if (packed.injection) require_close(*packed.injection, *dense.injection, 5e-3f, "rotating real MHC injection");
    }
    for (int round = 0; round < 3; ++round) {
        for (int order = 0; order < 2; ++order) {
            const bool packed = (round + order) % 2 == 0;
            const auto start = std::chrono::steady_clock::now();
            for (int repetition = 0; repetition < 4; ++repetition) {
                for (const auto& w : weights) execute(w, packed);
            }
            const auto milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "{\"round\":" << round << ",\"packed\":" << (packed ? "true" : "false")
                      << ",\"weight_sets\":" << weights.size() << ",\"normal_eval_ms_per_mhc\":"
                      << milliseconds / (4 * weights.size()) << "}\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--packed-mhc") {
            test_packed_mhc(argv[2]);
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--benchmark-packed-mhc") {
            benchmark_packed_mhc(argv[2]);
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--prefill-gate-only") {
            test_qsa_prefill_output_gate();
            std::cout << "QSA prefill output gate test passed\n";
            return 0;
        }
        test_decode_fast_path_matches_reference();
        test_prefill_mixing_epilogue();
        test_qsa_block_scores();
        test_qsa_prefill_output_gate();
        test_qsa_dense_attention_causal_boundaries();
        test_qsa_canonical_prefill_prefix();
        for (auto dtype : {mlx::core::float16, mlx::core::bfloat16,
                           mlx::core::float32}) {
            for (int rows = 1; rows <= 6; ++rows) {
                test_qsa_decode_prologue_matches_reference(dtype, 17, 1e7f, 1e-6f, rows);
                test_qsa_decode_prologue_matches_reference(dtype, 4099, 1e5f, 3e-5f, rows);
                test_qsa_decode_prologue_matches_reference(dtype, 0, 1e7f, 1e-6f, rows);
            }
        }
        test_mixed_storage_decode_matches_promoted_reference();
        test_residual_post_preserves_rounding();
        test_residual_write_norm_matches_composition();
        std::cout << "Qwen gated-HC fast path test passed\n";
        if (argc == 2 && std::string(argv[1]) == "--benchmark")
            benchmark_decode_path();
        if (argc == 2 && std::string(argv[1]) == "--benchmark-mixed")
            benchmark_mixed_decode_path();
        if (argc == 2 && std::string(argv[1]) == "--benchmark-qsa")
            benchmark_qsa_decode_prologue();
        if (argc == 2 && std::string(argv[1]) == "--benchmark-chain")
            benchmark_residual_write_norm_chain();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
