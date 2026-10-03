#include "qwen4_ops.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

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
    actual = mlx::core::astype(actual, mlx::core::float32);
    expected = mlx::core::astype(expected, mlx::core::float32);
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

void test_qsa_decode_prologue_matches_reference() {
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

    auto query_parts = split(reshape(
        query_gate,
        Shape{1, 1, query_heads, 2 * head_dimension}), 2, -1);
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
            Shape{1, 1, key_heads, head_dimension})), {0, 2, 1, 3}),
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
                Shape{1, 1, index_heads, index_dimension})),
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
    require_bit_exact(
        std::move(actual.query),
        std::move(reference_query),
        "Qwen QSA fused query");
    require_bit_exact(
        std::move(actual.output_gate),
        reshape(query_parts.at(1), Shape{1, 1, query_heads * head_dimension}),
        "Qwen QSA fused output gate");
    require_bit_exact(
        std::move(actual.key),
        std::move(reference_key),
        "Qwen QSA fused key");
    require_bit_exact(
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
    for (const int tokens : {1, 3}) {
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

} // namespace

int main(int argc, char** argv) {
    try {
        test_decode_fast_path_matches_reference();
        test_qsa_decode_prologue_matches_reference();
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
