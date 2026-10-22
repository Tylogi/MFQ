#include "mlx_csa2.h"
#include "mlx_deepseek_v41_engram.h"
#include "mlx_deepseek_v41_mhc.h"
#include "mlx_dsa_cache_ops.h"
#include "mlx_dsa.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {
using namespace mlx::core;
using namespace mfq::metal;
constexpr float eps = 1e-6f;

array sample(const Shape& shape, Dtype dtype, float phase = 0.0f) {
    std::size_t count = 1;
    for (int d : shape) count *= d;
    auto x = arange(static_cast<int>(count), float32);
    return reshape(astype(sin(x * 0.071f + phase) * 0.7f +
        cos(x * 0.013f - phase) * 0.2f, dtype), shape);
}

array part(const array& x, int axis, int begin, int end) {
    auto start = Shape(x.ndim(), 0), stop = x.shape();
    start[axis] = begin; stop[axis] = end;
    return slice(x, start, stop);
}

void close(array actual, array expected, float tolerance, const std::string& label) {
    if (actual.shape() != expected.shape()) throw std::runtime_error(label + " shape");
    actual = contiguous(astype(actual, float32));
    expected = contiguous(astype(expected, float32));
    eval(actual, expected);
    float worst = 0.0f;
    std::size_t failures = 0, first = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual.data<float>()[i] - expected.data<float>()[i]);
        if (!std::isfinite(error)) throw std::runtime_error(label + " non-finite");
        worst = std::max(worst, error);
        if (error > tolerance) { if (!failures) first = i; ++failures; }
    }
    if (worst > tolerance) throw std::runtime_error(label + " max error=" + std::to_string(worst) +
        " count=" + std::to_string(failures) + " first=" + std::to_string(first) +
        " actual=" + std::to_string(actual.data<float>()[first]) +
        " expected=" + std::to_string(expected.data<float>()[first]));
}

array rms(const array& x, const array& norm) {
    auto source = astype(x, float32);
    return astype(source * rsqrt(mean(source * source, -1, true) + eps) * norm, x.dtype());
}

array rotate(const array& x, const array& cosine, const array& sine, int rotary) {
    auto tail = mlx_rope_adjacent(part(x, 2, x.shape(2) - rotary, x.shape(2)), cosine, sine);
    return concatenate({part(x, 2, 0, x.shape(2) - rotary), tail}, 2);
}

void test_compress() {
    for (Dtype dtype : {float16, bfloat16, float32}) {
        for (int tokens : {1, 2, 3, 4, 5, 6, 17, 513, 4096}) {
            for (int pending : {0, 1}) {
                const int batch = tokens == 17 ? 2 : 1, width = 512;
                auto kv = sample({batch, tokens, width}, float32);
                auto score = sample(kv.shape(), float32, 1.3f);
                auto old = sample({batch, 2, width}, float32, 0.4f);
                auto old_score = sample(old.shape(), float32, 2.1f);
                auto norm = sample({width}, float32, 0.7f) + 1.0f;
                auto result = mlx_dsa_compress_window(kv, score, old, old_score, norm, pending, dtype, eps);
                const int count = (pending + tokens) / 2;
                if (count) {
                    auto all = concatenate({part(old, 1, 0, pending), kv}, 1);
                    auto scores = concatenate({part(old_score, 1, 0, pending), score}, 1);
                    auto values = reshape(part(all, 1, 0, count * 2), Shape{batch, count, 2, width});
                    auto logits = reshape(part(scores, 1, 0, count * 2), values.shape());
                    auto weights = softmax(logits, 2, true);
                    auto pooled = astype(sum(values * weights, 2), dtype);
                    close(result.latent, rms(pooled, norm), dtype == float32 ? 2e-6f : 0.016f, "C2 latent");
                } else result.latent.eval();
                for (int slot = 0; slot < 2; ++slot) {
                    const int last = tokens - 1 - ((pending + tokens - 1 + 2 - slot) % 2);
                    close(part(result.partial_kv, 1, slot, slot + 1),
                        part(last >= 0 ? kv : old, 1, last >= 0 ? last : slot,
                            (last >= 0 ? last : slot) + 1), 0.0f, "C2 tail");
                    close(part(result.partial_score, 1, slot, slot + 1),
                        part(last >= 0 ? score : old_score, 1, last >= 0 ? last : slot,
                            (last >= 0 ? last : slot) + 1), 0.0f, "C2 score tail");
                }
            }
        }
    }
    std::cout << "C2 aligned/unaligned and tail tests passed\n";
}

void test_pack_store() {
    for (Dtype dtype : {float16, bfloat16, float32}) {
        for (int width : {128, 512}) {
            for (int tokens : {1, 2, 6, 513}) {
                for (bool normalize : {false, true}) {
                    for (bool e4 : {false, true}) {
                        const int batch = tokens == 6 ? 2 : 1, rotary = 64, begin = 3;
                        const int gs = e4 ? 16 : 32, capacity = begin + tokens + 2;
                        auto x = sample({batch, tokens, width}, dtype);
                        auto norm = sample({width}, float32, 1.5f) + 1.0f;
                        auto angles = sample({tokens, rotary / 2}, float32, 0.3f);
                        auto cosine = cos(angles), sine = sin(angles);
                        auto normalized = normalize ? rms(x, norm) : x;
                        auto packed = mlx_csa2_pack(rotate(normalized, cosine, sine, rotary), 4, gs, e4);
                        auto cache = full({batch, capacity, width / 2 + width / gs}, 137, uint8);
                        auto update = mlx_dsa_norm_rope_pack_store(cache, x, norm, cosine, sine,
                            begin, rotary, gs, e4, eps, normalize, true);
                        close(update.normalized, normalized,
                            dtype == float32 ? 2e-6f : 0.016f, "cache pre-RoPE output");
                        close(part(update.cache, 1, begin, begin + tokens), packed, 0.0f,
                            "packed KV bytes dtype=" + std::to_string(static_cast<int>(dtype.val())) +
                            " D=" + std::to_string(width) + " M=" + std::to_string(tokens) +
                            " norm=" + std::to_string(normalize) + " e4=" + std::to_string(e4));
                        close(part(update.cache, 1, 0, begin),
                            full({batch, begin, cache.shape(2)}, 137, uint8), 0.0f, "cache prefix");
                        close(part(update.cache, 1, begin + tokens, capacity),
                            full({batch, 2, cache.shape(2)}, 137, uint8), 0.0f, "cache suffix");
                    }
                }
            }
        }
    }
    std::cout << "Norm/RoPE/FP4/store byte tests passed\n";
}

void test_engram() {
    for (Dtype dtype : {float16, bfloat16, float32}) {
        for (int tokens : {1, 2, 6, 65, 512}) {
            const int batch = tokens == 6 ? 2 : 1, streams = 4, hidden = 5120;
            auto x = sample({batch, tokens, streams, hidden}, dtype);
            auto kv = sample({batch, tokens, (streams + 1) * hidden}, dtype, 0.8f);
            auto weight = sample({streams, hidden}, float32, 0.3f);
            auto source = astype(x, float32);
            auto key = astype(reshape(part(kv, 2, 0, streams * hidden), source.shape()), float32);
            auto value = astype(reshape(part(kv, 2, streams * hidden, (streams + 1) * hidden),
                Shape{batch, tokens, 1, hidden}), float32);
            auto inverse = rsqrt(mean(source * source, -1) + eps) * rsqrt(mean(key * key, -1) + eps);
            auto score = sum(source * key * weight, -1) * inverse / std::sqrt(float(hidden));
            auto gate = sigmoid(sign(score) * sqrt(maximum(abs(score), array(1e-6f))));
            for (bool masked : {false, true}) {
                auto mask = reshape(equal(remainder(arange(batch * tokens), array(2)), array(0)), Shape{batch, tokens});
                auto g = masked ? gate * expand_dims(astype(mask, float32), -1) : gate;
                auto expected = astype(source + expand_dims(g, -1) * value, dtype);
                auto actual = deepseek_v41_engram_gate(x, kv, weight,
                    masked ? std::optional<array>(mask) : std::nullopt, eps);
                close(actual, expected, dtype == float32 ? 3e-6f : (dtype == float16 ? 0.002f : 0.016f), "Engram gate");
            }
        }
    }
    std::cout << "Engram gate/mask tests passed\n";
}

void test_mhc() {
    for (Dtype dtype : {float16, bfloat16}) {
        for (int tokens : {1, 2, 6, 65, 512}) {
            const int batch = tokens == 6 ? 2 : 1, hidden = 5120;
            auto residual = sample({batch, tokens, 4, hidden}, dtype);
            auto branch = sample({batch, tokens, hidden}, tokens == 2 ? float32 : dtype, 1.4f);
            auto post = sample({batch, tokens, 4}, float32, 0.1f) + 1.0f;
            auto combination = sample({batch, tokens, 4, 4}, float32, 0.7f) * 0.1f + 0.25f;
            auto pre = sample({batch, tokens, 4}, float32, 0.3f) * 0.2f + 0.5f;
            auto norm = sample({hidden}, float32, 0.9f) + 1.0f;
            auto expanded = deepseek_v41_hc_post(branch, residual, post, combination);
            auto expected = deepseek_v41_hc_collapse_norm(expanded, pre, norm, eps);
            auto actual = deepseek_v41_hc_post_collapse_norm(branch, residual,
                {post, combination, pre}, pre, norm, eps);
            const auto label = " dtype=" + std::to_string(static_cast<int>(dtype.val())) +
                " M=" + std::to_string(tokens);
            close(actual.residual, expanded, 0.0f, "mHC expanded bits" + label);
            close(actual.branch, expected, 0.0f, "mHC collapsed bits" + label);
        }
    }
    std::cout << "mHC Post/Pre/Norm bit-exact tests passed\n";
}

void benchmark(const std::string& name, const std::function<std::vector<array>()>& reference,
    const std::function<std::vector<array>()>& fused) {
    const auto measure = [](const auto& build) {
        const auto bundle = [&] {
            std::vector<array> outputs;
            for (int i = 0; i < 16; ++i) {
                auto result = build();
                outputs.insert(outputs.end(), result.begin(), result.end());
            }
            eval(outputs);
        };
        bundle();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) bundle();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 160;
    };
    const double old_us = measure(reference), new_us = measure(fused);
    std::cout << name << " reference_us=" << old_us << " fused_us=" << new_us
              << " reduction_pct=" << (1.0 - new_us / old_us) * 100.0 << '\n';
}

void run_benchmarks() {
    for (int tokens : {1, 6, 256}) {
        const std::string suffix = " M=" + std::to_string(tokens);
        auto residual = sample({1, tokens, 4, 5120}, float16);
        auto branch = sample({1, tokens, 5120}, float16, 0.6f);
        auto post = sample({1, tokens, 4}, float32) + 1.0f;
        auto comb = sample({1, tokens, 4, 4}, float32) * 0.1f + 0.25f;
        auto pre = post * 0.3f;
        auto norm = sample({5120}, float32) + 1.0f;
        eval(residual, branch, post, comb, pre, norm);
        benchmark("mHC" + suffix, [&] {
            auto expanded = deepseek_v41_hc_post(branch, residual, post, comb);
            return std::vector<array>{expanded, deepseek_v41_hc_collapse_norm(expanded, pre, norm, eps)};
        }, [&] {
            auto result = deepseek_v41_hc_post_collapse_norm(branch, residual, {post, comb, pre}, pre, norm, eps);
            return std::vector<array>{result.residual, result.branch};
        });

        auto kv = sample({1, tokens, 5 * 5120}, float16, 0.3f);
        auto weight = sample({4, 5120}, float32, 0.2f);
        eval(kv, weight);
        benchmark("Engram" + suffix, [&] {
            auto source = astype(residual, float32);
            auto key = astype(reshape(part(kv, 2, 0, 4 * 5120), source.shape()), float32);
            auto value = astype(reshape(part(kv, 2, 4 * 5120, 5 * 5120), Shape{1, tokens, 1, 5120}), float32);
            auto inv = rsqrt(mean(source * source, -1) + eps) * rsqrt(mean(key * key, -1) + eps);
            auto score = sum(source * key * weight, -1) * inv / std::sqrt(5120.0f);
            auto gate = sigmoid(sign(score) * sqrt(maximum(abs(score), array(1e-6f))));
            return std::vector<array>{astype(source + expand_dims(gate, -1) * value, float16)};
        }, [&] { return std::vector<array>{deepseek_v41_engram_gate(residual, kv, weight, std::nullopt, eps)}; });

        auto values = sample({1, tokens, 512}, float16);
        auto small_norm = sample({512}, float32) + 1.0f;
        auto angle = sample({tokens, 32}, float32);
        auto cosine = cos(angle), sine = sin(angle);
        auto cache = zeros({1, tokens + 4, 288}, uint8);
        auto rows = reshape(arange(2, 2 + tokens), Shape{1, tokens});
        eval(values, small_norm, cosine, sine, cache, rows);
        benchmark("KV-store" + suffix, [&] {
            auto normalized = rms(values, small_norm);
            auto packed = mlx_csa2_pack(rotate(normalized, cosine, sine, 64), 4, 16, true);
            return std::vector<array>{mlx_cache_write_inplace(cache, packed, rows), normalized};
        }, [&] {
            auto result = mlx_dsa_norm_rope_pack_store(cache, values, small_norm, cosine, sine,
                2, 64, 16, true, eps, true, true);
            return std::vector<array>{result.cache, result.normalized};
        });

        auto projected = astype(values, float32);
        auto scores = sample(projected.shape(), float32, 1.5f);
        auto old = sample({1, 2, 512}, float32, 0.3f);
        auto old_score = sample(old.shape(), float32, 1.1f);
        eval(projected, scores, old, old_score);
        benchmark("C2-pending" + suffix, [&] {
            auto state = old, state_score = old_score;
            std::vector<array> pooled;
            for (int i = 0; i < tokens; ++i) {
                auto row = full({1, 1}, (i + 1) % 2, int32);
                state = mlx_cache_write_inplace(state, part(projected, 1, i, i + 1), row);
                state_score = mlx_cache_write_inplace(state_score, part(scores, 1, i, i + 1), row);
                if (i % 2 == 0) pooled.push_back(rms(astype(sum(state * softmax(state_score, 1, true), 1, true), float16), small_norm));
            }
            return std::vector<array>{concatenate(pooled, 1), state, state_score};
        }, [&] {
            auto result = mlx_dsa_compress_window(projected, scores, old, old_score, small_norm, 1, float16, eps);
            return std::vector<array>{result.latent, result.partial_kv, result.partial_score};
        });
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--bench") { run_benchmarks(); return 0; }
        test_compress();
        test_pack_store();
        test_engram();
        test_mhc();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
