#include "mlx_linear_attention.h"

#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

void require_close(
    float actual,
    float expected,
    float tolerance = 2e-4f) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "linear-attention mismatch: actual=" +
            std::to_string(actual) +
            " expected=" +
            std::to_string(expected));
    }
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

mlx::core::array patterned_bfloat(
    std::size_t count,
    const mlx::core::Shape& shape,
    int multiplier,
    float scale) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int centered =
            static_cast<int>((index * multiplier) % 257) - 128;
        values[index] = static_cast<float>(centered) * scale;
    }
    return mlx::core::astype(
        mlx::core::array(
            values.begin(), shape, mlx::core::float32),
        mlx::core::bfloat16);
}

void require_bit_exact(
    mlx::core::array actual,
    mlx::core::array expected,
    const char* name) {
    if (actual.shape() != expected.shape() ||
        actual.dtype() != mlx::core::bfloat16 ||
        expected.dtype() != mlx::core::bfloat16) {
        throw std::runtime_error(
            std::string(name) + " shape/dtype mismatch");
    }
    mlx::core::eval(actual, expected);
    const auto* actual_bits = actual.data<std::uint16_t>();
    const auto* expected_bits = expected.data<std::uint16_t>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual_bits[index] != expected_bits[index]) {
            throw std::runtime_error(
                std::string(name) + " changed element " +
                std::to_string(index));
        }
    }
}

void require_vector_close(
    const float* actual,
    const std::vector<float>& expected,
    float tolerance,
    const char* name) {
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            throw std::runtime_error(
                std::string(name) + " mismatch at " +
                std::to_string(index) + ": actual=" +
                std::to_string(actual[index]) + " expected=" +
                std::to_string(expected[index]));
        }
    }
}

mfq::metal::MlxGatedDeltaGates reference_gates(
    const mlx::core::array& alpha,
    const mlx::core::array& beta,
    const mlx::core::array& bias,
    const mlx::core::array& decay) {
    using namespace mlx::core;
    auto input = astype(alpha, float32) + astype(bias, float32);
    auto softplus = maximum(input, array(0.0f)) +
        log1p(exp(-abs(input)));
    return {
        transpose(astype(decay, float32) * softplus, {0, 2, 1}),
        transpose(sigmoid(astype(beta, float32)), {0, 2, 1}),
    };
}

void test_gdn_gates() {
    using namespace mlx::core;
    for (const auto alpha_type : {float16, bfloat16, float32}) {
        for (const auto beta_type : {float16, bfloat16, float32}) {
            for (const auto weight_type : {float16, bfloat16, float32}) {
                for (const int tokens : {1, 7}) {
                    constexpr int batch = 2;
                    constexpr int heads = 17;
                    const Shape shape{batch, tokens, heads};
                    std::vector<float> values(batch * tokens * heads);
                    for (std::size_t i = 0; i < values.size(); ++i) {
                        values[i] = float(int(i % 101) - 50) * 0.713f;
                    }
                    values[0] = -1000.0f;
                    values[1] = 1000.0f;
                    auto alpha = astype(array(values.begin(), shape), alpha_type);
                    auto beta = astype(array(values.rbegin(), shape), beta_type);
                    auto bias = astype(arange(heads) * array(0.0193f), weight_type);
                    auto decay = astype(-exp(arange(heads) * array(0.0321f)), weight_type);
                    auto expected = reference_gates(alpha, beta, bias, decay);
                    auto actual = mfq::metal::gated_delta_gates(alpha, beta, bias, decay);
                    eval(actual.gate, actual.beta, expected.gate, expected.beta);
                    for (const auto& pair : {
                             std::pair{actual.gate, contiguous(expected.gate)},
                             std::pair{actual.beta, contiguous(expected.beta)}}) {
                        if (pair.first.dtype() != float32 ||
                            pair.first.shape() != Shape{batch, heads, tokens}) {
                            throw std::runtime_error("GDN gates output shape/dtype mismatch");
                        }
                        eval(pair.second);
                        const auto* a = pair.first.data<float>();
                        const auto* e = pair.second.data<float>();
                        for (std::size_t i = 0; i < pair.first.size(); ++i) {
                            if (!std::isfinite(a[i]) ||
                                a[i] != e[i]) {
                                throw std::runtime_error("GDN fused gates numerical mismatch: actual=" +
                                    std::to_string(a[i]) + " expected=" + std::to_string(e[i]) +
                                    " index=" + std::to_string(i) + " types=" +
                                    std::to_string(static_cast<int>(alpha_type.val())) + "," +
                                    std::to_string(static_cast<int>(beta_type.val())) + "," +
                                    std::to_string(static_cast<int>(weight_type.val())) +
                                    " delta=" + std::to_string((a[i] - e[i]) * 1e8f) + "e-8");
                            }
                        }
                    }
                }
            }
        }
    }
    bool rejected = false;
    try {
        mfq::metal::gated_delta_gates(
            zeros({1, 1, 4}), zeros({1, 1, 3}), zeros({4}), zeros({4}));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("GDN gates accepted invalid geometry");
}

void benchmark_gdn_gates() {
    using namespace mlx::core;
    constexpr int heads = 48;
    auto alpha = reshape(arange(heads) * array(0.0713f), Shape{1, 1, heads});
    auto beta = alpha - array(1.1f);
    auto bias = arange(heads) * array(0.0193f);
    auto decay = -exp(arange(heads) * array(0.0321f));
    eval(alpha, beta, bias, decay);
    auto execute = [&](bool fused) {
        auto gates = fused
            ? mfq::metal::gated_delta_gates(alpha, beta, bias, decay)
            : reference_gates(alpha, beta, bias, decay);
        eval(gates.gate, gates.beta);
    };
    for (int warm = 0; warm < 24; ++warm) execute(warm % 2 != 0);
    for (int pair = 0; pair < 12; ++pair) {
        double times[2]{};
        for (int arm = 0; arm < 2; ++arm) {
            const int fused = (pair + arm) % 2;
            const auto start = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 100; ++rep) execute(fused != 0);
            times[fused] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() / 100.0;
        }
        std::cout << "gdn_gates pair=" << pair << " reference_ms=" << times[0]
                  << " fused_ms=" << times[1] << '\n';
    }
}

void test_gdn_decode_step(
    int key_heads,
    int value_heads,
    mlx::core::Dtype projection_dtype,
    mlx::core::Dtype gate_dtype) {
    using namespace mlx::core;
    constexpr int dimension = 128;
    const int qk_width = 2 * key_heads * dimension;
    const int value_width = value_heads * dimension;
    const int channels = qk_width + value_width;

    std::vector<float> qk_data(qk_width);
    std::vector<float> value_data(value_width);
    std::vector<float> output_gate_data(value_width);
    std::vector<float> convolution_state_data(3 * channels);
    std::vector<float> convolution_weight_data(4 * channels);
    std::vector<float> recurrent_state_data(
        value_heads * dimension * dimension);
    std::vector<float> alpha_data(value_heads);
    std::vector<float> beta_data(value_heads);
    std::vector<float> bias_data(value_heads);
    std::vector<float> decay_data(value_heads);
    std::vector<float> norm_data(dimension);
    const auto fill = [](std::vector<float>& values, int stride, float scale) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] =
                static_cast<float>(static_cast<int>((index * stride) % 67) - 33) *
                scale;
        }
    };
    fill(qk_data, 11, 0.007f);
    fill(value_data, 13, 0.006f);
    fill(output_gate_data, 17, 0.009f);
    fill(convolution_state_data, 19, 0.002f);
    fill(convolution_weight_data, 23, 0.001f);
    fill(recurrent_state_data, 29, 0.0002f);
    fill(alpha_data, 31, 0.03f);
    fill(beta_data, 37, 0.04f);
    fill(bias_data, 41, 0.01f);
    fill(decay_data, 43, -0.02f);
    fill(norm_data, 47, 0.002f);
    for (auto& value : norm_data) value += 1.0f;
    for (auto& value : decay_data) value -= 0.4f;

    const auto qk = astype(
        array(qk_data.begin(), Shape{1, 1, qk_width}), projection_dtype);
    const auto value = astype(
        array(value_data.begin(), Shape{1, 1, value_width}), projection_dtype);
    const auto output_gate = astype(
        array(output_gate_data.begin(), Shape{1, 1, value_width}), projection_dtype);
    const auto alpha = astype(
        array(alpha_data.begin(), Shape{1, 1, value_heads}), gate_dtype);
    const auto beta = astype(
        array(beta_data.begin(), Shape{1, 1, value_heads}), gate_dtype);
    const array convolution_state(
        convolution_state_data.begin(), Shape{1, 3, channels});
    const array recurrent_state(
        recurrent_state_data.begin(),
        Shape{1, value_heads, dimension, dimension});
    const array convolution_weight(
        convolution_weight_data.begin(), Shape{channels, 4});
    const array bias(bias_data.begin(), Shape{value_heads});
    const array decay(decay_data.begin(), Shape{value_heads});
    const array norm(norm_data.begin(), Shape{dimension});

    const auto gates = mfq::metal::gated_delta_gates(
        alpha, beta, bias, decay);

    for (const bool silu_gate : {false, true}) {
        auto reference_conv_state = convolution_state;
        auto reference_recurrent_state = recurrent_state;
        auto actual_conv_state = convolution_state;
        auto actual_recurrent_state = recurrent_state;
        for (int step = 0; step < 8; ++step) {
            const float convolution_eps = step % 2 ? 1e-5f : 1e-6f;
            const float norm_eps = step % 3 ? 1e-4f : 1e-6f;
            const auto convolved = mfq::metal::linear_conv_qkv(
                reference_conv_state, qk, value, convolution_weight,
                key_heads, value_heads, dimension, dimension,
                std::nullopt, convolution_eps);
            const auto recurrent = mfq::metal::gated_delta_net(
                convolved.query, convolved.key, convolved.value,
                gates.gate, gates.beta, reference_recurrent_state, true);
            auto normalized = fast::rms_norm(
                recurrent.output,
                std::optional<array>(norm),
                norm_eps);
            auto gate = astype(
                transpose(
                    reshape(
                        output_gate,
                        Shape{1, 1, value_heads, dimension}),
                    {0, 2, 1, 3}),
                float32);
            gate = silu_gate ? gate * sigmoid(gate) : sigmoid(gate);
            auto reference_output = reshape(
                transpose(normalized * gate, {0, 2, 1, 3}),
                Shape{1, 1, value_width});
            auto actual = mfq::metal::gated_delta_decode_step(
                qk,
                value,
                output_gate,
                alpha,
                beta,
                actual_conv_state,
                actual_recurrent_state,
                convolution_weight,
                bias,
                decay,
                norm,
                key_heads,
                value_heads,
                dimension,
                convolution_eps,
                norm_eps,
                silu_gate);
            eval(
                reference_output,
                convolved.state,
                recurrent.state,
                actual.output,
                actual.convolution_state,
                actual.recurrent_state);
            require_vector_close(
                actual.output.data<float>(),
                std::vector<float>(
                    reference_output.data<float>(),
                    reference_output.data<float>() + reference_output.size()),
                5e-5f,
                silu_gate ? "GDN fused SiLU output" : "GDN fused sigmoid output");
            require_vector_close(
                actual.convolution_state.data<float>(),
                std::vector<float>(
                    convolved.state.data<float>(),
                    convolved.state.data<float>() + convolved.state.size()),
                0.0f,
                "GDN fused convolution state");
            require_vector_close(
                actual.recurrent_state.data<float>(),
                std::vector<float>(
                    recurrent.state.data<float>(),
                    recurrent.state.data<float>() + recurrent.state.size()),
                5e-6f,
                "GDN fused recurrent state");
            reference_conv_state = convolved.state;
            reference_recurrent_state = recurrent.state;
            actual_conv_state = std::move(actual.convolution_state);
            actual_recurrent_state = std::move(actual.recurrent_state);
        }
    }
    require_vector_close(
        convolution_state.data<float>(), convolution_state_data, 0.0f,
        "GDN retained convolution checkpoint");
    require_vector_close(
        recurrent_state.data<float>(), recurrent_state_data, 0.0f,
        "GDN retained recurrent checkpoint");

    bool rejected = false;
    try {
        mfq::metal::gated_delta_decode_step(
            qk,
            value,
            output_gate,
            alpha,
            beta,
            convolution_state,
            recurrent_state,
            convolution_weight,
            bias,
            decay,
            norm,
            key_heads,
            value_heads,
            64,
            1e-6f,
            1e-6f);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(
            "GDN decode step accepted unsupported geometry");
    }
}

void test_blocked_gdn_prefill(bool tiled_heads) {
    using namespace mlx::core;
    constexpr int batch = 1;
    constexpr int query_heads = 2;
    constexpr int value_heads = 4;
    constexpr int tokens = 67;
    constexpr int dimension = 128;
    const std::size_t query_size =
        batch * query_heads * tokens * dimension;
    const std::size_t value_size =
        batch * value_heads * tokens * dimension;
    const std::size_t gate_size = batch * value_heads * tokens;
    const std::size_t state_size =
        batch * value_heads * dimension * dimension;

    std::vector<float> query(query_size);
    std::vector<float> key(query_size);
    std::vector<float> value(value_size);
    std::vector<float> gate(gate_size);
    std::vector<float> beta(gate_size);
    std::vector<float> initial_state(state_size);
    for (std::size_t index = 0; index < query_size; ++index) {
        query[index] =
            static_cast<float>(static_cast<int>((index * 13) % 29) - 14) *
            0.0015f;
        key[index] =
            static_cast<float>(static_cast<int>((index * 17) % 31) - 15) *
            0.0012f;
    }
    for (std::size_t index = 0; index < value_size; ++index) {
        value[index] =
            static_cast<float>(static_cast<int>((index * 19) % 37) - 18) *
            0.002f;
    }
    for (std::size_t index = 0; index < gate_size; ++index) {
        gate[index] = -0.015f - static_cast<float>(index % 7) * 0.004f;
        beta[index] = 0.35f + static_cast<float>(index % 5) * 0.07f;
    }
    for (std::size_t index = 0; index < state_size; ++index) {
        initial_state[index] =
            static_cast<float>(static_cast<int>((index * 23) % 41) - 20) *
            0.00003f;
    }

    std::vector<float> expected_output(value_size, 0.0f);
    auto expected_state = initial_state;
    const float output_scale = 1.0f / std::sqrt(float(dimension));
    for (int value_head = 0; value_head < value_heads; ++value_head) {
        const int query_head = tiled_heads
            ? value_head % query_heads
            : value_head / (value_heads / query_heads);
        float* head_state = expected_state.data() +
            value_head * dimension * dimension;
        for (int token = 0; token < tokens; ++token) {
            const float* query_row = query.data() +
                (query_head * tokens + token) * dimension;
            const float* key_row = key.data() +
                (query_head * tokens + token) * dimension;
            const float decay = std::exp(
                gate[(value_head * tokens) + token]);
            const float beta_value =
                beta[(value_head * tokens) + token];
            for (int value_dimension = 0;
                 value_dimension < dimension;
                 ++value_dimension) {
                float projected_key = 0.0f;
                for (int key_dimension = 0;
                     key_dimension < dimension;
                     ++key_dimension) {
                    projected_key +=
                        head_state[key_dimension * dimension +
                                   value_dimension] *
                        key_row[key_dimension];
                }
                const std::size_t output_index =
                    (value_head * tokens + token) * dimension +
                    value_dimension;
                const float delta =
                    (value[output_index] - decay * projected_key) *
                    beta_value;
                float output = 0.0f;
                for (int key_dimension = 0;
                     key_dimension < dimension;
                     ++key_dimension) {
                    float& state_value =
                        head_state[key_dimension * dimension +
                                   value_dimension];
                    state_value = decay * state_value +
                        key_row[key_dimension] * delta;
                    output += state_value * query_row[key_dimension];
                }
                expected_output[output_index] = output * output_scale;
            }
        }
    }

    auto result = mfq::metal::gated_delta_net(
        array(query.begin(), Shape{batch, query_heads, tokens, dimension}),
        array(key.begin(), Shape{batch, query_heads, tokens, dimension}),
        array(value.begin(), Shape{batch, value_heads, tokens, dimension}),
        array(gate.begin(), Shape{batch, value_heads, tokens}),
        array(beta.begin(), Shape{batch, value_heads, tokens}),
        array(
            initial_state.begin(),
            Shape{batch, value_heads, dimension, dimension}),
        false,
        tiled_heads);
    eval(result.output, result.state);
    require_vector_close(
        result.output.data<float>(),
        expected_output,
        3e-5f,
        tiled_heads ? "blocked tiled GDN output" : "blocked GDN output");
    require_vector_close(
        result.state.data<float>(),
        expected_state,
        3e-5f,
        tiled_heads ? "blocked tiled GDN state" : "blocked GDN state");
}

void test_gdn_state_only() {
    using namespace mlx::core;
    const auto input = [](const Shape& shape, int multiplier, float scale) {
        std::size_t count = 1;
        for (int size : shape) count *= size;
        std::vector<float> values(count);
        for (std::size_t index = 0; index < count; ++index)
            values[index] = static_cast<float>(static_cast<int>((index * multiplier) % 257) - 128) * scale;
        return array(values.begin(), shape, float32);
    };
    for (const int dimension : {32, 64, 128}) {
        for (const int tokens : {0, 1, 2, 3, 4, 5, 6, 67}) {
            const Shape key_shape{2, 2, tokens, dimension};
            const Shape value_shape{2, 4, tokens, dimension};
            auto query = input(key_shape, 13, 0.0015f);
            auto key = input(key_shape, 17, 0.0012f);
            auto value = input(value_shape, 19, 0.002f);
            if (dimension == 64) { key = astype(key, float16); value = astype(value, float16); }
            auto state = input({2, 4, dimension, dimension}, 23, 0.0003f);
            auto beta = sigmoid(input({2, 4, tokens}, 29, 0.002f));
            for (const bool kda : {false, true}) {
                auto gate = -abs(input(kda ? value_shape : Shape{2, 4, tokens}, 31, 0.003f));
                for (const bool transposed : {false, true}) for (const bool tiled : {false, true}) {
                    auto expected = mfq::metal::gated_delta_net(query, key, value, gate, beta, state, transposed, tiled);
                    auto actual = mfq::metal::gated_delta_net_state(key, value, gate, beta, state, transposed, tiled);
                    auto matches = all(equal(actual, expected.state));
                    eval(matches);
                    if (!matches.item<bool>()) throw std::runtime_error("state-only GDN changed recurrent state");
                    auto second = mfq::metal::gated_delta_net_state(key, value, gate, beta, actual, transposed, tiled);
                    auto repeated = mfq::metal::gated_delta_net(query, key, value, gate, beta, expected.state, transposed, tiled);
                    matches = all(equal(second, repeated.state));
                    eval(matches);
                    if (!matches.item<bool>()) throw std::runtime_error("state-only GDN feedback changed recurrent state");
                }
            }
        }
    }
    const auto key = input({1, 1, 3, 32}, 13, 0.0015f);
    const auto value = input({1, 2, 3, 32}, 19, 0.002f);
    const auto gate = full({1, 2, 3}, -0.125f, float32);
    const auto beta = full({1, 2, 3}, 0.625f, float32);
    auto expected = mfq::metal::gated_delta_net(key, key, value, gate, beta);
    auto actual = mfq::metal::gated_delta_net_state(key, value, gate, beta);
    auto matches = all(equal(actual, expected.state));
    eval(matches);
    if (!matches.item<bool>()) throw std::runtime_error("state-only GDN zero-initialized state differs");
}

void test_cached_depthwise_dilated_decode() {
    using namespace mlx::core;
    constexpr int batch = 1;
    constexpr int channels = 10240;
    constexpr int kernel = 4;
    constexpr int dilation = 3;
    constexpr int state_length = (kernel - 1) * dilation;
    const auto input = patterned_bfloat(
        channels,
        Shape{batch, 1, channels},
        37,
        1.0f / 53.0f);
    const auto weight = patterned_bfloat(
        static_cast<std::size_t>(channels) * kernel,
        Shape{channels, kernel},
        29,
        1.0f / 4096.0f);
    const auto state = patterned_bfloat(
        static_cast<std::size_t>(batch) * state_length * channels,
        Shape{batch, state_length, channels},
        43,
        1.0f / 61.0f);

    auto combined = concatenate({state, input}, 1);
    auto reference_output = zeros(
        Shape{batch, 1, channels}, float32);
    auto weight_float = astype(weight, float32);
    for (int tap = 0; tap < kernel; ++tap) {
        auto source = slice(
            combined,
            Shape{0, tap * dilation, 0},
            Shape{batch, tap * dilation + 1, channels});
        auto coefficient = reshape(
            slice(
                weight_float,
                Shape{0, tap},
                Shape{channels, tap + 1}),
            Shape{1, 1, channels});
        reference_output = reference_output +
            astype(source, float32) * coefficient;
    }
    reference_output = astype(
        reference_output * sigmoid(reference_output),
        bfloat16);
    auto reference_state = contiguous(slice(
        combined,
        Shape{0, 1, 0},
        Shape{batch, state_length + 1, channels}));

    auto actual = mfq::metal::cached_depthwise_conv_silu(
        input,
        weight,
        std::optional<array>(state),
        dilation);
    require_bit_exact(
        std::move(actual.output),
        std::move(reference_output),
        "cached depthwise convolution output");
    require_bit_exact(
        std::move(actual.state),
        std::move(reference_state),
        "cached depthwise convolution state");
}

} // namespace

int main(int argc, char** argv) {
    try {
        using namespace mlx::core;

        test_gdn_gates();
        test_gdn_decode_step(1, 2, float16, float16);
        test_gdn_decode_step(16, 48, float16, float32);
        test_gdn_decode_step(16, 48, float16, bfloat16);
        test_gdn_decode_step(2, 6, bfloat16, bfloat16);
        test_gdn_decode_step(1, 1, float32, float32);
        if (argc == 2 && std::string(argv[1]) == "--benchmark-gates") {
            benchmark_gdn_gates();
            return 0;
        }
        test_cached_depthwise_dilated_decode();
        test_blocked_gdn_prefill(false);
        test_blocked_gdn_prefill(true);
        test_gdn_state_only();

        const array conv_input(
            {
                1.0f, 2.0f,
                3.0f, 4.0f,
                5.0f, 6.0f,
            },
            Shape{1, 3, 2});
        const array conv_weight(
            {
                1.0f, 2.0f,
                -1.0f, 0.5f,
            },
            Shape{2, 2});
        const array conv_bias(
            {0.5f, -0.5f},
            Shape{2});
        auto convolved = mfq::metal::ssm_conv_silu(
            conv_input,
            conv_weight,
            2,
            conv_bias);
        convolved.eval();
        const auto* convolved_values = convolved.data<float>();
        require_close(convolved_values[0], silu(7.5f));
        require_close(convolved_values[1], silu(-0.5f));
        require_close(convolved_values[2], silu(13.5f));
        require_close(convolved_values[3], silu(-1.5f));

        constexpr int dimension = 32;
        constexpr int qk_width = 2 * dimension;
        constexpr int value_width = dimension;
        constexpr int channels = qk_width + value_width;
        std::vector<float> qk_data(2 * qk_width, 1.0f);
        std::vector<float> value_data(2 * value_width, 1.0f);
        std::vector<float> weight_data(channels * 2, 0.0f);
        std::vector<float> state_data(channels, 0.0f);
        std::vector<float> gate_data(2, 0.0f);
        std::vector<float> beta_data(2, 1.0f);
        std::vector<float> recurrent_value_data(
            2 * dimension,
            1.0f);
        for (int channel = 0; channel < channels; ++channel) {
            weight_data[channel * 2 + 1] = 1.0f;
        }
        const array state(
            state_data.begin(),
            Shape{1, 1, channels});
        const array qk_values(qk_data.begin(), Shape{1, 2, qk_width});
        const array value_values(
            value_data.begin(),
            Shape{1, 2, value_width});
        const array weights(
            weight_data.begin(),
            Shape{channels, 2});

        auto projected = mfq::metal::linear_conv_qkv(
            state,
            qk_values,
            value_values,
            weights,
            1,
            1,
            dimension,
            dimension);
        projected.query.eval();
        projected.key.eval();
        projected.value.eval();
        projected.state.eval();
        const float normalized =
            1.0f / std::sqrt(static_cast<float>(dimension));
        for (int index = 0; index < 2 * dimension; ++index) {
            require_close(
                projected.query.data<float>()[index],
                normalized);
            require_close(
                projected.key.data<float>()[index],
                normalized);
            require_close(
                projected.value.data<float>()[index],
                silu(1.0f));
        }
        const auto* next_state = projected.state.data<float>();
        for (int index = 0; index < channels; ++index) {
            require_close(next_state[index], 1.0f);
        }

        const array gate(
            gate_data.begin(),
            Shape{1, 1, 2});
        const array beta(
            beta_data.begin(),
            Shape{1, 1, 2});
        const array recurrent_value(
            recurrent_value_data.begin(),
            Shape{1, 1, 2, dimension});
        auto recurrent = mfq::metal::gated_delta_net(
            projected.query,
            projected.key,
            recurrent_value,
            gate,
            beta);
        recurrent.output.eval();
        recurrent.state.eval();
        for (int index = 0; index < 2 * dimension; ++index) {
            require_close(
                recurrent.output.data<float>()[index],
                normalized);
        }
        for (std::size_t index = 0;
             index < recurrent.state.size();
             ++index) {
            require_close(
                recurrent.state.data<float>()[index],
                normalized);
        }

        const auto initial_recurrent = zeros(
            Shape{1, 1, dimension, dimension}, float32);
        mfq::metal::MlxGatedDeltaSpeculativeState transaction{
            state,
            initial_recurrent,
            qk_values,
            value_values,
            gate,
            beta,
            7,
            1,
            1,
            2,
        };
        auto restored =
            mfq::metal::replay_gated_delta_speculative_prefix(
                transaction,
                0,
                weights,
                1,
                1,
                dimension,
                dimension);
        auto prefix_convolution = mfq::metal::linear_conv_qkv(
            state,
            slice(
                qk_values,
                Shape{0, 0, 0},
                Shape{1, 1, qk_width}),
            slice(
                value_values,
                Shape{0, 0, 0},
                Shape{1, 1, value_width}),
            weights,
            1,
            1,
            dimension,
            dimension);
        auto prefix_recurrent = mfq::metal::gated_delta_net(
            prefix_convolution.query,
            prefix_convolution.key,
            prefix_convolution.value,
            slice(gate, Shape{0, 0, 0}, Shape{1, 1, 1}),
            slice(beta, Shape{0, 0, 0}, Shape{1, 1, 1}),
            initial_recurrent);
        eval(
            restored.convolution_state,
            restored.recurrent_state,
            prefix_convolution.state,
            prefix_recurrent.state);
        if (restored.position != 8) {
            throw std::runtime_error(
                "speculative recurrent replay position mismatch");
        }
        require_vector_close(
            restored.convolution_state.data<float>(),
            std::vector<float>(
                prefix_convolution.state.data<float>(),
                prefix_convolution.state.data<float>() +
                    prefix_convolution.state.size()),
            0.0f,
            "speculative convolution replay");
        require_vector_close(
            restored.recurrent_state.data<float>(),
            std::vector<float>(
                prefix_recurrent.state.data<float>(),
                prefix_recurrent.state.data<float>() +
                    prefix_recurrent.state.size()),
            2e-4f,
            "speculative recurrent replay");

        auto compiled_replay = mfq::metal::compile_gated_delta_speculative_replay(
            weights, 1, 1, dimension, dimension);
        for (int accepted = 0; accepted <= 1; ++accepted) {
            auto expected = mfq::metal::replay_gated_delta_speculative_prefix(
                transaction, accepted, weights, 1, 1, dimension, dimension);
            auto actual = compiled_replay(transaction, accepted);
            eval(expected.convolution_state, expected.recurrent_state,
                actual.convolution_state, actual.recurrent_state);
            if (actual.position != expected.position)
                throw std::runtime_error("compiled recurrent replay position mismatch");
            for (const auto& pair : {std::pair{expected.convolution_state, actual.convolution_state},
                    std::pair{expected.recurrent_state, actual.recurrent_state}}) {
                require_vector_close(pair.second.data<float>(),
                    std::vector<float>(pair.first.data<float>(),
                        pair.first.data<float>() + pair.first.size()),
                    0.0f, "compiled recurrent replay state");
            }
        }
        bool rejected_replay = false;
        try { (void)compiled_replay(transaction, 2); }
        catch (const std::runtime_error&) { rejected_replay = true; }
        if (!rejected_replay) throw std::runtime_error("compiled replay accepted an invalid prefix");

        std::cout
            << "MFQ C++ Gated DeltaNet Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
