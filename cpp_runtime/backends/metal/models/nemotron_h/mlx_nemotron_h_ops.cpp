#include "mlx_nemotron_h_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

// Bound on the in-flight [tokens, tokens] score block of one scan chunk.
constexpr std::int64_t kScanChunkTokens = 64;

array float32(const array& value) {
    return mlx::core::astype(value, mlx::core::float32);
}

array scaled_softplus(
    const array& dt,
    const array& dt_bias,
    double low,
    double high) {
    const auto gate = dt + mlx::core::reshape(float32(dt_bias), Shape{1, 1, -1});
    // Stable softplus: max(x, 0) + log1p(exp(-|x|)).
    const auto softplus =
        mlx::core::maximum(gate, array(0.0f, gate.dtype())) +
        mlx::core::log1p(mlx::core::exp(-mlx::core::abs(gate)));
    const auto low_value = array(static_cast<float>(low), gate.dtype());
    const auto high_value = std::isfinite(high)
        ? std::optional<array>(array(static_cast<float>(high), gate.dtype()))
        : std::nullopt;
    return mlx::core::clip(softplus, std::optional<array>(low_value), high_value);
}

array repeat_groups(const array& signal, int heads, int groups) {
    if (heads == groups) return signal;
    if (heads % groups != 0) {
        throw std::runtime_error(
            "Nemotron-H selective scan needs heads divisible by groups");
    }
    return mlx::core::repeat(signal, heads / groups, 2);
}

} // namespace

array round_to_float16(const array& input) {
    const auto rounded = mlx::core::astype(input, mlx::core::float16);
    return input.dtype() == mlx::core::float16
        ? rounded
        : mlx::core::astype(rounded, input.dtype());
}

array relu_squared(const array& input) {
    const auto zero = array(0.0f, input.dtype());
    const auto positive = mlx::core::maximum(input, zero);
    return positive * positive;
}

MlxSelectiveScanResult selective_state_scan(
    const array& x,
    const array& dt,
    const array& dt_bias,
    const array& decay,
    const array& b_signal,
    const array& c_signal,
    const array& skip,
    const std::optional<array>& state,
    double time_step_limit_low,
    double time_step_limit_high) {
    if (x.ndim() != 4 || dt.ndim() != 3 || b_signal.ndim() != 4 ||
        c_signal.ndim() != 4) {
        throw std::runtime_error(
            "Nemotron-H selective scan expects x [B,T,H,D], dt [B,T,H] "
            "and B/C [B,T,G,N]");
    }
    const int batch = x.shape(0);
    const int tokens = x.shape(1);
    const int heads = x.shape(2);
    const int head_dim = x.shape(3);
    const int state_width = b_signal.shape(3);
    const int groups = b_signal.shape(2);
    if (batch <= 0 || tokens <= 0 || heads <= 0 || head_dim <= 0 ||
        state_width <= 0 || dt.shape(0) != batch || dt.shape(1) != tokens ||
        dt.shape(2) != heads || c_signal.shape(0) != batch ||
        c_signal.shape(1) != tokens || c_signal.shape(2) != groups ||
        c_signal.shape(3) != state_width || skip.shape() != Shape{heads} ||
        decay.shape() != Shape{heads}) {
        throw std::runtime_error(
            "Nemotron-H selective scan geometry is inconsistent");
    }
    const auto scan_input = float32(x);
    const auto step = scaled_softplus(
        float32(dt), dt_bias, time_step_limit_low, time_step_limit_high);
    const auto log_decay =
        step * mlx::core::reshape(float32(decay), Shape{1, 1, heads});
    const auto b_values = repeat_groups(float32(b_signal), heads, groups);
    const auto c_values = repeat_groups(float32(c_signal), heads, groups);
    const auto state_shape =
        Shape{batch, heads, head_dim, state_width};
    array current_state = state
        ? mlx::core::contiguous(float32(*state))
        : mlx::core::zeros(state_shape, mlx::core::float32);
    if (current_state.shape() != state_shape) {
        throw std::runtime_error(
            "Nemotron-H selective scan state shape mismatch");
    }

    if (tokens == 1) {
        // Explicit single-step recurrence: h = h*exp(step*decay) + step*B x,
        // y = C.h + D x. Kept separate from the chunked form so a decode step
        // never depends on a triangular block of length one.
        const auto value = mlx::core::reshape(
            scan_input, Shape{batch, heads, head_dim});
        const auto b_step = mlx::core::reshape(
            b_values, Shape{batch, heads, state_width});
        const auto c_step = mlx::core::reshape(
            c_values, Shape{batch, heads, state_width});
        const auto step_one = mlx::core::reshape(step, Shape{batch, heads});
        const auto log_decay_one =
            step_one * mlx::core::reshape(
                            float32(decay), Shape{1, heads});
        // step * B x is an outer product per head.
        auto next_state =
            current_state * mlx::core::reshape(
                                mlx::core::exp(log_decay_one),
                                Shape{batch, heads, 1, 1}) +
            (mlx::core::reshape(step_one, Shape{batch, heads, 1, 1}) *
             mlx::core::reshape(b_step, Shape{batch, heads, 1, state_width}) *
             mlx::core::reshape(value, Shape{batch, heads, head_dim, 1}));
        auto output =
            mlx::core::sum(
                next_state * mlx::core::reshape(
                                  c_step, Shape{batch, heads, 1, state_width}),
                -1) +
            value * mlx::core::reshape(float32(skip), Shape{1, heads, 1});
        return {
            mlx::core::reshape(
                output, Shape{batch, 1, heads, head_dim}),
            mlx::core::contiguous(next_state),
        };
    }
    std::vector<array> pieces;
    pieces.reserve(static_cast<std::size_t>(
        (tokens + kScanChunkTokens - 1) / kScanChunkTokens));
    const auto mask = mlx::core::greater_equal(
        mlx::core::reshape(
            mlx::core::arange(tokens, mlx::core::int32), Shape{tokens, 1}),
        mlx::core::reshape(
            mlx::core::arange(tokens, mlx::core::int32), Shape{1, tokens}));
    for (int begin = 0; begin < tokens; begin += kScanChunkTokens) {
        const int length = std::min<std::int64_t>(
            kScanChunkTokens, tokens - begin);
        const auto take = [&](const array& value) {
            Shape start(value.ndim(), 0);
            start.at(1) = begin;
            Shape stop = value.shape();
            stop.at(1) = begin + length;
            return mlx::core::slice(value, start, stop);
        };
        const auto chunk_x = take(scan_input);
        const auto chunk_step = take(step);
        const auto chunk_log_decay = take(log_decay);
        const auto chunk_b = take(b_values);
        const auto chunk_c = take(c_values);
        // c[t] = sum_{i<=t} step_i * decay, so exp(c[t]-c[i]) is the decay
        // product from i+1..t (exactly 1 on the diagonal).
        const auto cumulative =
            mlx::core::cumsum(chunk_log_decay, 1, false, true);
        const auto relative = mlx::core::exp(
            mlx::core::reshape(cumulative, Shape{batch, length, 1, heads}) -
            mlx::core::reshape(cumulative, Shape{batch, 1, length, heads}));
        auto causal = mlx::core::where(
            mlx::core::reshape(
                mlx::core::slice(mask, Shape{0, 0}, Shape{length, length}),
                Shape{1, length, length, 1}),
            relative,
            array(0.0f, relative.dtype()));
        // Head-major views let the pairwise contractions stay plain matmuls;
        // MLX einsum rejects these repeated-label layouts.
        const auto to_head_major = [](const array& value) {
            return mlx::core::transpose(value, {0, 2, 1, 3});
        };
        const auto from_head_major = [](array value) {
            return mlx::core::transpose(std::move(value), {0, 2, 1, 3});
        };
        // Extra/extra: C_t . B_i for every (t,i) pair of the chunk.
        const auto c_dot_b = mlx::core::transpose(
            mlx::core::matmul(
                to_head_major(chunk_c),
                mlx::core::transpose(to_head_major(chunk_b), {0, 1, 3, 2})),
            {0, 2, 3, 1});
        const auto weights =
            causal * c_dot_b *
            mlx::core::reshape(chunk_step, Shape{batch, 1, length, heads});
        const auto inner = mlx::core::transpose(
            mlx::core::matmul(
                mlx::core::transpose(weights, {0, 3, 1, 2}),
                to_head_major(chunk_x)),
            {0, 2, 1, 3});
        // Contribution of the state entering this chunk: C_t . state.
        const auto carried = from_head_major(mlx::core::matmul(
            to_head_major(chunk_c),
            mlx::core::transpose(current_state, {0, 1, 3, 2})));
        auto output = inner +
            carried * mlx::core::reshape(
                          mlx::core::exp(cumulative),
                          Shape{batch, length, heads, 1});
        output = output +
            chunk_x * mlx::core::reshape(
                          float32(skip), Shape{1, 1, heads, 1});
        pieces.push_back(std::move(output));
        // Final state of the chunk: sum_i exp(c_last-c_i) step_i B_i x_i.
        const auto last = mlx::core::reshape(
            mlx::core::slice(
                cumulative, Shape{0, length - 1, 0},
                Shape{batch, length, heads}),
            Shape{batch, 1, heads});
        const auto tail_weights =
            mlx::core::exp(last - cumulative) * chunk_step;
        const auto b_scaled = to_head_major(chunk_b) *
            mlx::core::reshape(
                mlx::core::transpose(tail_weights, {0, 2, 1}),
                Shape{batch, heads, length, 1});
        current_state =
            mlx::core::matmul(
                mlx::core::transpose(
                    to_head_major(chunk_x), {0, 1, 3, 2}),
                b_scaled) +
            current_state * mlx::core::reshape(
                                mlx::core::exp(last),
                                Shape{batch, heads, 1, 1});
    }
    auto output = pieces.size() == 1
        ? pieces.front()
        : mlx::core::concatenate(std::move(pieces), 1);
    return {mlx::core::contiguous(output),
            mlx::core::contiguous(current_state)};
}

array grouped_gated_rms_norm(
    const array& scanned,
    const array& gate,
    const array& weight,
    std::int64_t group_width,
    float eps) {
    if (scanned.ndim() != 2 || gate.shape() != scanned.shape() ||
        group_width <= 0 || scanned.shape(1) % group_width != 0 ||
        weight.shape() != Shape{scanned.shape(1)}) {
        throw std::runtime_error(
            "Nemotron-H grouped gated RMSNorm geometry is invalid");
    }
    const int rows = scanned.shape(0);
    const int width = scanned.shape(1);
    const int groups = width / static_cast<int>(group_width);
    auto gated = scanned * (gate * mlx::core::sigmoid(gate));
    auto grouped = mlx::core::reshape(
        gated, Shape{rows, groups, static_cast<int>(group_width)});
    const auto variance = mlx::core::mean(
        grouped * grouped, -1, true);
    const auto normalized = grouped /
        mlx::core::sqrt(variance + array(eps, grouped.dtype()));
    return mlx::core::reshape(normalized, Shape{rows, width}) *
        mlx::core::reshape(weight, Shape{1, width});
}

} // namespace mfq::metal
