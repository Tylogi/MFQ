#pragma once

#include <cstdint>
#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

// Round through float16 at the activation boundaries the CUDA runtime uses
// (attention operands, MoE entry and the shared-expert output). Keeping this
// explicit lets a float32 test fixture reproduce the reference graph.
mlx::core::array round_to_float16(const mlx::core::array& input);

mlx::core::array relu_squared(const mlx::core::array& input);

struct MlxSelectiveScanResult {
    mlx::core::array output;
    mlx::core::array state;
};

// Mamba-2 selective state scan shared by every linear-attention layer.
//
//   step  = clamp(softplus(dt + dt_bias), low, high)      [B,T,heads]
//   state = state * exp(step * decay) + (step * B) x
//   y     = state . C + D * x
//
// x, B and C are already convolved, gate-free signals; decay is the
// precomputed -exp(a_log). The state keeps float32 [B,heads,head_dim,N] and is
// updated exactly once per call, so chunked prefills reproduce one pass.
MlxSelectiveScanResult selective_state_scan(
    const mlx::core::array& x,
    const mlx::core::array& dt,
    const mlx::core::array& dt_bias,
    const mlx::core::array& decay,
    const mlx::core::array& b_signal,
    const mlx::core::array& c_signal,
    const mlx::core::array& skip,
    const std::optional<mlx::core::array>& state,
    double time_step_limit_low,
    double time_step_limit_high);

// Grouped gated RMSNorm with the gate applied before normalization
// (norm_before_gate=false): each group of a head is normalized independently.
mlx::core::array grouped_gated_rms_norm(
    const mlx::core::array& scanned,
    const mlx::core::array& gate,
    const mlx::core::array& weight,
    std::int64_t group_width,
    float eps);

} // namespace mfq::metal
