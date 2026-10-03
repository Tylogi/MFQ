#pragma once

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxGatedDeltaNetResult {
    mlx::core::array output;
    mlx::core::array state;
};

struct MlxGatedDeltaGates {
    mlx::core::array gate;
    mlx::core::array beta;
};

struct MlxGatedDeltaDecodeResult {
    mlx::core::array output;
    mlx::core::array convolution_state;
    mlx::core::array recurrent_state;
};

// Projected inputs are [B,T,H]; outputs are FP32 [B,H,T]. The caller
// precomputes decay_scale = -exp(a_log), preserving the model's storage
// precision before promotion. No packed-weight format is involved here.
MlxGatedDeltaGates gated_delta_gates(
    const mlx::core::array& alpha,
    const mlx::core::array& beta,
    const mlx::core::array& dt_bias,
    const mlx::core::array& decay_scale);

// Single-row decode fast path for scalar-gated DeltaNet layers. The recurrent
// state uses the transposed [value,key] layout accepted by gated_delta_net.
MlxGatedDeltaDecodeResult gated_delta_decode_step(
    const mlx::core::array& qk,
    const mlx::core::array& value,
    const mlx::core::array& output_gate,
    const mlx::core::array& alpha,
    const mlx::core::array& beta,
    const mlx::core::array& convolution_state,
    const mlx::core::array& recurrent_state,
    const mlx::core::array& convolution_weight,
    const mlx::core::array& dt_bias,
    const mlx::core::array& decay_scale,
    const mlx::core::array& norm_weight,
    int key_heads,
    int value_heads,
    int dimension,
    float convolution_eps,
    float norm_eps,
    bool output_gate_silu = false);

struct MlxLinearConvQkvResult {
    mlx::core::array query;
    mlx::core::array key;
    mlx::core::array value;
    mlx::core::array state;
};

struct MlxCachedDepthwiseConvResult {
    mlx::core::array output;
    mlx::core::array state;
};

// Architecture-neutral transaction payload for speculative verification over
// a recurrent Gated DeltaNet cache.  The expensive input projections are
// evaluated for the complete [confirmed, drafts...] window once; rollback
// replays only the retained recurrent prefix from these projected values.
struct MlxGatedDeltaSpeculativeState {
    mlx::core::array convolution_state;
    mlx::core::array recurrent_state;
    std::optional<mlx::core::array> qk;
    std::optional<mlx::core::array> value;
    std::optional<mlx::core::array> gate;
    std::optional<mlx::core::array> beta;
    int position = 0;
    int batch = 0;
    int confirmed_tokens = 0;
    int total_tokens = 0;
};

struct MlxGatedDeltaCacheState {
    mlx::core::array convolution_state;
    mlx::core::array recurrent_state;
    int position = 0;
};

MlxGatedDeltaNetResult gated_delta_net(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    const mlx::core::array& gate,
    const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state = std::nullopt,
    bool transposed_state = false,
    bool tiled_heads = false);

mlx::core::array ssm_conv_silu(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    int tokens,
    const std::optional<mlx::core::array>& bias = std::nullopt);

MlxCachedDepthwiseConvResult cached_depthwise_conv_silu(
    const mlx::core::array& input,
    const mlx::core::array& weight,
    const std::optional<mlx::core::array>& state = std::nullopt,
    int dilation = 1,
    const std::optional<mlx::core::array>& bias = std::nullopt);

MlxLinearConvQkvResult linear_conv_qkv(
    const mlx::core::array& state,
    const mlx::core::array& qk,
    const mlx::core::array& value,
    const mlx::core::array& weight,
    int key_heads,
    int value_heads,
    int key_head_dimension,
    int value_head_dimension,
    const std::optional<mlx::core::array>& bias = std::nullopt,
    float eps = 1e-5f);

MlxGatedDeltaCacheState replay_gated_delta_speculative_prefix(
    const MlxGatedDeltaSpeculativeState& transaction,
    int accepted_tokens,
    const mlx::core::array& convolution_weight,
    int key_heads,
    int value_heads,
    int key_head_dimension,
    int value_head_dimension,
    const std::optional<mlx::core::array>& convolution_bias = std::nullopt,
    float eps = 1e-5f,
    bool transposed_state = false,
    bool tiled_heads = false);

} // namespace mfq::metal
