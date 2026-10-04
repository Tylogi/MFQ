#pragma once

// Qwen4-Exp model semantics implemented with native MLX/Metal primitives.

#include "mlx_tensor.h"

#include <optional>
#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxQwen4GatedResidualPre {
    mlx::core::array branch;
    mlx::core::array residual;
    std::optional<mlx::core::array> injection;
};

struct MlxQwen4QsaDecodePrologue {
    mlx::core::array query;
    mlx::core::array output_gate;
    mlx::core::array key;
    mlx::core::array index_query;
};

mlx::core::array qwen4_grouped_rms_norm(
    const mlx::core::array& value,
    const mlx::core::array& weight,
    int group_size,
    float eps = 1e-6f);

MlxQwen4GatedResidualPre qwen4_gated_residual_pre(
    const mlx::core::array& hyper_input,
    const mlx::core::array& norm_weight,
    const mlx::core::array& down_weight,
    const mlx::core::array& up_weight,
    const std::optional<mlx::core::array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps = 1e-6f);

MlxQwen4GatedResidualPre qwen4_gated_residual_pre(
    const mlx::core::array& hyper_input,
    const mlx::core::array& norm_weight,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps = 1e-6f);

// Apply the preceding branch's gated residual write while normalizing for
// the next hyper-connection. Decode keeps the exact promoted multiply/add
// and RMS reduction order while removing the intervening materialization
// dispatch.
MlxQwen4GatedResidualPre qwen4_gated_residual_pre_after(
    const mlx::core::array& previous_branch,
    const mlx::core::array& previous_residual,
    const mlx::core::array& previous_injection,
    const mlx::core::array& norm_weight,
    const mlx::core::array& down_weight,
    const mlx::core::array& up_weight,
    const std::optional<mlx::core::array>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps = 1e-6f);

MlxQwen4GatedResidualPre qwen4_gated_residual_pre_after(
    const mlx::core::array& previous_branch,
    const mlx::core::array& previous_residual,
    const mlx::core::array& previous_injection,
    const mlx::core::array& norm_weight,
    const MlxLinear& down_weight,
    const MlxLinear& up_weight,
    const std::optional<MlxLinear>& inject_weight,
    int hidden_size,
    int hc_count,
    float eps = 1e-6f);

mlx::core::array qwen4_gated_residual_post(
    const mlx::core::array& branch,
    const mlx::core::array& residual,
    const mlx::core::array& injection,
    int hc_count);

mlx::core::array qwen4_qsa_block_scores(
    const mlx::core::array& query,
    const mlx::core::array& pooled_keys);

// Decode-only QSA normalization/layout/RoPE prologue. Query, key, and index
// query share one multi-output dispatch; the strided query gate is copied to
// its final contiguous layout in the same pass.
MlxQwen4QsaDecodePrologue qwen4_qsa_decode_prologue(
    const mlx::core::array& query_gate,
    const mlx::core::array& key,
    const mlx::core::array& index_query_key,
    const mlx::core::array& query_norm_weight,
    const mlx::core::array& key_norm_weight,
    const mlx::core::array& index_query_norm_weight,
    const mlx::core::array& positions,
    int query_heads,
    int key_heads,
    int index_heads,
    int head_dimension,
    int index_dimension,
    int rotary_dimension,
    float rope_theta,
    float eps);

mlx::core::array qwen4_dense_gqa_attention(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    int query_offset);

} // namespace mfq::metal
