#pragma once

#include "nemotron_h_model.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxNemotronHRecurrentSnapshot {
    mlx::core::array convolution_state = mlx::core::array(0.0f);
    mlx::core::array recurrent_state = mlx::core::array(0.0f);
    int position = 0;
    int batch = 0;

    std::size_t nbytes() const noexcept {
        return convolution_state.nbytes() + recurrent_state.nbytes();
    }
};

// Mamba-2 mixer: fused in-projection (z, xBC, dt), causal depthwise
// convolution, selective state scan, grouped gated RMSNorm and out-projection.
// The convolution and scan states are float32 and are checkpointed across a
// speculative verification window.
class MlxNemotronHLinearAttentionBlock {
public:
    static MlxNemotronHLinearAttentionBlock load(
        const MfqContainer& model,
        const NemotronHConfig& config,
        const std::string& tensor_root,
        std::size_t layer_index);

    MlxNemotronHLinearAttentionBlock(
        NemotronHConfig config,
        MlxRmsNorm mixer_norm,
        MlxLinear in_projection,
        mlx::core::array convolution_weight,
        std::optional<mlx::core::array> convolution_bias,
        mlx::core::array decay,
        mlx::core::array skip,
        mlx::core::array dt_bias,
        mlx::core::array scan_weight,
        MlxLinear out_projection);

    // Returns the mixer branch, not the residual sum.
    mlx::core::array forward(
        const mlx::core::array& input,
        bool use_cache);
    mlx::core::array forward(
        const mlx::core::array& input,
        int position_offset,
        bool use_cache);
    // Verify a confirmed prefix plus a speculative suffix in one recurrent
    // chunk. Rejection replays only the retained window rows.
    mlx::core::array forward_speculative(
        const mlx::core::array& input,
        int position_offset,
        int confirmed_tokens);
    void commit_speculative() noexcept;
    void rollback_speculative(int accepted_tokens);

    void reset_cache(int batch);
    void materialize_cache();
    void clear_cache() noexcept;
    MlxNemotronHRecurrentSnapshot snapshot_cache() const;
    void restore_cache(const MlxNemotronHRecurrentSnapshot& snapshot);

    int cache_position() const noexcept {
        return cache_position_;
    }
    int cache_batch() const noexcept {
        return cache_batch_;
    }
    bool speculative_pending() const noexcept {
        return speculative_pending_;
    }

private:
    void validate_components() const;
    mlx::core::array project_and_scan(
        const mlx::core::array& input,
        int batch,
        int tokens);
    void replay_recurrent(int retained_tokens);

    NemotronHConfig config_;
    MlxRmsNorm mixer_norm_;
    MlxLinear in_projection_;
    mlx::core::array convolution_weight_;
    std::optional<mlx::core::array> convolution_bias_;
    mlx::core::array decay_;
    mlx::core::array skip_;
    mlx::core::array dt_bias_;
    mlx::core::array scan_weight_;
    MlxLinear out_projection_;
    std::optional<mlx::core::array> convolution_state_;
    std::optional<mlx::core::array> recurrent_state_;
    std::optional<mlx::core::array> zero_convolution_state_;
    std::optional<mlx::core::array> zero_recurrent_state_;
    int zero_cache_batch_ = 0;
    int cache_position_ = 0;
    int cache_batch_ = 0;
    // Checkpoint of the confirmed state plus the projected window rows.
    std::optional<mlx::core::array> checkpoint_convolution_;
    std::optional<mlx::core::array> checkpoint_recurrent_;
    mlx::core::array speculative_xbc_ = mlx::core::array(0.0f);
    mlx::core::array speculative_dt_ = mlx::core::array(0.0f);
    int speculative_start_ = -1;
    int speculative_confirmed_ = 0;
    int speculative_tokens_ = 0;
    bool speculative_pending_ = false;
};

} // namespace mfq::metal
