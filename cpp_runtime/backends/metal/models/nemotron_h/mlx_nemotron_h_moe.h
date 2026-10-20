#pragma once

#include "nemotron_h_model.h"
#include "mlx_moe.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// One pre-normed Nemotron-H MLP block: either the latent non-gated relu-squared
// MoE (router with e_score_correction_bias, sigmoid scores, latent in/out and
// one ungated shared expert) or a plain dense relu-squared MLP. Both read the
// residual stream and return only their branch.
class MlxNemotronHMoeBlock {
public:
    static MlxNemotronHMoeBlock load(
        const MfqContainer& model,
        const NemotronHConfig& config,
        NemotronHMixer kind,
        const std::string& tensor_root,
        std::size_t layer_index);

    // Dense relu-squared MLP block.
    MlxNemotronHMoeBlock(
        NemotronHConfig config,
        MlxRmsNorm ffn_norm,
        MlxLinear up,
        MlxLinear down);

    // Latent MoE block with either a dense expert bank or packed MFE records.
    MlxNemotronHMoeBlock(
        NemotronHConfig config,
        MlxRmsNorm ffn_norm,
        MlxLinear router,
        std::optional<mlx::core::array> router_bias,
        MlxLinear latent_in,
        MlxLinear latent_out,
        std::optional<MlxRoutedLinear> routed_up,
        std::optional<MlxRoutedLinear> routed_down,
        std::optional<mlx::core::array> dense_up,
        std::optional<mlx::core::array> dense_down,
        MlxLinear shared_up,
        MlxLinear shared_down);

    // Returns the mixer branch, not the residual sum.
    mlx::core::array forward(const mlx::core::array& input) const;

    bool is_moe() const noexcept {
        return is_moe_;
    }

private:
    void validate_components() const;
    mlx::core::array forward_moe(const mlx::core::array& input) const;
    mlx::core::array forward_dense(const mlx::core::array& input) const;

    NemotronHConfig config_;
    bool is_moe_ = false;
    MlxRmsNorm ffn_norm_;
    // Dense MLP.
    std::optional<MlxLinear> up_;
    std::optional<MlxLinear> down_;
    // MoE.
    std::optional<MlxLinear> router_;
    std::optional<mlx::core::array> router_bias_;
    std::optional<MlxLinear> latent_in_;
    std::optional<MlxLinear> latent_out_;
    std::optional<MlxRoutedLinear> routed_up_;
    std::optional<MlxRoutedLinear> routed_down_;
    std::optional<mlx::core::array> dense_up_;
    std::optional<mlx::core::array> dense_down_;
    std::optional<MlxLinear> shared_up_;
    std::optional<MlxLinear> shared_down_;
};

} // namespace mfq::metal
