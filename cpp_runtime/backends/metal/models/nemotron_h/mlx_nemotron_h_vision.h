#pragma once

#include "mfq_container.h"
#include "mlx_multimodal.h"
#include "mlx_pre_ln_vision.h"
#include "models/nemotron_h/vision.h"
#include "nemotron_h_model.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// RADIO patch generator + pre-LayerNorm ViT encoder. `encode` accepts
// normalized [patch_count, patch_width] pixels and returns the tower's
// `features` rows (the cls/register prefix is dropped again).
class MlxRadioEncoder {
public:
    static MlxRadioEncoder load(
        const MfqContainer& model,
        const NemotronHRadioConfig& config);

    MlxRadioEncoder(
        NemotronHRadioConfig config,
        mlx::core::array patch_projection,
        mlx::core::array position_embedding,
        mlx::core::array cls_register_token,
        std::vector<MlxPreLnVisionBlock> layers);

    mlx::core::array encode(
        const mlx::core::array& patches,
        std::int64_t rows,
        std::int64_t columns) const;
    // Patch projection + interpolated positions + cls/register prefix, before
    // the first encoder block.
    mlx::core::array embedding_input(
        const mlx::core::array& patches,
        std::int64_t rows,
        std::int64_t columns) const;

    const NemotronHRadioConfig& config() const noexcept {
        return config_;
    }

private:
    void validate_components() const;
    mlx::core::array interpolated_positions(
        std::int64_t rows,
        std::int64_t columns) const;

    NemotronHRadioConfig config_;
    mlx::core::array patch_projection_;
    mlx::core::array position_embedding_;
    mlx::core::array cls_register_token_;
    std::vector<MlxPreLnVisionBlock> layers_;
};

// `vision_final_layernorm` -> pixel shuffle v2 -> RMSNorm -> relu^2 MLP.
class MlxRadioProjector {
public:
    static MlxRadioProjector load(
        const MfqContainer& model,
        const NemotronHRadioConfig& config);

    MlxRadioProjector(
        NemotronHRadioConfig config,
        MlxVisionLayerNorm norm,
        MlxRmsNorm mlp_norm,
        MlxLinear up,
        MlxLinear down);

    mlx::core::array project(
        const mlx::core::array& features,
        std::int64_t rows,
        std::int64_t columns) const;

private:
    void validate_components() const;

    NemotronHRadioConfig config_;
    MlxVisionLayerNorm norm_;
    MlxRmsNorm mlp_norm_;
    MlxLinear up_;
    MlxLinear down_;
};

class MlxRadioVisionTower {
public:
    static MlxRadioVisionTower load(
        const MfqContainer& model,
        const NemotronHRadioConfig& config);

    mlx::core::array embedding_input(
        const mlx::core::array& patches,
        std::int64_t rows,
        std::int64_t columns) const;
    mlx::core::array features(
        const mlx::core::array& patches,
        std::int64_t rows,
        std::int64_t columns) const;
    mlx::core::array embed(
        const mlx::core::array& patches,
        std::int64_t rows,
        std::int64_t columns) const;
    std::int64_t tokens(std::int64_t rows, std::int64_t columns) const noexcept;
    const NemotronHRadioConfig& config() const noexcept {
        return encoder_.config();
    }

private:
    MlxRadioVisionTower(MlxRadioEncoder encoder, MlxRadioProjector projector)
        : encoder_(std::move(encoder)), projector_(std::move(projector)) {}

    MlxRadioEncoder encoder_;
    MlxRadioProjector projector_;
};

// Graph-selected RADIO prompt component: rewrites the `<image>` rows of the
// prompt embeddings. Nemotron-H is NoPE, so positions are plain offsets.
class MlxNemotronHRadioPromptComponent {
public:
    static MlxNemotronHRadioPromptComponent load(
        const MfqContainer& model,
        const NemotronHRadioConfig& config,
        std::string input_contract,
        std::string position_policy);

    MlxPreparedPrompt prepare(
        const std::vector<std::int64_t>& token_ids,
        const mlx::core::array& text_embeddings,
        const MlxGridMediaInput& media) const;

    const std::string& input_contract() const noexcept {
        return input_contract_;
    }
    const NemotronHRadioConfig& config() const noexcept {
        return tower_.config();
    }
    const MlxRadioVisionTower& tower() const noexcept {
        return tower_;
    }

private:
    MlxNemotronHRadioPromptComponent(
        MlxRadioVisionTower tower,
        NemotronHRadioConfig config,
        std::string input_contract)
        : tower_(std::move(tower)),
          config_(std::move(config)),
          input_contract_(std::move(input_contract)) {}

    std::vector<std::pair<std::int64_t, std::int64_t>> validated_grids(
        const MlxGridMediaInput& media) const;
    std::vector<std::vector<std::int64_t>> placeholder_runs(
        const std::vector<std::int64_t>& token_ids,
        const std::vector<std::pair<std::int64_t, std::int64_t>>& grids) const;

    MlxRadioVisionTower tower_;
    NemotronHRadioConfig config_;
    std::string input_contract_;
};

} // namespace mfq::metal
