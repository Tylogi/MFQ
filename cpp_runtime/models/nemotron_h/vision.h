#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace mfq::models::nemotron_h {

// RADIO (C-RADIO) vision tower geometry and the projection into the language
// model.  Every field comes from the released omni config: the tower from
// `vision_config`, the projector and the placeholder from the outer section.
struct RadioVisionConfig {
    std::int64_t hidden_size = 0;
    std::int64_t intermediate_size = 0;
    std::int64_t depth = 0;
    std::int64_t num_heads = 0;
    std::int64_t num_channels = 0;
    std::int64_t patch_size = 0;
    std::int64_t max_img_size = 0;
    std::int64_t num_cls_tokens = 0;
    std::int64_t num_registers = 0;
    double layer_norm_eps = 1e-6;
    // `downsample_ratio` 0.5 as the integer pixel-shuffle factor.
    std::int64_t downsample_factor = 0;
    std::int64_t projector_intermediate_size = 0;
    std::int64_t text_hidden_size = 0;
    // The projector's RMSNorm keeps the Megatron default epsilon.
    double projector_norm_eps = 1e-5;
    std::int64_t image_token_id = -1;

    std::int64_t head_dim() const noexcept { return hidden_size / num_heads; }
    // Side of the cropped-position table in patches (max_img_size / patch_size).
    std::int64_t position_side() const noexcept { return max_img_size / patch_size; }
    std::int64_t patch_width() const noexcept {
        return num_channels * patch_size * patch_size;
    }
    // cls + register tokens prepended by the patch generator; the tower's
    // `features` output drops exactly these.
    std::int64_t summary_tokens() const noexcept { return num_cls_tokens + num_registers; }
    // Width one pixel-shuffled patch group is projected from.
    std::int64_t shuffle_width() const noexcept {
        return hidden_size * downsample_factor * downsample_factor;
    }

    void validate() const;
    static RadioVisionConfig from_json(std::string_view payload);
};

// Four-tap bilinear interpolation of the cropped-position table onto one
// image's patch grid.  The reference (`RadioPatchEmbeddings.
// _interpolate_position_embedding`) resamples the square table to
// `max(rows, columns)` and then crops it, which is exactly a per-axis bilinear
// mapping with scale `position_side / max(rows, columns)`; both axes use
// `align_corners=False` semantics with the source coordinate clamped at zero.
struct RadioPositionInterpolation {
    std::vector<std::int32_t> indices; // flattened [patches, 4] table rows
    std::vector<float> weights; // flattened [patches, 4]
    std::int64_t patch_count = 0;
};

RadioPositionInterpolation make_radio_position_interpolation(
        std::int32_t rows, std::int32_t columns, std::int32_t position_side);

} // namespace mfq::models::nemotron_h
