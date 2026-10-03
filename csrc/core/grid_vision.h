#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mfq {

// Backend-neutral geometry and canonical tensor roles for grid_vit components.
struct GridShape {
    std::int32_t temporal = 0;
    std::int32_t height = 0;
    std::int32_t width = 0;

    bool operator==(const GridShape& other) const noexcept {
        return temporal == other.temporal && height == other.height &&
            width == other.width;
    }
    bool operator!=(const GridShape& other) const noexcept {
        return !(*this == other);
    }
};

struct GridVisionConfig {
    std::int64_t hidden_size = 0;
    std::int64_t intermediate_size = 0;
    std::int64_t depth = 0;
    std::int64_t num_heads = 0;
    std::int64_t in_channels = 3;
    std::int64_t patch_size = 0;
    std::int64_t temporal_patch_size = 0;
    std::int64_t spatial_merge_size = 0;
    std::int64_t out_hidden_size = 0;
    std::int64_t num_position_embeddings = 0;
    double rope_theta = 10'000.0;
    double layer_norm_eps = 1e-6;

    void validate() const;
    std::int64_t head_dim() const noexcept;
    std::int64_t patch_width() const noexcept;
};

enum class GridVisionTensorRole {
    patch_weight,
    patch_bias,
    position_weight,
    block_norm1_weight,
    block_norm1_bias,
    block_attention_qkv_weight,
    block_attention_qkv_bias,
    block_attention_output_weight,
    block_attention_output_bias,
    block_norm2_weight,
    block_norm2_bias,
    block_mlp_up_weight,
    block_mlp_up_bias,
    block_mlp_down_weight,
    block_mlp_down_bias,
    merger_norm_weight,
    merger_norm_bias,
    merger_mlp_up_weight,
    merger_mlp_up_bias,
    merger_mlp_down_weight,
    merger_mlp_down_bias,
};

std::string grid_vision_canonical_name(
    GridVisionTensorRole role,
    std::size_t block_index = 0);

// Patch order is merge-block-major within each frame, matching preprocessing.
struct GridVisionLayout {
    std::vector<std::int32_t> positions; // flattened [patches, 2] row/column
    std::vector<std::int32_t> segment_lengths; // one noncausal segment/frame
    std::int64_t patch_count = 0;
};

struct LearnedPositionInterpolation {
    std::vector<std::int32_t> indices; // flattened [patches, 4]
    std::vector<float> weights; // flattened [patches, 4]
    std::int64_t patch_count = 0;
};

GridVisionLayout make_grid_vision_layout(
    const std::vector<GridShape>& grids,
    std::int32_t spatial_merge_size);

LearnedPositionInterpolation make_learned_position_interpolation(
    const std::vector<GridShape>& grids,
    std::int32_t spatial_merge_size,
    std::int32_t position_side);

struct GridMropePositions {
    std::vector<std::int32_t> values; // axis-major [3, tokens]
    std::int64_t token_count = 0;
    int decode_delta = 0;
};

GridMropePositions build_grid_mrope_positions(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t image_token_id,
    std::int64_t video_token_id,
    std::int32_t spatial_merge_size,
    const std::vector<GridShape>& image_grids,
    const std::vector<GridShape>& video_grids);

} // namespace mfq
