#include "grid_vision.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq {
namespace {

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("invalid grid-ViT ") + name);
    }
    return static_cast<int>(value);
}

void validate_grid(const GridShape& grid, std::int32_t merge, bool image = false) {
    if (merge <= 0 || grid.temporal <= 0 || grid.height <= 0 || grid.width <= 0 ||
        grid.height % merge != 0 || grid.width % merge != 0 ||
        (image && grid.temporal != 1)) {
        throw std::invalid_argument(
            "grid-ViT THW must be positive and spatially divisible by merge size");
    }
    const auto patches = static_cast<std::int64_t>(grid.temporal) *
        grid.height * grid.width;
    if (patches > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("grid-ViT patch count exceeds native limits");
    }
}

std::vector<std::pair<std::int32_t, std::int32_t>> spatial_positions(
    const GridShape& grid,
    std::int32_t merge) {
    std::vector<std::pair<std::int32_t, std::int32_t>> result;
    result.reserve(static_cast<std::size_t>(grid.height * grid.width));
    for (std::int32_t block_row = 0; block_row < grid.height / merge; ++block_row) {
        for (std::int32_t block_column = 0; block_column < grid.width / merge;
             ++block_column) {
            for (std::int32_t inner_row = 0; inner_row < merge; ++inner_row) {
                for (std::int32_t inner_column = 0; inner_column < merge;
                     ++inner_column) {
                    result.emplace_back(
                        block_row * merge + inner_row,
                        block_column * merge + inner_column);
                }
            }
        }
    }
    return result;
}

} // namespace

void GridVisionConfig::validate() const {
    const int hidden = checked_int(hidden_size, "hidden size");
    const int heads = checked_int(num_heads, "head count");
    checked_int(intermediate_size, "intermediate size");
    checked_int(depth, "depth");
    checked_int(in_channels, "channel count");
    checked_int(patch_size, "patch size");
    checked_int(temporal_patch_size, "temporal patch size");
    checked_int(spatial_merge_size, "spatial merge size");
    checked_int(out_hidden_size, "output hidden size");
    const int positions = checked_int(num_position_embeddings, "position table size");
    const int side = static_cast<int>(std::sqrt(positions));
    if (hidden % heads != 0 || (hidden / heads) % 4 != 0 ||
        side * side != positions || !std::isfinite(rope_theta) ||
        rope_theta <= 0.0 || !std::isfinite(layer_norm_eps) ||
        layer_norm_eps <= 0.0) {
        throw std::invalid_argument("invalid grid-ViT geometry or numeric config");
    }
    checked_int(patch_width(), "flattened patch width");
}

std::int64_t GridVisionConfig::head_dim() const noexcept {
    return num_heads > 0 ? hidden_size / num_heads : 0;
}

std::int64_t GridVisionConfig::patch_width() const noexcept {
    return in_channels * temporal_patch_size * patch_size * patch_size;
}

std::string grid_vision_canonical_name(
    GridVisionTensorRole role,
    std::size_t block_index) {
    const auto block = [&](const char* suffix) {
        return "vision.block." + std::to_string(block_index) + suffix;
    };
    switch (role) {
    case GridVisionTensorRole::patch_weight: return "vision.patch_embedding.weight";
    case GridVisionTensorRole::patch_bias: return "vision.patch_embedding.bias";
    case GridVisionTensorRole::position_weight: return "vision.position_embedding.weight";
    case GridVisionTensorRole::block_norm1_weight: return block(".norm1.weight");
    case GridVisionTensorRole::block_norm1_bias: return block(".norm1.bias");
    case GridVisionTensorRole::block_attention_qkv_weight: return block(".attention.qkv.weight");
    case GridVisionTensorRole::block_attention_qkv_bias: return block(".attention.qkv.bias");
    case GridVisionTensorRole::block_attention_output_weight: return block(".attention.output.weight");
    case GridVisionTensorRole::block_attention_output_bias: return block(".attention.output.bias");
    case GridVisionTensorRole::block_norm2_weight: return block(".norm2.weight");
    case GridVisionTensorRole::block_norm2_bias: return block(".norm2.bias");
    case GridVisionTensorRole::block_mlp_up_weight: return block(".mlp.up.weight");
    case GridVisionTensorRole::block_mlp_up_bias: return block(".mlp.up.bias");
    case GridVisionTensorRole::block_mlp_down_weight: return block(".mlp.down.weight");
    case GridVisionTensorRole::block_mlp_down_bias: return block(".mlp.down.bias");
    case GridVisionTensorRole::merger_norm_weight: return "vision.merger.norm.weight";
    case GridVisionTensorRole::merger_norm_bias: return "vision.merger.norm.bias";
    case GridVisionTensorRole::merger_mlp_up_weight: return "vision.merger.mlp.up.weight";
    case GridVisionTensorRole::merger_mlp_up_bias: return "vision.merger.mlp.up.bias";
    case GridVisionTensorRole::merger_mlp_down_weight: return "vision.merger.mlp.down.weight";
    case GridVisionTensorRole::merger_mlp_down_bias: return "vision.merger.mlp.down.bias";
    }
    throw std::invalid_argument("unknown canonical grid-Vision tensor role");
}

GridVisionLayout make_grid_vision_layout(
    const std::vector<GridShape>& grids,
    std::int32_t spatial_merge_size) {
    if (grids.empty()) {
        throw std::invalid_argument("grid-ViT requires at least one THW item");
    }
    GridVisionLayout result;
    for (const auto& grid : grids) {
        validate_grid(grid, spatial_merge_size);
        const auto spatial = spatial_positions(grid, spatial_merge_size);
        for (std::int32_t frame = 0; frame < grid.temporal; ++frame) {
            result.segment_lengths.push_back(grid.height * grid.width);
            for (const auto [row, column] : spatial) {
                result.positions.push_back(row);
                result.positions.push_back(column);
                ++result.patch_count;
            }
        }
    }
    return result;
}

LearnedPositionInterpolation make_learned_position_interpolation(
    const std::vector<GridShape>& grids,
    std::int32_t spatial_merge_size,
    std::int32_t position_side) {
    if (grids.empty() || position_side <= 0) {
        throw std::invalid_argument(
            "grid-ViT learned-position interpolation geometry is invalid");
    }
    LearnedPositionInterpolation result;
    for (const auto& grid : grids) {
        validate_grid(grid, spatial_merge_size);
        const auto spatial = spatial_positions(grid, spatial_merge_size);
        for (std::int32_t frame = 0; frame < grid.temporal; ++frame) {
            for (const auto [row, column] : spatial) {
                const float row_source = static_cast<float>(row) *
                    static_cast<float>(position_side - 1) /
                    static_cast<float>(std::max(grid.height - 1, 1));
                const float column_source = static_cast<float>(column) *
                    static_cast<float>(position_side - 1) /
                    static_cast<float>(std::max(grid.width - 1, 1));
                const auto row_floor = static_cast<std::int32_t>(std::floor(row_source));
                const auto column_floor = static_cast<std::int32_t>(std::floor(column_source));
                const std::int32_t row_taps[2] = {
                    std::clamp(row_floor, 0, position_side - 1),
                    std::clamp(row_floor + 1, 0, position_side - 1),
                };
                const std::int32_t column_taps[2] = {
                    std::clamp(column_floor, 0, position_side - 1),
                    std::clamp(column_floor + 1, 0, position_side - 1),
                };
                const float row_weights[2] = {
                    static_cast<float>(row_floor + 1) - row_source,
                    row_source - static_cast<float>(row_floor),
                };
                const float column_weights[2] = {
                    static_cast<float>(column_floor + 1) - column_source,
                    column_source - static_cast<float>(column_floor),
                };
                for (int row_tap = 0; row_tap < 2; ++row_tap) {
                    for (int column_tap = 0; column_tap < 2; ++column_tap) {
                        result.indices.push_back(
                            row_taps[row_tap] * position_side + column_taps[column_tap]);
                        result.weights.push_back(
                            row_weights[row_tap] * column_weights[column_tap]);
                    }
                }
                ++result.patch_count;
            }
        }
    }
    return result;
}

GridMropePositions build_grid_mrope_positions(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t image_token_id,
    std::int64_t video_token_id,
    std::int32_t spatial_merge_size,
    const std::vector<GridShape>& image_grids,
    const std::vector<GridShape>& video_grids) {
    if (token_ids.empty() || spatial_merge_size <= 0 || image_token_id < 0 ||
        video_token_id < 0 || image_token_id == video_token_id) {
        throw std::invalid_argument("invalid multimodal position-policy configuration");
    }
    for (const auto& grid : image_grids) validate_grid(grid, spatial_merge_size, true);
    std::vector<GridShape> video_frames;
    for (const auto& grid : video_grids) {
        validate_grid(grid, spatial_merge_size);
        video_frames.insert(
            video_frames.end(), static_cast<std::size_t>(grid.temporal),
            GridShape{1, grid.height, grid.width});
    }
    if (token_ids.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("multimodal prompt is too long");
    }

    GridMropePositions result;
    result.token_count = static_cast<std::int64_t>(token_ids.size());
    result.values.resize(3 * token_ids.size());
    std::size_t image = 0;
    std::size_t video = 0;
    std::size_t begin = 0;
    int current = 0;
    while (begin < token_ids.size()) {
        const int modality = token_ids[begin] == image_token_id ? 1 :
            (token_ids[begin] == video_token_id ? 2 : 0);
        std::size_t end = begin + 1;
        while (end < token_ids.size()) {
            const int next = token_ids[end] == image_token_id ? 1 :
                (token_ids[end] == video_token_id ? 2 : 0);
            if (next != modality) break;
            ++end;
        }
        const auto count = end - begin;
        if (modality == 0) {
            for (std::size_t index = 0; index < count; ++index) {
                const auto value = static_cast<std::int32_t>(current + index);
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    result.values[axis * token_ids.size() + begin + index] = value;
                }
            }
            current += static_cast<int>(count);
        } else {
            const auto& grids = modality == 1 ? image_grids : video_frames;
            auto& selected = modality == 1 ? image : video;
            if (selected >= grids.size()) {
                throw std::invalid_argument(
                    "prompt contains more multimodal spans than grids");
            }
            const auto& grid = grids[selected++];
            const int rows = grid.height / spatial_merge_size;
            const int columns = grid.width / spatial_merge_size;
            const auto expected = static_cast<std::size_t>(
                grid.temporal * rows * columns);
            if (count != expected) {
                throw std::invalid_argument(
                    "multimodal placeholder span disagrees with its grid");
            }
            std::size_t index = 0;
            for (int temporal = 0; temporal < grid.temporal; ++temporal) {
                for (int row = 0; row < rows; ++row) {
                    for (int column = 0; column < columns; ++column, ++index) {
                        result.values[begin + index] = current + temporal;
                        result.values[token_ids.size() + begin + index] = current + row;
                        result.values[2 * token_ids.size() + begin + index] =
                            current + column;
                    }
                }
            }
            current += std::max(rows, columns);
        }
        begin = end;
    }
    if (image != image_grids.size() || video != video_frames.size()) {
        throw std::invalid_argument(
            "multimodal grids contain unused image/video entries");
    }
    const auto maximum = *std::max_element(result.values.begin(), result.values.end());
    result.decode_delta = static_cast<int>(maximum) + 1 -
        static_cast<int>(token_ids.size());
    return result;
}

} // namespace mfq
