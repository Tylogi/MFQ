#include "axial_patch_vision.h"

#include <cmath>
#include <limits>

namespace mfq::models::axial_patch_vision {
namespace {

int checked_positive(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid axial patch vision ") + name);
    }
    return static_cast<int>(value);
}

} // namespace

void Config::validate() const {
    const int selected_hidden = checked_positive(hidden, "hidden size");
    const int selected_heads = checked_positive(heads, "head count");
    checked_positive(layers, "layer count");
    checked_positive(intermediate, "intermediate size");
    checked_positive(patch_size, "patch size");
    checked_positive(downsample_ratio, "downsample ratio");
    checked_positive(output_hidden, "output hidden size");
    if (selected_hidden % selected_heads != 0 ||
        (selected_hidden / selected_heads) % 4 != 0 ||
        !std::isfinite(rope_theta) || rope_theta <= 0.0 ||
        !std::isfinite(rms_eps) || rms_eps <= 0.0) {
        throw std::invalid_argument(
            "invalid axial patch vision geometry or numeric config");
    }
}

AlignedGrid aligned_grid(int vit_height, int vit_width, int ratio) {
    if (vit_height <= 0 || vit_width <= 0 || ratio <= 0) {
        throw std::invalid_argument(
            "axial patch vision grid and downsample ratio must be positive");
    }
    return {
        (vit_height + ratio - 1) / ratio,
        (vit_width + ratio - 1) / ratio,
    };
}

std::vector<std::int32_t> gather_indices(
    int height,
    int width,
    int ratio) {
    const auto grid = aligned_grid(height, width, ratio);
    const int padding = height * width;
    std::vector<std::int32_t> indices;
    indices.reserve(static_cast<std::size_t>(grid.count() * ratio * ratio));
    for (int out_row = 0; out_row < grid.height; ++out_row) {
        for (int out_column = 0; out_column < grid.width; ++out_column) {
            for (int patch_row = 0; patch_row < ratio; ++patch_row) {
                for (int patch_column = 0; patch_column < ratio; ++patch_column) {
                    const int row = out_row * ratio + patch_row;
                    const int column = out_column * ratio + patch_column;
                    indices.push_back(
                        row < height && column < width
                            ? row * width + column
                            : padding);
                }
            }
        }
    }
    return indices;
}

std::vector<std::int64_t> span_token_types(
    int llm_height,
    int llm_width) {
    if (llm_height <= 0 || llm_width <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image span grid must be positive");
    }
    std::vector<std::int64_t> types;
    types.reserve(static_cast<std::size_t>(
        llm_height * (llm_width + 1) + 2));
    types.push_back(0);
    for (int row = 0; row < llm_height; ++row) {
        types.insert(
            types.end(), static_cast<std::size_t>(llm_width), 1);
        types.push_back(2);
    }
    types.push_back(3);
    return types;
}

void validate_span(
    int begin,
    int end,
    int vit_height,
    int vit_width,
    int llm_height,
    int llm_width,
    int prompt_tokens,
    int downsample_ratio,
    const std::vector<std::int64_t>& token_types) {
    if (begin < 0 || end <= begin || end > prompt_tokens) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image span positions are invalid");
    }
    if (vit_height <= 0 || vit_width <= 0 ||
        llm_height <= 0 || llm_width <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image span grid must be positive");
    }
    const auto aligned = aligned_grid(
        vit_height, vit_width, downsample_ratio);
    if (llm_height != aligned.height || llm_width != aligned.width) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image span grid disagrees with the aligner");
    }
    const auto expected = span_token_types(llm_height, llm_width);
    if (token_types.size() !=
            static_cast<std::size_t>(end - begin) ||
        token_types != expected) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image span token layout is invalid");
    }
}

} // namespace mfq::models::axial_patch_vision
