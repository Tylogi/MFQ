#include "text_processor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace mfq::engine {
namespace {
void require_media(bool valid, const char* message, const char* field = "mfq_multimodal") {
    if (!valid) throw InferenceInputError(InferenceInputErrorCode::Invalid, message, field);
}

template <class T>
void check_tensor(const std::vector<T>& values, const std::vector<int64_t>& shape) {
    if (shape.empty() && values.empty()) return;
    require_media(!shape.empty() && shape.size() <= 4, "media tensor rank is invalid");
    size_t count = 1;
    for (const auto dimension : shape) {
        require_media(dimension >= 0 && dimension <= std::numeric_limits<int>::max(), "media tensor dimension is invalid");
        require_media(!dimension || count <= std::numeric_limits<size_t>::max() / size_t(dimension), "media tensor size overflows");
        count *= size_t(dimension);
    }
    require_media(count == values.size(), "media tensor shape does not match its owned data");
    if constexpr (std::is_floating_point_v<T>)
        require_media(std::all_of(values.begin(), values.end(), [](T value) { return std::isfinite(value); }),
            "media tensor contains a non-finite value");
}

void validate_deepseek(const MfqMultimodalInput& media) {
    const auto& pixels = media.pixel_shape;
    require_media(pixels.size() == 3 && pixels[0] > 0 && pixels[0] <= 128 && pixels[2] == 3 * 14 * 14 &&
        media.patch_mask_shape == std::vector<int64_t>{pixels[0], pixels[1]} &&
        media.vision_grid_shape == std::vector<int64_t>{pixels[0], 4}, "DeepSeek multimodal tensor geometry is invalid");
    for (int64_t image = 0; image < pixels[0]; ++image) {
        const int64_t height = media.vision_grid[4 * image], width = media.vision_grid[4 * image + 1];
        require_media(height > 0 && width > 0 && height * width <= pixels[1] &&
            media.vision_grid[4 * image + 2] == (height + 2) / 3 &&
            media.vision_grid[4 * image + 3] == (width + 2) / 3,
            "DeepSeek vision grid disagrees with patch geometry", "mfq_multimodal.vision_grid");
        for (int64_t patch = 0; patch < pixels[1]; ++patch)
            require_media(media.patch_mask[image * pixels[1] + patch] == uint8_t(patch < height * width),
                "DeepSeek patch mask is not a contiguous active prefix", "mfq_multimodal.patch_mask");
    }
}

void validate_minicpmo(const MfqMultimodalInput& media) {
    const bool image = !media.pixel_shape.empty();
    const bool audio = !media.audio_features_shape.empty();
    require_media(image || audio, "MiniCPM-o input contains no media tensors");
    if (image) {
        const auto& pixels = media.pixel_shape;
        require_media(pixels.size() == 4 && pixels[0] > 0 && pixels[0] <= 576 && pixels[1] == 3 &&
            pixels[2] == 14 && pixels[3] % 14 == 0 &&
            media.patch_mask_shape == std::vector<int64_t>{pixels[0], pixels[3] / 14} &&
            media.target_sizes_shape == std::vector<int64_t>{pixels[0], 2}, "MiniCPM-o image tensor geometry is invalid");
        for (int64_t source = 0; source < pixels[0]; ++source) {
            const int64_t rows = media.target_sizes[2 * source], columns = media.target_sizes[2 * source + 1];
            const auto patches = media.patch_mask_shape[1];
            require_media(rows > 0 && columns > 0 && rows * columns <= patches,
                "MiniCPM-o target size disagrees with pixel geometry", "mfq_multimodal.target_sizes");
            for (int64_t patch = 0; patch < patches; ++patch)
                require_media(media.patch_mask[source * patches + patch] == uint8_t(patch < rows * columns),
                    "MiniCPM-o patch mask is not a contiguous active prefix", "mfq_multimodal.patch_mask");
        }
    } else require_media(media.patch_mask.empty() && media.target_sizes.empty(), "MiniCPM-o image tensors must be provided together");
    if (audio) {
        const auto& shape = media.audio_features_shape;
        require_media(shape.size() == 3 && shape[0] > 0 && shape[0] <= 128 && shape[1] == 80 &&
            shape[2] >= 9 && shape[2] <= 3000 && media.audio_lengths.size() == size_t(shape[0]),
            "MiniCPM-o audio tensor geometry is invalid");
        for (const auto length : media.audio_lengths)
            require_media(length >= 9 && length <= shape[2], "MiniCPM-o audio length is out of range", "mfq_multimodal.audio_lengths");
    } else require_media(media.audio_lengths.empty(), "MiniCPM-o audio tensors must be provided together");
}

void validate_grid(const MfqMultimodalInput& media) {
    const auto& pixels = media.pixel_shape;
    const auto& grid = media.vision_grid_shape;
    require_media(pixels.size() == 2 && pixels[0] > 0 && pixels[1] > 0 &&
        grid.size() == 2 && grid[0] > 0 && grid[1] == 3 && media.vision_types.size() == size_t(grid[0]),
        "grid-vision tensor geometry is invalid");
    uint64_t patches = 0;
    std::vector<int32_t> images, videos;
    for (int64_t item = 0; item < grid[0]; ++item) {
        const int64_t temporal = media.vision_grid[3 * item], rows = media.vision_grid[3 * item + 1], columns = media.vision_grid[3 * item + 2];
        const auto type = media.vision_types[item];
        require_media(temporal > 0 && rows > 0 && columns > 0 && (type == 1 || type == 2),
            "grid-vision dimensions or modality type are invalid");
        const auto available = uint64_t(pixels[0]) - patches;
        require_media(uint64_t(temporal) <= available / uint64_t(rows) / uint64_t(columns),
            "grid-vision patch count disagrees with pixel_values", "mfq_multimodal.pixel_values");
        patches += uint64_t(temporal) * uint64_t(rows) * uint64_t(columns);
        require_media(type != 1 || temporal == 1, "grid-vision images must have temporal size one");
        auto& destination = type == 1 ? images : videos;
        destination.insert(destination.end(), media.vision_grid.begin() + 3 * item, media.vision_grid.begin() + 3 * item + 3);
    }
    require_media(patches == uint64_t(pixels[0]), "grid-vision patch count disagrees with pixel_values");
    require_media(media.image_grid == images && media.video_grid == videos,
        "grid-vision split grids disagree with combined media order");
    for (const auto* shape : {&media.image_grid_shape, &media.video_grid_shape})
        require_media(shape->empty() || (shape->size() == 2 && (*shape)[1] == 3), "split vision grid must have [items,3] shape");
}
} // namespace

void TextProcessor::validate_media(const MfqMultimodalInput& media) {
    check_tensor(media.pixel_values, media.pixel_shape);
    check_tensor(media.patch_mask, media.patch_mask_shape);
    check_tensor(media.target_sizes, media.target_sizes_shape);
    check_tensor(media.vision_grid, media.vision_grid_shape);
    check_tensor(media.image_grid, media.image_grid_shape);
    check_tensor(media.video_grid, media.video_grid_shape);
    check_tensor(media.audio_features, media.audio_features_shape);
    switch (media.processor) {
        case MfqMultimodalProcessor::minicpmo: validate_minicpmo(media); return;
        case MfqMultimodalProcessor::deepseek_v4:
        case MfqMultimodalProcessor::deepseek_v41: validate_deepseek(media); return;
        case MfqMultimodalProcessor::grid_vision: validate_grid(media); return;
    }
    throw InferenceInputError(InferenceInputErrorCode::Unsupported, "unsupported media processor", "mfq_multimodal.processor");
}
} // namespace mfq::engine
