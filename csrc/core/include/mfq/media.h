#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mfq {

enum class MultimodalProcessor {
    minicpmo,
    deepseek_v4,
    deepseek_v41,
    grid_vision,
};

// Architecture-neutral named media payload. Family-specific fields are
// validated before this reaches a backend; prompt-position dependent metadata
// is produced after tokenizing.
struct MultimodalInput {
    MultimodalProcessor processor = MultimodalProcessor::minicpmo;
    std::string processor_name;
    std::vector<float> pixel_values;
    std::vector<int64_t> pixel_shape;
    std::vector<uint8_t> patch_mask;
    std::vector<int64_t> patch_mask_shape;
    std::vector<int32_t> target_sizes;
    std::vector<int64_t> target_sizes_shape;
    std::vector<int32_t> vision_grid;
    std::vector<int64_t> vision_grid_shape;
    std::vector<int32_t> vision_types;
    std::vector<int32_t> image_grid;
    std::vector<int64_t> image_grid_shape;
    std::vector<int32_t> video_grid;
    std::vector<int64_t> video_grid_shape;
    std::vector<int64_t> image_bounds;
    std::vector<int64_t> image_permutation;
    std::vector<int64_t> image_permutation_offsets;
    std::vector<float> audio_features;
    std::vector<int64_t> audio_features_shape;
    std::vector<int64_t> audio_lengths;
    std::vector<int64_t> audio_bounds;
};

// Source compatibility for out-of-tree MiniCPM-o integrations.
using VisionInput = MultimodalInput;

} // namespace mfq
