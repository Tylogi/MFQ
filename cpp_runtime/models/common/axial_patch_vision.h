#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::models::axial_patch_vision {

// Backend-neutral geometry for the shared axial patch ViT contract: patch
// linear, bidirectional attention with two-axis RoPE, RMSNorm, SwiGLU and a
// strided spatial aligner.
struct Config {
    std::int64_t hidden = 0;
    std::int64_t layers = 0;
    std::int64_t heads = 0;
    std::int64_t intermediate = 0;
    std::int64_t patch_size = 0;
    std::int64_t downsample_ratio = 0;
    std::int64_t output_hidden = 0;
    double rope_theta = 10'000.0;
    double rms_eps = 1e-6;

    void validate() const;
    std::int64_t patch_width() const noexcept {
        return 3 * patch_size * patch_size;
    }
    std::int64_t head_dim() const noexcept { return hidden / heads; }
};

struct AlignedGrid {
    std::int32_t height = 0;
    std::int32_t width = 0;

    std::int64_t count() const noexcept {
        return static_cast<std::int64_t>(height) * width;
    }
};

// ceil(vit/ratio) in both axes, matching spatial aligner output rows.
AlignedGrid aligned_grid(int vit_height, int vit_width, int ratio);

// Flattened [out, ratio, ratio] patch indices in row-major image order. The
// padding bucket is height * width, so callers append one zero row.
std::vector<std::int32_t> gather_indices(int height, int width, int ratio);

// Official V4.1 row-major span: 0=start, 1=image row, 2=newline, 3=end.
std::vector<std::int64_t> span_token_types(int llm_height, int llm_width);

// Validates one image span against its prompt positions, preprocessed ViT grid
// and the declared aligner grid. Inconsistent placeholder layouts must fail
// instead of shifting image rows into unrelated text tokens.
void validate_span(
    int begin,
    int end,
    int vit_height,
    int vit_width,
    int llm_height,
    int llm_width,
    int prompt_tokens,
    int downsample_ratio,
    const std::vector<std::int64_t>& token_types);

} // namespace mfq::models::axial_patch_vision
