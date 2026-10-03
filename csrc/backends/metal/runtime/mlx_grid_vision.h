#pragma once

#include "grid_vision.h"
#include "mfq_container.h"
#include "mlx_multimodal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

using ::mfq::GridVisionConfig;
using ::mfq::GridVisionLayout;
using ::mfq::LearnedPositionInterpolation;
using ::mfq::make_grid_vision_layout;
using ::mfq::make_learned_position_interpolation;
using GridThw = ::mfq::GridShape;

// Reusable learned-position, axial-RoPE grid ViT encoder. Its default loader
// consumes only the canonical MFQ namespace from mlx_tensor_schema.h.
class MlxGridVisionEncoder {
public:
  static MlxGridVisionEncoder load(const MfqContainer &model,
                                   const GridVisionConfig &config);

  MlxGridVisionEncoder(MlxGridVisionEncoder &&) noexcept;
  MlxGridVisionEncoder &operator=(MlxGridVisionEncoder &&) noexcept;
  ~MlxGridVisionEncoder();

  MlxGridVisionEncoder(const MlxGridVisionEncoder &) = delete;
  MlxGridVisionEncoder &operator=(const MlxGridVisionEncoder &) = delete;

  mlx::core::array encode(const mlx::core::array &pixel_values,
                          const std::vector<GridThw> &grid_thw) const;
  mlx::core::array encode(const MlxGridMediaInput &media) const;

  const GridVisionConfig &config() const noexcept;

private:
  struct Impl;
  explicit MlxGridVisionEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct QwenVisionOutput {
  mlx::core::array patch_features;
  mlx::core::array merged_features;
};

// Qwen's shared tower is a generic grid-ViT encoder followed by the learned
// patch merger: LayerNorm -> flatten merge^2 patches -> GELU -> projection.
class MlxQwenVisionTower {
public:
  static MlxQwenVisionTower load(const MfqContainer &model,
                                 const GridVisionConfig &config);

  MlxQwenVisionTower(MlxQwenVisionTower &&) noexcept;
  MlxQwenVisionTower &operator=(MlxQwenVisionTower &&) noexcept;
  ~MlxQwenVisionTower();

  MlxQwenVisionTower(const MlxQwenVisionTower &) = delete;
  MlxQwenVisionTower &operator=(const MlxQwenVisionTower &) = delete;

  QwenVisionOutput forward(const mlx::core::array &pixel_values,
                           const std::vector<GridThw> &grid_thw) const;
  QwenVisionOutput forward(const MlxGridMediaInput &media) const;

  const GridVisionConfig &config() const noexcept;

private:
  struct Impl;
  explicit MlxQwenVisionTower(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// A graph-selected prompt component. It owns the media tower and translates
// the shared grid_vision.v1 contract into the shared PreparedPrompt contract;
// the language model remains unaware of image/video preprocessing details.
class MlxGridVisionPromptComponent {
public:
  static MlxGridVisionPromptComponent load(
      const MfqContainer &model, const GridVisionConfig &config,
      std::int64_t image_token_id, std::int64_t video_token_id,
      std::string input_contract, std::string position_policy);

  MlxGridVisionPromptComponent(MlxGridVisionPromptComponent &&) noexcept;
  MlxGridVisionPromptComponent &
  operator=(MlxGridVisionPromptComponent &&) noexcept;
  ~MlxGridVisionPromptComponent();

  MlxGridVisionPromptComponent(const MlxGridVisionPromptComponent &) = delete;
  MlxGridVisionPromptComponent &
  operator=(const MlxGridVisionPromptComponent &) = delete;

  MlxPreparedPrompt prepare(
      const std::vector<std::int64_t> &token_ids,
      const mlx::core::array &text_embeddings,
      const MlxGridMediaInput &media) const;

  const std::string &input_contract() const noexcept;
  const std::string &position_policy() const noexcept;

private:
  struct Impl;
  explicit MlxGridVisionPromptComponent(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
