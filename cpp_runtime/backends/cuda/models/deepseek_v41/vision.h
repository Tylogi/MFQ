#pragma once

#include "core/axial_patch_vision.h"
#include "cuda_execution.h"
#include "mfq/runtime.h"
#include "models/common/axial_patch_vision.h"
#include "models/deepseek_v41/config.h"
#include "step_sequence.h"
#include "storage/weight_loader.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::deepseek_v41_vision {

using Tensor = mfq_tensor_backend::Tensor;
using axial_patch_vision::CudaAxialPatchVisionTower;

// V4.1-specific prompt adapter around the reusable axial patch ViT. It owns
// only the model's learned image sentinels and span layout contract.
class CudaDeepseekV41VisionPromptComponent {
  public:
    static CudaDeepseekV41VisionPromptComponent load(
        CudaExecutionContext& execution,
        const mfq::ModelSource& model,
        const mfq::models::deepseek_v41::Config& config,
        std::string input_contract,
        std::string position_policy);

    template <typename Model>
    mfq::StepSequence<CudaPreparedPrompt> prepare(
        Model& language,
        const std::vector<int64_t>& token_ids,
        const MfqMultimodalInput& media) const {
        using mfq::models::axial_patch_vision::span_token_types;
        using mfq::models::axial_patch_vision::validate_span;
        if (media.processor != MfqMultimodalProcessor::deepseek_v41 ||
            media.processor_name != "deepseek_v41") {
            throw std::invalid_argument(
                "DeepSeek-V4.1 received a foreign multimodal payload");
        }
        if (token_ids.empty() || media.pixel_shape.size() != 3 ||
            media.pixel_shape[0] <= 0 ||
            media.image_bounds.size() % 4 != 0 ||
            media.vision_grid.size() != media.image_bounds.size() ||
            media.pixel_shape[0] !=
                static_cast<std::int64_t>(media.image_bounds.size() / 4) ||
            media.pixel_shape[2] != config_.patch_width()) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 media payload geometry is invalid");
        }
        const std::int64_t images = media.pixel_shape[0];
        const std::int64_t padded_patches = media.pixel_shape[1];
        const std::int64_t patch_width = media.pixel_shape[2];
        if (padded_patches <= 0 ||
            media.pixel_values.size() !=
                static_cast<std::size_t>(images * padded_patches * patch_width)) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 pixel payload size is invalid");
        }
        if (!media.patch_mask.empty() &&
            (media.patch_mask_shape !=
                 std::vector<std::int64_t>{images, padded_patches} ||
             media.patch_mask.size() !=
                 static_cast<std::size_t>(images * padded_patches))) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 patch mask geometry is invalid");
        }
        auto embeddings = language.embed_forward(
            mfq_tensor_backend::tensor(
                token_ids,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCPU))
                .reshape({1, static_cast<std::int64_t>(token_ids.size())}));
        if (embeddings.dim() != 3 ||
            embeddings.size(1) != static_cast<std::int64_t>(token_ids.size()) ||
            embeddings.size(2) != config_.hidden) {
            throw std::runtime_error(
                "DeepSeek-V4.1 token embedding geometry mismatch");
        }
        std::vector<std::uint8_t> mask(
            token_ids.size(), static_cast<std::uint8_t>(0));
        int previous_end = 0;
        for (std::int64_t source = 0; source < images; ++source) {
            const auto bound_source = static_cast<int>(
                media.image_bounds[4 * source + 1]);
            const int begin = static_cast<int>(
                media.image_bounds[4 * source + 2]);
            const int end = static_cast<int>(
                media.image_bounds[4 * source + 3]);
            if (bound_source != static_cast<int>(source) ||
                begin < previous_end) {
                throw std::invalid_argument(
                    "DeepSeek-V4.1 image spans overlap or are unordered");
            }
            previous_end = end;
            const int vit_height = media.vision_grid[4 * source];
            const int vit_width = media.vision_grid[4 * source + 1];
            const int llm_height = media.vision_grid[4 * source + 2];
            const int llm_width = media.vision_grid[4 * source + 3];
            const auto types = span_token_types(llm_height, llm_width);
            validate_span(
                begin, end, vit_height, vit_width,
                llm_height, llm_width,
                static_cast<int>(token_ids.size()),
                static_cast<int>(config_.downsample_ratio),
                types);
            for (int position = begin; position < end; ++position) {
                if (token_ids[static_cast<std::size_t>(position)] !=
                    image_token_id_) {
                    throw std::invalid_argument(
                        "DeepSeek-V4.1 image span contains a text token");
                }
            }
            const std::int64_t active =
                static_cast<std::int64_t>(vit_height) * vit_width;
            if (active > padded_patches) {
                throw std::invalid_argument(
                    "DeepSeek-V4.1 vision grid exceeds the padded patch count");
            }
            if (!media.patch_mask.empty()) {
                for (std::int64_t patch = 0;
                     patch < padded_patches; ++patch) {
                    const auto value = media.patch_mask[
                        static_cast<std::size_t>(
                            source * padded_patches + patch)];
                    if (value > 1 ||
                        static_cast<bool>(value) != (patch < active)) {
                        throw std::invalid_argument(
                            "DeepSeek-V4.1 patch mask is not a contiguous "
                            "active prefix");
                    }
                }
            }
            const auto offset = static_cast<std::size_t>(
                source * padded_patches * patch_width);
            auto patches = mfq_tensor_backend::from_blob(
                               const_cast<float*>(
                                   media.pixel_values.data() + offset),
                               {static_cast<std::int64_t>(active), patch_width},
                               mfq_tensor_backend::TensorOptions()
                                   .dtype(mfq_tensor_backend::kFloat32)
                                   .device(mfq_tensor_backend::kCPU))
                               .clone()
                               .to(mfq_tensor_backend::kCUDA);
            auto aligned = tower_.encode(
                *language.execution, std::move(patches),
                vit_height, vit_width);
            if (aligned.dim() != 2 ||
                aligned.size(1) != config_.hidden) {
                throw std::runtime_error(
                    "DeepSeek-V4.1 aligned patch geometry mismatch");
            }
            const auto visual_count = static_cast<std::int64_t>(
                std::count(types.begin(), types.end(), 1));
            if (aligned.size(0) != visual_count) {
                throw std::invalid_argument(
                    "DeepSeek-V4.1 image tokens disagree with aligned patches");
            }
            const auto type_tensor = mfq_tensor_backend::tensor(
                types,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
            auto block = special_embeddings_
                             .index_select(0, type_tensor)
                             .contiguous();
            std::vector<std::int64_t> visual_rows;
            visual_rows.reserve(static_cast<std::size_t>(visual_count));
            for (std::size_t index = 0; index < types.size(); ++index) {
                if (types[index] == 1) {
                    visual_rows.push_back(
                        static_cast<std::int64_t>(index));
                }
            }
            const auto visual_index = mfq_tensor_backend::tensor(
                visual_rows,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
            block.index_copy_(
                0, visual_index,
                aligned.to(block.scalar_type()).contiguous());
            const auto prompt_index = mfq_tensor_backend::arange(
                begin, end,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCUDA));
            embeddings.index_copy_(
                1, prompt_index,
                block.to(embeddings.scalar_type()).unsqueeze(0));
            std::fill(
                mask.begin() + begin, mask.begin() + end,
                static_cast<std::uint8_t>(1));
            co_yield mfq::StepState::advanced;
        }
        auto positions = mfq_tensor_backend::arange(
                             0, static_cast<std::int64_t>(token_ids.size()),
                             mfq_tensor_backend::TensorOptions()
                                 .dtype(mfq_tensor_backend::kInt64)
                                 .device(mfq_tensor_backend::kCUDA))
                             .reshape(
                                 {1, static_cast<std::int64_t>(token_ids.size())});
        auto image_mask = mfq_tensor_backend::tensor(
                              mask,
                              mfq_tensor_backend::TensorOptions()
                                  .dtype(mfq_tensor_backend::kUInt8)
                                  .device(mfq_tensor_backend::kCPU))
                              .reshape(
                                  {1, static_cast<std::int64_t>(token_ids.size())})
                              .to(mfq_tensor_backend::kCUDA,
                                  mfq_tensor_backend::kBool);
        co_yield CudaPreparedPrompt{
            token_ids,
            std::move(embeddings),
            std::move(positions),
            std::move(image_mask),
            0,
            {},
        };
    }

    const std::string& input_contract() const noexcept {
        return input_contract_;
    }
    const std::string& position_policy() const noexcept {
        return position_policy_;
    }

  private:
    CudaDeepseekV41VisionPromptComponent(
        CudaAxialPatchVisionTower tower,
        Tensor special_embeddings,
        std::int64_t image_token_id,
        mfq::models::axial_patch_vision::Config config,
        std::string input_contract,
        std::string position_policy)
        : tower_(std::move(tower)),
          special_embeddings_(std::move(special_embeddings)),
          image_token_id_(image_token_id),
          config_(std::move(config)),
          input_contract_(std::move(input_contract)),
          position_policy_(std::move(position_policy)) {}

    CudaAxialPatchVisionTower tower_;
    // [4,hidden], with a zero placeholder at row 1 for dynamic ViT rows.
    Tensor special_embeddings_;
    std::int64_t image_token_id_ = -1;
    mfq::models::axial_patch_vision::Config config_;
    std::string input_contract_;
    std::string position_policy_;
};

} // namespace mfq::cuda::deepseek_v41_vision
