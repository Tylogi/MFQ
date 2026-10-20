#include "models/deepseek_v41/vision.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::cuda::deepseek_v41_vision {
namespace {

Tensor dense_vector(
    CudaExecutionContext& execution,
    const mfq::ModelSource& model,
    const std::string& name,
    std::int64_t width) {
    auto value = load_dense_native_gpu(execution, model, name)
                     .to(mfq_tensor_backend::kFloat32)
                     .contiguous();
    if (value.dim() != 1 || value.size(0) != width) {
        throw std::runtime_error(
            "DeepSeek-V4.1 vision sentinel shape mismatch: " + name);
    }
    return value;
}

} // namespace

CudaDeepseekV41VisionPromptComponent
CudaDeepseekV41VisionPromptComponent::load(
    CudaExecutionContext& execution,
    const mfq::ModelSource& model,
    const mfq::models::deepseek_v41::Config& config,
    std::string input_contract,
    std::string position_policy) {
    if (input_contract != kMfqDeepseekV41VisionInputContract) {
        throw std::invalid_argument(
            "unsupported DeepSeek-V4.1 vision input contract: " +
            input_contract);
    }
    if (position_policy != kMfqDeepseekV41PositionPolicy) {
        throw std::invalid_argument(
            "unsupported DeepSeek-V4.1 vision position policy: " +
            position_policy);
    }
    if (!config.has_vision()) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 config has no vision tower");
    }
    if (config.image_token_id < 0 ||
        config.image_token_id >= config.vocab) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image token id is outside the vocabulary");
    }
    mfq::models::axial_patch_vision::Config vision{
        config.vision.hidden,
        config.vision.n_layers,
        config.vision.n_heads,
        config.vision.intermediate,
        config.vision.patch_size,
        config.vision.downsample_ratio,
        config.hidden,
        config.vision.rope_theta,
        1e-6,
    };
    auto tower =
        CudaAxialPatchVisionTower::load(execution, model, vision);
    const auto hidden = config.hidden;
    auto start = dense_vector(
        execution, model, "vision.special_token.start", hidden);
    auto newline = dense_vector(
        execution, model, "vision.special_token.newline", hidden);
    auto end = dense_vector(
        execution, model, "vision.special_token.end", hidden);
    auto pad = mfq_tensor_backend::zeros(
        {hidden}, start.options());
    auto special_embeddings = mfq_tensor_backend::stack(
                                  {
                                      std::move(start),
                                      std::move(pad),
                                      std::move(newline),
                                      std::move(end),
                                  },
                                  0)
                                  .contiguous();
    if (tower.config().output_hidden != hidden) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 vision component geometry mismatch");
    }
    return CudaDeepseekV41VisionPromptComponent(
        std::move(tower),
        std::move(special_embeddings),
        config.image_token_id,
        vision,
        std::move(input_contract),
        std::move(position_policy));
}

} // namespace mfq::cuda::deepseek_v41_vision
