#include "runtime_components.h"

#include <iostream>
#include <stdexcept>

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
        Model& model,
        bool load_optional_components) {
    RuntimeComponents<Model> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (!load_optional_components) return result;

    switch (result.plan.vision) {
        case mfq::cuda::CudaVisionAdapter::none:
            break;
        case mfq::cuda::CudaVisionAdapter::grid_vit: {
            if constexpr (Model::backbone ==
                    mfq::cuda::CudaBackbone::generic_qwen) {
                const auto* component = result.graph.component("vision");
                const auto& config = model.config;
                if (component == nullptr || !config.grid_vision ||
                        !config.image_token_id || !config.video_token_id) {
                    throw std::runtime_error(
                        "CUDA grid-Vision configuration is incomplete");
                }
                result.grid_vision.emplace(
                    mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                        *model.source, *config.grid_vision,
                        *config.image_token_id, *config.video_token_id,
                        component->input_contract, component->position_policy));
                result.vision_available = true;
            } else {
                throw std::runtime_error(
                    "grid-Vision requires the Qwen CUDA backbone");
            }
            break;
        }
        case mfq::cuda::CudaVisionAdapter::minicpmo45:
            if constexpr (std::is_same_v<
                              Model, mfq::cuda::MiniCPMO45CausalLm>) {
                result.minicpmo.emplace(
                    MiniCPMO45Runtime::load_with_language(
                        std::move(model)));
                result.vision_available = true;
            } else {
                throw std::runtime_error(
                    "MiniCPM-o components require MiniCPMO45CausalLm");
            }
            break;
    }
    if constexpr (
            Model::backbone == mfq::cuda::CudaBackbone::generic_qwen) {
        if (result.plan.predictor ==
                mfq::cuda::CudaPredictorAdapter::qwen35) {
            const bool supported_placement = !g_layer_placement.enabled() &&
                g_dense_cpu_layer_count == 0 &&
                g_dsv4_cpu_offload_layers.empty() && !g_moe_expert_cache;
            if (supported_placement && model.num_experts() == 0 &&
                    model.supports_qwen_speculation()) {
                auto predictor = Qwen35Mtp::load_if_present(
                    *model.source, model.config);
                if (predictor) {
                    result.mtp = std::make_unique<Qwen35Mtp>(
                        std::move(*predictor));
                }
                result.mtp_available = static_cast<bool>(result.mtp);
            } else {
                std::cerr << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
            }
        }
    }
    if constexpr (
            Model::backbone == mfq::cuda::CudaBackbone::qwen4_exp ||
            Model::backbone == mfq::cuda::CudaBackbone::glm5_next) {
        if (result.plan.predictor ==
                mfq::cuda::CudaPredictorAdapter::flash_next) {
            MFQ_RUNTIME_CHECK(
                model.supports_qwen_speculation(),
                "invalid Flash-Next predictor backbone");
            using Predictor=std::conditional_t<
                Model::backbone==mfq::cuda::CudaBackbone::qwen4_exp,
                Qwen4ExpMtp,Glm5NextMtp>;
            auto predictor=Predictor::load_if_present(*model.source,model.config);
            if (predictor) result.mtp=std::make_unique<Predictor>(std::move(*predictor));
            result.mtp_available = static_cast<bool>(result.mtp);
        }
    }
    if constexpr (
            Model::backbone == mfq::cuda::CudaBackbone::deepseek_v41) {
        if (result.plan.predictor ==
                mfq::cuda::CudaPredictorAdapter::deepseek_v41_dspark) {
            MFQ_RUNTIME_CHECK(
                model.supports_deepseek_v41_speculation() && model.shared,
                "invalid DeepSeek-V4.1 DSpark backbone");
            result.mtp =
                mfq::cuda::deepseek_v41_runtime::load_dspark_if_present(
                    *model.source, model.shared->config);
            result.mtp_available = static_cast<bool>(result.mtp);
        }
    }
    return result;
}

#define MFQ_INSTANTIATE_COMPONENTS(TYPE)                                    \
    template RuntimeComponents<TYPE> load_runtime_components(               \
        TYPE&, bool)

MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_COMPONENTS
