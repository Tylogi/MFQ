#include "model_components.h"

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "models/deepseek_v41/dspark.h"
#include "models/glm5_next/mtp.h"
#include "models/qwen35/mtp.h"
#include "models/qwen4_exp/mtp.h"

#include <iostream>

namespace mfq::cuda {

template<>
RuntimeComponents<Qwen35CausalLm> load_runtime_components(
        Qwen35CausalLm& model, bool optional) {
    RuntimeComponents<Qwen35CausalLm> result{model.graph, model.plan};
    if (!optional) return result;
    if (result.plan.vision == CudaVisionAdapter::grid_vit) {
        const auto* component = result.graph.component("vision");
        const auto& config = model.config;
        if (!component || !config.grid_vision || !config.image_token_id || !config.video_token_id)
            throw std::runtime_error("CUDA grid-Vision configuration is incomplete");
        result.grid_vision.emplace(grid_vision_runtime::CudaGridVisionPromptComponent::load(
            *model.execution, *model.source, *config.grid_vision, *config.image_token_id,
            *config.video_token_id, component->input_contract, component->position_policy));
        result.vision_available = true;
    } else if (result.plan.vision != CudaVisionAdapter::none) {
        throw std::runtime_error("CUDA vision adapter is unsupported for Qwen");
    }
    if (result.plan.predictor == CudaPredictorAdapter::qwen35) {
        const auto& execution = *model.execution;
        if (!execution.layer_placement.enabled() && execution.dense_cpu_layer_count == 0 &&
                execution.dsv4_cpu_offload_layers.empty() && !execution.moe_expert_cache &&
                model.num_experts() == 0 && model.supports_speculation()) {
            if (auto predictor = Qwen35Mtp::load_if_present(
                    *model.source, model.config, *model.execution))
                result.mtp = std::make_unique<Qwen35Mtp>(std::move(*predictor));
            result.mtp_available = bool(result.mtp);
        } else {
            std::cerr << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    } else if (result.plan.predictor != CudaPredictorAdapter::none) {
        throw std::runtime_error("CUDA predictor adapter is unsupported for Qwen");
    }
    return result;
}

template<>
RuntimeComponents<Qwen4CausalLm> load_runtime_components(
        Qwen4CausalLm& model, bool optional) {
    RuntimeComponents<Qwen4CausalLm> result{model.graph, model.plan};
    if (!optional) return result;
    if (result.plan.vision != CudaVisionAdapter::none ||
            (result.plan.predictor != CudaPredictorAdapter::none &&
             result.plan.predictor != CudaPredictorAdapter::flash_next))
        throw std::runtime_error("unsupported Qwen4 CUDA component adapter");
    if (result.plan.predictor == CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(model.supports_speculation(), "invalid Flash-Next predictor model");
    if (auto predictor = qwen4_exp::Qwen4ExpMtp::load_if_present(
            *model.execution, *model.source, model.config))
        result.mtp = std::make_unique<qwen4_exp::Qwen4ExpMtp>(std::move(*predictor));
    result.mtp_available = bool(result.mtp);
    return result;
}

template<>
RuntimeComponents<Glm5CausalLm> load_runtime_components(
        Glm5CausalLm& model, bool optional) {
    RuntimeComponents<Glm5CausalLm> result{model.graph, model.plan};
    if (!optional) return result;
    if (result.plan.vision != CudaVisionAdapter::none ||
            (result.plan.predictor != CudaPredictorAdapter::none &&
             result.plan.predictor != CudaPredictorAdapter::flash_next))
        throw std::runtime_error("unsupported GLM5 CUDA component adapter");
    if (result.plan.predictor == CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(model.supports_speculation(), "invalid Flash-Next predictor model");
    if (auto predictor = glm5_next::Glm5NextMtp::load_if_present(
            *model.execution, *model.source, model.config))
        result.mtp = std::make_unique<glm5_next::Glm5NextMtp>(std::move(*predictor));
    result.mtp_available = bool(result.mtp);
    return result;
}

template<>
RuntimeComponents<DeepseekV41CausalLm> load_runtime_components(
        DeepseekV41CausalLm& model, bool optional) {
    RuntimeComponents<DeepseekV41CausalLm> result{model.graph, model.plan};
    if (!optional) return result;
    if (result.plan.vision != CudaVisionAdapter::none ||
            (result.plan.predictor != CudaPredictorAdapter::none &&
             result.plan.predictor != CudaPredictorAdapter::deepseek_v41_dspark))
        throw std::runtime_error("unsupported DeepSeek-V4.1 CUDA component adapter");
    if (result.plan.predictor == CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(model.supports_suffix_speculation() && model.shared,
        "invalid DeepSeek-V4.1 DSpark model");
    result.mtp = deepseek_v41_runtime::load_dspark_if_present(
        *model.execution, *model.source, model.shared->config);
    result.mtp_available = bool(result.mtp);
    return result;
}

template<>
RuntimeComponents<MiniCPMO45CausalLm> load_runtime_components(
        MiniCPMO45CausalLm& model, bool optional) {
    RuntimeComponents<MiniCPMO45CausalLm> result{model.graph, model.plan};
    if (!optional || result.plan.vision == CudaVisionAdapter::none) return result;
    if (result.plan.vision != CudaVisionAdapter::minicpmo45)
        throw std::runtime_error("unsupported MiniCPM-o CUDA vision adapter");
    result.composite = std::make_unique<minicpmo45::Components>(std::move(model));
    result.language_override = &result.composite->language();
    result.vision_available = true;
    return result;
}

template<>
RuntimeComponents<MiniCPMOTtsCausalLm> load_runtime_components(
        MiniCPMOTtsCausalLm& model, bool) {
    return {model.graph, model.plan};
}

} // namespace mfq::cuda
