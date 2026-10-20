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
#include "nlohmann/json.hpp"

namespace mfq::cuda {

template<>
RuntimeComponents load_runtime_components(
        Qwen35CausalLm& model, bool optional) {
    RuntimeComponents result{model.graph, model.plan};
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
                execution.dsv4_cpu_offload_layers.empty() && !execution.stores.moe_expert_cache &&
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

template <>
RuntimeComponents load_runtime_components(
        mfq::cuda::Qwen4CausalLm& model,
        bool optional) {
    RuntimeComponents result{model.graph, model.plan};
    if (!optional) return result;
    if (result.plan.vision == mfq::cuda::CudaVisionAdapter::grid_vit) {
        const auto document = nlohmann::json::parse(model.source->model_config_json());
        const auto& vision = document.at("vision_config");
        if (vision.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu_pytorch_tanh" ||
                !vision.value("deepstack_visual_indexes", std::vector<int>{}).empty())
            throw std::runtime_error("unsupported Qwen4 grid-ViT activation or deepstack");
        mfq::GridVisionConfig config;
        config.hidden_size = vision.at("hidden_size").get<int64_t>();
        config.intermediate_size = vision.at("intermediate_size").get<int64_t>();
        config.depth = vision.at("depth").get<int64_t>();
        config.num_heads = vision.at("num_heads").get<int64_t>();
        config.in_channels = vision.value("in_channels", int64_t{3});
        config.patch_size = vision.at("patch_size").get<int64_t>();
        config.temporal_patch_size = vision.at("temporal_patch_size").get<int64_t>();
        config.spatial_merge_size = vision.value("spatial_merge_size", int64_t{2});
        config.out_hidden_size = vision.at("out_hidden_size").get<int64_t>();
        config.num_position_embeddings = vision.at("num_position_embeddings").get<int64_t>();
        config.rope_theta = vision.value("rope_theta", 10000.0);
        if (vision.contains("rope_parameters"))
            config.rope_theta = vision.at("rope_parameters").value("rope_theta", config.rope_theta);
        config.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
        config.validate();
        const auto image = document.at("image_token_id").get<int64_t>();
        const auto video = document.at("video_token_id").get<int64_t>();
        if (config.out_hidden_size != model.hidden_size() || model.config.sections.size() != 3 ||
                image < 0 || video < 0 || image >= model.vocab_size() || video >= model.vocab_size() || image == video)
            throw std::runtime_error("invalid Qwen4 grid-ViT text geometry or placeholder IDs");
        const auto* component = result.graph.component("vision");
        result.grid_vision.emplace(
            mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                *model.execution, *model.source, config, image, video,
                component->input_contract, component->position_policy));
        result.vision_available = true;
    } else if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none) {
        throw std::runtime_error("unsupported Qwen4 CUDA vision adapter");
    }
    if (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 mfq::cuda::CudaPredictorAdapter::flash_next) {
        throw std::runtime_error(
            "unsupported Qwen4 CUDA component adapter");
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = mfq::cuda::qwen4_exp::Qwen4ExpMtp::load_if_present(
        *model.execution, *model.source, model.config);
    if (predictor) {
        result.mtp =
            std::make_unique<mfq::cuda::qwen4_exp::Qwen4ExpMtp>(
                std::move(*predictor));
    }
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

template<>
RuntimeComponents load_runtime_components(
        Glm5CausalLm& model, bool optional) {
    RuntimeComponents result{model.graph, model.plan};
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
RuntimeComponents load_runtime_components(
        DeepseekV41CausalLm& model, bool optional) {
    RuntimeComponents result{model.graph, model.plan};
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
RuntimeComponents load_runtime_components(
        MiniCPMO45CausalLm& model, bool optional) {
    RuntimeComponents result{model.graph, model.plan};
    if (!optional || result.plan.vision == CudaVisionAdapter::none) return result;
    if (result.plan.vision != CudaVisionAdapter::minicpmo45)
        throw std::runtime_error("unsupported MiniCPM-o CUDA vision adapter");
    result.composite = std::make_unique<minicpmo45::Components>(std::move(model));
    result.vision_available = true;
    return result;
}

template<>
RuntimeComponents load_runtime_components(
        MiniCPMOTtsCausalLm& model, bool) {
    return {model.graph, model.plan};
}

} // namespace mfq::cuda
