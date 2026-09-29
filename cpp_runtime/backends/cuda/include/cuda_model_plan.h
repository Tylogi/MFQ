#pragma once

#include "mfq/model_graph.h"

#include <string>
#include <string_view>
#include <utility>

namespace mfq::cuda {

// CUDA resolves semantic implementation IDs from model_graph.json.  These are
// execution implementations, not checkpoint/model aliases.
enum class CudaBackbone {
    generic_qwen,
    minicpmo45,
    minicpmo_tts,
    gemma4,
    glm_dsa,
    glm5_next,
    qwen4_exp,
    deepseek_v4,
    deepseek_v41,
    unsupported,
};

enum class CudaVisionAdapter {
    none,
    grid_vit,
    minicpmo45,
};

enum class CudaPredictorAdapter {
    none,
    qwen35,
    flash_next,
    deepseek_v41_dspark,
};

struct CudaModelPlan {
    CudaVisionAdapter vision = CudaVisionAdapter::none;
    CudaPredictorAdapter predictor = CudaPredictorAdapter::none;
};

struct CudaComponentState {
    bool vision_declared = false;
    bool vision_supported = false;
    bool vision_available = false;
    bool vision_enabled = false;
    bool mtp_declared = false;
    bool mtp_supported = false;
    bool mtp_available = false;
    bool mtp_enabled = false;
};

inline constexpr CudaBackbone cuda_backbone(
        std::string_view implementation) noexcept {
    if (implementation == "qwen3_5" ||
        implementation == "generic_qwen") {
        return CudaBackbone::generic_qwen;
    }
    if (implementation == "minicpmo45") {
        return CudaBackbone::minicpmo45;
    }
    if (implementation == "minicpmo_tts") {
        return CudaBackbone::minicpmo_tts;
    }
    if (implementation == "gemma4") return CudaBackbone::gemma4;
    if (implementation == "glm_dsa") return CudaBackbone::glm_dsa;
    if (implementation == "glm5_next") return CudaBackbone::glm5_next;
    if (implementation == "qwen4_exp") return CudaBackbone::qwen4_exp;
    if (implementation == "deepseek_v4") {
        return CudaBackbone::deepseek_v4;
    }
    if (implementation == "deepseek_v41") {
        return CudaBackbone::deepseek_v41;
    }
    return CudaBackbone::unsupported;
}

inline CudaModelPlan cuda_model_plan(
        const ModelGraph& graph) noexcept {
    CudaModelPlan result;
    const auto* vision = graph.component("vision");
    if (graph.backbone == "qwen3_5" && vision != nullptr &&
        vision->tensor_root == "vision" &&
        vision->implementation == "grid_vit" &&
        vision->input_contract == kMfqGridVisionInputContract &&
        vision->position_policy == kMfqGridMropePositionPolicy) {
        result.vision = CudaVisionAdapter::grid_vit;
    }
    const auto* audio = graph.component("audio_input");
    const auto* tts = graph.component("audio_output");
    // The current CUDA MiniCPM adapter is one composite implementation.  Do
    // not advertise a partially declared graph as supported until those
    // weighted components have independent adapters.
    if (vision != nullptr && audio != nullptr && tts != nullptr &&
        vision->implementation == "minicpmo45_vision" &&
        audio->implementation == "minicpmo45_audio" &&
        tts->implementation == "minicpmo45_tts") {
        result.vision = CudaVisionAdapter::minicpmo45;
    }
    const auto* predictor = graph.component("predictor");
    if (graph.backbone == "qwen3_5" && predictor != nullptr &&
        predictor->implementation == "next_token_prediction" &&
        predictor->tensor_root == "predictor") {
        result.predictor = CudaPredictorAdapter::qwen35;
    }
    if ((graph.backbone=="qwen4_exp" || graph.backbone=="glm5_next") && predictor!=nullptr &&
        predictor->implementation=="next_token_prediction" && predictor->tensor_root=="predictor") {
        result.predictor=CudaPredictorAdapter::flash_next;
    }
    if (graph.backbone == "deepseek_v41" && predictor != nullptr &&
        predictor->implementation == "deepseek_v41_dspark" &&
        predictor->tensor_root == "predictor") {
        result.predictor = CudaPredictorAdapter::deepseek_v41_dspark;
    }
    return result;
}

inline CudaComponentState cuda_component_state(
        const ModelGraph& graph,
        const CudaModelPlan& plan,
        bool vision_loaded,
        bool mtp_loaded,
        bool enable_vision = true,
        bool enable_mtp = true) noexcept {
    CudaComponentState result;
    result.vision_declared = graph.has_component("vision");
    result.vision_supported = result.vision_declared &&
        plan.vision != CudaVisionAdapter::none;
    result.vision_available = result.vision_supported && vision_loaded;
    result.vision_enabled = result.vision_available && enable_vision;
    result.mtp_declared = graph.has_component("predictor");
    result.mtp_supported = result.mtp_declared &&
        plan.predictor != CudaPredictorAdapter::none;
    result.mtp_available = result.mtp_supported && mtp_loaded;
    result.mtp_enabled = result.mtp_available && enable_mtp;
    return result;
}

} // namespace mfq::cuda
