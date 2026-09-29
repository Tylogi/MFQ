#include "cuda_model_plan.h"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

mfq::ModelGraph graph_with(
        std::string backbone,
        std::string vision = {},
        std::string predictor = {}) {
    mfq::ModelGraph graph;
    graph.backbone = std::move(backbone);
    graph.components.push_back({
        "text", "model", graph.backbone, "decoder", {}, {}, {}});
    if (!vision.empty()) {
        graph.components.push_back({
            "vision", "vision", std::move(vision), "optional", {}, {}, {}});
    }
    if (!predictor.empty()) {
        graph.components.push_back({
            "predictor", "predictor", std::move(predictor), "optional",
            {}, {}, {}});
    }
    return graph;
}

} // namespace

int main() {
    try {
        using namespace mfq::cuda;

        auto qwen = graph_with(
            "qwen3_5", "grid_vit", "next_token_prediction");
        qwen.components.at(1).input_contract =
            mfq::kMfqGridVisionInputContract;
        qwen.components.at(1).position_policy =
            mfq::kMfqGridMropePositionPolicy;
        const auto qwen_plan = cuda_model_plan(qwen);
        require(
            cuda_backbone(qwen.backbone) == CudaBackbone::generic_qwen,
            "Qwen semantic backbone did not resolve");
        const auto qwen_state = cuda_component_state(
            qwen, qwen_plan, true, true);
        require(
            qwen_state.vision_declared && qwen_state.mtp_declared,
            "graph components were not preserved");
        require(
            qwen_state.vision_supported && qwen_state.mtp_supported &&
                qwen_state.vision_available && qwen_state.mtp_available,
            "Qwen predictor/vision availability did not match execution adapters");
        const auto qwen_unloaded = cuda_component_state(qwen, qwen_plan, false, false);
        require(qwen_unloaded.mtp_supported && !qwen_unloaded.mtp_available && !qwen_unloaded.mtp_enabled,
            "unloaded Qwen predictor was reported available");
        const auto require_grid_rejected = [](const mfq::ModelGraph& graph) {
            require(
                cuda_model_plan(graph).vision ==
                    CudaVisionAdapter::none,
                "mutated grid-Vision registration was accepted");
        };
        auto wrong_backbone = qwen;
        wrong_backbone.backbone = "generic_qwen";
        require_grid_rejected(wrong_backbone);
        auto wrong_root = qwen;
        wrong_root.components.at(1).tensor_root = "other_vision";
        require_grid_rejected(wrong_root);
        auto wrong_implementation = qwen;
        wrong_implementation.components.at(1).implementation = "other_vit";
        require_grid_rejected(wrong_implementation);
        auto wrong_input = qwen;
        wrong_input.components.at(1).input_contract = "other_input.v1";
        require_grid_rejected(wrong_input);
        auto wrong_position = qwen;
        wrong_position.components.at(1).position_policy = "other_positions";
        require_grid_rejected(wrong_position);
        const auto unknown_predictor = graph_with("qwen3_5", {}, "experimental_draft");
        require(cuda_model_plan(unknown_predictor).predictor == CudaPredictorAdapter::none,
            "unknown predictor implementation was inferred from backbone name");

        auto minicpm = graph_with("minicpmo45", "minicpmo45_vision");
        minicpm.components.push_back({
            "audio_input", "audio", "minicpmo45_audio", "optional",
            {}, {}, {}});
        minicpm.components.push_back({
            "audio_output", "tts", "minicpmo45_tts", "optional",
            {}, {}, {}});
        const auto minicpm_plan = cuda_model_plan(minicpm);
        require(
            minicpm_plan.vision == CudaVisionAdapter::minicpmo45,
            "MiniCPM semantic Vision adapter did not resolve");
        const auto defaults = cuda_component_state(
            minicpm, minicpm_plan, true, false);
        require(
            defaults.vision_supported && defaults.vision_available &&
                defaults.vision_enabled,
            "loaded MiniCPM Vision did not default on");
        const auto disabled = cuda_component_state(
            minicpm, minicpm_plan, true, false, false, true);
        require(
            disabled.vision_available && !disabled.vision_enabled,
            "manual switch changed component availability");

        const auto partial_minicpm = graph_with(
            "minicpmo45", "minicpmo45_vision");
        require(
            cuda_model_plan(partial_minicpm).vision ==
                CudaVisionAdapter::none,
            "partial MiniCPM graph selected a composite CUDA adapter");

        const auto unsupported = graph_with("unknown_experimental", "grid_vit");
        require(
            cuda_backbone(unsupported.backbone) ==
                CudaBackbone::unsupported,
            "unimplemented CUDA backbone was accepted");
        const auto qwen4 = graph_with("qwen4_exp", "grid_vit", "next_token_prediction");
        const auto qwen4_plan = cuda_model_plan(qwen4);
        require(cuda_backbone(qwen4.backbone) == CudaBackbone::qwen4_exp &&
            qwen4_plan.vision == CudaVisionAdapter::none && qwen4_plan.predictor == CudaPredictorAdapter::flash_next,
            "Qwen4 text/optional-component selection mismatch");

        const auto glm_next = graph_with("glm5_next", "grid_vit", "next_token_prediction");
        const auto glm_plan = cuda_model_plan(glm_next);
        require(cuda_backbone(glm_next.backbone) == CudaBackbone::glm5_next,
            "GLM Flash-Next text adapter was not selected");
        require(glm_plan.vision == CudaVisionAdapter::none &&
            glm_plan.predictor == CudaPredictorAdapter::flash_next,
            "GLM optional component selection mismatch");
        for (const auto& graph:{qwen4,glm_next}) {
            const auto plan=cuda_model_plan(graph);
            const auto absent=cuda_component_state(graph,plan,false,false);
            const auto loaded=cuda_component_state(graph,plan,false,true);
            require(absent.mtp_supported && !absent.mtp_available && loaded.mtp_enabled,
                "Flash-Next predictor declaration was confused with loaded weights");
        }

        const auto deepseek_v41 = graph_with(
            "deepseek_v41", "deepseek_v41_vision", "deepseek_v41_dspark");
        const auto deepseek_v41_plan = cuda_model_plan(deepseek_v41);
        require(
            cuda_backbone(deepseek_v41.backbone) ==
                CudaBackbone::deepseek_v41,
            "DeepSeek-V4.1 text adapter was not selected");
        require(
            deepseek_v41_plan.vision == CudaVisionAdapter::none &&
                deepseek_v41_plan.predictor ==
                    CudaPredictorAdapter::deepseek_v41_dspark,
            "DeepSeek-V4.1 optional component selection mismatch");
        const auto deepseek_v41_state = cuda_component_state(
            deepseek_v41, deepseek_v41_plan, false, true);
        require(
            deepseek_v41_state.mtp_supported &&
                deepseek_v41_state.mtp_enabled &&
                !deepseek_v41_state.vision_supported,
            "DeepSeek-V4.1 DSpark availability state mismatch");

        std::cout << "MFQ CUDA model plan tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ CUDA model plan test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
