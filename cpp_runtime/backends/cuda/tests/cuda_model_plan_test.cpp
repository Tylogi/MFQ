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

        const auto require_capabilities = [](
                const CudaCapabilityMatrix& actual,
                const CudaCapabilityMatrix& expected,
                const char* message) {
            require(
                actual.text == expected.text &&
                actual.image_input == expected.image_input &&
                actual.video_input == expected.video_input &&
                actual.audio_input == expected.audio_input &&
                actual.audio_output == expected.audio_output &&
                actual.full_duplex == expected.full_duplex &&
                actual.mtp == expected.mtp,
                message);
        };

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
        require_capabilities(
            cuda_runtime_capabilities(
                qwen, qwen_plan, qwen_state, true, false),
            {true, true, false, false, false, false, true},
            "Qwen3.5 CUDA capability matrix mismatch");
        const auto qwen_unloaded = cuda_component_state(
            qwen, qwen_plan, false, false);
        require(qwen_unloaded.mtp_supported &&
            !qwen_unloaded.mtp_available && !qwen_unloaded.mtp_enabled,
            "unloaded Qwen predictor was reported available");
        require_capabilities(
            cuda_runtime_capabilities(
                qwen, qwen_plan, qwen_unloaded, true, false),
            {true, false, false, false, false, false, false},
            "unloaded Qwen components were advertised by model name");
        auto flash = qwen;
        flash.backbone = "qwen4_exp";
        const auto flash_plan = cuda_model_plan(flash);
        require_capabilities(cuda_runtime_capabilities(flash, flash_plan,
            cuda_component_state(flash, flash_plan, true, true), true, false),
            {true, true, false, false, false, false, true},
            "Qwen4 grid-ViT and MTP must be available together");
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
        minicpm.components.push_back({
            "duplex", "model", "minicpmo45_duplex", "optional",
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
        require_capabilities(
            cuda_runtime_capabilities(
                minicpm, minicpm_plan, defaults, true, true),
            {true, true, true, true, true, true, false},
            "MiniCPM-o CUDA capability matrix mismatch");
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
        require_capabilities(
            cuda_capability_matrix(cuda_model_plan(unsupported)), {},
            "unknown model name inferred CUDA capabilities");

        for (const char* backbone : {
                 "gemma4", "glm_dsa", "deepseek_v4", "nemotron_h"}) {
            auto text_model = graph_with(backbone);
            require_capabilities(
                cuda_capability_matrix(cuda_model_plan(text_model)),
                {true, false, false, false, false, false, false},
                "text-only CUDA capability matrix mismatch");
        }
        const auto nemotron = graph_with("nemotron_h", "radio",
                                         "nemotron_h_mtp");
        const auto nemotron_plan = cuda_model_plan(nemotron);
        const auto nemotron_state = cuda_component_state(
            nemotron, nemotron_plan, true, true);
        // A `radio` implementation without the cropped-position data contract
        // is not executable: the adapter resolves vision from the contract.
        require(
            nemotron_plan.vision == CudaVisionAdapter::none &&
                nemotron_plan.predictor == CudaPredictorAdapter::nemotron_h &&
                nemotron_state.vision_declared && !nemotron_state.vision_supported &&
                nemotron_state.mtp_declared && nemotron_state.mtp_supported &&
                nemotron_state.mtp_enabled,
            "Nemotron-H optional component selection mismatch");
        auto nemotron_vision = nemotron;
        nemotron_vision.components.at(1).input_contract =
            mfq::kMfqRadioCpeInputContract;
        const auto nemotron_vision_plan = cuda_model_plan(nemotron_vision);
        require(
            nemotron_vision_plan.vision == CudaVisionAdapter::radio &&
                nemotron_vision_plan.backbone == CudaBackbone::nemotron_h,
            "Nemotron-H RADIO adapter was not selected from its data contract");
        require_capabilities(
            cuda_runtime_capabilities(
                nemotron_vision, nemotron_vision_plan,
                cuda_component_state(nemotron_vision, nemotron_vision_plan, true, true),
                true, false),
            {true, true, false, false, false, false, true},
            "Nemotron-H RADIO media capabilities mismatch");
        // The same component on another backbone stays unselected: the adapter
        // list is per backend, not per contract string alone.
        auto radio_on_qwen = nemotron_vision;
        radio_on_qwen.backbone = "qwen3_5";
        require(
            cuda_model_plan(radio_on_qwen).vision == CudaVisionAdapter::radio,
            "RADIO data contract was not resolved independently of the backbone");
        auto radio_with_position_policy = nemotron_vision;
        radio_with_position_policy.components.at(1).position_policy = "grid_mrope";
        require(
            cuda_model_plan(radio_with_position_policy).vision ==
                CudaVisionAdapter::none,
            "RADIO adapter accepted a multimodal position policy");
        auto radio_with_other_contract = nemotron_vision;
        radio_with_other_contract.components.at(1).input_contract = "other.v1";
        require(
            cuda_model_plan(radio_with_other_contract).vision ==
                CudaVisionAdapter::none,
            "RADIO adapter accepted an unknown patch contract");
        const auto nemotron_text = graph_with("nemotron_h");
        require(
            cuda_model_plan(nemotron_text).predictor ==
                CudaPredictorAdapter::none,
            "Nemotron-H text-only graph selected a predictor");
        auto deepseek_v4_with_predictor = graph_with(
            "deepseek_v4", {}, "next_token_prediction");
        require_capabilities(
            cuda_capability_matrix(
                cuda_model_plan(deepseek_v4_with_predictor)),
            {true, false, false, false, false, false, false},
            "DeepSeek-V4 inferred an unimplemented predictor");

        const auto qwen4 = graph_with("qwen4_exp", "grid_vit", "next_token_prediction");
        const auto qwen4_plan = cuda_model_plan(qwen4);
        require(cuda_backbone(qwen4.backbone) == CudaBackbone::qwen4_exp &&
            qwen4_plan.vision == CudaVisionAdapter::none && qwen4_plan.predictor == CudaPredictorAdapter::flash_next,
            "Qwen4 text/optional-component selection mismatch");
        require_capabilities(
            cuda_capability_matrix(qwen4_plan),
            {true, false, false, false, false, false, true},
            "Qwen4 inferred unimplemented media capabilities");

        const auto glm_next = graph_with("glm5_next", "grid_vit", "next_token_prediction");
        const auto glm_plan = cuda_model_plan(glm_next);
        require(cuda_backbone(glm_next.backbone) == CudaBackbone::glm5_next,
            "GLM Flash-Next text adapter was not selected");
        require(glm_plan.vision == CudaVisionAdapter::none &&
            glm_plan.predictor == CudaPredictorAdapter::flash_next,
            "GLM optional component selection mismatch");
        require_capabilities(
            cuda_capability_matrix(glm_plan),
            {true, false, false, false, false, false, true},
            "GLM5 inferred unimplemented media capabilities");
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
        require_capabilities(
            cuda_runtime_capabilities(
                deepseek_v41, deepseek_v41_plan,
                deepseek_v41_state, false, false),
            {true, false, false, false, false, false, true},
            "DeepSeek-V4.1 inferred an unimplemented Vision adapter");

        std::cout << "MFQ CUDA model plan tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ CUDA model plan test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
