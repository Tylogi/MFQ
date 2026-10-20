#include "models/cuda_model_plan.h"

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

mfq::ModelGraph grid_vision_graph(std::string backbone = "qwen3_5") {
    auto graph = graph_with(std::move(backbone), "grid_vit", "next_token_prediction");
    graph.components.at(1).input_contract = mfq::kGridVisionInputContract;
    graph.components.at(1).position_policy = mfq::kGridMropePositionPolicy;
    return graph;
}

const mfq::cuda::CudaModelPlan k_text_plan{};
const mfq::cuda::CudaModelPlan k_grid_plan{
    mfq::cuda::CudaVisionAdapter::grid_vit,
    mfq::cuda::CudaPredictorAdapter::qwen35};
const mfq::cuda::CudaModelPlan k_minicpm_plan{
    mfq::cuda::CudaVisionAdapter::minicpmo45,
    mfq::cuda::CudaPredictorAdapter::none};
const mfq::cuda::CudaModelPlan k_flash_next_plan{
    mfq::cuda::CudaVisionAdapter::none,
    mfq::cuda::CudaPredictorAdapter::flash_next};
const mfq::cuda::CudaModelPlan k_dspark_plan{
    mfq::cuda::CudaVisionAdapter::none,
    mfq::cuda::CudaPredictorAdapter::deepseek_v41_dspark};

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

        // Which graph a family accepts is that family's own accepts_backbone();
        // what is decided here is only whether a declared component matches a
        // contract CUDA has an adapter for.
        const auto qwen = grid_vision_graph();
        require(declares_grid_vision(qwen), "grid Vision contract was rejected");
        const auto require_grid_rejected = [](const mfq::ModelGraph& graph) {
            require(
                !declares_grid_vision(graph),
                "mutated grid-Vision registration was accepted");
        };
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

        require(
            declares_next_token_predictor(
                graph_with("qwen3_5", {}, "next_token_prediction")),
            "next-token predictor contract was rejected");
        require(
            !declares_next_token_predictor(
                graph_with("qwen3_5", {}, "experimental_draft")),
            "unknown predictor implementation was inferred from backbone name");
        require(
            !declares_next_token_predictor(
                graph_with("deepseek_v41", {}, "deepseek_v41_dspark")),
            "DSpark was treated as a next-token predictor");
        require(
            declares_deepseek_v41_dspark(
                graph_with("deepseek_v41", {}, "deepseek_v41_dspark")),
            "DSpark predictor contract was rejected");

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
        require(
            declares_minicpmo45_components(minicpm),
            "MiniCPM composite component contract was rejected");
        require(
            !declares_minicpmo45_components(
                graph_with("minicpmo45", "minicpmo45_vision")),
            "partial MiniCPM graph selected a composite CUDA adapter");

        // Component availability is the graph declaration, this family's adapter
        // plan, and whether the adapter weights actually loaded.
        require(qwen.has_component("vision") && qwen.has_component("predictor"),
            "graph components were not preserved");
        require(k_grid_plan.vision != CudaVisionAdapter::none &&
                k_grid_plan.predictor != CudaPredictorAdapter::none,
            "Qwen predictor/vision adapters were not planned");
        require_capabilities(
            cuda_runtime_capabilities(qwen, k_grid_plan, true, true, true, false),
            {true, true, false, false, false, false, true},
            "Qwen3.5 CUDA capability matrix mismatch");
        require_capabilities(
            cuda_runtime_capabilities(qwen, k_grid_plan, false, false, true, false),
            {true, false, false, false, false, false, false},
            "unloaded Qwen components were advertised by model name");

        require_capabilities(
            cuda_runtime_capabilities(minicpm, k_minicpm_plan, true, false, true, true),
            {true, true, true, true, true, true, false},
            "MiniCPM-o CUDA capability matrix mismatch");

        // A plan exists only once a family is selected, so text is supported;
        // what an unadapted plan must not infer is media and MTP.
        require_capabilities(cuda_capability_matrix(k_text_plan),
            {true, false, false, false, false, false, false},
            "an unadapted plan inferred media or MTP capabilities");
        for (const auto& graph : {grid_vision_graph("gemma4"),
                                  graph_with("glm_dsa"),
                                  graph_with("deepseek_v4"),
                                  graph_with("deepseek_v4", {}, "next_token_prediction")}) {
            require_capabilities(
                cuda_runtime_capabilities(graph, k_text_plan, true, true, true, true),
                {true, false, false, false, false, false, false},
                "text-only CUDA capability matrix mismatch");
        }

        // A declared predictor must not advertise MTP until its weights loaded.
        for (const auto& graph : {graph_with("qwen4_exp", "grid_vit", "next_token_prediction"),
                                  graph_with("glm5_next", "grid_vit", "next_token_prediction")}) {
            require(k_flash_next_plan.predictor != CudaPredictorAdapter::none,
                "Flash-Next predictor adapter was not planned");
            require_capabilities(
                cuda_runtime_capabilities(graph, k_flash_next_plan, false, false, false, false),
                {true, false, false, false, false, false, false},
                "unloaded Flash-Next predictor was advertised");
            require_capabilities(
                cuda_runtime_capabilities(graph, k_flash_next_plan, false, true, false, false),
                {true, false, false, false, false, false, true},
                "Flash-Next predictor availability mismatch");
            require_capabilities(
                cuda_capability_matrix(k_flash_next_plan),
                {true, false, false, false, false, false, true},
                "Flash-Next inferred unimplemented media capabilities");
        }

        const auto deepseek_v41 = graph_with(
            "deepseek_v41", "deepseek_v41_vision", "deepseek_v41_dspark");
        require(k_dspark_plan.predictor != CudaPredictorAdapter::none &&
                k_dspark_plan.vision == CudaVisionAdapter::none,
            "DeepSeek-V4.1 DSpark plan mismatch");
        require_capabilities(
            cuda_runtime_capabilities(deepseek_v41, k_dspark_plan, false, true, false, false),
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
