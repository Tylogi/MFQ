#pragma once

#include "mfq/model_graph.h"

#include <string>
#include <string_view>
#include <utility>

namespace mfq::cuda {

// CUDA resolves semantic implementation IDs from model_graph.json.  These are
// execution implementations, not checkpoint/model aliases.
//
// There is no backbone enum here on purpose: a graph's family is the CausalLm
// type whose accepts_backbone() takes graph.backbone. These enums name only the
// component adapters a family has CUDA code for.
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

// How the engine runs one CUDA model family. Every family states exactly one of
// these, as `policy`, in its own models/*/ops.h. The engine reads these instead
// of comparing model types, so adding a family never edits the engine.
struct CudaFamilyPolicy {
    // Bind the engine state to the continuous-batching executor.
    bool continuous_batching = false;
    // May use the streamed QSA KV store.
    bool qsa_kv_offload = false;
    // Prefill runs layer-major over the whole prompt instead of in chunks.
    bool layer_major_prefill = false;
    // Decode replays this family's captured decode graph.
    bool decode_graph = false;
    // One shared rope cache covers every layer, so it is built once at load.
    bool common_rope = false;
    // Dense (non-MoE) blocks may be placed on CPU by --n-gpu-layers.
    bool dense_cpu_offload = false;
    // The MTP predictor adapter this family implements, if it has one.
    CudaPredictorAdapter predictor = CudaPredictorAdapter::none;
};

struct CudaCapabilityMatrix {
    bool text = false;
    bool image_input = false;
    bool video_input = false;
    bool audio_input = false;
    bool audio_output = false;
    bool full_duplex = false;
    bool mtp = false;
};

// Graph component contracts that have a CUDA adapter. A family's
// plan_components() calls these; an adapter is never inferred from the model
// name alone.
inline bool declares_grid_vision(const ModelGraph& graph) noexcept {
    const auto* vision = graph.component("vision");
    return vision != nullptr && vision->tensor_root == "vision" &&
        vision->implementation == "grid_vit" &&
        vision->input_contract == mfq::kMfqGridVisionInputContract &&
        vision->position_policy == mfq::kMfqGridMropePositionPolicy;
}

// MiniCPM-o's CUDA adapter is one composite implementation, so a graph that
// declares only some of its weighted components is not supported.
inline bool declares_minicpmo45_components(const ModelGraph& graph) noexcept {
    const auto* vision = graph.component("vision");
    const auto* audio = graph.component("audio_input");
    const auto* tts = graph.component("audio_output");
    return vision != nullptr && audio != nullptr && tts != nullptr &&
        vision->implementation == "minicpmo45_vision" &&
        audio->implementation == "minicpmo45_audio" &&
        tts->implementation == "minicpmo45_tts";
}

inline bool declares_next_token_predictor(const ModelGraph& graph) noexcept {
    const auto* predictor = graph.component("predictor");
    return predictor != nullptr &&
        predictor->implementation == "next_token_prediction" &&
        predictor->tensor_root == "predictor";
}

inline bool declares_deepseek_v41_dspark(const ModelGraph& graph) noexcept {
    const auto* predictor = graph.component("predictor");
    return predictor != nullptr &&
        predictor->implementation == "deepseek_v41_dspark" &&
        predictor->tensor_root == "predictor";
}

// What the loaded model advertises. Derived from the graph plus this family's
// adapter plan plus which adapter weights actually loaded; RuntimeComponents
// already holds all three, so there is no separate component-lifecycle struct.
inline constexpr CudaCapabilityMatrix cuda_capability_matrix(
        const CudaModelPlan& plan) noexcept {
    CudaCapabilityMatrix result;
    // A plan only exists once a family has been selected, so text is supported;
    // cuda_runtime_capabilities() gates it on the graph declaring text.
    result.text = true;
    result.image_input =
        plan.vision == CudaVisionAdapter::grid_vit ||
        plan.vision == CudaVisionAdapter::minicpmo45;
    result.video_input = plan.vision == CudaVisionAdapter::minicpmo45;
    result.audio_input = plan.vision == CudaVisionAdapter::minicpmo45;
    result.audio_output = plan.vision == CudaVisionAdapter::minicpmo45;
    result.full_duplex = plan.vision == CudaVisionAdapter::minicpmo45;
    result.mtp = plan.predictor != CudaPredictorAdapter::none;
    return result;
}

inline CudaCapabilityMatrix cuda_runtime_capabilities(
        const ModelGraph& graph,
        const CudaModelPlan& plan,
        bool vision_available,
        bool mtp_available,
        bool multimodal_available,
        bool duplex_available) noexcept {
    auto result = cuda_capability_matrix(plan);
    result.text = result.text && graph.has_component("text");
    result.image_input = result.image_input && vision_available && multimodal_available;
    result.video_input = result.video_input && vision_available && multimodal_available;
    result.audio_input = result.audio_input && multimodal_available &&
        graph.has_component("audio_input");
    result.audio_output = result.audio_output && duplex_available &&
        graph.has_component("audio_output");
    result.full_duplex = result.full_duplex && duplex_available &&
        graph.has_component("duplex");
    result.mtp = result.mtp && mtp_available;
    return result;
}

} // namespace mfq::cuda
