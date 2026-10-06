#pragma once

#include "models/common/causal_model_ops.h"
#include "models/common/mtp.h"
#include "models/common/grid_vision_component.h"
#include "minicpmo_components.h"

#include <memory>
#include <optional>
#include <stdexcept>

namespace mfq::cuda {

template <typename Model> struct RuntimeComponents {
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    std::optional<grid_vision_runtime::CudaGridVisionPromptComponent> grid_vision;
    std::unique_ptr<MtpModule> mtp;
    Model* language_override = nullptr;
    std::unique_ptr<minicpmo45::Components> composite;
    bool vision_available = false, mtp_available = false;
    Model& language(Model& fallback) { return language_override ? *language_override : fallback; }
    CudaComponentState state() const noexcept { return cuda_component_state(graph, plan, vision_available, mtp_available); }
};

template <typename Model>
RuntimeComponents<Model> load_runtime_components(Model& model, bool optional) {
    RuntimeComponents<Model> result{model.graph, model.plan};
    if (optional && (result.plan.vision != CudaVisionAdapter::none || result.plan.predictor != CudaPredictorAdapter::none))
        throw std::runtime_error("CUDA component adapter is unsupported for this model");
    return result;
}

template<> RuntimeComponents<Qwen35CausalLm> load_runtime_components(Qwen35CausalLm&, bool);
template<> RuntimeComponents<Qwen4CausalLm> load_runtime_components(Qwen4CausalLm&, bool);
template<> RuntimeComponents<Glm5CausalLm> load_runtime_components(Glm5CausalLm&, bool);
template<> RuntimeComponents<DeepseekV41CausalLm> load_runtime_components(DeepseekV41CausalLm&, bool);
template<> RuntimeComponents<MiniCPMO45CausalLm> load_runtime_components(MiniCPMO45CausalLm&, bool);
template<> RuntimeComponents<MiniCPMOTtsCausalLm> load_runtime_components(MiniCPMOTtsCausalLm&, bool);
} // namespace mfq::cuda
