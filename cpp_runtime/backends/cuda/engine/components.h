#pragma once

#include "models/causal_models.h"
#include "models/minicpmo45/ops.h"
#include "models/mtp.h"
#include "models/grid_vision_component.h"
#include "cuda_runtime_config.h"

#include <memory>
#include <optional>

struct CudaExecutionContext;
struct DecodeGraphCache;
namespace mfq::cuda::internal { class TextSessionCache; }

template <typename Model>
struct RuntimeComponents {
    mfq::ModelGraph graph;
    mfq::cuda::CudaModelPlan plan;
    std::optional<
        mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<MtpModule> mtp;
    Model* language_override = nullptr;
    std::unique_ptr<mfq::cuda::minicpmo45::Components> composite;
    bool vision_available = false;
    bool mtp_available = false;

    Model& language(Model& fallback) {
        return language_override ? *language_override : fallback;
    }

    mfq::cuda::CudaComponentState state() const noexcept {
        return mfq::cuda::cuda_component_state(
            graph, plan, vision_available, mtp_available);
    }
};

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
    Model& model,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::Qwen35CausalLm>
load_runtime_components(
    mfq::cuda::Qwen35CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::Qwen4CausalLm>
load_runtime_components(
    mfq::cuda::Qwen4CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::Glm5CausalLm>
load_runtime_components(
    mfq::cuda::Glm5CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::DeepseekV41CausalLm>
load_runtime_components(
    mfq::cuda::DeepseekV41CausalLm& model,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMO45CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMOTtsCausalLm& model,
    bool load_optional_components);
