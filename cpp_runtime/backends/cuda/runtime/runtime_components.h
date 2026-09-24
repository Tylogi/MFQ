#pragma once

#include "causal_lm.h"
#include "grid_vision_runtime.h"
#include "mtp.h"
#include "../models/deepseek_v41/deepseek_v41_dspark.h"
#include "../models/glm5_next/mtp.h"
#include "../models/qwen4_exp/mtp.h"
#include "../models/minicpmo45/minicpmo45_runtime.h"
#include "../models/qwen35/mtp.h"

#include <memory>
#include <optional>
#include <type_traits>

template <typename Model>
struct RuntimeComponents {
    mfq::ModelGraph graph;
    mfq::cuda::CudaModelPlan plan;
    std::optional<MiniCPMO45Runtime> minicpmo;
    std::optional<
        mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<MtpModule> mtp;
    bool vision_available = false;
    bool mtp_available = false;

    Model& language(Model& fallback) {
        if constexpr (std::is_same_v<
                          Model, mfq::cuda::MiniCPMO45CausalLm>) {
            return minicpmo ? minicpmo->language : fallback;
        }
        return fallback;
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
