#pragma once

#include "causal_lm.h"
#include "grid_vision_runtime.h"
#include "mtp.h"
#include "mfq/runtime.h"
#include "../models/deepseek_v41/deepseek_v41_dspark.h"
#include "../models/glm5_next/mtp.h"
#include "../models/qwen4_exp/mtp.h"
#include "../models/qwen35/mtp.h"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>

template <typename Model>
struct RuntimeComponents {
    mfq::ModelGraph graph;
    mfq::cuda::CudaModelPlan plan;
    std::optional<
        mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<MtpModule> mtp;
    Model* language_override = nullptr;
    std::function<void(MfqInferenceEngine&, std::mutex&)> engine_binder;
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

#include "../models/minicpmo45/minicpmo45_components.h"
