#pragma once

#include "models/causal_models.h"
#include "models/mtp.h"
#include "models/grid_vision_component.h"
#include "cuda_runtime_config.h"
#include "models/deepseek_v41/dspark.h"
#include "models/glm5_next/mtp.h"
#include "models/qwen4_exp/mtp.h"
#include "models/qwen35/mtp.h"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>

struct CudaExecutionContext;
struct DecodeGraphCache;
namespace mfq::cuda::internal { class TextSessionCache; }

struct CudaRuntimeBindings {
    MfqMultimodalGenerateFn multimodal_generate;
    MfqDuplexBackend duplex;
};

template <typename Model>
struct RuntimeComponents {
    mfq::ModelGraph graph;
    mfq::cuda::CudaModelPlan plan;
    std::optional<
        mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<MtpModule> mtp;
    Model* language_override = nullptr;
    std::function<CudaRuntimeBindings(std::mutex&)> bind_runtime;
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

template <typename Model>
std::unique_ptr<mfq::engine::ContinuousBatching>
make_cuda_continuous_batching(
    Model& model,
    CudaExecutionContext& execution,
    std::mutex& model_mutex,
    DecodeGraphCache& decode_graph,
    mfq::cuda::internal::TextSessionCache& session_cache,
    RuntimeComponents<Model>& components,
    const mfq::cuda::CudaRuntimeConfig& config);

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
std::unique_ptr<mfq::engine::ContinuousBatching>
make_cuda_continuous_batching(
    mfq::cuda::Qwen35CausalLm& model,
    CudaExecutionContext& execution,
    std::mutex& model_mutex,
    DecodeGraphCache& decode_graph,
    mfq::cuda::internal::TextSessionCache& session_cache,
    RuntimeComponents<mfq::cuda::Qwen35CausalLm>& components,
    const mfq::cuda::CudaRuntimeConfig& config);

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
