#include "components.h"

#include "models/minicpmo45/causal_lm.h"
#include "cuda_batching.h"
#include "generation.h"

#include <iostream>
#include <stdexcept>

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
        Model& model,
        bool load_optional_components) {
    RuntimeComponents<Model> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (load_optional_components &&
            (result.plan.vision != mfq::cuda::CudaVisionAdapter::none ||
             result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none)) {
        throw std::runtime_error(
            "CUDA component adapter is unsupported for this model");
    }
    return result;
}

template <>
RuntimeComponents<mfq::cuda::Qwen35CausalLm> load_runtime_components(
        mfq::cuda::Qwen35CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::Qwen35CausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (!load_optional_components) return result;

    if (result.plan.vision == mfq::cuda::CudaVisionAdapter::grid_vit) {
        const auto* component = result.graph.component("vision");
        const auto& config = model.config;
        if (component == nullptr || !config.grid_vision ||
                !config.image_token_id || !config.video_token_id) {
            throw std::runtime_error(
                "CUDA grid-Vision configuration is incomplete");
        }
        result.grid_vision.emplace(
            mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                *model.execution, *model.source, *config.grid_vision,
                *config.image_token_id, *config.video_token_id,
                component->input_contract, component->position_policy));
        result.vision_available = true;
    } else if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none) {
        throw std::runtime_error(
            "CUDA vision adapter is unsupported for Qwen");
    }

    if (result.plan.predictor == mfq::cuda::CudaPredictorAdapter::qwen35) {
        const auto& model_execution = *model.execution;
        const bool supported_placement =
            !model_execution.layer_placement.enabled() &&
            model_execution.dense_cpu_layer_count == 0 &&
            model_execution.dsv4_cpu_offload_layers.empty() &&
            !model_execution.moe_expert_cache;
        if (supported_placement && model.num_experts() == 0 &&
                model.supports_speculation()) {
            auto predictor = Qwen35Mtp::load_if_present(
                *model.source, model.config, *model.execution);
            if (predictor) {
                result.mtp = std::make_unique<Qwen35Mtp>(
                    std::move(*predictor));
            }
            result.mtp_available = static_cast<bool>(result.mtp);
        } else {
            std::cerr << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    } else if (result.plan.predictor !=
            mfq::cuda::CudaPredictorAdapter::none) {
        throw std::runtime_error(
            "CUDA predictor adapter is unsupported for Qwen");
    }
    return result;
}

template <>
RuntimeComponents<mfq::cuda::Qwen4CausalLm> load_runtime_components(
        mfq::cuda::Qwen4CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::Qwen4CausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (!load_optional_components) return result;
    if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none ||
            (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 mfq::cuda::CudaPredictorAdapter::flash_next)) {
        throw std::runtime_error(
            "unsupported Qwen4 CUDA component adapter");
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = mfq::cuda::qwen4_exp::Qwen4ExpMtp::load_if_present(
        *model.execution, *model.source, model.config);
    if (predictor) {
        result.mtp =
            std::make_unique<mfq::cuda::qwen4_exp::Qwen4ExpMtp>(
                std::move(*predictor));
    }
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

template <>
RuntimeComponents<mfq::cuda::Glm5CausalLm> load_runtime_components(
        mfq::cuda::Glm5CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::Glm5CausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (!load_optional_components) return result;
    if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none ||
            (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 mfq::cuda::CudaPredictorAdapter::flash_next)) {
        throw std::runtime_error(
            "unsupported GLM5 CUDA component adapter");
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = mfq::cuda::glm5_next::Glm5NextMtp::load_if_present(
        *model.execution, *model.source, model.config);
    if (predictor) {
        result.mtp =
            std::make_unique<mfq::cuda::glm5_next::Glm5NextMtp>(
                std::move(*predictor));
    }
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

template <>
RuntimeComponents<mfq::cuda::DeepseekV41CausalLm> load_runtime_components(
        mfq::cuda::DeepseekV41CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::DeepseekV41CausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (!load_optional_components) return result;
    if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none ||
            (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 mfq::cuda::CudaPredictorAdapter::deepseek_v41_dspark)) {
        throw std::runtime_error(
            "unsupported DeepSeek-V4.1 CUDA component adapter");
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(
        model.supports_suffix_speculation() && model.shared,
        "invalid DeepSeek-V4.1 DSpark model");
    result.mtp =
        mfq::cuda::deepseek_v41_runtime::load_dspark_if_present(
            *model.execution, *model.source, model.shared->config);
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
        mfq::cuda::MiniCPMO45CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    if (!load_optional_components ||
            result.plan.vision == mfq::cuda::CudaVisionAdapter::none) {
        return result;
    }
    if (result.plan.vision !=
            mfq::cuda::CudaVisionAdapter::minicpmo45) {
        throw std::runtime_error(
            "unsupported MiniCPM-o CUDA vision adapter");
    }

    auto state = std::make_shared<mfq::cuda::minicpmo45::Components>(
        std::move(model));
    result.language_override = &state->language();
    result.bind_runtime = [state](std::mutex& model_mutex) {
        return CudaRuntimeBindings{
            state->multimodal_generate(model_mutex),
            state->duplex(model_mutex),
        };
    };
    result.vision_available = true;
    return result;
}

template <>
RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm>
load_runtime_components(
        mfq::cuda::MiniCPMOTtsCausalLm& model,
        bool) {
    RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm> result;
    result.graph = model.graph;
    result.plan = model.plan;
    return result;
}

template <typename Model>
std::unique_ptr<mfq::engine::ContinuousBatching>
make_cuda_continuous_batching(
        Model&,
        CudaExecutionContext&,
        std::mutex&,
        DecodeGraphCache&,
        mfq::cuda::internal::TextSessionCache&,
        RuntimeComponents<Model>&,
        const mfq::cuda::CudaRuntimeConfig&) {
    return {};
}

template <>
std::unique_ptr<mfq::engine::ContinuousBatching>
make_cuda_continuous_batching(
        mfq::cuda::Qwen35CausalLm& model,
        CudaExecutionContext& execution,
        std::mutex& model_mutex,
        DecodeGraphCache& decode_graph,
        mfq::cuda::internal::TextSessionCache& session_cache,
        RuntimeComponents<mfq::cuda::Qwen35CausalLm>& components,
        const mfq::cuda::CudaRuntimeConfig& config) {
    auto exclusive_generation = [
            &model, &decode_graph, &session_cache, &components, &config](
            const std::vector<int64_t>& prompt,
            const MfqMultimodalInput* media,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback& on_prefill,
            const MfqPromptCachePlan& cache_plan,
            const MfqTokenConstraintPtr& token_constraint,
            const MfqCancellationCheck& cancelled) {
        mfq::cuda::internal::PreparedPromptFactory<
            mfq::cuda::Qwen35CausalLm> prepare;
        if (media != nullptr) {
            if (!components.grid_vision) {
                throw std::runtime_error(
                    "continuous batching received unavailable grid vision input");
            }
            prepare = [&components, &prompt, media](auto& language) {
                return std::optional<CudaPreparedPrompt>{
                    components.grid_vision->prepare(
                        language, prompt, *media)};
            };
        }
        std::mutex already_locked_model;
        return mfq::cuda::internal::generate(
            model, already_locked_model, decode_graph, session_cache,
            config, prompt, sampling, on_token, on_prefill,
            cache_plan, token_constraint,
            sampling.enable_mtp ? components.mtp.get() : nullptr,
            std::move(prepare), cancelled);
    };
    return std::make_unique<mfq::cuda::QwenBatchExecutor>(
        model, execution, model_mutex, config.continuous_batch,
        config.generation.prefill_chunk_size,
        std::move(exclusive_generation));
}

#define MFQ_INSTANTIATE_COMPONENTS(TYPE)                                  \
    template RuntimeComponents<TYPE> load_runtime_components(TYPE&, bool)

MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::DeepseekV4CausalLm);

#define MFQ_INSTANTIATE_BATCHING(TYPE)                                      \
    template std::unique_ptr<mfq::engine::ContinuousBatching>               \
    make_cuda_continuous_batching(                                          \
        TYPE&, CudaExecutionContext&, std::mutex&, DecodeGraphCache&,       \
        mfq::cuda::internal::TextSessionCache&, RuntimeComponents<TYPE>&,    \
        const mfq::cuda::CudaRuntimeConfig&)

MFQ_INSTANTIATE_BATCHING(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_BATCHING(mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_BATCHING
#undef MFQ_INSTANTIATE_COMPONENTS
