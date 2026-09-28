#include "cuda_engine.h"

#include "models/causal_lm.h"
#include "cuda_execution.h"
#include "decode_graph.h"
#include "generation.h"
#include "runtime_config.h"
#include "storage/moe_expert_cache.h"
#include "models/qwen35/batch_executor.h"
#include "models/loader.h"
#include "models/components.h"
#include "mtp_metrics.h"
#include "text_session_cache.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda {
namespace {

using namespace mfq::cuda::internal;

template <CudaBackbone Backbone>
struct CudaEngineState {
    using Model = CausalLmFor<Backbone>;

    CudaEngineState(
            Model loaded_model,
            RuntimeComponents<Model> loaded_components,
            CudaRuntimeConfig config)
        : execution(*loaded_model.execution),
          runtime_config(std::move(config)),
          model(std::move(loaded_model)),
          components(std::move(loaded_components)),
          language(&components.language(model)),
          decode_graph(language->max_position_embeddings()),
          session_cache(
              runtime_config.session_cache,
              runtime_config.prefix_cache,
              make_cuda_paged_prefix_cache(
                  *language->source,
                  language->max_position_embeddings(),
                  language->supports_paged_text_session_state(),
                  runtime_config.prefix_cache),
              language->supports_text_session_state() &&
                  runtime_config.continuous_batch.scheduling.max_sequences == 0,
              language->supports_text_session_state()
                  ? (runtime_config.continuous_batch.scheduling.max_sequences == 0
                        ? 0 : 2)
                  : 1) {
        if (runtime_config.continuous_batch.scheduling.max_sequences == 0) {
            return;
        }
        if constexpr (Backbone == CudaBackbone::generic_qwen) {
            auto qwen_executor = std::make_unique<
                qwen35::QwenBatchExecutor>(
                    *language, execution, model_mutex,
                    runtime_config.continuous_batch,
                    runtime_config.generation);
            std::cerr
                << "continuous_batching enabled=1 max_sequences="
                << runtime_config.continuous_batch.scheduling.max_sequences
                << " prefill_chunk_size="
                << runtime_config.generation.prefill_chunk_size
                << " decode=target_only mtp=disabled"
                << " moe="
                << (qwen35::qwen_continuous_batch_has_moe(
                        *language) ? 1 : 0)
                << " moe_expert_cache="
                << (qwen35::qwen_continuous_batch_has_cached_moe(
                        *language) ? 1 : 0)
                << " paged_kv="
                << (qwen_executor->paged_kv_enabled() ? 1 : 0)
                << " page_size="
                << qwen_executor->paged_kv_page_size()
                << " prefix_cache=fresh_prefill\n";
            batch_executor = std::move(qwen_executor);
        } else {
            throw std::runtime_error(
                "continuous batching requires Qwen35CausalLm");
        }
    }

    CudaExecutionContext& execution;
    const CudaRuntimeConfig runtime_config;
    mfq_tensor_backend::NoGradGuard no_grad;
    Model model;
    RuntimeComponents<Model> components;
    Model* language = nullptr;
    std::mutex model_mutex;
    DecodeGraphCache decode_graph;
    TextSessionCache session_cache;
    std::unique_ptr<mfq::engine::ContinuousBatchExecutor> batch_executor;
};

template <CudaBackbone Backbone>
std::vector<std::pair<std::string, double>> engine_metrics(
        const std::shared_ptr<CudaEngineState<Backbone>>& state) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    MFQ_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto memory = mfq_cuda_memory_stats(mfq_current_cuda_device());
    const auto components = state->components.state();
    const bool mtp_available =
        components.mtp_available && !state->batch_executor;
    std::vector<std::pair<std::string, double>> result{
        {"device_free_bytes", static_cast<double>(free_bytes)},
        {"device_total_bytes", static_cast<double>(total_bytes)},
        {"cuda_allocated_bytes", static_cast<double>(
            memory.allocated_bytes)},
        {"cuda_reserved_bytes", static_cast<double>(
            memory.reserved_bytes)},
        {"vision_declared", components.vision_declared ? 1.0 : 0.0},
        {"vision_supported", components.vision_supported ? 1.0 : 0.0},
        {"vision_available", components.vision_available ? 1.0 : 0.0},
        {"mtp_declared", components.mtp_declared ? 1.0 : 0.0},
        {"mtp_supported", components.mtp_supported ? 1.0 : 0.0},
        {"mtp_available", mtp_available ? 1.0 : 0.0},
    };
    std::unique_lock lock(state->model_mutex, std::try_to_lock);
    if (mtp_available && lock.owns_lock() && state->components.mtp) {
        mfq::engine::mtp::append_generation_metrics(
            result, state->components.mtp->last_stats);
    }
    if (state->batch_executor) {
        auto batching = state->batch_executor->metrics();
        result.insert(result.end(), batching.begin(), batching.end());
    }
    return result;
}

template <CudaBackbone Backbone>
CudaInferenceEngine make_cuda_inference_engine(
        CausalLmFor<Backbone> model,
        RuntimeComponents<CausalLmFor<Backbone>> components,
        CudaRuntimeConfig config) {
    using State = CudaEngineState<Backbone>;
    auto state = std::make_shared<State>(
        std::move(model), std::move(components), std::move(config));

    CudaInferenceEngine engine;
    engine.max_concurrent_requests = std::max<std::size_t>(
        1, state->runtime_config.continuous_batch.scheduling.max_sequences);
    // Both CUDA entry points use one internal generate request. Keep the
    // external callbacks while Metal and the legacy CUDA path remain intact.
    const auto generate_request = [state](
            const std::vector<int64_t>& prompt,
            const MfqMultimodalInput* media,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback& on_prefill,
            const MfqPromptCachePlan& cache_plan,
            const MfqTokenConstraintPtr& token_constraint,
            const MfqCancellationCheck& cancelled) {
        PreparedPromptFactory<typename State::Model> prepare;
        if (media) {
            if constexpr (Backbone == CudaBackbone::generic_qwen) {
                if (!state->components.grid_vision) {
                    throw std::invalid_argument("CUDA vision component is unavailable");
                }
                prepare = [state, &prompt, media](auto& language) {
                    return std::optional<CudaPreparedPrompt>{
                        state->components.grid_vision->prepare(
                            language, prompt, *media)};
                };
            } else {
                throw std::invalid_argument("CUDA backbone has no prepared vision component");
            }
        } else if (state->batch_executor) {
            return state->batch_executor->submit(
                prompt, sampling, on_token, on_prefill,
                cache_plan, token_constraint, cancelled);
        }
        return mfq::cuda::internal::generate(
            *state->language, state->model_mutex,
            state->decode_graph, state->session_cache,
            state->runtime_config, prompt, sampling, on_token, on_prefill,
            cache_plan, token_constraint,
            state->batch_executor && media
                ? nullptr : state->components.mtp.get(),
            std::move(prepare), cancelled);
    };
    engine.generate = [generate_request](
            const std::vector<int64_t>& prompt,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback& on_prefill,
            const MfqPromptCachePlan& cache_plan,
            const MfqTokenConstraintPtr& token_constraint,
            const MfqCancellationCheck& cancelled) {
        return generate_request(prompt, nullptr, sampling, on_token,
                                on_prefill, cache_plan, token_constraint,
                                cancelled);
    };
    engine.session_control = {
        [state](const std::string& source_session_id,
                const std::string& target_session_id) {
            std::lock_guard<std::mutex> lock(state->model_mutex);
            return state->session_cache.fork_session(
                source_session_id, target_session_id);
        },
        [state](const std::string& session_id) {
            std::lock_guard<std::mutex> lock(state->model_mutex);
            return state->session_cache.close_session(session_id);
        },
        [state] {
            return state->session_cache.metrics();
        },
        [state] {
            std::lock_guard<std::mutex> lock(state->model_mutex);
            return state->session_cache.clear();
        },
        [state](uint64_t target_bytes) {
            return state->session_cache.trim_hot(target_bytes);
        },
    };
    if (state->components.engine_binder) {
        state->components.engine_binder(
            engine, state->model_mutex);
    } else if (state->components.grid_vision) {
        engine.multimodal_generate = [generate_request](
                const std::vector<int64_t>& prompt,
                const MfqMultimodalInput& media,
                const MfqSamplingParams& sampling,
                const MfqTokenCallback& on_token,
                const MfqPrefillCallback& on_prefill,
                const MfqPromptCachePlan& cache_plan,
                const MfqTokenConstraintPtr& token_constraint,
                const MfqCancellationCheck& cancelled) {
            return generate_request(prompt, &media, sampling, on_token,
                                    on_prefill, cache_plan, token_constraint,
                                    cancelled);
        };
    }
    engine.runtime_metrics = [state] {
        return engine_metrics(state);
    };

    const auto component_state = state->components.state();
    const bool model_adapter_loaded =
        static_cast<bool>(state->components.engine_binder);
    engine.metadata.source = state->language->source;
    engine.metadata.architecture = state->components.graph.architecture;
    engine.metadata.model_type = state->language->model_type();
    engine.metadata.max_context = state->language->max_position_embeddings();
    engine.metadata.vocab_size = state->language->vocab_size();
    engine.metadata.capabilities.text =
        state->components.graph.has_component("text") &&
        state->components.plan.backbone != CudaBackbone::unsupported;
    engine.metadata.capabilities.image_input = component_state.vision_available;
    engine.metadata.capabilities.video_input =
        component_state.vision_available &&
        !state->components.grid_vision.has_value();
    engine.metadata.capabilities.audio_input = model_adapter_loaded &&
        state->components.graph.has_component("audio_input");
    engine.metadata.capabilities.audio_output = model_adapter_loaded &&
        state->components.graph.has_component("audio_output");
    engine.metadata.capabilities.full_duplex = model_adapter_loaded &&
        state->components.graph.has_component("duplex");
    engine.metadata.capabilities.mtp =
        component_state.mtp_available && !state->batch_executor;
    return engine;
}

} // namespace

CudaInferenceEngine load_cuda_engine(CudaEngineOptions options) {
    if (options.context_size == 0) options.context_size = 32768;
    auto runtime_config = resolve_cuda_runtime_config(options);
    g_profiler.enabled = false;
    mfq_tensor_backend::NoGradGuard no_grad;
    return with_loaded_cuda_model(
        options, true,
        [&]<CudaBackbone Backbone>(auto& model,
                auto& components, auto, auto) {
            return make_cuda_inference_engine<Backbone>(
                std::move(model), std::move(components),
                std::move(runtime_config));
        });
}

} // namespace mfq::cuda
