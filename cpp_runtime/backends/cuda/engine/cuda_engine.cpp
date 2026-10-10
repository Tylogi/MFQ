#include "cuda_runtime_config.h"
#include "cuda_execution.h"
#include "generation.h"
#include "storage/model_loader.h"
#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"
#include "runtime/kv_offload.h"
#include "runtime/execution_options.h"
#include "cuda_batching.h"
#include "storage/text_session_cache.h"
#include "mtp_metrics.h"
#include "request_executor.h"
#include "generation_flow.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <variant>

namespace mfq::cuda {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

namespace {
template <typename Model>
struct CudaEngineState {
    std::shared_ptr<CudaExecutionContext> execution;
    CudaRuntimeConfig config;
    Model model;
    RuntimeComponents<Model> components;
    Model& language;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::unique_ptr<ContinuousBatch<QwenBatchOperations>> batching;
    bool duplex_active = false;

    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents<Model> optional, CudaRuntimeConfig runtime_config)
        : execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(components.language(model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1,
                cuda_has_separate_memory()) {
        if (config.continuous_batch.max_sequences) {
            if constexpr (std::is_same_v<Model, Qwen35CausalLm>)
                batching = std::make_unique<ContinuousBatch<QwenBatchOperations>>(
                    config.generation.prefill_chunk_size, config.continuous_batch.prefill_token_budget,
                    language, *execution, config.continuous_batch);
            else throw std::invalid_argument("continuous batching is unavailable for this model");
        }
    }
    bool can_batch(const EngineRequest& request) const {
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), bool(components.mtp));
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) {
        if (batching) batching->step(eligible);
    }
    using Prepared = CudaPreparedPrompt;
    mfq::StepSequence<Prepared> prepare(InferenceRequest& input) {
        if (components.composite) return components.composite->prepare(input.prompt, *input.vision);
        if (components.grid_vision) return components.grid_vision->prepare(language, input.prompt, *input.vision);
        throw std::invalid_argument("model has no multimodal component");
    }
    auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
        PrefillCudaTimer timer;
        auto step = preparation.next();
        MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, std::optional<Prepared> prepared,
                             const RequestId& id, bool batched) {
        return internal::generate(language, graph, cache, config,
                                  input, output, batched ? nullptr : components.mtp.get(), std::move(prepared),
                                  batched ? batching.get() : nullptr, id);
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        if (request.batched) graph.invalidate();
        return generate_prepared(*this, request.input, request.output, id, request.batched);
    }
    template <class T> ControlResult control(T value) {
        if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            size_t free = 0, total = 0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            auto memory = mfq_cuda_memory_stats(mfq_current_cuda_device());
            const auto capabilities = components.state();
            Metrics metrics{{"device_free_bytes", double(free)}, {"device_total_bytes", double(total)},
                {"cuda_allocated_bytes", double(memory.allocated_bytes)}, {"cuda_reserved_bytes", double(memory.reserved_bytes)},
                {"vision_declared", double(capabilities.vision_declared)}, {"vision_supported", double(capabilities.vision_supported)},
                {"vision_available", double(capabilities.vision_available)}, {"mtp_declared", double(capabilities.mtp_declared)},
                {"mtp_supported", double(capabilities.mtp_supported)}, {"mtp_available", double(capabilities.mtp_available)}};
            auto experts = moe_expert_memory_metrics(execution->moe_expert_cache);
            metrics.insert(metrics.end(), experts.begin(), experts.end());
            metrics.insert(metrics.end(), {{"ssd_expert_enabled", 0.0}, {"ssd_expert_payload_bytes", 0.0},
                {"qsa_kv_offload_supported", std::is_same_v<Model, Qwen4CausalLm> ? 1.0 : 0.0}});
            if (execution->qsa_kv_store) {
                auto kv = execution->qsa_kv_store->metrics();
                metrics.insert(metrics.end(), kv.begin(), kv.end());
                for (const auto& value : kv)
                    if (value.first == "qsa_kv_resident_bytes") metrics.emplace_back("kv_cache_bytes", value.second);
                metrics.emplace_back("kv_cache_contexts", language.cache_pos > 0 ? 1.0 : 0.0);
            } else metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
                {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0}, {"qsa_kv_ssd_bytes", 0.0}});
            if (components.mtp) mtp::append_generation_metrics(metrics, components.mtp->last_stats);
            if (batching) {
                auto batch = batching->metrics(); metrics.insert(metrics.end(), batch.begin(), batch.end());
            }
            return metrics;
        } else {
            if (!components.composite) throw std::invalid_argument("model has no duplex component");
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                components.composite->start(value); duplex_active = true;
            } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                return components.composite->step(value);
            } else { components.composite->stop(); duplex_active = false; }
            return std::monostate{};
        }
    }

};
using State = std::variant<
    std::unique_ptr<CudaEngineState<Qwen35CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMO45CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMOTtsCausalLm>>,
    std::unique_ptr<CudaEngineState<Gemma4CausalLm>>,
    std::unique_ptr<CudaEngineState<GlmDsaCausalLm>>,
    std::unique_ptr<CudaEngineState<Glm5CausalLm>>,
    std::unique_ptr<CudaEngineState<Qwen4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV41CausalLm>>>;

struct CudaBackend {
    using Options = CudaEngineOptions;
    Options options;
    State state;
    template <class F> decltype(auto) visit(F&& run) {
        return std::visit([&](auto& state) -> decltype(auto) {
            if (!state) throw std::runtime_error("CUDA engine is unloaded");
            MfqCudaGuard guard(state->execution->layer_placement.primary_device());
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*state);
        }, state);
    }
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> load() {
        if (!options.context_size) options.context_size = 32768;
        auto execution = std::make_shared<CudaExecutionContext>();
        setup_cuda_load(options, *execution);
        execution->profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        auto config = resolve_cuda_runtime_config(options);
        const auto kv_options = runtime_options::qsa_kv_offload();
        if (kv_options.gpu_budget_bytes) {
            KvOffloadConfig kv;
            kv.gpu_budget_bytes = kv_options.gpu_budget_bytes;
            kv.ram_budget_bytes = cuda_has_separate_memory() ? kv_options.ram_budget_bytes : 0;
            kv.buffer_bytes = std::min<std::size_t>(kv.buffer_bytes, kv.gpu_budget_bytes / 4);
            kv.directory = kv_options.directory;
            execution->qsa_kv_store = std::make_shared<KvOffloadStore>(std::move(kv));
        }
        state = with_loaded_cuda_model(*execution, options, true,
            [&](auto& model, auto& components, auto, auto) -> State {
                using Model = std::decay_t<decltype(model)>;
                if constexpr (!std::is_same_v<Model, Qwen4CausalLm>)
                    if (execution->qsa_kv_store) throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                return std::make_unique<CudaEngineState<Model>>(execution, std::move(model), std::move(components), config);
            });
        return visit([&](auto& state) {
            EngineInfo info;
            info.max_requests = std::max<std::size_t>(1, config.continuous_batch.max_sequences);
            info.max_context = state.language.max_position_embeddings();
            info.vocab_size = static_cast<int32_t>(state.language.vocab_size());
            info.model_type = state.language.model_type();
            info.multimodal = state.components.grid_vision.has_value() || bool(state.components.composite);
            info.reload = true;
            const auto caps = cuda_runtime_capabilities(state.components.graph, state.components.plan,
                state.components.state(), info.multimodal, bool(state.components.composite));
            info.duplex = caps.full_duplex;
            info.capabilities = {state.components.graph.architecture, caps.text,
                caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
                caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
            auto text = TextProcessor::load(*state.language.source, options.tokenizer_model,
                info.vocab_size, info.model_type);
            return std::pair{std::move(info), std::move(text)};
        });
    }
    bool exclusive() const {
        return std::visit([](const auto& state) { return state && state->exclusive(); }, state);
    }
    void unload() {
        state = std::unique_ptr<CudaEngineState<Qwen35CausalLm>>{};
    }
};
} // namespace

std::unique_ptr<Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<EngineInstance<CudaBackend>>(std::move(options));
}
} // namespace mfq::cuda
