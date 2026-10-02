#include "cuda_engine.h"
#include "cuda_execution.h"
#include "generation.h"
#include "storage/model_loader.h"
#include "storage/weight_loader.h"
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
    std::unique_ptr<QwenBatchExecutor> batching;
    bool duplex_active = false;

    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents<Model> optional, CudaRuntimeConfig runtime_config)
        : execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(components.language(model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1) {
        if (config.continuous_batch.max_sequences) {
            if constexpr (std::is_same_v<Model, Qwen35CausalLm>)
                batching = std::make_unique<QwenBatchExecutor>(language, *execution,
                    config.continuous_batch, config.generation.prefill_chunk_size);
            else throw std::invalid_argument("continuous batching is unavailable for this model");
        }
    }
    bool can_batch(const EngineRequest& request) const {
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), bool(components.mtp));
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void admit_batch(const RequestId& id, ExecutionRequest& request) {
        graph.invalidate();
        batching->admit(id, request);
    }
    void step_batch(const std::vector<RequestId>& eligible) { batching->step(eligible); }
    void reset() {
        language.reset(1);
        if (components.mtp) components.mtp->reset(1);
        graph.invalidate();
    }
    using Prepared = CudaPreparedPrompt;
    std::pair<Prepared, double> prepare(InferenceRequest& input) {
        PrefillCudaTimer timer;
        Prepared prepared;
        if (components.composite) prepared = components.composite->prepare(input.prompt, *input.vision);
        else if (components.grid_vision) prepared = components.grid_vision->prepare(language, input.prompt, *input.vision);
        else throw std::invalid_argument("model has no multimodal component");
        MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return {std::move(prepared), timer.elapsed_ms()};
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, std::optional<Prepared> prepared) {
        return internal::generate(language, graph, cache, config,
                                  input, output, components.mtp.get(), std::move(prepared));
    }
    Generation generate(InferenceRequest& input, InferenceOutput& output) {
        return generate_prepared(*this, input, output);
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
    CudaEngineMetadata metadata;
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
        EngineInfo info;
        auto execution = std::make_shared<CudaExecutionContext>();
        setup_cuda_load(options, *execution);
        execution->profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        auto config = resolve_cuda_runtime_config(options);
        state = with_loaded_cuda_model(*execution, options, true,
            [&](auto& model, auto& components, auto, auto) -> State {
                using Model = std::decay_t<decltype(model)>;
                return std::make_unique<CudaEngineState<Model>>(execution, std::move(model), std::move(components), config);
            });
        visit([&](auto& state) {
            metadata.source = state.language.source;
            metadata.architecture = state.components.graph.architecture;
            metadata.model_type = state.language.model_type();
            metadata.max_context = state.language.max_position_embeddings();
            metadata.vocab_size = state.language.vocab_size();
            const bool multimodal = state.components.grid_vision.has_value() || bool(state.components.composite);
            const auto caps = cuda_runtime_capabilities(state.components.graph, state.components.plan,
                state.components.state(), multimodal, bool(state.components.composite));
            metadata.capabilities = {caps.text, caps.image_input, caps.video_input, caps.audio_input,
                                    caps.audio_output, caps.full_duplex, caps.mtp};
            info = {std::max<std::size_t>(1, config.continuous_batch.max_sequences),
                metadata.max_context, static_cast<int32_t>(metadata.vocab_size), multimodal, caps.full_duplex, true, {}};
        });
        auto text = TextProcessor::load(*metadata.source, options.tokenizer_model,
            static_cast<int32_t>(metadata.vocab_size), metadata.model_type);
        return {std::move(info), std::move(text)};
    }
    bool exclusive() const {
        return std::visit([](const auto& state) { return state && state->exclusive(); }, state);
    }
    void unload() {
        state = std::unique_ptr<CudaEngineState<Qwen35CausalLm>>{};
        metadata.source.reset();
    }
};
} // namespace

struct CudaEngine::Impl : EngineInstance<CudaBackend> {
    using EngineInstance::EngineInstance;
};

CudaEngine::CudaEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaEngine::CudaEngine(CudaEngine&&) noexcept = default;
CudaEngine& CudaEngine::operator=(CudaEngine&&) noexcept = default;
CudaEngine::~CudaEngine() = default;
const CudaEngineMetadata& CudaEngine::metadata() const { return impl_->backend.metadata; }
EngineInfo CudaEngine::info() const { return impl_->info(); }
EngineStatus CudaEngine::status() const { return impl_->status(); }
Admission CudaEngine::admit(EngineRequest request) { return impl_->admit(std::move(request)); }
void CudaEngine::cancel(const RequestId& id) { impl_->cancel(id); }
EngineStepResult CudaEngine::step(const std::vector<RequestId>& eligible) { return impl_->step(eligible); }
SessionResult CudaEngine::session(const SessionCommand& command) { return impl_->session(command); }
ControlResult CudaEngine::control(ControlRequest request) { return impl_->control(std::move(request)); }
std::int64_t CudaEngine::reload(std::int64_t context) { return impl_->reload(context); }
void CudaEngine::shutdown() { impl_->shutdown(); }

CudaEngine load_cuda_engine(CudaEngineOptions options) {
    return CudaEngine(std::make_unique<CudaEngine::Impl>(std::move(options)));
}
} // namespace mfq::cuda
