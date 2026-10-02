#include "cuda_engine.h"
#include "cuda_execution.h"
#include "generation.h"
#include "model_loader.h"
#include "models/registry.h"
#include "cuda_batching.h"
#include "text_session_cache.h"
#include "mtp_metrics.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <variant>
#include <unordered_map>

namespace mfq::cuda {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

namespace {
struct RequestState {
    InferenceRequest input;
    InferenceOutput output;
    Generation generation;
    bool started = false, batched = false;
    RequestState(InferenceRequest prepared, const MfqTokenizer* tokenizer, const RequestId& id)
        : input(std::move(prepared)), output(input, tokenizer, id) {}
};

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
    std::unordered_map<RequestId, std::unique_ptr<RequestState>> requests;
    bool duplex_active = false;
    bool healthy = true;

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
    bool special(const EngineRequest& request) const {
        const auto& input = request.input;
        return input.media || !input.cache_plan.session_id.empty() ||
            ((request.token_ids.empty() || input.cache_plan.stable_prefix_tokens) && cache.persistent_prefix_enabled()) ||
            (input.sampling.enable_mtp && components.mtp);
    }
    EngineStatus status() const {
        if (!healthy) return {0, false};
        const auto capacity = std::max<std::size_t>(1, config.continuous_batch.max_sequences);
        if (duplex_active) return {0, true};
        for (const auto& [id, state] : requests) if (!state->batched) return {0, true};
        return {capacity - requests.size(), true};
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
}

struct CudaEngine::Impl {
    State state;
    std::unique_ptr<TextProcessor> text;
    CudaEngineOptions options;
    EngineInfo info;
    template <class F> decltype(auto) visit(F&& run) {
        return std::visit([&](auto& state) -> decltype(auto) {
            if (!state) throw std::runtime_error("CUDA engine is unloaded");
            MfqCudaGuard guard(state->execution->layer_placement.primary_device());
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*state);
        }, state);
    }
};

CudaEngine::CudaEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaEngine::CudaEngine(CudaEngine&&) noexcept = default;
CudaEngine& CudaEngine::operator=(CudaEngine&&) noexcept = default;
CudaEngine::~CudaEngine() = default;
EngineInfo CudaEngine::info() const { return impl_->info; }
EngineStatus CudaEngine::status() const {
    return std::visit([](const auto& state) { return state ? state->status() : EngineStatus{0, false}; }, impl_->state);
}

Admission CudaEngine::admit(EngineRequest request) {
    return impl_->visit([&](auto& state) {
        if (state.requests.contains(request.id)) throw std::invalid_argument("duplicate request ID");
        if (!state.status().available || ((!state.batching || state.special(request)) && !state.requests.empty()))
            return Admission::deferred;
        const bool batched = state.batching && !state.special(request);
        const bool raw = !request.token_ids.empty();
        InferenceRequest input;
        if (raw) {
            input.prompt = std::move(request.token_ids);
            input.sampling = request.input.sampling;
            input.cache_plan = request.input.cache_plan;
        } else input = impl_->text->prepare(std::move(request.input), metadata.max_context);
        auto plan = plan_generation(input.prompt, metadata.vocab_size,
            metadata.max_context, input.sampling.max_tokens, input.cache_plan.stable_prefix_tokens);
        input.sampling.max_tokens = plan.generation_tokens;
        input.cache_plan.stable_prefix_tokens = plan.stable_prefix_tokens;
        const auto* tokenizer = raw ? nullptr : &impl_->text->tokenizer();
        auto current = std::make_unique<RequestState>(std::move(input), tokenizer, request.id);
        current->output.metrics.mtp.available = bool(state.components.mtp);
        current->batched = batched;
        if (current->batched) {
            state.graph.invalidate();
            state.batching->admit(request.id, current->input, current->output);
        }
        state.requests.emplace(request.id, std::move(current));
        return Admission::accepted;
    });
}

void CudaEngine::cancel(const RequestId& id) {
    impl_->visit([&](auto& state) {
        auto found = state.requests.find(id);
        if (found != state.requests.end()) found->second->output.result.cancelled = true;
    });
}

EngineStepResult CudaEngine::step(const std::vector<RequestId>& eligible) {
    return impl_->visit([&](auto& state) {
        EngineStepResult result;
        bool has_batch = false;
        for (const auto& [id, request] : state.requests) has_batch |= request->batched;
        if (has_batch) {
            result = state.batching->step(eligible);
            for (const auto& event : result.events)
                if (terminal(event.data)) state.requests.erase(event.id);
        } else {
            for (auto it = state.requests.begin(); it != state.requests.end();) {
                auto& current = *it->second;
                auto& input = current.input;
                if (!current.output.result.cancelled && std::find(eligible.begin(), eligible.end(), it->first) == eligible.end()) {
                    ++it; continue;
                }
                try {
                    if (!current.started) {
                        current.started = true;
                        std::optional<CudaPreparedPrompt> prepared;
                        if (input.vision && !current.output.result.cancelled) {
                            PrefillCudaTimer timer;
                            if (state.components.composite) prepared = state.components.composite->prepare(input.prompt, *input.vision);
                            else if (state.components.grid_vision) prepared = state.components.grid_vision->prepare(state.language, input.prompt, *input.vision);
                            else throw std::invalid_argument("model has no multimodal component");
                            MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
                            current.output.metrics.multimodal_ms = timer.elapsed_ms();
                        }
                        current.generation = generate(state.language, state.graph, state.cache, state.config,
                            input, current.output, state.components.mtp.get(), std::move(prepared));
                        if (input.vision && !current.output.result.cancelled) {
                            result.events.push_back({it->first, PrefillProgress{{0, 0.0,
                                current.output.metrics.multimodal_ms, current.output.metrics.multimodal_ms}}});
                            result.advanced.push_back(it->first);
                            ++it; continue;
                        }
                    }
                    if (auto event = current.generation.next()) {
                        if (auto* progress = std::get_if<PrefillProgress>(&*event)) {
                            progress->timing.multimodal_ms = current.output.metrics.multimodal_ms;
                            progress->timing.model_ms += progress->timing.multimodal_ms;
                            current.output.metrics.mark_prefill(progress->timing);
                        }
                        result.events.push_back({it->first, std::move(*event)});
                        result.advanced.push_back(it->first);
                        ++it; continue;
                    }
                    current.generation = {};
                    auto delta = current.output.finish();
                    if (!delta.diffs.empty()) result.events.push_back({it->first, std::move(delta)});
                    result.events.push_back({it->first, UsageUpdate{input.prompt.size(),
                        static_cast<std::size_t>(current.output.result.completion_tokens)}});
                    if (current.output.result.cancelled)
                        result.events.push_back({it->first, Cancelled{{input.prompt.size(),
                            static_cast<std::size_t>(current.output.result.completion_tokens)}, current.output.metrics}});
                    else result.events.push_back({it->first, Completed{current.output.result.finish_reason,
                        {input.prompt.size(), static_cast<std::size_t>(current.output.result.completion_tokens)}, current.output.metrics}});
                } catch (const std::exception& error) {
                    current.generation = {};
                    try { state.language.reset(1); if (state.components.mtp) state.components.mtp->reset(1); state.graph.invalidate(); } catch (...) {}
                    result.events.push_back({it->first, Failed{"backend_failure", error.what()}});
                }
                it = state.requests.erase(it);
            }
        }
        for (const auto& event : result.events)
            if (const auto* failure = std::get_if<Failed>(&event.data);
                    failure && failure->code == "backend_failure") state.healthy = false;
        result.status = state.status();
        result.has_work = !state.requests.empty();
        return result;
    });
}

SessionResult CudaEngine::session(const SessionCommand& command) {
    return impl_->visit([&](auto& state) -> SessionResult {
        switch (command.kind) {
            case SessionCommand::Kind::fork: return {state.cache.fork_session(command.source, command.target), {}};
            case SessionCommand::Kind::close: return {state.cache.close_session(command.source), {}};
            case SessionCommand::Kind::clear: return {state.cache.clear(), {}};
            case SessionCommand::Kind::trim: return {state.cache.trim_hot(command.bytes), {}};
            case SessionCommand::Kind::metrics: return {0, state.cache.metrics()};
        }
        throw std::invalid_argument("unknown session command");
    });
}

ControlResult CudaEngine::control(ControlRequest request) {
    return impl_->visit([&](auto& state) -> ControlResult {
        return std::visit([&](auto&& value) -> ControlResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, DecodeTokens>)
                return impl_->text->decode_tokens(value.tokens, value.excluded);
            else if constexpr (std::is_same_v<T, PrepareDuplex>) {
                impl_->text->prepare_duplex_session(value.prompt, value.parameters); return std::move(value.parameters);
            } else if constexpr (std::is_same_v<T, PrepareDuplexStep>) {
                impl_->text->prepare_duplex_step(value.text, value.input); return std::move(value.input);
            } else if constexpr (std::is_same_v<T, RuntimeMetrics>) {
                size_t free = 0, total = 0;
                MFQ_CUDA_CHECK(cudaMemGetInfo(&free, &total));
                auto memory = mfq_cuda_memory_stats(mfq_current_cuda_device());
                const auto components = state.components.state();
                Metrics metrics{{"device_free_bytes", double(free)}, {"device_total_bytes", double(total)},
                    {"cuda_allocated_bytes", double(memory.allocated_bytes)}, {"cuda_reserved_bytes", double(memory.reserved_bytes)},
                    {"vision_declared", double(components.vision_declared)}, {"vision_supported", double(components.vision_supported)},
                    {"vision_available", double(components.vision_available)}, {"mtp_declared", double(components.mtp_declared)},
                    {"mtp_supported", double(components.mtp_supported)}, {"mtp_available", double(components.mtp_available)}};
                if (state.components.mtp) mtp::append_generation_metrics(metrics, state.components.mtp->last_stats);
                if (state.batching) {
                    auto batch = state.batching->metrics(); metrics.insert(metrics.end(), batch.begin(), batch.end());
                }
                return metrics;
            } else {
                if (!state.components.composite) throw std::invalid_argument("model has no duplex component");
                if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                    state.components.composite->start(value); state.duplex_active = true;
                } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                    return state.components.composite->step(value);
                } else { state.components.composite->stop(); state.duplex_active = false; }
                return std::monostate{};
            }
        }, std::move(request));
    });
}

std::int64_t CudaEngine::reload(std::int64_t context) {
    if (context < 1) throw std::invalid_argument("reload context must be positive");
    std::visit([](const auto& state) {
        if (state && !state->requests.empty()) throw std::runtime_error("reload requires a quiescent engine");
    }, impl_->state);
    auto options = impl_->options;
    options.context_size = context;
    shutdown();
    auto loaded = load_cuda_engine(std::move(options));
    *this = std::move(loaded);
    return metadata.max_context;
}

void CudaEngine::shutdown() {
    impl_->state = std::unique_ptr<CudaEngineState<Qwen35CausalLm>>{};
    impl_->text.reset();
    metadata.source.reset();
}

CudaEngine load_cuda_engine(CudaEngineOptions options) {
    if (!options.context_size) options.context_size = 32768;
    auto impl = std::make_unique<CudaEngine::Impl>();
    impl->options = options;
    auto execution = std::make_shared<CudaExecutionContext>();
    setup_cuda_load(options, *execution);
    execution->profiler.enabled = false;
    mfq_tensor_backend::NoGradGuard no_grad;
    auto config = resolve_cuda_runtime_config(options);
    impl->state = with_loaded_cuda_model(*execution, options, true,
        [&](auto& model, auto& components, auto, auto) -> State {
            using Model = std::decay_t<decltype(model)>;
            return std::make_unique<CudaEngineState<Model>>(execution, std::move(model), std::move(components), config);
        });
    CudaEngine engine(std::move(impl));
    engine.impl_->visit([&](auto& state) {
        auto& metadata = engine.metadata;
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
        engine.impl_->info = {std::max<std::size_t>(1, config.continuous_batch.max_sequences),
            metadata.max_context, static_cast<int32_t>(metadata.vocab_size), multimodal, caps.full_duplex, true, {}};
    });
    if (engine.metadata.source->has_asset(kTokenizerGgufAsset)) {
        const auto bytes = engine.metadata.source->read_asset(kTokenizerGgufAsset);
        engine.impl_->text = std::make_unique<TextProcessor>(std::vector<uint8_t>(
            reinterpret_cast<const uint8_t*>(bytes.data()), reinterpret_cast<const uint8_t*>(bytes.data()) + bytes.size()),
            static_cast<int32_t>(engine.metadata.vocab_size), engine.metadata.model_type);
    } else engine.impl_->text = std::make_unique<TextProcessor>(options.tokenizer_model,
        static_cast<int32_t>(engine.metadata.vocab_size), engine.metadata.model_type);
    engine.impl_->info.chat = engine.impl_->text->chat_template_capabilities();
    return engine;
}
} // namespace mfq::cuda
