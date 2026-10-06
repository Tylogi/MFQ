#include "load_options.h"
#include "cuda_runtime_config.h"
#include "../ops/cuda_execution.h"
#include "generation.h"
#include "model_registry.h"
#include "storage/weight_loader.h"
#include "storage/text_session_cache.h"
#include "mtp_metrics.h"
#include "request_executor.h"
#include "generation_flow.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <variant>
#include <type_traits>

namespace mfq::cuda {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

namespace internal {
template <typename Model>
struct CudaEngineState {
    std::shared_ptr<CudaExecutionContext> execution;
    CudaRuntimeConfig config;
    Model model;
    RuntimeComponents<Model> components;
    Model& language;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::conditional_t<Model::adapter_supports_batching,
        std::unique_ptr<typename Model::Batch>, std::nullptr_t> batching = nullptr;
    bool duplex_active = false, duplex_pending = false, duplex_stop = false;
    mfq::StepSequence<ControlCompletion> duplex_sequence;
    std::optional<AdmissionBudget::Lease> duplex_memory;
    AdmissionBudget memory;
    std::vector<std::size_t> request_bytes;

    void prepare_memory(std::size_t limit) {
        int device_count = 0;
        MFQ_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        auto plan = language.adapter_resource_plan(language.blocks,
            config.continuous_batch.max_sequences != 0, device_count, components.mtp.get());
        cache.limit_snapshot_bytes(plan.snapshot_bytes);
        if (components.composite) plan.add(execution->layer_placement.primary_device(),
            {components.composite->state_memory_bytes()});
        for (const auto* devices : {&execution->tensor_parallel.devices,
                                   &execution->expert_parallel.devices,
                                   &execution->layer_placement.devices})
            for (int device : *devices) if (!plan.request_bytes.at(device)) plan.add(device, {1});
        const auto chunk = std::min<std::size_t>(language.max_position_embeddings(),
            config.generation.prefill_chunk_size);
        std::size_t capacity = std::max<std::size_t>(1, config.continuous_batch.max_sequences);
        for (int device = 0; device < device_count; ++device) if (plan.request_bytes[device]) {
            // Prefill activations and graph temporaries share this execution envelope.
            plan.add(device, {256ULL * 1024 * 1024});
            plan.add(device, {64, chunk, std::size_t(language.hidden_size())});
            MfqCudaGuard guard(device);
            std::size_t free = 0, total = 0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            const auto available = limit ? std::min(limit, free - free / 10) : free - free / 10;
            capacity = std::min(capacity, available / plan.request_bytes[device]);
        }
        if (!capacity) throw ResourceExhausted("model context and workspace exceed the available execution memory budget");
        CudaResourcePlan budgets(device_count);
        for (int device = 0; device < device_count; ++device) if (plan.request_bytes[device]) {
            budgets.add(device, {capacity, plan.request_bytes[device]});
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            MfqCudaGuard guard(device);
            mfq::cuda::default_context(device)->reserve_execution_memory(budgets.request_bytes[device]);
#endif
        }
        request_bytes = std::move(plan.request_bytes);
        memory = AdmissionBudget(std::move(budgets.request_bytes));
        if (config.continuous_batch.max_sequences) config.continuous_batch.max_sequences = capacity;
    }
    auto reserve(const EngineRequest&) { return memory.reserve(request_bytes); }
    std::size_t available() const { return memory.available(request_bytes); }


    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents<Model> optional, CudaRuntimeConfig runtime_config, std::size_t memory_limit)
        : execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(components.language(model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1) {
        prepare_memory(memory_limit);
        if (config.continuous_batch.max_sequences) {
            if constexpr (Model::adapter_supports_batching)
                batching = language.adapter_create_batch(language, *execution,
                    config.generation.prefill_chunk_size, config.continuous_batch);
            else throw std::invalid_argument("continuous batching is unavailable for this model");
        }
    }
    bool can_batch(const EngineRequest&) const {
        return bool(batching);
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) {
        if constexpr (Model::adapter_supports_batching)
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
        typename Model::Batch* batch = nullptr;
        if constexpr (Model::adapter_supports_batching)
            if (batched) batch = batching.get();
        return internal::generate(language, graph, cache, config,
                                  input, output, batched ? nullptr : components.mtp.get(), std::move(prepared),
                                  batch, id);
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        if constexpr (Model::adapter_supports_batching) {
            if (request.batched && language.adapter_owns_generation(request.input, cache, bool(components.mtp)))
                return generate_owned(id, request);
        }
        return generate_prepared(*this, request.input, request.output, id, request.batched);
    }
    Generation generate_owned(const RequestId& id, ExecutionRequest& request)
        requires Model::adapter_supports_batching {
        auto native = language.adapter_request_state(language, components.mtp.get());
        DecodeGraphCache request_graph(language.max_position_embeddings());
        struct Ops {
            using Prepared = CudaPreparedPrompt;
            CudaEngineState &owner;
            DecodeGraphCache &graph;
            auto prepare(InferenceRequest &input) { return owner.prepare(input); }
            auto advance_preparation(mfq::StepSequence<Prepared> &sequence) {
                return owner.advance_preparation(sequence);
            }
            Generation generate_text(InferenceRequest &input, InferenceOutput &output,
                    std::optional<Prepared> prepared, const RequestId &id, bool) {
                return internal::generate(owner.language, graph, owner.cache, owner.config,
                    input, output, owner.components.mtp.get(), std::move(prepared), nullptr, id);
            }
        } ops{*this, request_graph};
        // A suspended coroutine never leaves its state attached to shared weights.
        // MTP verification remains a per-request quantum; ordinary rows still batch.
        auto sequence = generate_prepared(ops, request.input, request.output, id, false);
        ExecutionCleanup cleanup{request.output.cleanup_failure, [&] {
            auto binding = native.bind();
            sequence = {};
        }};
        while (true) {
            decltype(sequence.next()) step;
            {
                auto binding = native.bind();
                step = sequence.next();
            }
            if (!step) break;
            co_yield std::move(step);
        }
        cleanup.finish();
    }
    mfq::StepSequence<ControlCompletion> start_duplex(MfqDuplexSessionParams parameters) {
        auto start = components.composite->start(std::move(parameters));
        while (auto step = start.next()) co_yield step.state;
        co_yield ControlCompletion{std::monostate{}};
    }
    mfq::StepSequence<ControlCompletion> step_duplex(MfqDuplexStepInput input) {
        auto generation = components.composite->step(std::move(input));
        while (auto step = generation.next()) {
            if (step.value) co_yield ControlCompletion{std::move(*step.value)};
            else co_yield step.state;
        }
    }
    void advance_control(EngineStepResult& result) {
        if (!duplex_pending && !duplex_stop) return;
        try {
            if (duplex_stop) {
                duplex_sequence = {};
                components.composite->stop();
                if (duplex_pending) result.control = Cancelled{};
                duplex_pending = duplex_active = duplex_stop = false;
                duplex_memory.reset();
            } else {
                auto step = duplex_sequence.next();
                result.control_advanced = step.state == mfq::StepState::advanced;
                if (step.value) {
                    result.control = std::move(*step.value);
                    duplex_pending = false;
                    duplex_sequence = {};
                } else if (!step) throw std::runtime_error("duplex completed without a result");
                else if (step.state == mfq::StepState::waiting)
                    result.wake_at = Clock::now() + std::chrono::milliseconds(1);
            }
        } catch (const std::exception& error) {
            result.control = Failed{"duplex_failed", error.what(), false};
            duplex_sequence = {};
            duplex_pending = duplex_active = duplex_stop = false;
            components.composite->stop();
            duplex_memory.reset();
        }
        result.has_control_work = duplex_pending || duplex_stop;
    }

    template <class T> ControlResult control(T value) {
        if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            size_t free = 0, total = 0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            auto device_memory = mfq_cuda_memory_stats(mfq_current_cuda_device());
            const auto capabilities = components.state();
            Metrics metrics{{"device_free_bytes", double(free)}, {"device_total_bytes", double(total)},
                {"cuda_allocated_bytes", double(device_memory.allocated_bytes)}, {"cuda_reserved_bytes", double(device_memory.reserved_bytes)},
                {"vision_declared", double(capabilities.vision_declared)}, {"vision_supported", double(capabilities.vision_supported)},
                {"vision_available", double(capabilities.vision_available)}, {"mtp_declared", double(capabilities.mtp_declared)},
                {"mtp_supported", double(capabilities.mtp_supported)}, {"mtp_available", double(capabilities.mtp_available)}};
            for (size_t device = 0; device < request_bytes.size(); ++device) if (request_bytes[device]) {
                const auto prefix = "admission_device_" + std::to_string(device);
                metrics.emplace_back(prefix + "_capacity_bytes", memory.capacity()[device]);
                metrics.emplace_back(prefix + "_reserved_bytes", memory.used()[device]);
                metrics.emplace_back(prefix + "_request_bytes", request_bytes[device]);
            }
            if (components.mtp) mtp::append_generation_metrics(metrics, components.mtp->last_stats);
            if constexpr (Model::adapter_supports_batching) {
                if (batching) {
                    auto batch = batching->metrics(); metrics.insert(metrics.end(), batch.begin(), batch.end());
                }
            }
            return metrics;
        } else {
            if (!components.composite) throw std::invalid_argument("model has no duplex component");
            if constexpr (std::is_same_v<T, StopDuplex>) {
                duplex_stop = true;
            } else {
                if (duplex_pending || duplex_stop) throw std::runtime_error("duplex operation is already pending");
                if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                    if (!duplex_memory) {
                        auto reservation = memory.reserve(request_bytes);
                        if (!reservation) throw ResourceExhausted("duplex has no execution memory budget");
                        duplex_memory.emplace(std::move(*reservation));
                    }
                    duplex_active = true;
                    duplex_sequence = start_duplex(std::move(value));
                } else {
                    if (!duplex_active) throw std::invalid_argument("duplex session is not prepared");
                    duplex_sequence = step_duplex(std::move(value));
                }
                duplex_pending = true;
                return ControlPending{};
            }
            return std::monostate{};
        }
    }

};
} // namespace internal

namespace {
using State = CudaModelStates<CudaEngineState>;

struct CudaBackend {
    using Options = CudaEngineOptions;
    Options options;
    State state;
    template <class F> decltype(auto) visit(F&& run) {
        return std::visit([&](auto& state) -> decltype(auto) {
            if (!state) throw std::runtime_error("CUDA engine is unloaded");
            CudaExecutionScope guard(*state->execution);
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*state);
        }, state);
    }
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> load() {
        if (!options.context_size) options.context_size = 32768;
        auto execution = std::make_shared<CudaExecutionContext>();
        CudaExecutionScope guard(*execution);
        setup_cuda_load(options, *execution);
        execution->profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        auto config = resolve_cuda_runtime_config(options);
        state = with_loaded_cuda_model(*execution, options, true,
            [&](auto& model, auto& components, auto, auto) -> State {
                using Model = std::decay_t<decltype(model)>;
                return std::make_unique<CudaEngineState<Model>>(execution, std::move(model), std::move(components), config, options.memory_budget_bytes);
            });
        return visit([&](auto& state) {
            EngineInfo info;
            info.backend = "cuda";
            info.max_requests = state.available();
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
    std::size_t available() const {
        return std::visit([](const auto& state) { return state ? state->available() : 0; }, state);
    }
    void unload() {
        state = State{};
    }
};
} // namespace

std::unique_ptr<Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<EngineInstance<CudaBackend>>(std::move(options));
}
} // namespace mfq::cuda
