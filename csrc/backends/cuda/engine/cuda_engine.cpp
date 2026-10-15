#include "load_options.h"
#include "cuda_runtime_config.h"
#include "native/env.h"
#include "ops/execution/cuda_execution.h"
#include "generation.h"
#include "model_registry.h"
#include "batch_execution.h"
#include "models/qwen35/batch_state.h"
#include "storage/weights/loaders/weight.h"
#include "storage/weights/moe/expert_cache.h"
#include "storage/state/offload.h"
#include "ops/execution/execution_options.h"
#include "storage/session/text_cache.h"
#include "mtp_metrics.h"
#include "request_executor.h"
#include "generation_flow.h"
#include "engine_instance.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <variant>
#include <type_traits>

namespace mfq::cuda {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

namespace internal {
// The model family is decided once, in CudaBackend::load(). Everything after
// that talks to this interface, so no engine operation re-identifies which
// family is loaded. The family-specific work stays inside CudaEngineState,
// where its compile-time policy branches still resolve per family.
struct CudaEngineOps {
    virtual ~CudaEngineOps() = default;
    virtual CudaExecutionContext& engine_execution() noexcept = 0;
    virtual TextSessionCache& session_cache() noexcept = 0;
    virtual std::pair<EngineInfo, std::unique_ptr<TextProcessor>> describe(
        const std::string& tokenizer_model) = 0;
    virtual bool exclusive() const = 0;
    virtual std::size_t available() const = 0;
    virtual bool mtp_available() const = 0;
    virtual bool can_overlap(const EngineRequest&) const = 0;
    virtual std::optional<AdmissionBudget::Lease> reserve(const EngineRequest&) = 0;
    virtual void execute(const std::vector<RequestId>&) = 0;
    virtual Generation generate(const RequestId&, ExecutionRequest&) = 0;
    virtual ControlResult control(ControlRequest&&) = 0;
    virtual void advance_control(EngineStepResult&) = 0;
};

template <typename Model, class Batch = std::monostate, class RequestState = std::monostate>
struct CudaEngineState : CudaEngineOps {
    static constexpr bool supports_batch = !std::is_same_v<Batch, std::monostate>;
    std::shared_ptr<CudaExecutionContext> execution;
    CudaRuntimeConfig config;
    Model model;
    RuntimeComponents components;
    Model& language;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::unique_ptr<Batch> batching;
    bool duplex_active = false, duplex_pending = false, duplex_stop = false;
    mfq::StepSequence<ControlCompletion> duplex_sequence;
    std::optional<AdmissionBudget::Lease> duplex_memory;
    AdmissionBudget memory;
    std::vector<std::size_t> request_bytes;

    void prepare_memory(std::size_t limit) {
        int device_count = 0;
        MFQ_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        auto plan = language.resource_plan(language.blocks,
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
    auto reserve(const EngineRequest&) -> std::optional<AdmissionBudget::Lease> override {
        return memory.reserve(request_bytes);
    }
    std::size_t available() const override { return memory.available(request_bytes); }

    CudaExecutionContext& engine_execution() noexcept override { return *execution; }
    TextSessionCache& session_cache() noexcept override { return cache; }
    // Engine info is built by the state that owns the model and its components,
    // so the interface never has to expose either.
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> describe(
            const std::string& tokenizer_model) override {
        EngineInfo info;
        info.backend = "cuda";
        info.max_requests = available();
        info.max_context = language.max_position_embeddings();
        info.vocab_size = static_cast<int32_t>(language.vocab_size());
        info.model_type = language.model_type();
        info.multimodal = components.grid_vision.has_value() || bool(components.composite);
        info.reload = true;
        const auto caps = cuda_runtime_capabilities(components.graph, components.plan,
            components.vision_available, components.mtp_available, info.multimodal,
            bool(components.composite));
        info.duplex = caps.full_duplex;
        info.capabilities = {components.graph.architecture, caps.text,
            caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
            caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
        auto text = TextProcessor::load(*language.source, tokenizer_model,
            info.vocab_size, info.model_type);
        return std::pair{std::move(info), std::move(text)};
    }
    // MiniCPM-o's composite owns the language model it actually runs; every
    // other family runs the model it loaded. Bound once here so the hot paths
    // keep a concrete reference; a mismatch is a bug, so it throws.
    static Model& bind_language(RuntimeComponents& loaded_components, Model& loaded) {
        if (!loaded_components.composite) return loaded;
        return dynamic_cast<Model&>(loaded_components.composite->language());
    }
    ControlResult control(ControlRequest&& command) override {
        return std::visit([this](auto&& value) {
            return control_command(std::forward<decltype(value)>(value));
        }, std::move(command));
    }


    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents optional, CudaRuntimeConfig runtime_config, std::size_t memory_limit)
        : execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(bind_language(components, model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1,
                cuda_has_separate_memory()) {
        if (config.continuous_batch.max_sequences && !supports_batch)
            throw std::invalid_argument("continuous batching is unavailable for this model");
        prepare_memory(memory_limit);
        if constexpr (supports_batch) {
            if (config.continuous_batch.max_sequences)
                batching = std::make_unique<Batch>(config.generation.prefill_chunk_size,
                    config.continuous_batch.prefill_token_budget, language, *execution,
                    config.continuous_batch);
        }
    }
    bool can_overlap(const EngineRequest&) const override {
        return bool(batching);
    }
    bool exclusive() const override { return duplex_active; }
    bool mtp_available() const override { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) override {
        if constexpr (supports_batch) if (batching) batching->step(eligible);
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
        std::optional<Generation> physical;
        if constexpr (supports_batch) if (batched) physical.emplace(batching->generate(id, input, output));
        return internal::generate(language, graph, cache, config, input, output,
                                  components.mtp.get(), std::move(prepared), std::move(physical));
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) override {
        if (request.concurrent && batch_requires_owned_generation(request.input, cache, bool(components.mtp))) {
            if constexpr (!std::is_same_v<RequestState, std::monostate>) return generate_owned(id, request);
            else throw InferenceInputError(InferenceInputErrorCode::Unsupported,
                "this request requires independent native state for concurrent execution");
        }
        return generate_prepared(*this, request.input, request.output, id, request.concurrent);
    }
    Generation generate_owned(const RequestId& id, ExecutionRequest& request)
        requires (!std::is_same_v<RequestState, std::monostate>) {
        RequestState native(language, components.mtp.get());
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
                    std::optional<Prepared> prepared, const RequestId&, bool) {
                return internal::generate(owner.language, graph, owner.cache, owner.config,
                    input, output, owner.components.mtp.get(), std::move(prepared));
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
    void advance_control(EngineStepResult& result) override {
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

    template <class T> ControlResult control_command(T value) {
        if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            size_t free = 0, total = 0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            auto device_memory = mfq_cuda_memory_stats(mfq_current_cuda_device());
            const bool vision_declared = components.graph.has_component("vision");
            const bool mtp_declared = components.graph.has_component("predictor");
            Metrics metrics{{"device_free_bytes", double(free)}, {"device_total_bytes", double(total)},
                {"cuda_allocated_bytes", double(device_memory.allocated_bytes)}, {"cuda_reserved_bytes", double(device_memory.reserved_bytes)},
                {"vision_declared", double(vision_declared)},
                {"vision_supported", double(vision_declared && components.plan.vision != CudaVisionAdapter::none)},
                {"vision_available", double(components.vision_available)}, {"mtp_declared", double(mtp_declared)},
                {"mtp_supported", double(mtp_declared && components.plan.predictor != CudaPredictorAdapter::none)},
                {"mtp_available", double(components.mtp_available)}};
            for (size_t device = 0; device < request_bytes.size(); ++device) if (request_bytes[device]) {
                const auto prefix = "admission_device_" + std::to_string(device);
                metrics.emplace_back(prefix + "_capacity_bytes", memory.capacity()[device]);
                metrics.emplace_back(prefix + "_reserved_bytes", memory.used()[device]);
                metrics.emplace_back(prefix + "_request_bytes", request_bytes[device]);
            }
            auto experts = moe_expert_memory_metrics(execution->moe_expert_cache);
            metrics.insert(metrics.end(), experts.begin(), experts.end());
            metrics.insert(metrics.end(), {{"ssd_expert_enabled", 0.0}, {"ssd_expert_payload_bytes", 0.0},
                {"qsa_kv_offload_supported", Model::policy.qsa_kv_offload ? 1.0 : 0.0}});
            if (execution->qsa_kv_store) {
                auto kv = execution->qsa_kv_store->metrics();
                metrics.insert(metrics.end(), kv.begin(), kv.end());
                for (const auto& value : kv)
                    if (value.first == "qsa_kv_resident_bytes")
                        metrics.emplace_back("kv_cache_bytes", value.second);
                metrics.emplace_back("kv_cache_contexts", language.cache_pos > 0 ? 1.0 : 0.0);
            } else {
                metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
                    {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0},
                    {"qsa_kv_ssd_bytes", 0.0}});
            }
            if (components.mtp) {
                mtp::append_generation_metrics(metrics, components.mtp->last_stats);
                metrics.emplace_back("mtp_predictor_graph_replays",
                    double(components.mtp->graph_replay_count()));
            }
            if constexpr (supports_batch) if (batching) {
                auto batch = batching->metrics(); metrics.insert(metrics.end(), batch.begin(), batch.end());
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
                } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                    if (!duplex_active) throw std::invalid_argument("duplex session is not prepared");
                    duplex_sequence = step_duplex(std::move(value));
                } else {
                    throw std::invalid_argument("probability scoring is unavailable for this model");
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
// Concrete execution types are selected only here, at composition. Neither
// the generation interface nor the shared batch policy knows a model family.
template <class Model> using BoundState = std::conditional_t<
    Model::policy.continuous_batching,
    CudaEngineState<Model, ContinuousBatch<CudaBatchExecution<Model>>,
                    qwen35::Qwen35GenerationState>,
    CudaEngineState<Model>>;
using State = std::unique_ptr<CudaEngineOps>;

struct CudaBackend {
    using Options = CudaEngineOptions;
    Options options;
    State state;
    template <class F> decltype(auto) visit(F&& run) {
        if (!state) throw std::runtime_error("CUDA engine is unloaded");
        CudaExecutionScope guard(state->engine_execution());
        mfq_tensor_backend::NoGradGuard no_grad;
        return run(*state);
    }
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> load() {
        // Knobs are consulted from the launch paths, so pin them for the rest of
        // the process: a graph capture and its replays must see the same values.
        const bool dump_env = mfq::cuda::env_flag("MFQ_CUDA_ENV_DUMP", false);
        mfq::cuda::env_freeze();
        if (dump_env) std::cerr << "cuda knobs at load:\n" << mfq::cuda::env_dump();
        if (!options.context_size) options.context_size = 32768;
        auto execution = std::make_shared<CudaExecutionContext>();
        CudaExecutionScope guard(*execution);
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
                if constexpr (!Model::policy.qsa_kv_offload)
                    if (execution->qsa_kv_store)
                        throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                return std::make_unique<BoundState<Model>>(execution, std::move(model),
                    std::move(components), config, options.memory_budget_bytes);
            });
        return visit([&](CudaEngineOps& state) { return state.describe(options.tokenizer_model); });
    }
    bool exclusive() const { return state && state->exclusive(); }
    std::size_t available() const { return state ? state->available() : 0; }
    void unload() {
        if (mfq::cuda::env_frozen() && mfq::cuda::env_flag("MFQ_CUDA_ENV_DUMP", false))
            std::cerr << "cuda knobs in use:\n" << mfq::cuda::env_dump();
        state.reset();
    }
};
} // namespace

std::unique_ptr<Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<EngineInstance<CudaBackend>>(std::move(options));
}
} // namespace mfq::cuda
