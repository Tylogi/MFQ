#include "../storage/load_options.h"
#include "cuda_runtime_config.h"
#include "../ops/cuda_execution.h"
#include "generation.h"
#include "storage/model_loader.h"
#include "storage/weight_loader.h"
#include "cuda_batching.h"
#include "storage/text_session_cache.h"
#include "mtp_metrics.h"
#include "request_executor.h"
#include "generation_flow.h"
#include "models/qwen35/linear_attention.h"
#include "models/common/full_block.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <variant>

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
    std::unique_ptr<ContinuousBatch<QwenBatchOperations>> batching;
    bool duplex_active = false;
    AdmissionBudget memory;
    std::vector<std::size_t> request_bytes;

    void prepare_memory(std::size_t limit) {
        int device_count = 0;
        MFQ_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        request_bytes.assign(device_count, 0);
        const auto product = [](std::initializer_list<std::size_t> factors) {
            std::size_t result = 1;
            for (auto factor : factors) {
                if (factor && result > std::size_t(-1) / factor)
                    throw ResourceExhausted("model memory geometry overflows size_t");
                result *= factor;
            }
            return result;
        };
        const auto add = [&](int device, std::size_t bytes) {
            if (request_bytes.at(device) > std::size_t(-1) - bytes)
                throw ResourceExhausted("model memory geometry overflows size_t");
            request_bytes[device] += bytes;
        };
        const auto context = std::size_t(language.max_position_embeddings());
        const auto chunk = std::min<std::size_t>(context, config.generation.prefill_chunk_size);
        const auto hidden = std::size_t(language.hidden_size());
        const bool batch = config.continuous_batch.max_sequences != 0;
        const std::size_t kv_copies = batch ? (config.continuous_batch.paged_kv ? 1 : 4) : 3;
        const std::size_t recurrent_copies = batch ? 4 : 3;
        const std::size_t graph_copies = batch ? 8 : 1;
        for (const auto& block : language.blocks) {
            if (block->cpu_offloaded) continue;
            const int device = block->cuda_device;
            if (const auto* full = dynamic_cast<const FullBlock*>(block.get())) {
                const auto capacity = full->sliding ? std::size_t(full->attention_window) : context;
                // Dense/paged KV, snapshot/rollback, and packed batch state.
                add(device, product({kv_copies, 2, 2, capacity + 1024, std::size_t(full->kv_heads),
                    std::size_t(full->attention_head_dim)}));
                // Split attention output plus MMA mask/metadata, retained in graph buckets.
                add(device, product({graph_copies, std::size_t(full->attention_heads), 16,
                    std::size_t(full->attention_head_dim + 2), 4}));
                add(device, product({graph_copies, std::size_t(full->kv_heads), (context + 63) / 64, 8,
                    std::size_t(2 + full->attention_head_dim / 2), 8}));
            } else if (const auto* linear = dynamic_cast<const qwen35::LinearAttentionBlock*>(block.get())) {
                const auto& c = linear->qwen_config;
                add(device, product({recurrent_copies, 4, std::size_t(c.linear_conv_kernel_dim - 1),
                    std::size_t(2 * c.linear_k_size() + c.linear_v_size())}));
                add(device, product({recurrent_copies, 4, std::size_t(c.linear_num_value_heads),
                    std::size_t(c.linear_value_head_dim), std::size_t(c.linear_value_head_dim)}));
            } else {
                // Other native families are serial; reserve a dense upper envelope
                // for their compressed, recurrent or speculative cache layouts.
                add(device, product({8, context, hidden}));
            }
        }
        // At most one full request state may remain in the snapshot cache.
        // The per-request copy factors cover active state, capture and cached state.
        std::size_t snapshot_budget = std::size_t(-1);
        for (auto bytes : request_bytes) if (bytes)
            snapshot_budget = std::min(snapshot_budget, bytes / std::max(kv_copies, recurrent_copies));
        cache.limit_snapshot_bytes(snapshot_budget == std::size_t(-1) ? 0 : snapshot_budget);
        const auto primary = execution->layer_placement.primary_device();
        add(primary, product({8, std::size_t(language.vocab_size())}));
        for (const auto* devices : {&execution->tensor_parallel.devices,
                                   &execution->expert_parallel.devices,
                                   &execution->layer_placement.devices})
            for (int device : *devices) if (!request_bytes.at(device)) add(device, 1);
        std::size_t capacity = std::max<std::size_t>(1, config.continuous_batch.max_sequences);
        for (int device = 0; device < device_count; ++device) if (request_bytes[device]) {
            // Prefill activations and graph temporaries share this execution envelope.
            add(device, 256ULL * 1024 * 1024);
            add(device, product({64, chunk, hidden}));
            MfqCudaGuard guard(device);
            std::size_t free = 0, total = 0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            const auto available = limit ? std::min(limit, free - free / 10) : free - free / 10;
            capacity = std::min(capacity, available / request_bytes[device]);
        }
        if (!capacity) throw ResourceExhausted("model context and workspace exceed the available execution memory budget");
        std::vector<std::size_t> budgets(device_count);
        for (int device = 0; device < device_count; ++device) if (request_bytes[device]) {
            budgets[device] = product({capacity, request_bytes[device]});
#ifdef MFQ_NATIVE_CUDA_RUNTIME
            MfqCudaGuard guard(device);
            mfq::cuda::default_context(device)->reserve_execution_memory(budgets[device]);
#endif
        }
        memory = AdmissionBudget(std::move(budgets));
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
} // namespace internal

namespace {
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
        state = std::unique_ptr<CudaEngineState<Qwen35CausalLm>>{};
    }
};
} // namespace

std::unique_ptr<Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<EngineInstance<CudaBackend>>(std::move(options));
}
} // namespace mfq::cuda
