#include "load_options.h"
#include "cuda_runtime_config.h"
#include "native/env.h"
#include "execution/cuda_execution.h"
#include "generation.h"
#include "model_registry.h"
#include "batch_execution.h"
#include "engine/qwen4_batch_execution.h"
#include "models/qwen35/batch_state.h"
#include "storage/weights/loaders/weight.h"
#include "storage/weights/moe/expert_cache.h"
#include "storage/state/offload.h"
#include "storage/weights/moe/kv_expert_loan.h"
#include "native/resident_allocations.h"
#include "ops/execution/execution_options.h"
#include "storage/session/text_cache.h"
#include "csrc/engine/include/mtp_metrics.h"
#include "table.h"
#include "flow.h"
#include "instance.h"
#include "native/weight_memory.h"

#include <cuda_runtime_api.h>
#include "kernels/mfq_cuda_kv_quantization.h"
#include <algorithm>
#include <iostream>
#include <variant>
#include <type_traits>
#include <limits>
#include <set>

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
    virtual SessionResult session(const SessionCommand&) = 0;
    virtual std::pair<EngineInfo, std::unique_ptr<TextProcessor>> describe(
        const std::string& tokenizer_model, const std::string& config_path) = 0;
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
    std::shared_ptr<WeightMemoryLedger> weight_memory;
    std::shared_ptr<CudaExecutionContext> execution;
    CudaRuntimeConfig config;
    Model model;
    RuntimeComponents components;
    Model& language;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::optional<batch_storage_t<Model>> storage;
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
            const std::string& tokenizer_model, const std::string& config_path) override {
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
        info.stop_token_ids = model_stop_token_ids(load_model_config_json(*language.source, config_path), info.vocab_size);
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


    std::size_t resident_budget=0, weight_budget=0, initial_expert_budget=0, allocation_ceiling=0;
    std::shared_ptr<KvExpertLoan> graph_expert_loan;
    mfq_tensor_backend::Tensor graph_expert_storage;
    std::optional<std::size_t> graph_expert_region;
    struct WorkspaceExpertRegion {mfq_tensor_backend::Tensor storage;std::size_t region;};
    std::vector<WorkspaceExpertRegion> workspace_expert_regions;
    std::uint64_t workspace_expert_reclaimed_used_bytes=0;
    int graph_draft_depth=3;
    bool preparing_graph_depth=true;

    std::size_t workspace_expert_capacity() const {
        std::size_t bytes=0;for(const auto& part:workspace_expert_regions)bytes+=part.storage.nbytes();return bytes;
    }
    std::size_t workspace_expert_used() const {
        std::size_t bytes=0;
        if(graph_expert_loan)for(const auto& part:workspace_expert_regions)bytes+=graph_expert_loan->region_bytes(part.region);
        return bytes;
    }
    void release_graph_expert_space(std::size_t required_available=std::numeric_limits<std::size_t>::max()) {
        if(!graph_expert_region && workspace_expert_regions.empty())return;
        finish_moe_expert_exchanges(execution->stores.moe_expert_cache);
        MFQ_CUDA_CHECK(cudaDeviceSynchronize());
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const bool all=required_available==std::numeric_limits<std::size_t>::max();
        while(!workspace_expert_regions.empty() && (all || context->available_memory()<required_available)) {
            const auto& part=workspace_expert_regions.back();
            const auto bytes=part.storage.nbytes();
            const auto used=graph_expert_loan->region_bytes(part.region);
            workspace_expert_reclaimed_used_bytes+=used;
            graph_expert_loan->remove(part.region);
            workspace_expert_regions.pop_back();
            context->reclaim_idle_memory();
            std::cerr<<"runtime_prefill_workspace_cache action=reclaim bytes="<<bytes<<" used_bytes="<<used<<'\n';
        }
        if(graph_expert_region && (all || context->available_memory()<required_available)) {
            const auto bytes=graph_expert_storage.nbytes();
            graph_expert_loan->remove(*graph_expert_region);
            graph_expert_region.reset();graph_expert_storage={};
            std::cerr<<"runtime_mtp_workspace_cache action=reclaim bytes="<<bytes<<'\n';
        }
        context->reclaim_idle_memory();
    }

    void restore_mtp_expert_space() {
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(preparing_graph_depth || graph_expert_region || !graph_expert_loan || !components.mtp)return;
            const auto maximum=qwen4_mtp_graph_reserve(language,components.mtp.get());
            const auto active=qwen4_mtp_graph_reserve(language,components.mtp.get(),graph_draft_depth);
            if(active>=maximum)return;
            auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
            const auto available=context->available_memory(),headroom=std::size_t(256)<<20;
            auto bytes=std::min(maximum-active,available>headroom?available-headroom:0);
            bytes=bytes/KvExpertLoan::page*KvExpertLoan::page;
            if(!bytes)return;
            // An ordinary resident tensor supplies revocable expert pages;
            // no decode graph captures an address into this allocation.
            ResidentAllocationScope resident;
            try {
                graph_expert_storage=mfq_tensor_backend::empty({int64_t(bytes)},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kUInt8));
            } catch(const std::exception& error) {
                const std::string message=error.what();
                if(message.find("VRAM limit")==std::string::npos && message.find("out of memory")==std::string::npos)throw;
                std::cerr<<"runtime_mtp_workspace_cache action=skip reason=memory_pressure\n";
                return;
            }
            graph_expert_region=graph_expert_loan->add(graph_expert_storage,int64_t(bytes),1,true,1,true,true);
            std::cerr<<"runtime_mtp_workspace_cache action=lend depth="<<graph_draft_depth<<" bytes="<<bytes<<'\n';
        }
    }

    void restore_graph_expert_space() {
        if(preparing_graph_depth)return;
        prepare_moe_decode_residency(execution->stores.moe_expert_cache);
        mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
        restore_mtp_expert_space();
        if(!graph_expert_loan)return;
        bool ssd=false;std::size_t transfer=0,streamed=0;
        const auto expert_metrics=moe_expert_memory_metrics(execution->stores.moe_expert_cache);
        for(const auto& [name,value]:expert_metrics) {
            if(name=="ssd_expert_enabled")ssd=value!=0;
            if(name=="expert_decode_stage_budget_bytes")transfer=std::size_t(value);
            if(name=="ram_expert_payload_bytes" || name=="ssd_expert_payload_bytes")streamed+=std::size_t(value);
        }
        if(!ssd)return;
        if(const auto* profile=mfq::cuda::env_lookup("MFQ_PROFILE_SSD_EXPERTS");profile && profile[0]=='1' && language.source) {
            const auto io=language.source->file_read_stats();
            std::cerr<<"runtime_model_io_stats boundary=prefill_end calls="<<io.calls
                <<" logical_bytes="<<io.logical_bytes<<" physical_bytes="<<io.physical_bytes
                <<" read_ms="<<double(io.read_nanoseconds)/1e6<<'\n';
            std::cerr<<"runtime_moe_pipeline_stats boundary=prefill_end";
            for(const auto& [name,value]:expert_metrics)
                if(name=="expert_route_wait_ms" || name=="expert_plan_ms" ||
                   name=="expert_host_fetch_ms" || name=="expert_logical_transfer_bytes" ||
                   name=="ssd_expert_read_wait_ms")std::cerr<<' '<<name<<'='<<value;
            std::cerr<<'\n';
        }
        // Spare device memory can hold revocable copies of streamed experts.
        // Primary arenas retain their configured budget. Prefill, KV growth
        // and online resident budgets return these chunks before allocation.
        const auto reserve=runtime_options::workspace_reserve_bytes();
        const auto decode=(std::size_t(1280)<<20)+transfer;
        const auto existing=workspace_expert_capacity();
        const auto target=std::max(streamed,reserve>decode?reserve-decode:0);
        if(existing>=target)return;
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto available=context->available_memory(),headroom=std::size_t(256)<<20;
        auto bytes=std::min(target-existing,available>headroom?available-headroom:0);
        bytes=bytes/KvExpertLoan::page*KvExpertLoan::page;
        if(!bytes)return;
        ResidentAllocationScope resident;
        // Small KV or prefix growth returns only the required chunks, keeping
        // the rest of the decoded expert working set available to MTP.
        while(bytes) {
            const auto chunk=std::min(bytes,std::size_t(256)<<20);
            mfq_tensor_backend::Tensor storage;
            try {
                storage=mfq_tensor_backend::empty({int64_t(chunk)},
                    mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kUInt8));
            }catch(const std::exception& error) {
                const std::string message=error.what();
                if(message.find("VRAM limit")==std::string::npos && message.find("out of memory")==std::string::npos)throw;
                break;
            }
            workspace_expert_regions.reserve(workspace_expert_regions.size()+1);
            const auto region=graph_expert_loan->add(storage,int64_t(chunk),1,true,1,true,true);
            workspace_expert_regions.push_back({std::move(storage),region});
            bytes-=chunk;
        }
        std::cerr<<"runtime_prefill_workspace_cache action=lend bytes="<<workspace_expert_capacity()-existing
            <<" capacity_bytes="<<workspace_expert_capacity()<<" reserve_bytes="<<reserve<<'\n';
    }

    void prepare_graph_depth(const InferenceRequest& input) {
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(!components.mtp)return;
            const auto* warmup=mfq::cuda::env_lookup("MFQ_QWEN4_WARMUP_GRAPHS");
            if(!language.reuse_decode_graphs || !execution->config.moe_pipeline || (warmup && warmup[0]=='0'))return;
            const auto& sampling=input.sampling;
            const bool wide=sampling.temperature<=0 || sampling.top_k==1 ||
                (sampling.top_k>0 && sampling.top_k<=64);
            const int requested=sampling.enable_mtp && wide
                ?std::clamp(int(sampling.mtp_max_draft_tokens),1,components.mtp->maximum_draft_depth()):1;
            // Short prefill shares the existing four-row graphs. Only draft
            // caps above three need additional verification storage.
            const int depth=std::max(3,requested);
            if(depth==graph_draft_depth && language.graph_registry)return;
            const auto started=std::chrono::steady_clock::now();
            preparing_graph_depth=true;
            try {
                release_graph_expert_space();
                graph.invalidate();
                qwen4_release_decode_graphs(language,components.mtp.get());
                release_moe_decode_buffers(execution->stores.moe_expert_cache);
                language.reset(1);components.mtp->reset(1);
                auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
                context->reclaim_idle_memory();
                reclaim_prefix_workspace((std::size_t(2)<<30)+qwen4_mtp_graph_reserve(language,components.mtp.get(),depth));
                qwen4_warmup_decode_graphs(language,components.mtp.get(),depth);
                graph_draft_depth=depth;
                preparing_graph_depth=false;
                restore_graph_expert_space();
            } catch(...) {
                graph_draft_depth=0;preparing_graph_depth=false;
                throw;
            }
            std::cerr<<"runtime_mtp_depth_plan requested_depth="<<requested<<" prepared_depth="<<depth
                <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
        }
    }

    bool online_budget_supported() const {
        return execution->layer_placement.devices.size()<=1 && !execution->tensor_parallel.enabled()
            && !execution->expert_parallel.enabled();
    }

    void reclaim_prefix_workspace(std::size_t bytes) {
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto available=context->available_memory();
        if(available>=bytes)return;
        if(graph_expert_region || !workspace_expert_regions.empty()) {
            std::cerr<<"runtime_workspace_cache_reclaim reason=workspace requested_bytes="<<bytes
                <<" available_bytes="<<available<<'\n';
            release_graph_expert_space(bytes);
        }
        if(context->available_memory()>=bytes)return;
        std::size_t hot=0;
        for(const auto& [name,value]:cache.metrics())
            if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
        const auto shortage=bytes-context->available_memory();
        const auto released=cache.trim_hot(hot>shortage?hot-shortage:0);
        context->reclaim_idle_memory();
        if(released)std::cerr<<"runtime_workspace_reclaim requested_bytes="<<bytes
            <<" available_bytes="<<context->available_memory()<<" prefix_released_bytes="<<released<<'\n';
    }

    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents optional, CudaRuntimeConfig runtime_config, std::size_t memory_limit, std::shared_ptr<WeightMemoryLedger> weights)
        : weight_memory(std::move(weights)), execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(bind_language(components, model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1,
                cuda_has_separate_memory()) {
        initial_expert_budget=moe_expert_cache_budget_bytes(execution->stores.moe_expert_cache);
        if (execution->stores.qsa_kv_store)
            execution->stores.qsa_kv_store->set_ram_reservation([this](std::size_t bytes) {
                cache.reserve_ram(bytes);reclaim_moe_optional_host(execution->stores.moe_expert_cache,bytes);
            });
        allocation_ceiling=mfq::cuda::default_context(execution->layer_placement.primary_device())->memory_stats().limit;
        if (config.continuous_batch.max_sequences && !supports_batch)
            throw std::invalid_argument("continuous batching is unavailable for this model");
        prepare_memory(memory_limit);
        if constexpr (supports_batch) {
            if (config.continuous_batch.max_sequences) {
                if constexpr(std::is_same_v<batch_storage_t<Model>,std::monostate>) {
                    batching=std::make_unique<Batch>(config.generation.prefill_chunk_size,
                        config.continuous_batch.prefill_token_budget,language,*execution,config.continuous_batch);
                } else {
                storage.emplace(language, config.continuous_batch.max_sequences,
                                config.continuous_batch.paged_kv);
                batching = std::make_unique<Batch>(config.generation.prefill_chunk_size,
                    config.continuous_batch.prefill_token_budget, language, *execution,
                    config.continuous_batch, *storage);
                }
            }
        }
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        if(execution->stores.moe_expert_cache) {
            graph_expert_loan=execution->stores.kv_memory?execution->stores.kv_memory->expert_loan():nullptr;
            if(!graph_expert_loan && (components.mtp || moe_expert_cache_ssd_experts(execution->stores.moe_expert_cache))) {
                graph_expert_loan=std::make_shared<KvExpertLoan>();
                set_moe_kv_expert_loan(execution->stores.moe_expert_cache,graph_expert_loan);
            }
            language.restore_decode_workspace=[this]{restore_graph_expert_space();};
            set_moe_workspace_reservation(execution->stores.moe_expert_cache,
                [this](std::size_t bytes) {reclaim_prefix_workspace(bytes);},
                [this](std::size_t bytes) {
                    cache.reserve_ram(bytes);reclaim_moe_optional_host(execution->stores.moe_expert_cache,bytes);
                });
        }
        if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
            language.reserve_kv_workspace = [this](std::size_t bytes) {
                const auto device = execution->layer_placement.primary_device();
                auto context = mfq::cuda::default_context(device);
                if (context->available_memory() >= bytes) return;
                if(graph_expert_region || !workspace_expert_regions.empty()) {
                    std::cerr<<"runtime_workspace_cache_reclaim reason=kv_growth requested_bytes="<<bytes
                        <<" available_bytes="<<context->available_memory()<<'\n';
                    release_graph_expert_space(bytes);
                }
                if(context->available_memory()>=bytes)return;
                std::size_t hot = 0;
                for (const auto& [name,value] : cache.metrics())
                    if (name == "prefix_cache_hot_bytes") hot = std::size_t(value);
                const auto shortage = bytes - context->available_memory();
                cache.trim_hot(hot > shortage ? hot - shortage : 0);
                context->reclaim_idle_memory();
                if (context->available_memory() < bytes)
                    throw InferenceInputError(InferenceInputErrorCode::ContextLength,
                        "insufficient VRAM for context growth; reduce concurrent requests or prompt length");
            };
            if(execution->stores.kv_memory)execution->stores.kv_memory->set_reservation(language.reserve_kv_workspace);
        }
#endif
        if(const auto bytes=runtime_options::resident_budget_bytes()) {
            SessionCommand command;
            command.kind=SessionCommand::Kind::memory_budget;
            command.bytes=bytes;
            session(command);
        }
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            qwen4_warmup_decode_graphs(language,components.mtp.get());
            // Exercise the same draft/verify/commit loop as a real request.
            // Graph-only warmup cannot cover its history replay, sampling and
            // expert exchange boundaries. No synthetic policy/history survives.
            const auto* warmup=mfq::cuda::env_lookup("MFQ_QWEN4_WARMUP_MTP");
            if(language.graph_registry && components.mtp &&
                    language.max_position_embeddings()>=48 && (!warmup || warmup[0]!='0')) {
                struct CacheLearningScope {
                    std::shared_ptr<MoeExpertCache> cache;bool previous;
                    explicit CacheLearningScope(std::shared_ptr<MoeExpertCache> value):cache(std::move(value)),
                        previous(set_moe_reuse_learning(cache,false)) {}
                    ~CacheLearningScope(){set_moe_reuse_learning(cache,previous);}
                } learning(execution->stores.moe_expert_cache);
                auto& predictor=*components.mtp;
                const auto saved_policy=predictor.policy_state;
                const auto start=std::chrono::steady_clock::now();
                InferenceRequest request;
                request.chat=false;
                request.sampling.temperature=0.7;
                request.sampling.top_p=0.8;
                request.sampling.top_k=20;
                request.sampling.seed=1234;
                request.sampling.max_tokens=static_cast<int32_t>(
                    std::min<int64_t>(64,language.max_position_embeddings()-16));
                request.sampling.mtp_max_draft_tokens=3;
                for(int i=0;i<16;++i)request.prompt.push_back((i*7919+17)%language.vocab_size());
                InferenceOutput output(request,nullptr,"cuda-mtp-warmup");
                auto scope=language.execution_scope();
                auto generation=run_mtp_generation(language,predictor,request,output,16);
                while(generation.next()) {}
                finish_moe_expert_exchanges(execution->stores.moe_expert_cache);
                predictor.reset(1);
                language.reset(1);
                predictor.policy_state=saved_policy;
                predictor.last_stats={};
                predictor.last_cycles=predictor.last_accepted=predictor.last_rejected=0;
                MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
                std::cerr<<"runtime_qwen4_mtp_warmup tokens="<<output.result.completion_tokens
                    <<" ms="<<std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count()<<'\n';
                release_moe_prefill_buffers(execution->stores.moe_expert_cache);
                mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
            }
        }
        preparing_graph_depth=false;
        restore_graph_expert_space();
    }
    ~CudaEngineState() {
        language.restore_decode_workspace={};
        try {release_graph_expert_space();}
        catch(const std::exception& error) {std::cerr<<"runtime_mtp_workspace_cache cleanup_error="<<error.what()<<'\n';}
        if(execution->stores.kv_memory)execution->stores.kv_memory->set_reservation({});
        set_moe_workspace_reservation(execution->stores.moe_expert_cache,{});
        if (execution->stores.qsa_kv_store) execution->stores.qsa_kv_store->set_ram_reservation({});
    }
    SessionResult session(const SessionCommand& command) override {
        if(command.kind!=SessionCommand::Kind::memory_budget)return control_session(cache,command);
        if(!online_budget_supported())throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        const auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto dense=weight_memory->resident_bytes();
        const auto minimum=moe_expert_cache_minimum_bytes(execution->stores.moe_expert_cache);
        const auto requested_weights=command.weight_bytes.value_or(weight_budget);
        if(requested_weights && requested_weights<dense+minimum)
            throw std::invalid_argument("weight budget is below dense weights and minimum expert working set");
        auto limit=command.bytes ? allocation_ceiling ? std::min<std::size_t>(command.bytes,allocation_ceiling) : command.bytes : allocation_ceiling;
        if(limit) {
            std::size_t free=0,total=0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free,&total));
            limit=std::min(limit,std::min(total,free+context->resident_memory_usage()));
        }
        if(limit && limit<dense+minimum)
            throw std::invalid_argument("resident memory budget is below non-offloadable weights");
        const auto previous_limit=context->memory_stats().limit;
        const auto previous_experts=moe_expert_cache_budget_bytes(execution->stores.moe_expert_cache);
        const auto before=context->resident_memory_usage();
        release_graph_expert_space();
        finish_moe_expert_exchanges(execution->stores.moe_expert_cache);
        graph.invalidate();
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            qwen4_release_decode_graphs(language,components.mtp.get());
        if constexpr(supports_batch) if(batching)batching->release_idle_buffers();
        if(components.mtp)components.mtp->reset(1);
        language.reset(1);
        release_moe_decode_buffers(execution->stores.moe_expert_cache);
        context->reclaim_idle_memory();
        auto target=requested_weights ? requested_weights-dense : initial_expert_budget;
        std::size_t hot=0,expert_bytes=0;
        for(const auto& [name,value]:cache.metrics())if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
        for(const auto& [name,value]:moe_expert_memory_metrics(execution->stores.moe_expert_cache))
            if(name=="resident_expert_cache_bytes")expert_bytes=std::size_t(value);
        auto used=context->resident_memory_usage();
        if(limit && used>limit) {
            cache.trim_hot(hot>used-limit ? hot-(used-limit) : 0);
            context->reclaim_idle_memory();used=context->resident_memory_usage();
        }
        const auto fixed=used>expert_bytes?used-expert_bytes:0;
        if(limit) {
            if(fixed>limit || minimum>limit-fixed)throw std::invalid_argument("resident memory budget is below live KV and workspace requirements");
            target=std::min(target,limit-fixed);
        }
        try {
            if(!limit || limit>previous_limit)context->set_memory_limit(limit);
            if(execution->stores.moe_expert_cache)resize_moe_expert_cache(execution->stores.moe_expert_cache,std::max(target,minimum));
            context->set_memory_limit(limit);
        } catch(...) {
            context->set_memory_limit(0);
            if(execution->stores.moe_expert_cache)resize_moe_expert_cache(execution->stores.moe_expert_cache,previous_experts);
            context->set_memory_limit(previous_limit);
            throw;
        }
        resident_budget=command.bytes;weight_budget=requested_weights;
        const auto after=context->resident_memory_usage();
        return {before>after?before-after:0,{}};
    }
    bool can_overlap(const EngineRequest& request) const override {
        if constexpr(std::is_same_v<RequestState,std::monostate>) {
            const auto& input=request.input;
            if(input.media || !input.cache_plan.session_id.empty() ||
                (input.sampling.enable_mtp && components.mtp))return false;
        }
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
        if(!batched)prepare_graph_depth(input);
        std::optional<Generation> physical;
        if constexpr (supports_batch) if (batched) physical.emplace(batching->generate(id, input, output));
        return internal::generate(language, graph, cache, config, input, output,
                                  components.mtp.get(), std::move(prepared), std::move(physical));
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) override {
        bool owned=batch_requires_owned_generation(request.input,cache,bool(components.mtp));
        if constexpr(std::is_same_v<RequestState,std::monostate>)
            owned=request.input.vision || !request.input.cache_plan.session_id.empty() ||
                (request.input.sampling.enable_mtp && components.mtp);
        if (request.concurrent && owned) {
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
            const auto batch_metrics = [&] { if constexpr(supports_batch) { if(batching)return batching->metrics(); } return Metrics{}; }();
            size_t free = 0, total = 0;
            MFQ_CUDA_CHECK(cudaMemGetInfo(&free, &total));
            auto device_memory = mfq_cuda_memory_stats(mfq_current_cuda_device());
            const bool vision_declared = components.graph.has_component("vision");
            const bool mtp_declared = components.graph.has_component("predictor");
            Metrics metrics{

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
            auto experts = moe_expert_memory_metrics(execution->stores.moe_expert_cache);
            metrics.insert(metrics.end(), experts.begin(), experts.end());
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                auto phases=qwen4_decode_graph_profile_metrics(language,components.mtp.get());
                metrics.insert(metrics.end(),phases.begin(),phases.end());
            }
            if(language.source) {
                const auto io=language.source->file_read_stats();
                metrics.insert(metrics.end(),{{"model_io_direct",io.mode==mfq::FileReadMode::Direct?1.0:0.0},
                    {"model_io_read_calls",double(io.calls)},{"model_io_logical_bytes",double(io.logical_bytes)},
                    {"model_io_physical_bytes",double(io.physical_bytes)},{"model_io_read_ms",double(io.read_nanoseconds)/1e6},
                    {"model_io_staging_bytes",double(io.staging_bytes)},{"model_io_cached_staging_bytes",double(io.staging_cached_bytes)},
                    {"model_io_staging_allocations",double(io.staging_allocations)}});
            }
            if(execution->stores.kv_memory) {
                auto kv=execution->stores.kv_memory->metrics();
                metrics.insert(metrics.end(),kv.begin(),kv.end());
            }
            std::size_t resident_weights=weight_memory->resident_bytes();
            for(const auto& metric:experts)
                if(metric.first=="resident_expert_cache_bytes")resident_weights+=std::size_t(metric.second);
            const auto ple=weight_memory->ple_bytes();
            metrics.insert(metrics.end(),{{"resident_weight_bytes",double(resident_weights)},
                {"ssd_ple_enabled",ple>0?1.0:0.0},{"ssd_ple_payload_bytes",double(ple)}});
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                auto cache_memory=qwen4_cache_memory_metrics(language,components.mtp.get());
                for (auto& [name,value] : cache_memory)
                    for (const auto& [batch_name,batch_value] : batch_metrics) {
                        if (name == "kv_cache_bytes" && batch_name == "continuous_batching_request_kv_bytes")
                            value += batch_value;
                        if (name == "kv_cache_contexts" && batch_name == "continuous_batching_request_contexts")
                            value += batch_value;
                    }
                metrics.insert(metrics.end(),cache_memory.begin(),cache_memory.end());
            }
            const auto metric_bytes = [&](const std::string& key) {
                for(const auto& [name,value]:metrics)if(name==key)return std::size_t(value);
                return std::size_t(0);
            };
            std::set<int> devices(execution->layer_placement.devices.begin(),execution->layer_placement.devices.end());
            devices.insert(execution->tensor_parallel.devices.begin(),execution->tensor_parallel.devices.end());
            devices.insert(execution->expert_parallel.devices.begin(),execution->expert_parallel.devices.end());
            const auto primary=execution->layer_placement.primary_device();
            devices.insert(primary);
            auto prefix_bytes=cache.device_hot_bytes();
            std::size_t allocated=0,reserved=0,free_bytes=0,total_bytes=0,buffers=0;
            for(const int device:devices) {
                MfqCudaGuard guard(device);
                size_t free=0,total=0;
                MFQ_CUDA_CHECK(cudaMemGetInfo(&free,&total));
                const auto memory=mfq_cuda_memory_stats(device);
                const auto external=device==primary?metric_bytes("kv_memory_vmm_bytes"):0;
                const auto resident=std::max(memory.reserved_bytes,memory.allocated_bytes)+external;
                const auto stem="gpu_"+std::to_string(device)+"_";
                auto weights=weight_memory->resident_bytes(device);
                if(device==primary)weights+=metric_bytes("resident_expert_cache_bytes");
                const auto hot=prefix_bytes[device];
                auto live=metric_bytes(stem+"live_kv_bytes");
                if(device==primary)for(const auto& [name,value]:batch_metrics)
                    if(name=="continuous_batching_request_kv_bytes")live+=std::size_t(value);
                metrics.insert(metrics.end(),{{stem+"resident_bytes",double(resident)},
                    {stem+"resident_weight_bytes",double(weights)}, {stem+"prefix_cache_bytes",double(hot)},
                    {stem+"memory_total_bytes",double(total)}, {stem+"memory_available_bytes",double(free)}});
                if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                    const auto categorized=weights+live+hot;
                    const auto buffer=resident>categorized?resident-categorized:0;
                    metrics.emplace_back(stem+"resident_buffer_bytes",double(buffer));
                    metrics.emplace_back(stem+"kv_bytes",double(live+hot));
                    buffers+=buffer;
                }
                allocated+=memory.allocated_bytes+external;reserved+=memory.reserved_bytes+external;free_bytes+=free;total_bytes+=total;
            }
            metrics.insert(metrics.end(),{{"device_free_bytes",double(free_bytes)}, {"device_total_bytes",double(total_bytes)},
                {"cuda_allocated_bytes",double(allocated)}, {"cuda_reserved_bytes",double(reserved)}});
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)metrics.emplace_back("resident_buffer_bytes",double(buffers));
            const auto expert_bytes=metric_bytes("resident_expert_cache_bytes");
            const auto minimum=moe_expert_cache_minimum_bytes(execution->stores.moe_expert_cache);
            metrics.insert(metrics.end(),{{"resident_memory_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_weight_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_memory_budget_bytes",double(resident_budget)}, {"resident_weight_budget_bytes",double(weight_budget)},
                {"resident_memory_used_bytes",double(std::max(allocated,reserved))},
                {"resident_reclaimable_weight_bytes",double(expert_bytes>minimum?expert_bytes-minimum:0)}});
            metrics.emplace_back("qsa_kv_offload_supported",std::is_same_v<Model, Qwen4CausalLm> ? 1.0 : 0.0);
            if (execution->stores.qsa_kv_store) {
                auto kv = execution->stores.qsa_kv_store->metrics();
                metrics.insert(metrics.end(), kv.begin(), kv.end());
            } else metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
                {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0}, {"qsa_kv_ssd_bytes", 0.0}});
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
    CudaEngineState<Model, ContinuousBatch<batch_executor_t<Model>>,
                    std::conditional_t<std::is_same_v<Model,Qwen35CausalLm>,qwen35::Qwen35GenerationState,std::monostate>>,
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
        auto weight_memory=std::make_shared<WeightMemoryLedger>();
        const WeightMemoryLoadScope memory_load(*weight_memory);
        mfq_tensor_backend::NoGradGuard no_grad;
        auto config = resolve_cuda_runtime_config(options);
        const auto kv_options = runtime_options::qsa_kv_offload();
        execution->stores.kv_memory=std::make_shared<KvMemoryManager>();
        if (kv_options.gpu_budget_bytes) {
            execution->stores.kv_memory.reset();
            KvOffloadConfig kv;
            kv.gpu_budget_bytes = kv_options.gpu_budget_bytes;
            kv.ram_budget_bytes = cuda_has_separate_memory() ? kv_options.ram_budget_bytes : 0;
            kv.buffer_bytes = std::min<std::size_t>(kv.buffer_bytes, kv.gpu_budget_bytes / 4);
            kv.directory = kv_options.directory;
            execution->stores.qsa_kv_store = std::make_shared<KvOffloadStore>(std::move(kv));
        }
        state = with_loaded_cuda_model(*execution, options, true,
            [&](auto& model, auto& components, auto, auto) -> State {
                using Model = std::decay_t<decltype(model)>;
                if constexpr (!Model::policy.qsa_kv_offload)
                    if (execution->stores.qsa_kv_store)
                        throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                return std::make_unique<BoundState<Model>>(execution, std::move(model),
                    std::move(components), config, options.memory_budget_bytes, weight_memory);
            });
        return visit([&](CudaEngineOps& state) { return state.describe(options.tokenizer_model, options.config_path); });
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
