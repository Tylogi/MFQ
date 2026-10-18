#include "generation.h"
#include "generation_flow.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "core/full_block.h"
#include "cuda_sampling.h"
#include "storage/text_session_cache.h"
#include "mfq_cuda_sampling_ops.h"
#include "storage/moe_expert_cache.h"
#include "runtime/execution_options.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace mfq::cuda::internal {
using mfq::engine::Generation;
namespace {
template <typename Model>
static mfq_tensor_backend::Tensor sample_token(Model &model, mfq_tensor_backend::Tensor ids,
    mfq::cuda::Sampler &sampler, mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr &token_constraint) {
    if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
        return model.next_token(ids);
    }

    auto logits = model.last_logits(ids).contiguous().view({1, -1});
    return mfq::cuda::sample_logits(sampler, std::move(logits), counts, token_constraint);
}

template <typename Model> struct CudaGenerationOps {
    Model &model;
    DecodeGraphCache &graph;
    const MfqTokenConstraintPtr &constraint;
    mfq_tensor_backend::Tensor pending, counts;
    mfq_tensor_backend::Tensor known_token_host;
    mfq::cuda::Sampler sampler;
    bool has_penalties;
    const CudaPreparedPrompt *prepared;
    const CudaDecodeGraphConfig &graph_config;
    mfq_tensor_backend::Tensor full_ids;
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;
    int64_t prefill_compute_chunk=1024;
    mfq::engine::InferenceOutput* output=nullptr;

    CudaGenerationOps(Model &model, DecodeGraphCache &graph,
        const mfq::engine::InferenceRequest &request, const CudaPreparedPrompt *prepared,
        const CudaDecodeGraphConfig &graph_config,int64_t compute_chunk=1024,
        mfq::engine::InferenceOutput* inference_output=nullptr)
        : model(model), graph(graph), constraint(request.token_constraint),
          sampler(request.sampling),
          has_penalties(sampler.has_penalties()), prepared(prepared), graph_config(graph_config),
          prefill_compute_chunk(compute_chunk),output(inference_output) {
        full_ids = mfq_tensor_backend::tensor(request.prompt,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA))
                       .reshape({1, -1})
                       .contiguous();
        graph.ensure_storage(model.vocab_size());
        if (has_penalties) {
            counts = mfq::cuda::SamplingOps::token_counts(full_ids, model.vocab_size(), graph.counts);
        }
        generation_limit = request.sampling.max_tokens;
        graph_active = graph_eligible();
    }

    double prefill(const mfq::engine::PrefillChunk &chunk) {
        using Tensor = mfq_tensor_backend::Tensor;
        PrefillCudaTimer timer;
        bool prefill_finished=false;
        const auto finish_prefill=[&] {
            if(prefill_finished)return;
            MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            prefill_finished=true;
        };
        auto ids = full_ids.narrow(1, chunk.offset, chunk.count).contiguous();
        Tensor hidden;
        bool layer_major=false,short_windows=false;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if((!prepared || !prepared->transformed()) && model.reuse_decode_graphs &&
                    model.graph_registry && chunk.count<=128) {
                // The same bounded prompt windows as MTP avoid materializing
                // a whole expert layer before a short prompt has any scratch.
                short_windows=true;
                for(int64_t offset=0;offset<chunk.count;) {
                    if(output && output->stopped())break;
                    const auto rows=model.cache_pos==0?int64_t(1):std::min<int64_t>(4,chunk.count-offset);
                    auto part=ids.narrow(1,offset,rows);
                    Tensor window_logits;
                    if(auto result=qwen4_decode_graph_hidden(model,part,nullptr,rows>1?1:0,
                            nullptr,{},&window_logits))
                        hidden=std::move(*result);
                    else hidden=model.hidden_forward(part,mfq_nullopt,mfq_nullopt,nullptr,
                        mfq_nullopt,nullptr,rows>1?1:0);
                    if(rows>1)model.commit_speculative();
                    offset+=rows;
                }
            } else if((!prepared || !prepared->transformed()) &&
                qwen4_layer_major_prefill_eligible(model,chunk.count)) {
                layer_major=true;
                hidden=qwen4_layer_major_prefill_hidden(model,ids,prefill_compute_chunk,true,
                    [&] {return output && output->stopped();},nullptr,false,finish_prefill);
            }
        }
        if(output && output->stopped()) {
            finish_prefill();
            return timer.elapsed_ms();
        }
        if (hidden.defined()) {
            // The layer-major path has already advanced every physical cache.
        } else if (prepared && prepared->transformed()) {
            MoePrefillWorkspaceLoan workspace(model.execution->moe_expert_cache);
            bool borrowed=false;
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
                borrowed=workspace.begin((std::size_t(4)<<30)+
                    std::size_t(chunk.count)*model.hidden_size()*model.hc_mult()*sizeof(float)*4)>0;
            hidden = model.hidden_forward_inputs(ids,
                prepared->embeddings.narrow(1, chunk.offset, chunk.count).contiguous(),
                prepared->positions.defined()
                    ? MfqOptional<Tensor>(prepared->positions.narrow(-1, chunk.offset, chunk.count).contiguous())
                    : MfqOptional<Tensor>(mfq_nullopt),
                mfq_nullopt,
                nullptr,
                mfq_nullopt,
                true);
            finish_prefill();
            if(borrowed) {
                workspace.suspend();
                hidden=hidden.narrow(1,hidden.size(1)-1,1).clone();
                workspace.finish();
            }
            model.decode_position_delta = prepared->decode_position_delta;
        } else {
            MfqOptional<Tensor> length = mfq_nullopt;
            if (model.adapter_uses_decode_sequence_length() && chunk.offset > 0 && chunk.count == 1)
                length = mfq_tensor_backend::full({1}, chunk.offset + 1, full_ids.options());
            hidden = model.hidden_forward(ids, mfq_nullopt, length);
        }
        // Match prompt-progress timing: first-token head/sampling and expert
        // restoration remain in TTFT, outside reported prefill throughput.
        finish_prefill();
        if (chunk.offset + chunk.count == full_ids.size(1)) {
            auto logits = model
                              .logits_from_hidden(hidden.index({Slice(), -1, Slice()})
                                      .to(mfq_tensor_backend::kFloat16)
                                      .contiguous())
                              .reshape({1, -1});
            pending = mfq::cuda::sample_logits(sampler, std::move(logits), counts, constraint);
        }
        const auto elapsed=timer.elapsed_ms();
        if(graph_config.trace)std::cerr<<"runtime_prefill_chunk offset="<<chunk.offset
            <<" tokens="<<chunk.count<<" layer_major="<<layer_major
            <<" short_windows="<<short_windows
            <<" compute_chunk="<<prefill_compute_chunk<<" elapsed_ms="<<elapsed
            <<" timing=before_cache_restore"<<'\n';
        return elapsed;
    }
    std::int64_t first_token() {
        const auto token=pending.template item<int64_t>();
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            // prefill()'s sequence-sized temporaries have now retired. Cache
            // restoration remains before token publication, inside TTFT.
            if(model.restore_decode_workspace)model.restore_decode_workspace();
        }
        return token;
    }

    bool graph_eligible() const {
        const auto &execution = *model.execution;
        return graph_config.enabled && (!prepared || !prepared->transformed()) && !constraint &&
               !model.metadata.flash_next && mfq_cuda_graph_capture_supported() &&
               execution.dsv4_cpu_offload_layers.empty() && execution.dense_cpu_layer_count == 0 &&
               !execution.moe_expert_cache && model_parallel_cuda_graph_enabled(execution) &&
               generation_limit >= graph_config.minimum_generation_tokens &&
               generation_limit <= graph.generated_capacity;
    }
    void prepare_graph() {
        const auto &execution = *model.execution;
        graph.ensure_compute_streams(execution.tensor_parallel.enabled()
                                         ? execution.tensor_parallel
                                         : execution.expert_parallel);
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        cudaStream_t graph_stream = graph.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }

        std::int64_t position = model.cache_pos;
        std::int64_t length = position + 1;
        std::int64_t step = 1;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(graph.static_input.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(),
            sizeof(int64_t),
            cudaMemcpyDeviceToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(graph.generated.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(),
            sizeof(int64_t),
            cudaMemcpyDeviceToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(graph.static_pos.template data_ptr<int64_t>(),
            &position,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(graph.static_len.template data_ptr<int64_t>(),
            &length,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(graph.static_step.template data_ptr<int64_t>(),
            &step,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            graph_stream));
        *sampler.ops().random_host().template data_ptr<float>() = 0.5f;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(graph.random.template data_ptr<float>(),
            sampler.ops().random_host().template data_ptr<float>(),
            sizeof(float),
            cudaMemcpyHostToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream));

        const std::int64_t requested = model.cache_pos + generation_limit;
        const std::int64_t planned =
            decode_graph_bucket(requested, model.max_position_embeddings());
        const std::int64_t parts =
            decode_graph_attention_parts(planned, FullBlock::kDecodeAttentionMaxParts);
        const bool greedy = sampler.greedy();
        const auto sample_static = [&]() {
            if (greedy && !has_penalties) {
                return model.next_token_static(
                    graph.static_input, graph.static_pos, graph.static_len, {planned, parts});
            }
            auto logits =
                model
                    .last_logits_static(
                        graph.static_input, graph.static_pos, graph.static_len, {planned, parts})
                    .contiguous()
                    .view({1, -1});
            if (has_penalties) {
                logits = sampler.apply_penalties(std::move(logits), counts);
            }
            if (greedy) {
                return sampler.ops().sample_greedy(std::move(logits));
            }
            return sampler.ops().sample_stochastic(
                std::move(logits), graph.random, sampler.params());
        };
        const bool hit = graph.ensure_captured(model, planned, sampler.params(), greedy, [&]() {
            return sample_static();
        }, [&](const mfq_tensor_backend::Tensor &next) {
            if (has_penalties) {
                sample_token_counts_add_cuda(counts, next.contiguous());
            }
            decode_graph_commit_cuda(next,
                graph.generated,
                graph.static_step,
                graph.static_input,
                graph.static_pos,
                graph.static_len);
        });
        if (!hit)
            report_cuda_memory(execution.config, "runtime_graph_capture");
        if (graph_config.trace) {
            std::cerr << "runtime_cuda_graph action=" << (hit ? "reuse" : "capture")
                      << " requested_len=" << requested << " planned_len=" << planned
                      << " captures=" << graph.captures << " reuses=" << graph.reuses << std::endl;
        }
    }
    std::int64_t advance() {
        if (!graph_active) {
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                if(known_token_host.defined()) {
                    const bool greedy=sampler.greedy() && !has_penalties && !constraint;
                    if(auto result=qwen4_decode_graph_logits(model,pending.reshape({1,1}),greedy?2:1,known_token_host)) {
                        pending=greedy?std::move(*result):mfq::cuda::sample_logits(sampler,
                            result->contiguous().view({1,-1}),counts,constraint);
                        return pending.template item<int64_t>();
                    }
                }
            }
            pending = sample_token(model, pending.reshape({1, 1}), sampler, counts, constraint);
            return pending.template item<int64_t>();
        }

        if (!graph_prepared) {
            prepare_graph();
            graph_prepared = true;
        }
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        cudaStream_t graph_stream = graph.stream.stream();
        if (!sampler.greedy()) {
            *sampler.ops().random_host().template data_ptr<float>() = sampler.next_uniform_float();
            MFQ_CUDA_CHECK(cudaMemcpyAsync(graph.random.template data_ptr<float>(),
                sampler.ops().random_host().template data_ptr<float>(),
                sizeof(float),
                cudaMemcpyHostToDevice,
                graph_stream));
        }
        graph.graph->replay();
        const auto token = graph.static_next.template item<int64_t>();
        ++model.cache_pos;
        return token;
    }
    void accept(std::int64_t token) {
        if (has_penalties && !graph_active) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            const auto* option=std::getenv("MFQ_DECODE_EARLY_PLE");
            if((option && option[0]=='0') || !model.reuse_decode_graphs || !model.graph_registry ||
               (prepared && prepared->transformed()) || (output && output->stopped()))return;
            if(!known_token_host.defined())known_token_host=mfq_tensor_backend::empty({1,1},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kInt64));
            known_token_host.template data_ptr<int64_t>()[0]=token;
            if(qwen4_prefetch_decode_ple(model,known_token_host)) {
                static bool reported=false;
                if(!reported) {
                    reported=true;
                    std::cerr<<"runtime_decode_early_ple enabled=1 host_token_reuse=1\n";
                }
            }
        }
    }
};

} // namespace

template <class Model> struct CudaGenerationContext {
    using Hidden = mfq_tensor_backend::Tensor;
    using Restore = TextSessionRestore;
    Model &model;
    DecodeGraphCache &graph;
    TextSessionCache &cache;
    const CudaRuntimeConfig &config;
    MtpModule *mtp;
    std::optional<CudaPreparedPrompt> prepared;
    mfq::engine::ContinuousBatch<CudaBatchOperations<Model>>* batching;
    std::string request_id;

    bool batched() const { return batching != nullptr; }
    void trace_expert_cache(const char* boundary) {
        // Request boundaries only: these existing counters need no events or
        // extra per-token synchronization. Diagnostic output drains exchanges.
        static const bool enabled=[] {
            const char* value=std::getenv("MFQ_TRACE_MOE_SUMMARY");
            return value && value[0] && value[0]!='0';
        }();
        if (!enabled || batching || std::uncaught_exceptions()) return;
        std::cerr << "moe_request_cache_stats boundary=" << boundary << '\n';
        print_moe_expert_cache_stats(model.execution->moe_expert_cache,std::cerr);
        if(model.source) {
            const auto io=model.source->file_read_stats();
            std::cerr<<"runtime_model_io_stats boundary="<<boundary<<" calls="<<io.calls
                <<" logical_bytes="<<io.logical_bytes<<" physical_bytes="<<io.physical_bytes
                <<" read_ms="<<double(io.read_nanoseconds)/1e6<<'\n';
        }
    }
    void prepare_request(const mfq::engine::InferenceRequest& input) {
        trace_expert_cache("begin");
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            model.decode_reserve_until=std::min<int64_t>(model.max_position_embeddings(),
                int64_t(input.prompt.size())+std::min<int64_t>(input.sampling.max_tokens,256));
            if(!batching && input.prompt.size()>1024) {
                finish_moe_expert_exchanges(model.execution->moe_expert_cache);
                if(model.cache_pos==0 && !model.graph_registry)
                    release_moe_decode_buffers(model.execution->moe_expert_cache);
                else release_moe_prefill_buffers(model.execution->moe_expert_cache);
                mfq::cuda::default_context(mfq_current_cuda_device())->reclaim_idle_memory();
            }
        }
#endif
    }
    void reserve_workspace(std::size_t bytes) {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            reserve_moe_workspace(model.execution->moe_expert_cache,bytes);
            auto context=mfq::cuda::default_context(mfq_current_cuda_device());
            auto available=context->available_memory();
            if(available>=bytes)return;
            std::size_t hot=0;
            for(const auto& metric:cache.metrics())
                if(metric.first=="prefix_cache_hot_bytes")hot=std::size_t(metric.second);
            const auto shortage=bytes-available;
            const auto released=cache.trim_hot(hot>shortage?hot-shortage:0);
            context->reclaim_idle_memory();
            std::cerr<<"runtime_workspace_budget requested_bytes="<<bytes
                <<" available_bytes="<<context->available_memory()
                <<" prefix_released_bytes="<<released<<'\n';
        }
#endif
    }
    bool prepare_snapshot() {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            // Retire prefill buffers before copying KV/recurrent state. Evict
            // old hot entries before allocating the incoming checkpoint.
            release_moe_prefill_buffers(model.execution->moe_expert_cache);
            // A prefix stores occupied rows, not the full preallocated context
            // or graph-owned speculative checkpoints.
            const auto incoming=qwen4_prefix_snapshot_bytes(model,mtp);
            std::size_t hot=0,limit=0;
            for(const auto& metric:cache.metrics()) {
                if(metric.first=="prefix_cache_hot_bytes")hot=std::size_t(metric.second);
                if(metric.first=="prefix_cache_max_bytes")limit=std::size_t(metric.second);
            }
            const auto target=limit>incoming?limit-incoming:0;
            if(hot>target)cache.trim_hot(target);
            // Transposing graph KV can temporarily hold a contiguous copy in
            // addition to the checkpoint. Include that peak and metadata.
            const auto needed=incoming*2+(std::size_t(64)<<20);
            reserve_workspace(needed);
            const auto available=mfq::cuda::default_context(mfq_current_cuda_device())->available_memory();
            if(available<needed) {
                std::cerr<<"runtime_session_cache action=skip reason=workspace_pressure"
                    <<" requested_bytes="<<needed<<" available_bytes="<<available<<'\n';
                return false;
            }
        }
#endif
        return true;
    }
    void invalidate_plan() { graph.invalidate(); }
    bool has_hidden(const Hidden &value) const { return value.defined(); }
    void reset() {
        if (batching) return;
        model.reset(1);
        if (mtp)
            mtp->reset(1);
    }
    void release() {
        if (batching) return; // Each row owns its slot cleanup.
        trace_expert_cache("end");
        try { reset(); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }
    Generation plain(mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output,
                     size_t reused, size_t stable) {
        if (batching) return batching->generate(request_id, input, output);
        if(input.prompt.size()-reused>1024)
            reserve_workspace(mfq::cuda::runtime_options::workspace_reserve_bytes());
        auto logical_chunk=config.generation.prefill_chunk_size;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            // Bound residual/FFN storage as well as projection tiles. Long
            // prompts advance each bounded chunk through all layers.
            logical_chunk=std::min<int64_t>(logical_chunk,8192);
            if(reused==0 && (stable==0 || stable==input.prompt.size()) &&
                (!prepared || !prepared->transformed()) &&
                qwen4_layer_major_prefill_eligible(model,input.prompt.size()))
                logical_chunk=input.prompt.size();
        }
        return mfq::engine::generate_sequence(
            CudaGenerationOps<Model>(model, graph, input, prepared ? &*prepared : nullptr,
                config.decode_graph,config.generation.prefill_chunk_size,&output),
            output, input.prompt.size(), reused, stable,logical_chunk);
    }
    Generation speculate(mfq::engine::InferenceRequest &input, mfq::engine::InferenceOutput &output,
        size_t reused, Hidden restored, Hidden *committed) {
        if(input.prompt.size()-reused>1024)
            reserve_workspace(mfq::cuda::runtime_options::workspace_reserve_bytes());
        return run_mtp_generation(model,
            *mtp,
            input,
            output,
            config.generation.prefill_chunk_size,
            prepared ? &*prepared : nullptr,
            reused,
            std::move(restored),
            committed);
    }
};

template <typename Model>
Generation generate(Model &model, DecodeGraphCache &graph, TextSessionCache &cache,
    const CudaRuntimeConfig &config, mfq::engine::InferenceRequest &request,
    mfq::engine::InferenceOutput &output, MtpModule *mtp,
    std::optional<CudaPreparedPrompt> prepared,
    mfq::engine::ContinuousBatch<CudaBatchOperations<Model>>* batching, std::string request_id) {
    return mfq::engine::generate_request(
        CudaGenerationContext<Model>{model, graph, cache, config, mtp, std::move(prepared), batching,
                                     std::move(request_id)},
        request,
        output);
}

#define MFQ_INSTANTIATE_FLOW(MODEL)                                                                \
    template Generation generate(MODEL &,                                                          \
        DecodeGraphCache &,                                                                        \
        TextSessionCache &,                                                                        \
        const CudaRuntimeConfig &,                                                                 \
        mfq::engine::InferenceRequest &,                                                           \
        mfq::engine::InferenceOutput &,                                                            \
        MtpModule *,                                                                               \
        std::optional<CudaPreparedPrompt>,                                                         \
        mfq::engine::ContinuousBatch<CudaBatchOperations<MODEL>>*, std::string);
MFQ_INSTANTIATE_FLOW(mfq::cuda::Qwen35CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::MiniCPMO45CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::MiniCPMOTtsCausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::Gemma4CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::GlmDsaCausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::Glm5CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::Qwen4CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::DeepseekV4CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::DeepseekV41CausalLm)

#undef MFQ_INSTANTIATE_FLOW
} // namespace mfq::cuda::internal
