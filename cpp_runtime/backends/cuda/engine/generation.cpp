#include "generation.h"

#include "inference.h"
#include "models/causal_models.h"
#include "cuda_sampling.h"
#include "text_session_cache.h"
#include "mtp.h"
#include "../models/qwen35/linear_attention.h"
#include "mfq_cuda_ops.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::internal {
namespace {
template <typename Model>
static mfq_tensor_backend::Tensor sample_token(
    Model& model,
    mfq_tensor_backend::Tensor ids,
    mfq::cuda::Sampler& sampler,
    mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr & token_constraint,
    cudaEvent_t prefill_finished = nullptr)
{
    if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
        auto next = model.next_token(ids);
        if (prefill_finished != nullptr) {
            MFQ_CUDA_CHECK(cudaEventRecord(
                prefill_finished, mfq_get_current_cuda_stream()));
        }
        return next;
    }

    auto logits = model.last_logits(ids).contiguous().view({1, -1});
    if (prefill_finished != nullptr) {
        MFQ_CUDA_CHECK(cudaEventRecord(
            prefill_finished, mfq_get_current_cuda_stream()));
    }
    return mfq::cuda::sample_logits(
        sampler, std::move(logits), counts, token_constraint);
}

template <typename Model>
static mfq_tensor_backend::Tensor prefill_tail(
    Model& model,
    mfq_tensor_backend::Tensor ids,
    int64_t chunk_size,
    const MfqCancellationCheck& cancelled) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "runtime prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "runtime prefill IDs must have shape [1, tokens]");
    int64_t offset = 0;
    for (;;) {
        const auto chunk = mfq::engine::next_prefill_chunk(
            ids.size(1), offset, chunk_size);
        if (cancelled && cancelled()) {
            throw mfq::engine::InferenceCancelled{};
        }
        if (chunk.last) {
            return offset == 0
                ? ids
                : ids.narrow(1, offset, chunk.count).contiguous();
        }
        (void)model.hidden_forward(
            ids.narrow(1, chunk.offset, chunk.count).contiguous());
        offset += chunk.count;
    }
}

template <typename Model>
static mfq_tensor_backend::Tensor hidden_forward_prepared_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor& ids,
    const CudaPreparedPrompt& prepared,
    int64_t chunk_size,
    const MfqCancellationCheck& cancelled,
    mfq_tensor_backend::Tensor* raw_hidden = nullptr,
    int64_t prepared_offset = 0) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0 && prepared.transformed() && prepared_offset >= 0 &&
            ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0 &&
            prepared.embeddings.defined() && prepared.positions.defined() &&
            prepared.embeddings.dim() == 3 &&
            prepared.embeddings.size(0) == 1 &&
            prepared_offset + ids.size(1) <= prepared.embeddings.size(1) &&
            prepared.embeddings.size(2) == model.hidden_size() &&
            (prepared.positions.dim() == 1 ||
             prepared.positions.dim() == 2) &&
            prepared_offset + ids.size(1) <= prepared.positions.size(-1),
        "prepared CUDA prefill tensors disagree with prompt geometry");
    std::vector<mfq_tensor_backend::Tensor> raw_chunks;
    if (raw_hidden != nullptr) {
        raw_chunks.reserve(static_cast<std::size_t>(
            (ids.size(1) + chunk_size - 1) / chunk_size));
    }
    mfq_tensor_backend::Tensor hidden;
    for (int64_t offset = 0; offset < ids.size(1);) {
        if (cancelled && cancelled()) {
            throw mfq::engine::InferenceCancelled{};
        }
        const auto chunk = mfq::engine::next_prefill_chunk(
            ids.size(1), offset, chunk_size);
        mfq_tensor_backend::Tensor raw_chunk;
        hidden = model.hidden_forward_inputs(
            ids.narrow(1, chunk.offset, chunk.count).contiguous(),
            prepared.embeddings.narrow(
                1, prepared_offset + chunk.offset, chunk.count).contiguous(),
            prepared.positions.narrow(
                -1, prepared_offset + chunk.offset, chunk.count).contiguous(),
            mfq_nullopt, nullptr, mfq_nullopt, true, mfq_nullopt,
            raw_hidden != nullptr ? &raw_chunk : nullptr);
        if (raw_hidden != nullptr) raw_chunks.push_back(std::move(raw_chunk));
        if (cancelled && cancelled()) {
            throw mfq::engine::InferenceCancelled{};
        }
        offset += chunk.count;
    }
    model.decode_position_delta = prepared.decode_position_delta;
    if (raw_hidden != nullptr) {
        *raw_hidden = raw_chunks.size() == 1
            ? std::move(raw_chunks.front())
            : mfq_tensor_backend::cat(raw_chunks, 1).contiguous();
    }
    return hidden;
}

} // namespace

namespace {

template <typename Model>
struct CudaGenerationOps {
    Model& model;
    TextSessionCache& cache;
    DecodeGraphCache& graph;
    const std::vector<int64_t>& prompt;
    const MfqPromptCachePlan& plan;
    const MfqTokenConstraintPtr& constraint;
    int64_t chunk_size;
    mfq_tensor_backend::Tensor& full_ids;
    mfq_tensor_backend::Tensor& pending;
    mfq_tensor_backend::Tensor& counts;
    mfq_tensor_backend::Tensor& random_host;
    mfq::cuda::Sampler& sampler;
    bool has_penalties;
    mfq_tensor_backend::TensorOptions options;
    const CudaPreparedPrompt* prepared;
    const std::string& input_key;
    const MfqCancellationCheck& cancelled;
    const CudaDecodeGraphConfig& graph_config;
    double multimodal_ms;
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;

    bool supports_cache() const {
        return model.supports_text_session_state() &&
            (!prepared || !prepared->transformed() || !input_key.empty());
    }
    bool persistent_prefix_enabled() const {
        return cache.persistent_prefix_enabled();
    }
    std::size_t prompt_size() const { return prompt.size(); }
    std::size_t restore(std::size_t stable) {
        return cache.restore_best(
            model, nullptr, plan.session_id, prompt, stable, input_key).tokens;
    }
    void reset() { model.reset(1); }
    std::int64_t cache_position() const { return model.cache_pos; }
    void snapshot(std::vector<int64_t> tokens) {
        try {
            auto state = model.capture_text_session_state(tokens);
            state.input_key = input_key;
            cache.store(plan.session_id, std::move(state));
        } catch (const std::exception& error) {
            std::cerr << "runtime_session_cache action=skip session="
                      << plan.session_id << " error=" << error.what()
                      << std::endl;
        }
    }
    mfq::engine::PrefillResult prefill(
            std::size_t reused, std::size_t stable,
            const std::function<void(std::size_t)>& checkpoint) {
        PrefillCudaTimer timer;
        auto ids = full_ids.narrow(
            1, static_cast<int64_t>(reused),
            static_cast<int64_t>(prompt.size() - reused)).contiguous();
        if (stable > 0 && stable < prompt.size()) {
            if (reused < stable) {
                auto prefix = full_ids.narrow(
                    1, static_cast<int64_t>(reused),
                    static_cast<int64_t>(stable - reused)).contiguous();
                if (prepared && prepared->transformed()) {
                    (void)hidden_forward_prepared_chunked(
                        model, prefix, *prepared, chunk_size, cancelled,
                        nullptr, static_cast<int64_t>(reused));
                } else {
                    prefix = prefill_tail(
                        model, std::move(prefix), chunk_size, cancelled);
                    MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt;
                    if (model.adapter_uses_decode_sequence_length() &&
                            model.cache_pos > 0 &&
                            prefix.size(1) == 1) {
                        seq_len = mfq_tensor_backend::full(
                            {1}, model.cache_pos + 1, options);
                    }
                    (void)model.hidden_forward(prefix, mfq_nullopt, seq_len);
                }
            }
            checkpoint(stable);
            ids = full_ids.narrow(
                1, static_cast<int64_t>(stable),
                static_cast<int64_t>(prompt.size() - stable)).contiguous();
        }
        if (prepared && prepared->transformed()) {
            const auto offset = stable > 0 && stable < prompt.size()
                ? stable : reused;
            auto hidden = hidden_forward_prepared_chunked(
                model, ids, *prepared, chunk_size, cancelled, nullptr,
                static_cast<int64_t>(offset));
            auto logits = model.lm_head.forward(
                *model.execution,
                hidden.index({Slice(), -1, Slice()})
                    .to(mfq_tensor_backend::kFloat16).contiguous())
                .contiguous().view({1, -1});
            MFQ_CUDA_CHECK(cudaEventRecord(
                timer.finished_event(), mfq_get_current_cuda_stream()));
            pending = mfq::cuda::sample_logits(
                sampler, std::move(logits), counts, constraint);
        } else {
            ids = prefill_tail(
                model, std::move(ids), chunk_size, cancelled);
            pending = sample_token(model, ids, sampler, counts, constraint,
                                   timer.finished_event());
        }
        const auto token = pending.template item<int64_t>();
        const double prefill_ms = timer.elapsed_ms();
        return {token, {prompt.size() - reused, prefill_ms, multimodal_ms,
                        prefill_ms + multimodal_ms}};
    }
    bool graph_eligible() const {
        const auto& execution = *model.execution;
        return graph_config.enabled &&
            (!prepared || !prepared->transformed()) && !constraint &&
            !model.metadata.flash_next && mfq_cuda_graph_capture_supported() &&
            execution.dsv4_cpu_offload_layers.empty() &&
            execution.dense_cpu_layer_count == 0 &&
            !execution.moe_expert_cache &&
            model_parallel_cuda_graph_enabled(
                execution.tensor_parallel,
                execution.expert_parallel,
                execution.model_parallel_collectives) &&
            generation_limit >= graph_config.minimum_generation_tokens &&
            generation_limit <= graph.generated_capacity;
    }
    void prepare_graph() {
        const auto& execution = *model.execution;
        graph.ensure_compute_streams(
            execution.tensor_parallel.enabled()
                ? execution.tensor_parallel
                : execution.expert_parallel);
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards =
            activate_cuda_graph_compute_streams(graph.compute_streams);
        cudaStream_t graph_stream = graph.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }

        std::int64_t position = model.cache_pos;
        std::int64_t length = position + 1;
        std::int64_t step = 1;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph.static_input.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(), sizeof(int64_t),
            cudaMemcpyDeviceToDevice, graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph.generated.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(), sizeof(int64_t),
            cudaMemcpyDeviceToDevice, graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph.static_pos.template data_ptr<int64_t>(), &position,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph.static_len.template data_ptr<int64_t>(), &length,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph.static_step.template data_ptr<int64_t>(), &step,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_stream));
        *random_host.template data_ptr<float>() = 0.5f;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph.random.template data_ptr<float>(),
            random_host.template data_ptr<float>(), sizeof(float),
            cudaMemcpyHostToDevice, graph_stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream));

        const std::int64_t requested = model.cache_pos + generation_limit;
        const std::int64_t planned = decode_graph_bucket(
            requested, model.max_position_embeddings());
        const std::int64_t parts = decode_graph_attention_parts(
            planned, FullBlock::kDecodeAttentionMaxParts);
        const bool greedy = sampler.greedy();
        const auto sample_static = [&]() {
            if (greedy && !has_penalties) {
                return model.next_token_static(
                    graph.static_input, graph.static_pos,
                    graph.static_len, planned, parts);
            }
            auto logits = model.last_logits_static(
                    graph.static_input, graph.static_pos,
                    graph.static_len, planned, parts)
                .contiguous().view({1, -1});
            if (has_penalties) {
                logits = sampler.apply_penalties(std::move(logits), counts);
            }
            if (greedy) {
                return sampler.ops().sample_greedy(std::move(logits));
            }
            return sampler.ops().sample_stochastic(
                std::move(logits), graph.random, sampler.params());
        };
        const bool hit = graph.ensure_captured(
            model, planned, sampler.params(), greedy,
            [&]() { return sample_static(); },
            [&](const mfq_tensor_backend::Tensor& next) {
                if (has_penalties) {
                    sample_token_counts_add_cuda(counts, next.contiguous());
                }
                decode_graph_commit_cuda(
                    next, graph.generated, graph.static_step,
                    graph.static_input, graph.static_pos,
                    graph.static_len);
            });
        if (!hit) report_cuda_memory("runtime_graph_capture");
        if (graph_config.trace) {
            std::cerr << "runtime_cuda_graph action="
                      << (hit ? "reuse" : "capture")
                      << " requested_len=" << requested
                      << " planned_len=" << planned
                      << " captures=" << graph.captures
                      << " reuses=" << graph.reuses << std::endl;
        }
    }
    std::int64_t advance() {
        if (!graph_active) {
            pending = sample_token(
                model, pending.reshape({1, 1}), sampler, counts, constraint);
            return pending.template item<int64_t>();
        }

        if (!graph_prepared) {
            prepare_graph();
            graph_prepared = true;
        }
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards =
            activate_cuda_graph_compute_streams(graph.compute_streams);
        cudaStream_t graph_stream = graph.stream.stream();
        if (!sampler.greedy()) {
            *random_host.template data_ptr<float>() =
                sampler.next_uniform_float();
            MFQ_CUDA_CHECK(cudaMemcpyAsync(
                graph.random.template data_ptr<float>(),
                random_host.template data_ptr<float>(), sizeof(float),
                cudaMemcpyHostToDevice, graph_stream));
        }
        graph.graph->replay();
        const auto token = graph.static_next.template item<int64_t>();
        ++model.cache_pos;
        return token;
    }
    void accept(std::int64_t) {
        if (has_penalties && !graph_active) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }
    }
    std::int32_t generate(
            std::size_t reused, std::size_t stable,
            const std::function<void(std::size_t)>& checkpoint,
            const MfqTokenCallback& emit,
            const MfqPrefillCallback& on_prefill,
            std::int32_t max_tokens,
            const MfqCancellationCheck& cancellation) {
        generation_limit = max_tokens;
        graph_active = graph_eligible();
        return mfq::engine::generate_target(
            *this, reused, stable, checkpoint, emit,
            on_prefill, max_tokens, cancellation);
    }
};

template <typename Model>
struct CudaMtpGenerationOps {
    Model& model;
    MtpModule& mtp;
    TextSessionCache& cache;
    const std::vector<int64_t>& prompt;
    const MfqSamplingParams& sampling;
    const MfqPromptCachePlan& plan;
    const MfqTokenConstraintPtr& constraint;
    int64_t chunk_size;
    const CudaPreparedPrompt* prepared;
    const std::string& input_key;
    double multimodal_ms;
    TextSessionRestore restored;
    mfq_tensor_backend::Tensor last_target_hidden;

    bool supports_cache() const {
        return model.supports_text_session_state() &&
            (!prepared || !prepared->transformed() || !input_key.empty());
    }
    bool persistent_prefix_enabled() const {
        return cache.persistent_prefix_enabled();
    }
    std::size_t restore(std::size_t stable) {
        if (!mtp.supports_session_state()) return 0;
        restored = cache.restore_best(
            model, &mtp, plan.session_id, prompt, stable, input_key);
        return restored.tokens;
    }
    void reset() { model.reset(1); mtp.reset(1); }
    std::int64_t cache_position() const { return model.cache_pos; }
    void snapshot(std::vector<int64_t> tokens) {
        if (!last_target_hidden.defined() || model.cache_pos <= 1) return;
        try {
            auto state = model.capture_text_session_state(tokens);
            state.input_key = input_key;
            state.mtp = mtp.capture_session_state(
                model.cache_pos, last_target_hidden);
            state.bytes += state.mtp->bytes;
            cache.store(plan.session_id, std::move(state));
        } catch (const std::exception& error) {
            std::cerr << "runtime_session_cache action=skip session="
                      << plan.session_id << " error=" << error.what()
                      << std::endl;
        }
    }
    std::int32_t generate(
            std::size_t reused, std::size_t,
            const std::function<void(std::size_t)>&,
            const MfqTokenCallback& emit,
            const MfqPrefillCallback& on_prefill, std::int32_t,
            const MfqCancellationCheck& cancelled) {
        return run_mtp_generation(
            model, mtp, prompt, sampling, emit, on_prefill,
            chunk_size, constraint, prepared, reused,
            restored.mtp_last_target_hidden, &last_target_hidden,
            multimodal_ms, cancelled);
    }
};

} // namespace

template <typename Model>
int32_t generate(
    Model& model,
    std::mutex& model_mutex,
    DecodeGraphCache& graph_cache,
    TextSessionCache& session_cache,
    const CudaRuntimeConfig& config,
    const std::vector<int64_t>& prompt,
    const MfqSamplingParams& sampling,
    const MfqTokenCallback& on_token,
    const MfqPrefillCallback& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    MtpModule* mtp,
    PreparedPromptFactory<Model> prepare_prompt,
    MfqCancellationCheck cancelled) {
    std::lock_guard<std::mutex> lock(model_mutex);
    if (prompt.empty() || config.generation.prefill_chunk_size <= 0) {
        throw std::invalid_argument(
            "CUDA generate needs a prompt and positive prefill chunk size");
    }
    std::optional<CudaPreparedPrompt> prepared;
    double multimodal_ms = 0.0;
    if (cancelled && cancelled()) return 0;
    if (prepare_prompt) {
        PrefillCudaTimer timer;
        prepared = prepare_prompt(model);
        MFQ_CUDA_CHECK(cudaEventRecord(
            timer.finished_event(), mfq_get_current_cuda_stream()));
        multimodal_ms = timer.elapsed_ms();
    }
    if (prepared && prepared->token_ids != prompt) {
        throw std::invalid_argument(
            "prepared prompt token IDs disagree with the rendered prompt");
    }
    const auto logical_context = static_cast<std::int64_t>(prompt.size()) +
        (prepared && prepared->transformed()
            ? prepared->decode_position_delta : 0);
    if (logical_context < 0) {
        throw std::invalid_argument(
            "prepared prompt decode position is negative");
    }
    const auto plan = mfq::engine::plan_generation(
        prompt, model.vocab_size(), model.max_position_embeddings(),
        sampling.max_tokens, cache_plan.stable_prefix_tokens,
        std::max<std::int64_t>(
            static_cast<std::int64_t>(prompt.size()), logical_context));
    auto effective_sampling = sampling;
    effective_sampling.max_tokens = plan.generation_tokens;
    auto effective_cache_plan = cache_plan;
    effective_cache_plan.stable_prefix_tokens = plan.stable_prefix_tokens;
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    if (mtp != nullptr) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    const bool use_mtp = mtp != nullptr && effective_sampling.enable_mtp &&
        effective_sampling.max_tokens > 1 &&
        mfq_token_constraint_supports_speculation(token_constraint);
    if (use_mtp) {
        CudaMtpGenerationOps<Model> ops{
            model, *mtp, session_cache, prompt, effective_sampling,
            effective_cache_plan, token_constraint,
            config.generation.prefill_chunk_size,
            prepared ? &*prepared : nullptr, input_key, multimodal_ms, {}, {}};
        return mfq::engine::generate(
            ops, prompt, effective_sampling, on_token, on_prefill,
            effective_cache_plan, cancelled);
    }
    auto options = mfq_tensor_backend::TensorOptions()
        .dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto full_ids = mfq_tensor_backend::tensor(prompt, options)
        .reshape({1, -1}).contiguous();
    graph_cache.ensure_storage(model.vocab_size());
    auto random_host = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kFloat32)
            .device(mfq_tensor_backend::kCPU).pinned_memory(true));
    auto random_cuda = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kFloat32)
            .device(mfq_tensor_backend::kCUDA));
    mfq::cuda::Sampler sampler(
        effective_sampling,
        mfq::cuda::SamplingOps(random_host, std::move(random_cuda)));
    const bool has_penalties = sampler.has_penalties();
    auto counts = has_penalties ? graph_cache.counts : mfq_tensor_backend::Tensor{};
    if (has_penalties) {
        counts.zero_();
        sample_token_counts_add_cuda(counts, full_ids);
    }
    mfq_tensor_backend::Tensor pending;

    CudaGenerationOps<Model> ops{
        model, session_cache, graph_cache, prompt, effective_cache_plan,
        token_constraint, config.generation.prefill_chunk_size,
        full_ids, pending, counts, random_host, sampler, has_penalties,
        options, prepared ? &*prepared : nullptr, input_key, cancelled,
        config.decode_graph, multimodal_ms};
    return mfq::engine::generate(
        ops, prompt, effective_sampling, on_token, on_prefill,
        effective_cache_plan, cancelled);
}

#define MFQ_INSTANTIATE_FLOW(MODEL)                                       \
    template int32_t generate(                                             \
        MODEL&, std::mutex&, DecodeGraphCache&, TextSessionCache&,          \
        const CudaRuntimeConfig&, const std::vector<int64_t>&,              \
        const MfqSamplingParams&, const MfqTokenCallback&,                  \
        const MfqPrefillCallback&, const MfqPromptCachePlan&,               \
        const MfqTokenConstraintPtr&, MtpModule*,                           \
        PreparedPromptFactory<MODEL>, MfqCancellationCheck);

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
