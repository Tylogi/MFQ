#include "generation.h"
#include "models/causal_models.h"
#include "models/full_block.h"
#include "cuda_sampling.h"
#include "text_session_cache.h"
#include "mtp.h"
#include "mfq_cuda_ops.h"

#include <algorithm>
#include <iostream>

namespace mfq::cuda::internal {
namespace {
template <typename Model>
static mfq_tensor_backend::Tensor sample_token(
    Model& model,
    mfq_tensor_backend::Tensor ids,
    mfq::cuda::Sampler& sampler,
    mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr & token_constraint)
{
    if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
        return model.next_token(ids);
    }

    auto logits = model.last_logits(ids).contiguous().view({1, -1});
    return mfq::cuda::sample_logits(
        sampler, std::move(logits), counts, token_constraint);
}


template <typename Model>
struct CudaGenerationOps {
    Model& model;
    DecodeGraphCache& graph;
    const MfqTokenConstraintPtr& constraint;
    mfq_tensor_backend::Tensor& pending;
    mfq_tensor_backend::Tensor& counts;
    mfq_tensor_backend::Tensor& random_host;
    mfq::cuda::Sampler& sampler;
    bool has_penalties;
    const CudaPreparedPrompt* prepared;
    const CudaDecodeGraphConfig& graph_config;
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;

    bool graph_eligible() const {
        const auto& execution = *model.execution;
        return graph_config.enabled &&
            (!prepared || !prepared->transformed()) && !constraint &&
            !model.metadata.flash_next && mfq_cuda_graph_capture_supported() &&
            execution.dsv4_cpu_offload_layers.empty() &&
            execution.dense_cpu_layer_count == 0 &&
            !execution.moe_expert_cache &&
            model_parallel_cuda_graph_enabled(execution) &&
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
        if (!hit) report_cuda_memory(execution.config, "runtime_graph_capture");
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
};

}

template <typename Model>
Generation generate(
        Model& model, DecodeGraphCache& graph_cache, TextSessionCache& session_cache,
        const CudaRuntimeConfig& config, mfq::engine::InferenceRequest& request,
        mfq::engine::InferenceOutput& output, MtpModule* mtp,
        std::optional<CudaPreparedPrompt> prepared) {
    using namespace mfq::engine;
    using Tensor = mfq_tensor_backend::Tensor;
    const auto& prompt = request.prompt;
    auto& sampling = request.sampling;
    const auto& cache_plan = request.cache_plan;
    const auto& token_constraint = request.token_constraint;
    if (config.generation.prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    if (prepared && prepared->token_ids != prompt)
        throw std::invalid_argument("prepared prompt token IDs disagree");
    const auto occupied = static_cast<int64_t>(prompt.size()) +
        (prepared && prepared->transformed() ? prepared->decode_position_delta : 0);
    const auto plan = plan_generation(prompt, model.vocab_size(),
        model.max_position_embeddings(), sampling.max_tokens,
        cache_plan.stable_prefix_tokens, std::max<int64_t>(prompt.size(), occupied));
    sampling.max_tokens = plan.generation_tokens;
    if (output.stopped()) co_return;
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    const bool caching = model.supports_text_session_state() &&
        (!prepared || !prepared->transformed() || !input_key.empty()) &&
        plan.stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() || session_cache.persistent_prefix_enabled());
    const bool use_mtp = mtp && sampling.enable_mtp && sampling.max_tokens > 1;
    if (mtp) { mtp->last_stats = {}; mtp->last_stats.available = true; }
    TextSessionRestore restored;
    if (caching && (!use_mtp || mtp->supports_session_state()))
        restored = session_cache.restore_best(model, use_mtp ? mtp : nullptr,
            cache_plan.session_id, prompt, plan.stable_prefix_tokens, input_key);
    if (restored.tokens || use_mtp) graph_cache.invalidate();
    if (!restored.tokens) { model.reset(1); if (mtp) mtp->reset(1); }
    struct ResetOnFailure {
        Model& model; MtpModule* mtp; bool success = false;
        ~ResetOnFailure() {
            if (!success) try { model.reset(1); if (mtp) mtp->reset(1); } catch (...) {}
        }
    } cleanup{model, mtp};
    std::vector<int64_t> history = prompt;
    Tensor last_target_hidden;
    std::size_t last_snapshot = 0;
    const auto snapshot = [&] {
        const auto position = model.cache_pos;
        if (!caching || output.result.cancelled || position <= 0 ||
                static_cast<std::size_t>(position) > history.size() ||
                static_cast<std::size_t>(position) == last_snapshot) return;
        try {
            std::vector<int64_t> tokens(history.begin(), history.begin() + position);
            auto state = model.capture_text_session_state(tokens);
            state.input_key = input_key;
            if (use_mtp) {
                if (!last_target_hidden.defined() || position <= 1) return;
                state.mtp = mtp->capture_session_state(position, last_target_hidden);
                state.bytes += state.mtp->bytes;
            }
            session_cache.store(cache_plan.session_id, std::move(state));
            last_snapshot = static_cast<std::size_t>(position);
        } catch (const std::exception& error) {
            std::cerr << "runtime_session_cache action=skip error=" << error.what() << '\n';
        }
    };
    if (use_mtp) {
        auto generation = run_mtp_generation(model, *mtp, request, output,
            config.generation.prefill_chunk_size, prepared ? &*prepared : nullptr,
            restored.tokens, restored.mtp_last_target_hidden, &last_target_hidden);
        while (auto event = generation.next()) {
            if (auto* delta = std::get_if<OutputDelta>(&*event))
                history.insert(history.end(), delta->token_ids.begin(), delta->token_ids.end());
            co_yield std::move(*event);
        }
    } else {
        auto options = mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
        auto full_ids = mfq_tensor_backend::tensor(prompt, options).reshape({1, -1}).contiguous();
        graph_cache.ensure_storage(model.vocab_size());
        auto random_host = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU).pinned_memory(true));
        auto random_cuda = mfq_tensor_backend::empty({1}, mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCUDA));
        mfq::cuda::Sampler sampler(sampling, mfq::cuda::SamplingOps(random_host, std::move(random_cuda)));
        const bool has_penalties = sampler.has_penalties();
        auto counts = has_penalties ? graph_cache.counts : Tensor{};
        if (has_penalties) { counts.zero_(); sample_token_counts_add_cuda(counts, full_ids); }
        Tensor pending;
        CudaGenerationOps<Model> ops{
            model, graph_cache, token_constraint, pending, counts, random_host,
            sampler, has_penalties, prepared ? &*prepared : nullptr, config.decode_graph};
        ops.generation_limit = sampling.max_tokens;
        ops.graph_active = ops.graph_eligible();
        double elapsed = 0.0;
        auto offset = static_cast<int64_t>(restored.tokens);
        Tensor hidden;
        while (offset < full_ids.size(1) && !output.stopped()) {
            auto end = full_ids.size(1);
            if (caching && offset < static_cast<int64_t>(plan.stable_prefix_tokens))
                end = std::min(end, static_cast<int64_t>(plan.stable_prefix_tokens));
            const auto chunk = next_prefill_chunk(end, offset, config.generation.prefill_chunk_size);
            PrefillCudaTimer timer;
            auto ids = full_ids.narrow(1, chunk.offset, chunk.count).contiguous();
            if (prepared && prepared->transformed()) {
                hidden = model.hidden_forward_inputs(ids,
                    prepared->embeddings.narrow(1, chunk.offset, chunk.count).contiguous(),
                    prepared->positions.narrow(-1, chunk.offset, chunk.count).contiguous(),
                    mfq_nullopt, nullptr, mfq_nullopt, true);
                model.decode_position_delta = prepared->decode_position_delta;
            } else {
                MfqOptional<Tensor> length = mfq_nullopt;
                if (model.adapter_uses_decode_sequence_length() && offset > 0 && chunk.count == 1)
                    length = mfq_tensor_backend::full({1}, offset + 1, options);
                hidden = model.hidden_forward(ids, mfq_nullopt, length);
            }
            offset += chunk.count;
            if (offset == full_ids.size(1)) {
                auto logits = model.logits_from_hidden(hidden.index({Slice(), -1, Slice()})
                    .to(mfq_tensor_backend::kFloat16).contiguous()).reshape({1, -1});
                MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
                pending = mfq::cuda::sample_logits(sampler, std::move(logits), counts, token_constraint);
            } else MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
            elapsed += timer.elapsed_ms();
            if (caching && offset == static_cast<int64_t>(plan.stable_prefix_tokens)) snapshot();
            co_yield PrefillProgress{{static_cast<std::size_t>(offset) - restored.tokens, elapsed, 0.0, elapsed}};
        }
        if (!output.stopped()) {
            auto token = pending.template item<int64_t>();
            while (!output.stopped()) {
                auto delta = output.append(std::vector<int64_t>{token});
                history.insert(history.end(), delta.token_ids.begin(), delta.token_ids.end());
                ops.accept(token);
                co_yield std::move(delta);
                if (output.stopped()) break;
                token = ops.advance();
            }
        }
    }
    snapshot();
    model.reset(1);
    if (mtp) mtp->reset(1);
    cleanup.success = true;
}

#define MFQ_INSTANTIATE_FLOW(MODEL) \
    template Generation generate(MODEL&, DecodeGraphCache&, TextSessionCache&, \
        const CudaRuntimeConfig&, mfq::engine::InferenceRequest&, \
        mfq::engine::InferenceOutput&, MtpModule*, std::optional<CudaPreparedPrompt>);
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
