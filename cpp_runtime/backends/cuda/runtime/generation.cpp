#include "generation.h"

#include "inference.h"
#include "causal_lm.h"
#include "cuda_sampling.h"
#include "text_session_cache.h"
#include "mtp.h"
#include "../models/qwen35/qwen35_linear_attention.h"
#include "mfq_cuda_ops.h"

#include <algorithm>
#include <cstdlib>
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
    int64_t chunk_size) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "runtime prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "runtime prefill IDs must have shape [1, tokens]");
    int64_t offset = 0;
    while (ids.size(1) - offset > chunk_size) {
        (void)model.hidden_forward(
            ids.narrow(1, offset, chunk_size).contiguous());
        offset += chunk_size;
    }
    return offset == 0
        ? ids
        : ids.narrow(1, offset, ids.size(1) - offset).contiguous();
}

template <typename Model>
static mfq_tensor_backend::Tensor hidden_forward_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor & ids,
    int64_t chunk_size,
    mfq_tensor_backend::Tensor * raw_hidden = nullptr) {
    MFQ_RUNTIME_CHECK(
        chunk_size > 0,
        "runtime prefill chunk size must be positive");
    MFQ_RUNTIME_CHECK(
        ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0,
        "runtime prefill IDs must have shape [1, tokens]");
    std::vector<mfq_tensor_backend::Tensor> raw_chunks;
    if (raw_hidden != nullptr) {
        raw_chunks.reserve(static_cast<std::size_t>(
            (ids.size(1) + chunk_size - 1) / chunk_size));
    }
    mfq_tensor_backend::Tensor hidden;
    for (int64_t offset = 0; offset < ids.size(1); offset += chunk_size) {
        const int64_t count = std::min(chunk_size, ids.size(1) - offset);
        mfq_tensor_backend::Tensor raw_chunk;
        hidden = model.hidden_forward(
            ids.narrow(1, offset, count).contiguous(),
            mfq_nullopt,
            mfq_nullopt,
            nullptr,
            mfq_nullopt,
            raw_hidden != nullptr ? &raw_chunk : nullptr);
        if (raw_hidden != nullptr) {
            raw_chunks.push_back(std::move(raw_chunk));
        }
    }
    if (raw_hidden != nullptr) {
        *raw_hidden = raw_chunks.size() == 1
            ? std::move(raw_chunks.front())
            : mfq_tensor_backend::cat(raw_chunks, 1).contiguous();
    }
    return hidden;
}

template <typename Model>
static mfq_tensor_backend::Tensor hidden_forward_prepared_chunked(
    Model& model,
    const mfq_tensor_backend::Tensor& ids,
    const CudaPreparedPrompt& prepared,
    int64_t chunk_size,
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
    for (int64_t offset = 0; offset < ids.size(1); offset += chunk_size) {
        const int64_t count = std::min(chunk_size, ids.size(1) - offset);
        mfq_tensor_backend::Tensor raw_chunk;
        hidden = model.hidden_forward_inputs(
            ids.narrow(1, offset, count).contiguous(),
            prepared.embeddings.narrow(
                1, prepared_offset + offset, count).contiguous(),
            prepared.positions.narrow(
                -1, prepared_offset + offset, count).contiguous(),
            mfq_nullopt, nullptr, mfq_nullopt, true, mfq_nullopt,
            raw_hidden != nullptr ? &raw_chunk : nullptr);
        if (raw_hidden != nullptr) raw_chunks.push_back(std::move(raw_chunk));
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

PrefillCudaTimer::PrefillCudaTimer()
    : stream_(mfq_get_current_cuda_stream()) {
    MFQ_CUDA_CHECK(cudaEventCreate(&started_));
    try {
        MFQ_CUDA_CHECK(cudaEventCreate(&finished_));
        MFQ_CUDA_CHECK(cudaEventRecord(started_, stream_));
    } catch (...) {
        if (finished_ != nullptr) cudaEventDestroy(finished_);
        cudaEventDestroy(started_);
        finished_ = nullptr;
        started_ = nullptr;
        throw;
    }
}

PrefillCudaTimer::~PrefillCudaTimer() {
    if (finished_ != nullptr) cudaEventDestroy(finished_);
    if (started_ != nullptr) cudaEventDestroy(started_);
}

cudaEvent_t PrefillCudaTimer::finished_event() const {
    return finished_;
}

double PrefillCudaTimer::elapsed_ms() const {
    MFQ_CUDA_CHECK(cudaEventSynchronize(finished_));
    float elapsed = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, started_, finished_));
    return static_cast<double>(elapsed);
}

template <typename Model>
int32_t generate_tokens(
    Model& model,
    std::mutex & model_mutex,
    DecodeGraphCache & graph_cache,
    TextSessionCache & session_cache,
    const std::vector<int64_t> & prompt,
    const MfqSamplingParams & sampling,
    const MfqTokenCallback & on_token,
    const MfqPrefillCallback & on_prefill,
    const MfqPromptCachePlan & cache_plan,
    const MfqTokenConstraintPtr & token_constraint,
    MtpModule* mtp,
    int64_t prefill_chunk_size,
    PreparedPromptFactory<Model> prepare_prompt)
{
    std::lock_guard<std::mutex> lock(model_mutex);
    std::optional<CudaPreparedPrompt> prepared;
    double multimodal_ms = 0.0;
    if (prepare_prompt) {
        PrefillCudaTimer multimodal_timer;
        prepared = prepare_prompt(model);
        MFQ_CUDA_CHECK(cudaEventRecord(
            multimodal_timer.finished_event(),
            mfq_get_current_cuda_stream()));
        multimodal_ms = multimodal_timer.elapsed_ms();
    }
    if (prepared && prepared->token_ids != prompt) {
        throw std::invalid_argument(
            "prepared prompt token IDs disagree with the rendered prompt");
    }
    const bool transformed_prompt = prepared && prepared->transformed();
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    if (mtp != nullptr) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    const char* mtp_reprefill = std::getenv("MFQ_RUNTIME_REPREFILL");
    const char* mtp_trace = std::getenv("MFQ_RUNTIME_TRACE_INCREMENTAL");
    const bool use_mtp =
        mtp != nullptr && sampling.enable_mtp && sampling.max_tokens > 1 &&
        mfq_token_constraint_supports_speculation(token_constraint) &&
        !(mtp_reprefill != nullptr && mtp_reprefill[0] == '1') &&
        !(mtp_trace != nullptr && mtp_trace[0] == '1');
    const size_t stable_prefix_tokens = std::min(
        cache_plan.stable_prefix_tokens, prompt.size());
    const bool cache_enabled =
        stable_prefix_tokens > 0 &&
        (!transformed_prompt || !input_key.empty()) &&
        (!cache_plan.session_id.empty() ||
         session_cache.persistent_prefix_enabled()) &&
        model.supports_text_session_state();
    const bool can_restore_mtp = use_mtp && mtp->supports_session_state();
    const auto restored = cache_enabled && (!use_mtp || can_restore_mtp)
        ? session_cache.restore_best(
            model, use_mtp ? mtp : nullptr,
            cache_plan.session_id, prompt, stable_prefix_tokens, input_key)
        : TextSessionRestore{};
    const size_t reused_tokens = restored.tokens;
    if (reused_tokens == 0) {
        model.reset(1);
        if (use_mtp) mtp->reset(1);
    }
    if (use_mtp) {
        if constexpr (
                Model::backbone == mfq::cuda::CudaBackbone::generic_qwen ||
                Model::backbone == mfq::cuda::CudaBackbone::qwen4_exp ||
                Model::backbone == mfq::cuda::CudaBackbone::glm5_next ||
                Model::backbone == mfq::cuda::CudaBackbone::deepseek_v41) {
            std::vector<int64_t> history = prompt;
            const auto tracking_callback = [&](int32_t token) {
                const bool keep_going = !on_token || on_token(token);
                if (keep_going) history.push_back(token);
                return keep_going;
            };
            mfq_tensor_backend::Tensor last_target_hidden;
            const int32_t generated = run_mtp_generation<Model::backbone>(
                model, *mtp, prompt, sampling, tracking_callback, on_prefill,
                prefill_chunk_size, token_constraint,
                transformed_prompt ? &*prepared : nullptr,
                reused_tokens, restored.mtp_last_target_hidden,
                &last_target_hidden, multimodal_ms);
            if (cache_enabled && last_target_hidden.defined() &&
                    model.cache_pos > 1 &&
                    model.cache_pos <= static_cast<int64_t>(history.size())) {
                try {
                    history.resize(static_cast<size_t>(model.cache_pos));
                    auto state = model.capture_text_session_state(history);
                    state.input_key = input_key;
                    state.mtp = mtp->capture_session_state(
                        model.cache_pos, last_target_hidden);
                    state.bytes += state.mtp->bytes;
                    session_cache.store(
                        cache_plan.session_id, std::move(state));
                } catch (const std::exception& error) {
                    std::cerr
                        << "runtime_session_cache action=skip session="
                        << cache_plan.session_id
                        << " error=" << error.what() << std::endl;
                }
            }
            return generated;
        }
        throw std::runtime_error(
            "MTP is unavailable for this causal LM type");
    }
    auto options = mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA);
    auto full_ids = mfq_tensor_backend::tensor(prompt, options)
        .reshape({1, -1}).contiguous();
    auto ids = full_ids.narrow(
        1, static_cast<int64_t>(reused_tokens),
        static_cast<int64_t>(prompt.size() - reused_tokens)).contiguous();
    graph_cache.ensure_storage(model.vocab_size());
    auto random_host = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCPU).pinned_memory(true));
    auto random_cuda = mfq_tensor_backend::empty(
        {1}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32).device(mfq_tensor_backend::kCUDA));
    mfq::cuda::Sampler sampler(
        sampling,
        mfq::cuda::SamplingOps(random_host, std::move(random_cuda)));
    const bool has_penalties = sampler.has_penalties();
    auto counts = has_penalties ? graph_cache.counts : mfq_tensor_backend::Tensor();
    if (has_penalties) {
        counts.zero_();
        sample_token_counts_add_cuda(counts, full_ids);
    }
    const auto store_session_snapshot = [&](
            const std::vector<int64_t>& snapshot_tokens) {
        if (!cache_enabled || model.cache_pos !=
                static_cast<int64_t>(snapshot_tokens.size())) {
            return;
        }
        try {
            auto state = model.capture_text_session_state(snapshot_tokens);
            state.input_key = input_key;
            session_cache.store(
                cache_plan.session_id, std::move(state));
        } catch (const std::exception & error) {
            std::cerr << "runtime_session_cache action=skip session="
                      << cache_plan.session_id
                      << " error=" << error.what() << std::endl;
        }
    };
    auto sample_first_token = [&]() {
        PrefillCudaTimer prefill_timer;
        mfq_tensor_backend::Tensor next;
        if (transformed_prompt) {
            size_t begin = reused_tokens;
            mfq_tensor_backend::Tensor hidden;
            if (cache_enabled && stable_prefix_tokens < prompt.size()) {
                if (begin < stable_prefix_tokens) {
                    auto stable_ids = full_ids.narrow(
                        1, static_cast<int64_t>(begin),
                        static_cast<int64_t>(stable_prefix_tokens - begin))
                        .contiguous();
                    hidden = hidden_forward_prepared_chunked(
                        model, stable_ids, *prepared, prefill_chunk_size,
                        nullptr, static_cast<int64_t>(begin));
                }
                store_session_snapshot(std::vector<int64_t>(
                    prompt.begin(), prompt.begin() +
                        static_cast<std::ptrdiff_t>(stable_prefix_tokens)));
                begin = stable_prefix_tokens;
            }
            if (begin < prompt.size()) {
                auto remaining_ids = full_ids.narrow(
                    1, static_cast<int64_t>(begin),
                    static_cast<int64_t>(prompt.size() - begin)).contiguous();
                hidden = hidden_forward_prepared_chunked(
                    model, remaining_ids, *prepared, prefill_chunk_size,
                    nullptr, static_cast<int64_t>(begin));
            }
            MFQ_RUNTIME_CHECK(
                hidden.defined(),
                "prepared CUDA prefill produced no hidden state");
            auto logits = model.lm_head.forward(
                hidden.index({Slice(), -1, Slice()})
                    .to(mfq_tensor_backend::kFloat16).contiguous())
                .contiguous().view({1, -1});
            MFQ_CUDA_CHECK(cudaEventRecord(
                prefill_timer.finished_event(),
                mfq_get_current_cuda_stream()));
            next = mfq::cuda::sample_logits(
                sampler, std::move(logits), counts, token_constraint);
        } else {
            if (cache_enabled && stable_prefix_tokens < prompt.size()) {
                if (reused_tokens < stable_prefix_tokens) {
                    auto stable_suffix = full_ids.narrow(
                        1, static_cast<int64_t>(reused_tokens),
                        static_cast<int64_t>(
                            stable_prefix_tokens - reused_tokens)).contiguous();
                    stable_suffix = prefill_tail(
                        model, std::move(stable_suffix), prefill_chunk_size);
                    MfqOptional<mfq_tensor_backend::Tensor> stable_seq_len = mfq_nullopt;
                    if (!Model::is_minicpmo45 && model.cache_pos > 0 &&
                            stable_suffix.size(1) == 1) {
                        stable_seq_len = mfq_tensor_backend::full(
                            {1}, model.cache_pos + 1, options);
                    }
                    (void)model.hidden_forward(
                        stable_suffix, mfq_nullopt, stable_seq_len);
                }
                store_session_snapshot(std::vector<int64_t>(
                    prompt.begin(), prompt.begin() +
                        static_cast<std::ptrdiff_t>(stable_prefix_tokens)));
                ids = full_ids.narrow(
                    1, static_cast<int64_t>(stable_prefix_tokens),
                    static_cast<int64_t>(
                        prompt.size() - stable_prefix_tokens)).contiguous();
            }
            ids = prefill_tail(
                model, std::move(ids), prefill_chunk_size);
            next = sample_token(
                model, ids, sampler, counts, token_constraint,
                prefill_timer.finished_event());
        }
        const int64_t token = next.template item<int64_t>();
        const double prefill_ms = prefill_timer.elapsed_ms();
        if (stable_prefix_tokens == prompt.size()) {
            store_session_snapshot(prompt);
        }
        if (on_prefill) {
            on_prefill(MfqPrefillTiming{
                prompt.size() - reused_tokens,
                prefill_ms,
                multimodal_ms,
                prefill_ms + multimodal_ms});
        }
        return std::make_pair(std::move(next), token);
    };
    const char * reprefill_env = std::getenv("MFQ_RUNTIME_REPREFILL");
    const bool reprefill = !transformed_prompt &&
        reprefill_env != nullptr && reprefill_env[0] == '1';
    std::vector<int64_t> history = prompt;
    const auto store_live_history = [&]() {
        if (model.cache_pos <= 0 ||
                model.cache_pos > static_cast<int64_t>(history.size())) {
            return;
        }
        store_session_snapshot(std::vector<int64_t>(
            history.begin(), history.begin() + model.cache_pos));
    };
    const char * trace_incremental_env =
        std::getenv("MFQ_RUNTIME_TRACE_INCREMENTAL");
    const bool trace_incremental =
        trace_incremental_env != nullptr && trace_incremental_env[0] == '1';
    if (!transformed_prompt && trace_incremental && sampling.max_tokens > 0) {
        auto [first, first_token] = sample_first_token();
        if (!on_token(first_token)) return 1;

        std::vector<mfq_tensor_backend::Tensor> incremental_trace;
        std::vector<mfq_tensor_backend::Tensor> full_trace;
        std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> incremental_gemma_trace;
        std::vector<std::pair<std::string, mfq_tensor_backend::Tensor>> full_gemma_trace;
        const int64_t decode_len = model.cache_pos + 1;
        auto seq_len = mfq_tensor_backend::tensor({decode_len}, options);
        g_gemma_trace_layer = 0;
        g_gemma_stage_trace = &incremental_gemma_trace;
        auto incremental_hidden = model.hidden_forward(
            first.reshape({1, 1}), mfq_nullopt, seq_len, &incremental_trace);
        g_gemma_stage_trace = nullptr;

        history.push_back(first_token);
        model.reset(1);
        auto full_ids = mfq_tensor_backend::tensor(history, options).reshape({1, -1}).contiguous();
        g_gemma_stage_trace = &full_gemma_trace;
        auto full_hidden = model.hidden_forward(
            full_ids, mfq_nullopt, mfq_nullopt, &full_trace);
        g_gemma_stage_trace = nullptr;
        g_gemma_trace_layer = -1;
        mfq_cuda_synchronize();

        if (incremental_trace.size() != full_trace.size()) {
            throw std::runtime_error("incremental trace stage count mismatch");
        }
        for (size_t i = 0; i < incremental_trace.size(); ++i) {
            auto got = incremental_trace[i].reshape({-1}).to(mfq_tensor_backend::kFloat64);
            auto ref = full_trace[i].index({Slice(), -1, Slice()}).reshape({-1}).to(mfq_tensor_backend::kFloat64);
            const double denominator = std::max(ref.norm().template item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
            const double cosine = mfq_tensor_backend::dot(got, ref).template item<double>() /
                std::max(got.norm().template item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_trace stage="
                      << (i == 0 ? "embedding" : "block_" + std::to_string(i - 1))
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        if (incremental_gemma_trace.size() != full_gemma_trace.size()) {
            throw std::runtime_error("incremental Gemma stage count mismatch");
        }
        for (size_t i = 0; i < incremental_gemma_trace.size(); ++i) {
            const auto & got_tensor = incremental_gemma_trace[i].second;
            const auto & full_tensor = full_gemma_trace[i].second;
            if (incremental_gemma_trace[i].first != full_gemma_trace[i].first ||
                full_tensor.numel() < got_tensor.numel()) {
                throw std::runtime_error("incremental Gemma stage layout mismatch");
            }
            auto got = got_tensor.reshape({-1}).to(mfq_tensor_backend::kFloat64);
            auto full_flat = full_tensor.reshape({-1});
            auto ref = full_flat.narrow(
                0, full_flat.numel() - got_tensor.numel(), got_tensor.numel()).to(mfq_tensor_backend::kFloat64);
            const double denominator = std::max(ref.norm().template item<double>(), 1.0e-30);
            const double relative_l2 = (got - ref).norm().template item<double>() / denominator;
            const double cosine = mfq_tensor_backend::dot(got, ref).template item<double>() /
                std::max(got.norm().template item<double>() * denominator, 1.0e-30);
            std::cerr << "incremental_gemma_trace stage="
                      << incremental_gemma_trace[i].first
                      << " relative_l2=" << relative_l2
                      << " cosine=" << cosine << std::endl;
        }
        auto incremental_logits = model.lm_head.forward(
            incremental_hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous());
        auto full_logits = model.lm_head.forward(
            full_hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous());
        const double logits_relative_l2 =
            (incremental_logits.to(mfq_tensor_backend::kFloat64) - full_logits.to(mfq_tensor_backend::kFloat64)).norm().template item<double>() /
            std::max(full_logits.to(mfq_tensor_backend::kFloat64).norm().template item<double>(), 1.0e-30);
        std::cerr << "incremental_trace logits_relative_l2=" << logits_relative_l2
                  << " incremental_top=" << incremental_logits.argmax(-1).template item<int64_t>()
                  << " full_top=" << full_logits.argmax(-1).template item<int64_t>() << std::endl;
        return 1;
    }

    const char * graph_env = std::getenv("MFQ_RUNTIME_CUDA_GRAPH");
    const bool graph_enabled =
        (graph_env == nullptr || graph_env[0] != '0') &&
        !Model::is_flash_next &&
        mfq_cuda_graph_capture_supported() &&
        g_dsv4_cpu_offload_layers.empty() &&
        g_dense_cpu_layer_count == 0 &&
        !g_moe_expert_cache &&
        model_parallel_cuda_graph_enabled();
    const char * graph_min_env =
        std::getenv("MFQ_RUNTIME_CUDA_GRAPH_MIN_TOKENS");
    const int32_t graph_min_tokens = graph_min_env != nullptr
        ? std::max<int32_t>(2, std::atoi(graph_min_env))
        : 16;
    // Grammar state advances on the CPU and may require a one-off full-logit
    // mask, so constrained requests cannot be replayed as a fixed CUDA graph.
    // Unconstrained decode keeps the existing graph fast path unchanged.
    const bool graph_eligible = !transformed_prompt && graph_enabled && !reprefill &&
        !token_constraint &&
        sampling.max_tokens >= graph_min_tokens &&
        sampling.max_tokens <= graph_cache.generated_capacity;
    if (graph_eligible) {
        const bool greedy = sampler.greedy();
        auto [first, first_token] = sample_first_token();
        int32_t generated = 1;
        if (!on_token(first_token)) {
            store_live_history();
            return generated;
        }
        history.push_back(first_token);
        if (generated >= sampling.max_tokens) {
            store_live_history();
            return generated;
        }

        graph_cache.ensure_compute_streams();
        MfqCudaGuard graph_device_guard(
            graph_cache.stream.device_index());
        auto graph_stream_guards =
            activate_cuda_graph_compute_streams(
                graph_cache.compute_streams);
        cudaStream_t graph_raw_stream = graph_cache.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(graph_cache.counts, first.contiguous());
        }

        int64_t pos_h = model.cache_pos;
        int64_t len_h = pos_h + 1;
        int64_t step_h = 1;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_input.template data_ptr<int64_t>(), first.template data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.generated.template data_ptr<int64_t>(), first.template data_ptr<int64_t>(),
            sizeof(int64_t), cudaMemcpyDeviceToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_pos.template data_ptr<int64_t>(), &pos_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_len.template data_ptr<int64_t>(), &len_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.static_step.template data_ptr<int64_t>(), &step_h,
            sizeof(int64_t), cudaMemcpyHostToDevice, graph_raw_stream));
        *random_host.template data_ptr<float>() = 0.5f;
        MFQ_CUDA_CHECK(cudaMemcpyAsync(
            graph_cache.random.template data_ptr<float>(), random_host.template data_ptr<float>(),
            sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_raw_stream));

        const int64_t requested_len = model.cache_pos + sampling.max_tokens;
        const int64_t planned_len = decode_graph_bucket(
            requested_len, model.max_position_embeddings());
        const int64_t attention_parts = decode_graph_attention_parts(
            planned_len, FullBlock::kDecodeAttentionMaxParts);
        auto sample_static = [&]() {
            if (greedy && !has_penalties) {
                return model.next_token_static(
                    graph_cache.static_input, graph_cache.static_pos,
                    graph_cache.static_len, planned_len, attention_parts);
            }
            auto logits = model.last_logits_static(
                    graph_cache.static_input, graph_cache.static_pos,
                    graph_cache.static_len, planned_len, attention_parts)
                .contiguous().view({1, -1});
            if (has_penalties) {
                logits = sampler.apply_penalties(
                    std::move(logits), graph_cache.counts);
            }
            if (greedy) {
                return sampler.ops().sample_greedy(std::move(logits));
            }
            return sampler.ops().sample_stochastic(
                std::move(logits), graph_cache.random, sampler.params());
        };

        const bool cache_hit = graph_cache.ensure_captured(
            model, planned_len, sampling, greedy,
            [&]() { return sample_static(); },
            [&](const mfq_tensor_backend::Tensor& next) {
                if (has_penalties) {
                    sample_token_counts_add_cuda(
                        graph_cache.counts, next.contiguous());
                }
                decode_graph_commit_cuda(
                    next, graph_cache.generated, graph_cache.static_step,
                    graph_cache.static_input, graph_cache.static_pos,
                    graph_cache.static_len);
            });
        if (!cache_hit) report_cuda_memory("runtime_graph_capture");
        if (trace_cuda_graph()) {
            std::cerr << "runtime_cuda_graph action=" << (cache_hit ? "reuse" : "capture")
                      << " requested_len=" << requested_len
                      << " planned_len=" << planned_len
                      << " captures=" << graph_cache.captures
                      << " reuses=" << graph_cache.reuses << std::endl;
        }

        while (generated < sampling.max_tokens) {
            if (!greedy) {
                *random_host.template data_ptr<float>() = sampler.next_uniform_float();
                MFQ_CUDA_CHECK(cudaMemcpyAsync(
                    graph_cache.random.template data_ptr<float>(), random_host.template data_ptr<float>(),
                    sizeof(float), cudaMemcpyHostToDevice, graph_raw_stream));
            }
            graph_cache.graph->replay();
            const int64_t token =
                graph_cache.static_next.template item<int64_t>();
            ++generated;
            if (!on_token(token)) break;
            history.push_back(token);
        }
        model.cache_pos += generated - 1;
        store_live_history();
        return generated;
    }

    int32_t generated = 0;
    while (generated < sampling.max_tokens) {
        if (reprefill && generated > 0) {
            model.reset(1);
            ids = mfq_tensor_backend::tensor(history, options).reshape({1, -1}).contiguous();
        }
        mfq_tensor_backend::Tensor next;
        int64_t token = 0;
        if (generated == 0) {
            auto first = sample_first_token();
            next = std::move(first.first);
            token = first.second;
        } else {
            next = sample_token(
                model, ids, sampler, counts, token_constraint);
            token = next.template item<int64_t>();
        }
        ++generated;
        if (!on_token(token)) break;
        history.push_back(token);
        if (has_penalties) sample_token_counts_add_cuda(counts, next.contiguous());
        ids = next.reshape({1, 1});
    }
    store_live_history();
    return generated;
}



template <typename Model>
int32_t generate(
    Model& model,
    std::mutex& model_mutex,
    DecodeGraphCache& graph_cache,
    TextSessionCache& session_cache,
    const std::vector<int64_t>& prompt,
    const MfqSamplingParams& sampling,
    const MfqTokenCallback& on_token,
    const MfqPrefillCallback& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    MtpModule* mtp,
    int64_t prefill_chunk_size,
    PreparedPromptFactory<Model> prepare_prompt) {
    std::lock_guard<std::mutex> lock(model_mutex);
    if (prompt.empty() || prefill_chunk_size <= 0) {
        throw std::invalid_argument("CUDA generate needs a prompt and positive prefill chunk size");
    }
    std::optional<CudaPreparedPrompt> prepared;
    double multimodal_ms = 0.0;
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
    const bool transformed = prepared && prepared->transformed();
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    if (mtp != nullptr) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    const bool use_mtp = mtp != nullptr && sampling.enable_mtp &&
        sampling.max_tokens > 1 &&
        mfq_token_constraint_supports_speculation(token_constraint);
    if (use_mtp) {
        if constexpr (
                Model::backbone == mfq::cuda::CudaBackbone::generic_qwen ||
                Model::backbone == mfq::cuda::CudaBackbone::qwen4_exp ||
                Model::backbone == mfq::cuda::CudaBackbone::glm5_next ||
                Model::backbone == mfq::cuda::CudaBackbone::deepseek_v41) {
        struct MtpOps {
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
            bool persistent_prefix_enabled() const { return cache.persistent_prefix_enabled(); }
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
                    const MfqPrefillCallback& on_prefill, std::int32_t) {
                return run_mtp_generation<Model::backbone>(
                    model, mtp, prompt, sampling, emit, on_prefill,
                    chunk_size, constraint, prepared, reused,
                    restored.mtp_last_target_hidden, &last_target_hidden,
                    multimodal_ms);
            }
        } ops{model, *mtp, session_cache, prompt, sampling, cache_plan,
              token_constraint, prefill_chunk_size,
              prepared ? &*prepared : nullptr, input_key, multimodal_ms, {}, {}};
        return mfq::engine::generate(
            ops, prompt, sampling, on_token, on_prefill, cache_plan);
        }
        throw std::runtime_error("MTP is unavailable for this causal LM type");
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
        sampling, mfq::cuda::SamplingOps(
            std::move(random_host), std::move(random_cuda)));
    const bool has_penalties = sampler.has_penalties();
    auto counts = has_penalties ? graph_cache.counts : mfq_tensor_backend::Tensor{};
    if (has_penalties) {
        counts.zero_();
        sample_token_counts_add_cuda(counts, full_ids);
    }
    mfq_tensor_backend::Tensor pending;

    struct Ops {
        Model& model;
        TextSessionCache& cache;
        const std::vector<int64_t>& prompt;
        const MfqPromptCachePlan& plan;
        const MfqTokenConstraintPtr& constraint;
        int64_t chunk_size;
        mfq_tensor_backend::Tensor& full_ids;
        mfq_tensor_backend::Tensor& pending;
        mfq_tensor_backend::Tensor& counts;
        mfq::cuda::Sampler& sampler;
        bool has_penalties;
        mfq_tensor_backend::TensorOptions options;
        const CudaPreparedPrompt* prepared;
        const std::string& input_key;
        double multimodal_ms;

        bool supports_cache() const {
            return model.supports_text_session_state() &&
                (!prepared || !prepared->transformed() || !input_key.empty());
        }
        bool persistent_prefix_enabled() const { return cache.persistent_prefix_enabled(); }
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
                            model, prefix, *prepared, chunk_size, nullptr,
                            static_cast<int64_t>(reused));
                    } else {
                        prefix = prefill_tail(model, std::move(prefix), chunk_size);
                        MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt;
                        if (model.cache_pos > 0 && prefix.size(1) == 1) {
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
                    model, ids, *prepared, chunk_size, nullptr,
                    static_cast<int64_t>(offset));
                auto logits = model.lm_head.forward(
                    hidden.index({Slice(), -1, Slice()})
                        .to(mfq_tensor_backend::kFloat16).contiguous())
                    .contiguous().view({1, -1});
                MFQ_CUDA_CHECK(cudaEventRecord(
                    timer.finished_event(), mfq_get_current_cuda_stream()));
                pending = mfq::cuda::sample_logits(
                    sampler, std::move(logits), counts, constraint);
            } else {
                ids = prefill_tail(model, std::move(ids), chunk_size);
                pending = sample_token(model, ids, sampler, counts, constraint,
                                       timer.finished_event());
            }
            const auto token = pending.template item<int64_t>();
            const double prefill_ms = timer.elapsed_ms();
            return {token, {prompt.size() - reused, prefill_ms, multimodal_ms,
                            prefill_ms + multimodal_ms}};
        }
        std::int64_t advance() {
            pending = sample_token(
                model, pending.reshape({1, 1}), sampler, counts, constraint);
            return pending.template item<int64_t>();
        }
        void accept(std::int64_t) {
            if (has_penalties) {
                sample_token_counts_add_cuda(counts, pending.contiguous());
            }
        }
        std::int32_t generate(
                std::size_t reused, std::size_t stable,
                const std::function<void(std::size_t)>& checkpoint,
                const MfqTokenCallback& emit,
                const MfqPrefillCallback& on_prefill,
                std::int32_t max_tokens) {
            const auto first = prefill(reused, stable, checkpoint);
            if (stable == prompt.size()) checkpoint(stable);
            if (on_prefill) on_prefill(first.timing);
            std::int32_t generated = 0;
            std::int64_t token = first.token;
            while (generated < max_tokens) {
                ++generated;
                if (!emit(token)) break;
                accept(token);
                if (generated == max_tokens) break;
                token = advance();
            }
            return generated;
        }
    } ops{model, session_cache, prompt, cache_plan, token_constraint,
          prefill_chunk_size, full_ids, pending, counts, sampler,
          has_penalties, options, prepared ? &*prepared : nullptr,
          input_key, multimodal_ms};
    return mfq::engine::generate(
        ops, prompt, sampling, on_token, on_prefill, cache_plan);
}

#define MFQ_INSTANTIATE_FLOW(BACKBONE)                                    \
    template int32_t generate(                                              \
        mfq::cuda::CausalLmFor<BACKBONE>&, std::mutex&, DecodeGraphCache&,  \
        TextSessionCache&, const std::vector<int64_t>&,                     \
        const MfqSamplingParams&, const MfqTokenCallback&,                  \
        const MfqPrefillCallback&, const MfqPromptCachePlan&,               \
        const MfqTokenConstraintPtr&, MtpModule*, int64_t,                  \
        PreparedPromptFactory<mfq::cuda::CausalLmFor<BACKBONE>>);

MFQ_INSTANTIATE_FLOW(mfq::cuda::CudaBackbone::generic_qwen)
MFQ_INSTANTIATE_FLOW(mfq::cuda::CudaBackbone::gemma4)
MFQ_INSTANTIATE_FLOW(mfq::cuda::CudaBackbone::glm_dsa)
MFQ_INSTANTIATE_FLOW(mfq::cuda::CudaBackbone::glm5_next)
MFQ_INSTANTIATE_FLOW(mfq::cuda::CudaBackbone::qwen4_exp)
MFQ_INSTANTIATE_FLOW(mfq::cuda::CudaBackbone::deepseek_v4)
MFQ_INSTANTIATE_FLOW(mfq::cuda::CudaBackbone::deepseek_v41)

#undef MFQ_INSTANTIATE_FLOW

#define MFQ_INSTANTIATE_GENERATION(BACKBONE)                              \
    template int32_t generate_tokens(                                    \
        mfq::cuda::CausalLmFor<BACKBONE>&, std::mutex&,                  \
        DecodeGraphCache&, TextSessionCache&,                            \
        const std::vector<int64_t>&, const MfqSamplingParams&,           \
        const MfqTokenCallback&, const MfqPrefillCallback&,              \
        const MfqPromptCachePlan&, const MfqTokenConstraintPtr&,         \
        MtpModule*, int64_t,                                             \
        PreparedPromptFactory<mfq::cuda::CausalLmFor<BACKBONE>>);

MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::generic_qwen)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::minicpmo45)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::minicpmo_tts)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::gemma4)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::glm_dsa)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::glm5_next)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::qwen4_exp)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::deepseek_v4)
MFQ_INSTANTIATE_GENERATION(mfq::cuda::CudaBackbone::deepseek_v41)

#undef MFQ_INSTANTIATE_GENERATION

} // namespace mfq::cuda::internal
