#pragma once

#include "cuda_sampling.h"
#include "moe_expert_cache.h"
#include "qwen_paged_kv.h"
#include "../models/qwen35/qwen35_linear_attention.h"

// Included by cuda_runtime.cpp after the CUDA Qwen model and sampler are defined.
// The scheduler owns request concurrency; the adapter below owns the hybrid
// full-attention/recurrent state carried between decode iterations.

namespace mfq::cuda::continuous {

using Tensor = mfq_tensor_backend::Tensor;
using LinearBlock = mfq::cuda::qwen35::LinearAttentionBlock;

struct QwenBatchLayerState {
    enum class Kind { FullAttention, Recurrent };
    Kind kind = Kind::FullAttention;
    Tensor first;
    Tensor second;
    bool ring = false;
    bool paged = false;
};

struct QwenBatchState {
    int64_t batch = 0;
    std::vector<QwenBatchLayerState> layers;
};

static void clear_full_attention_decode_workspaces(FullBlock & block) {
    block.decode_partial_o = Tensor();
    block.decode_partial_m = Tensor();
    block.decode_partial_l = Tensor();
    block.decode_mma_mask = Tensor();
    block.decode_mma_kv_max = Tensor();
    block.decode_mma_meta = Tensor();
}

static bool qwen_continuous_batch_has_moe(
        const mfq::cuda::Qwen35CausalLm & model) {
    for (const auto & block : model.blocks) {
        if (const auto * full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->ffn.is_moe) return true;
        } else if (const auto * linear =
                dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.is_moe) return true;
        }
    }
    return false;
}

static bool qwen_continuous_batch_has_cached_moe(
        const mfq::cuda::Qwen35CausalLm & model) {
    for (const auto & block : model.blocks) {
        if (const auto * full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->ffn.uses_moe_expert_cache()) return true;
        } else if (const auto * linear =
                dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.uses_moe_expert_cache()) return true;
        }
    }
    return false;
}

class MoeContinuousBatchCacheScope {
public:
    explicit MoeContinuousBatchCacheScope(bool enabled)
        : previous_(g_moe_continuous_batch_cache_serial) {
        g_moe_continuous_batch_cache_serial = enabled;
    }

    ~MoeContinuousBatchCacheScope() {
        g_moe_continuous_batch_cache_serial = previous_;
    }

    MoeContinuousBatchCacheScope(
        const MoeContinuousBatchCacheScope &) = delete;
    MoeContinuousBatchCacheScope & operator=(
        const MoeContinuousBatchCacheScope &) = delete;

private:
    bool previous_ = false;
};

static std::string qwen_continuous_batching_incompatibility(
        const mfq::cuda::Qwen35CausalLm & model) {
    if (model.blocks.empty()) {
        return "continuous batching requires at least one model block";
    }
    if (g_dense_cpu_layer_count != 0 || !g_dsv4_cpu_offload_layers.empty()) {
        return "continuous batching requires GPU-resident model blocks";
    }
    for (const auto & block : model.blocks) {
        if (block->cpu_offloaded) {
            return "continuous batching cannot use CPU-offloaded blocks";
        }
        if (const auto * full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->sliding) {
                return "continuous batching requires non-sliding Qwen attention";
            }
            if (full->ffn.uses_moe_expert_cache() &&
                    full->ffn.moe_top_k >
                        g_moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        if (const auto * linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.uses_moe_expert_cache() &&
                    linear->ffn.moe_top_k >
                        g_moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        return "continuous batching encountered an unsupported Qwen block";
    }
    return {};
}

static bool qwen_continuous_batch_greedy_enabled() {
    const char * environment = std::getenv(
        "MFQ_CONTINUOUS_BATCH_GREEDY");
    return environment == nullptr || std::atoi(environment) != 0;
}

static bool qwen_continuous_batch_packed_metadata_enabled() {
    const char * environment = std::getenv(
        "MFQ_CONTINUOUS_BATCH_PACKED_METADATA");
    return environment == nullptr || std::atoi(environment) != 0;
}

static bool qwen_continuous_batch_cuda_graph_enabled(const mfq::cuda::Qwen35CausalLm & model) {
    const char * environment = std::getenv(
        "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH");
    const char * runtime_environment =
        std::getenv("MFQ_RUNTIME_CUDA_GRAPH");
    return (environment == nullptr || std::atoi(environment) != 0) &&
        (runtime_environment == nullptr || runtime_environment[0] != '0') &&
        !qwen_continuous_batch_has_cached_moe(model) &&
        mfq_cuda_graph_capture_supported() &&
        model_parallel_cuda_graph_enabled();
}

static bool qwen_continuous_paged_kv_enabled() {
    const char * environment = std::getenv("MFQ_CONTINUOUS_PAGED_KV");
    return environment == nullptr || std::atoi(environment) != 0;
}

static QwenBatchState take_qwen_batch_state(
        mfq::cuda::Qwen35CausalLm & model, int64_t batch, QwenPagedKvArena * paged_kv) {
    MFQ_RUNTIME_CHECK(model.speculative_start < 0,
        "continuous batching cannot detach speculative state");
    QwenBatchState state;
    state.batch = batch;
    state.layers.reserve(model.blocks.size());
    for (auto & block : model.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        QwenBatchLayerState layer;
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            layer.kind = QwenBatchLayerState::Kind::FullAttention;
            layer.paged = paged_kv != nullptr;
            if (layer.paged) {
                MFQ_RUNTIME_CHECK(full->cache.is_paged() &&
                    full->cache.batch_size() == batch,
                    "continuous batching paged KV state is unavailable");
            } else {
                MFQ_RUNTIME_CHECK(
                    full->cache.k.defined() && full->cache.v.defined() &&
                    full->cache.k.dim() == 4 &&
                    full->cache.k.size(0) == batch &&
                    full->cache.v.sizes() == full->cache.k.sizes(),
                    "continuous batching full-attention state is unavailable");
                layer.first = full->cache.k;
                layer.second = full->cache.v;
                layer.ring = full->cache.ring;
            }
            full->cache = KVCache();
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(linear->conv_state.defined() &&
                linear->gdn_state.defined() &&
                linear->conv_state.size(0) == batch &&
                linear->gdn_state.size(0) == batch &&
                !linear->speculative_pending,
                "continuous batching recurrent state is unavailable");
            layer.kind = QwenBatchLayerState::Kind::Recurrent;
            layer.first = linear->conv_state;
            layer.second = linear->gdn_state;
            linear->conv_state = Tensor();
            linear->gdn_state = Tensor();
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
        } else {
            throw std::runtime_error(
                "continuous batching encountered an unsupported block state");
        }
        state.layers.push_back(std::move(layer));
    }
    model.cache_pos = 0;
    return state;
}

static Tensor merge_batch_tensors(
        const std::vector<QwenBatchState> & states, size_t layer,
        bool second) {
    std::vector<Tensor> values;
    values.reserve(states.size());
    for (const auto & state : states) {
        values.push_back(second
            ? state.layers[layer].second
            : state.layers[layer].first);
    }
    return values.size() == 1 ? values.front() :
        mfq_tensor_backend::cat(values, 0).contiguous();
}

static void restore_qwen_batch_states(
        mfq::cuda::Qwen35CausalLm & model, const std::vector<QwenBatchState> & states,
        int64_t cache_position, QwenPagedKvArena * paged_kv) {
    MFQ_RUNTIME_CHECK(!states.empty(),
        "continuous batching cannot restore an empty state list");
    int64_t batch = 0;
    for (const auto & state : states) {
        MFQ_RUNTIME_CHECK(state.batch > 0 &&
            state.layers.size() == model.blocks.size(),
            "continuous batching state layout changed");
        batch += state.batch;
    }
    for (size_t layer_index = 0;
            layer_index < model.blocks.size(); ++layer_index) {
        auto & block = model.blocks[layer_index];
        MfqCudaGuard guard(block->cuda_device);
        const auto kind = states.front().layers[layer_index].kind;
        for (const auto & state : states) {
            MFQ_RUNTIME_CHECK(state.layers[layer_index].kind == kind,
                "continuous batching mixed incompatible layer states");
        }
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(kind ==
                QwenBatchLayerState::Kind::FullAttention,
                "continuous batching full-attention state kind changed");
            if (paged_kv != nullptr) {
                for (const auto & state : states) {
                    MFQ_RUNTIME_CHECK(state.layers[layer_index].paged,
                        "continuous batching lost paged KV state");
                }
                full->cache = KVCache();
                clear_full_attention_decode_workspaces(*full);
                continue;
            }
            const auto shape = states.front().layers[layer_index].first.sizes();
            for (const auto & state : states) {
                const auto & saved = state.layers[layer_index];
                MFQ_RUNTIME_CHECK(!saved.paged && !saved.ring &&
                    saved.first.dim() == 4 &&
                    saved.second.sizes() == saved.first.sizes() &&
                    saved.first.size(1) == shape[1] &&
                    saved.first.size(2) == shape[2] &&
                    saved.first.size(3) == shape[3],
                    "continuous batching KV cache geometry changed");
            }
            full->cache.k = merge_batch_tensors(
                states, layer_index, false);
            full->cache.v = merge_batch_tensors(
                states, layer_index, true);
            full->cache.ring = false;
            MFQ_RUNTIME_CHECK(full->cache.k.size(0) == batch,
                "continuous batching KV merge produced the wrong batch");
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(kind == QwenBatchLayerState::Kind::Recurrent,
                "continuous batching recurrent state kind changed");
            linear->conv_state = merge_batch_tensors(
                states, layer_index, false);
            linear->gdn_state = merge_batch_tensors(
                states, layer_index, true);
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
            MFQ_RUNTIME_CHECK(linear->conv_state.size(0) == batch &&
                linear->gdn_state.size(0) == batch,
                "continuous batching recurrent merge produced the wrong batch");
        } else {
            throw std::runtime_error(
                "continuous batching restore encountered an unsupported block");
        }
    }
    model.cache_pos = cache_position;
    model.speculative_start = -1;
    model.speculative_confirmed = 0;
}

static void compact_qwen_batch_state(
        mfq::cuda::Qwen35CausalLm & model, const std::vector<int64_t> & rows,
        int64_t cache_position, QwenPagedKvArena * paged_kv) {
    MFQ_RUNTIME_CHECK(!rows.empty(),
        "continuous batching cannot compact to an empty batch");
    for (auto & block : model.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            if (paged_kv != nullptr) {
                full->cache = KVCache();
                clear_full_attention_decode_workspaces(*full);
                continue;
            }
            auto indices = mfq_tensor_backend::tensor(
                rows, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::Device(
                        mfq_tensor_backend::kCUDA, block->cuda_device)));
            full->cache.k = full->cache.k.index_select(
                0, indices).contiguous();
            full->cache.v = full->cache.v.index_select(
                0, indices).contiguous();
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            auto indices = mfq_tensor_backend::tensor(
                rows, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::Device(
                        mfq_tensor_backend::kCUDA, block->cuda_device)));
            linear->conv_state = linear->conv_state
                .index_select(0, indices).contiguous();
            linear->gdn_state = linear->gdn_state
                .index_select(0, indices).contiguous();
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
        }
    }
    model.cache_pos = cache_position;
}

static Tensor qwen_logits_from_last_hidden(mfq::cuda::Qwen35CausalLm & model, Tensor hidden) {
    auto last = hidden.index({Slice(), -1, Slice()})
        .to(mfq_tensor_backend::kFloat16).contiguous();
    return model.apply_final_logit_softcap(
        model.lm_head.forward(last));
}

static std::vector<const void *> qwen_decode_state_addresses(mfq::cuda::Qwen35CausalLm & model) {
    std::vector<const void *> addresses;
    addresses.reserve(2 * model.blocks.size());
    for (auto & block : model.blocks) {
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            if (full->cache.is_paged()) {
                addresses.push_back(full->cache.k_chunk_ptrs.data_ptr());
                addresses.push_back(full->cache.v_chunk_ptrs.data_ptr());
                addresses.push_back(full->cache.page_table.data_ptr());
            } else {
                addresses.push_back(full->cache.k.data_ptr());
                addresses.push_back(full->cache.v.data_ptr());
            }
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            addresses.push_back(linear->conv_state.data_ptr());
            addresses.push_back(linear->gdn_state.data_ptr());
        }
    }
    return addresses;
}

struct QwenContinuousDecodeGraph {
    decltype(mfq_get_stream_from_pool(false)) stream;
    std::vector<MfqCudaStream> compute_streams;
    std::unique_ptr<MfqCudaGraph> graph;
    Tensor static_next;
    std::vector<const void *> state_addresses;
    int64_t batch = 0;
    int64_t planned_len = 0;
    bool valid = false;

    QwenContinuousDecodeGraph()
        : stream(mfq_get_stream_from_pool(false)) {}

    void ensure_compute_streams() {
        if (compute_streams.empty()) {
            compute_streams = make_cuda_graph_compute_streams(stream);
        }
    }

    std::vector<MfqCudaStream> participant_streams() const {
        return cuda_graph_participant_streams(compute_streams);
    }

    bool matches(
            int64_t candidate_batch, int64_t candidate_len,
            const std::vector<const void *> & candidate_addresses) const {
        return valid && batch == candidate_batch &&
            planned_len == candidate_len &&
            state_addresses == candidate_addresses;
    }

    void set_key(
            int64_t candidate_batch, int64_t candidate_len,
            std::vector<const void *> candidate_addresses) {
        batch = candidate_batch;
        planned_len = candidate_len;
        state_addresses = std::move(candidate_addresses);
        valid = true;
    }

    void invalidate() {
        if (graph) graph->reset();
        graph.reset();
        static_next = Tensor();
        state_addresses.clear();
        valid = false;
    }
};

class CudaContinuousBatcher {
public:
    CudaContinuousBatcher(
            mfq::cuda::Qwen35CausalLm & model, std::mutex & model_mutex,
            int32_t max_sequences,
            int64_t prefill_chunk_size = 2048,
            std::chrono::microseconds initial_batch_wait =
                std::chrono::microseconds(1000))
        : model_(model), model_mutex_(model_mutex),
          max_sequences_(max_sequences),
          prefill_chunk_size_(prefill_chunk_size),
          moe_enabled_(qwen_continuous_batch_has_moe(model)),
          cached_moe_enabled_(
              qwen_continuous_batch_has_cached_moe(model)),
          initial_batch_wait_(initial_batch_wait) {
        if (max_sequences_ < 1) {
            throw std::invalid_argument(
                "continuous batching max sequences must be positive");
        }
        if (prefill_chunk_size_ < 1) {
            throw std::invalid_argument(
                "continuous batching prefill chunk size must be positive");
        }
        const auto incompatibility =
            qwen_continuous_batching_incompatibility(model_);
        if (!incompatibility.empty()) {
            throw std::runtime_error(incompatibility);
        }
        if (qwen_continuous_paged_kv_enabled()) {
            paged_kv_ = std::make_unique<QwenPagedKvArena>(
                model_, max_sequences_);
        }
        worker_ = std::thread([this] { worker_main(); });
    }

    ~CudaContinuousBatcher() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_ready_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    CudaContinuousBatcher(const CudaContinuousBatcher &) = delete;
    CudaContinuousBatcher & operator=(
        const CudaContinuousBatcher &) = delete;

    int32_t submit(
            const std::vector<int64_t> & prompt,
            const MfqSamplingParams & sampling,
            const MfqTokenCallback & on_token,
            const MfqPrefillCallback & on_prefill,
            const MfqPromptCachePlan & cache_plan,
            const MfqTokenConstraintPtr & token_constraint) {
        if (prompt.empty() ||
                prompt.size() > static_cast<size_t>(
                    model_.max_position_embeddings())) {
            throw std::invalid_argument(
                "continuous batching prompt length is invalid");
        }
        if (sampling.max_tokens < 0) {
            throw std::invalid_argument(
                "continuous batching max_tokens cannot be negative");
        }
        for (const auto token : prompt) {
            if (token < 0 || token >= model_.vocab_size()) {
                throw std::invalid_argument(
                    "continuous batching prompt token is outside the vocabulary");
            }
        }
        if (sampling.max_tokens == 0 ||
                prompt.size() == static_cast<size_t>(
                    model_.max_position_embeddings())) {
            return 0;
        }
        auto request = std::make_shared<Request>(
            prompt, sampling, token_constraint);
        request->generation_limit = static_cast<int32_t>(
            std::min<int64_t>(sampling.max_tokens,
                model_.max_position_embeddings() -
                    static_cast<int64_t>(prompt.size())));
        if (!cache_plan.session_id.empty() ||
                cache_plan.stable_prefix_tokens != 0) {
            ++prefix_cache_bypasses_;
        }
        if (sampling.enable_mtp) ++mtp_bypasses_;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stopping_) {
                throw std::runtime_error(
                    "continuous batching scheduler is stopping");
            }
            pending_.push_back(request);
            queued_.fetch_add(1, std::memory_order_relaxed);
        }
        queue_ready_.notify_one();
        int32_t delivered = 0;
        bool callbacks_enabled = true;
        std::exception_ptr callback_error;
        std::exception_ptr producer_error;
        for (;;) {
            std::optional<MfqPrefillTiming> prefill;
            std::optional<int64_t> token;
            bool producer_done = false;
            {
                std::unique_lock<std::mutex> lock(request->mutex);
                request->output_ready.wait(lock, [&] {
                    return request->prefill_timing.has_value() ||
                        !request->output_tokens.empty() || request->done;
                });
                if (request->prefill_timing.has_value()) {
                    prefill = std::move(request->prefill_timing);
                    request->prefill_timing.reset();
                } else if (!request->output_tokens.empty()) {
                    token = request->output_tokens.front();
                    request->output_tokens.pop_front();
                } else {
                    producer_done = request->done;
                    producer_error = request->error;
                }
            }
            if (producer_done) break;
            if (!callbacks_enabled) continue;
            try {
                if (prefill.has_value()) {
                    if (on_prefill) on_prefill(*prefill);
                } else if (token.has_value()) {
                    ++delivered;
                    if (on_token && !on_token(*token)) {
                        callbacks_enabled = false;
                        request->cancel_requested.store(
                            true, std::memory_order_release);
                        queue_ready_.notify_one();
                    }
                }
            } catch (...) {
                callback_error = std::current_exception();
                callbacks_enabled = false;
                request->cancel_requested.store(
                    true, std::memory_order_release);
                queue_ready_.notify_one();
            }
        }
        if (callback_error) std::rethrow_exception(callback_error);
        if (producer_error) std::rethrow_exception(producer_error);
        return delivered;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        return {
            {"continuous_batching_max_sequences",
                static_cast<double>(max_sequences_)},
            {"continuous_batching_active",
                static_cast<double>(active_count_.load())},
            {"continuous_batching_prefilling",
                static_cast<double>(prefilling_count_.load())},
            {"continuous_batching_queued",
                static_cast<double>(queued_.load())},
            {"continuous_batching_requests",
                static_cast<double>(requests_.load())},
            {"continuous_batching_decode_batches",
                static_cast<double>(decode_batches_.load())},
            {"continuous_batching_decode_tokens",
                static_cast<double>(decode_tokens_.load())},
            {"continuous_batching_max_batch",
                static_cast<double>(max_batch_seen_.load())},
            {"continuous_batching_admissions",
                static_cast<double>(admissions_.load())},
            {"continuous_batching_prefill_chunks",
                static_cast<double>(prefill_chunks_.load())},
            {"continuous_batching_prefill_yields",
                static_cast<double>(prefill_yields_.load())},
            {"continuous_batching_interleaved_admissions",
                static_cast<double>(interleaved_admissions_.load())},
            {"continuous_batching_compactions",
                static_cast<double>(compactions_.load())},
            {"continuous_batching_batched_greedy_batches",
                static_cast<double>(batched_greedy_batches_.load())},
            {"continuous_batching_packed_metadata_batches",
                static_cast<double>(packed_metadata_batches_.load())},
            {"continuous_batching_cuda_graph_captures",
                static_cast<double>(cuda_graph_captures_.load())},
            {"continuous_batching_cuda_graph_replays",
                static_cast<double>(cuda_graph_replays_.load())},
            {"continuous_batching_mtp_target_only_requests",
                static_cast<double>(mtp_bypasses_.load())},
            {"continuous_batching_moe",
                moe_enabled_ ? 1.0 : 0.0},
            {"continuous_batching_moe_cached_row_serial",
                cached_moe_enabled_ ? 1.0 : 0.0},
            {"continuous_batching_prefix_cache_bypasses",
                static_cast<double>(prefix_cache_bypasses_.load())},
            {"continuous_batching_paged_kv",
                paged_kv_ ? 1.0 : 0.0},
            {"paged_kv_page_size",
                paged_kv_ ? static_cast<double>(paged_kv_->page_size()) : 0.0},
            {"paged_kv_live_pages",
                paged_kv_ ? static_cast<double>(paged_kv_->live_pages()) : 0.0},
            {"paged_kv_peak_live_pages",
                paged_kv_ ? static_cast<double>(paged_kv_->peak_live_pages()) : 0.0},
            {"paged_kv_capacity_pages",
                paged_kv_ ? static_cast<double>(paged_kv_->capacity_pages()) : 0.0},
            {"paged_kv_reserved_bytes",
                paged_kv_ ? static_cast<double>(paged_kv_->reserved_bytes()) : 0.0},
            {"paged_kv_page_allocations",
                paged_kv_ ? static_cast<double>(paged_kv_->allocation_count()) : 0.0},
            {"paged_kv_page_reuses",
                paged_kv_ ? static_cast<double>(paged_kv_->reuse_count()) : 0.0},
            {"paged_kv_page_releases",
                paged_kv_ ? static_cast<double>(paged_kv_->release_count()) : 0.0},
            {"paged_kv_table_updates",
                paged_kv_ ? static_cast<double>(paged_kv_->page_table_updates()) : 0.0},
        };
    }

    int64_t queued_requests() const {
        return queued_.load(std::memory_order_relaxed);
    }

    bool paged_kv_enabled() const noexcept { return paged_kv_ != nullptr; }
    int64_t paged_kv_page_size() const noexcept {
        return paged_kv_ ? paged_kv_->page_size() : 0;
    }

private:
    struct Request {
        Request(
                const std::vector<int64_t> & input_prompt,
                const MfqSamplingParams & input_sampling,
                const MfqTokenConstraintPtr & input_constraint)
            : prompt(input_prompt), sampling(input_sampling),
              token_constraint(input_constraint) {}

        std::vector<int64_t> prompt;
        MfqSamplingParams sampling;
        MfqTokenConstraintPtr token_constraint;
        std::optional<mfq::cuda::Sampler> sampler;
        Tensor counts;
        Tensor prefill_ids;
        std::optional<QwenBatchState> prefill_state;
        std::vector<std::unique_ptr<PrefillCudaTimer>> prefill_timers;
        int64_t prefill_offset = 0;
        int32_t generation_limit = 0;
        int32_t produced = 0;
        int64_t pending_token = 0;
        int64_t cache_length = 0;
        QwenPagedKvSequence paged_kv;
        std::mutex mutex;
        std::condition_variable output_ready;
        std::optional<MfqPrefillTiming> prefill_timing;
        std::deque<int64_t> output_tokens;
        std::atomic<bool> cancel_requested{false};
        bool done = false;
        std::exception_ptr error;
    };

    static void complete_request(
            const std::shared_ptr<Request> & request,
            std::exception_ptr error = {}) {
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            if (request->done) return;
            request->error = error;
            request->done = true;
        }
        request->output_ready.notify_one();
    }

    static void publish_prefill(
            const std::shared_ptr<Request> & request,
            const MfqPrefillTiming & timing) {
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            request->prefill_timing = timing;
        }
        request->output_ready.notify_one();
    }

    static void publish_token(
            const std::shared_ptr<Request> & request, int64_t token) {
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            request->output_tokens.push_back(token);
        }
        request->output_ready.notify_one();
    }

    void bind_paged_requests(
            const std::vector<std::shared_ptr<Request>> & requests) {
        if (!paged_kv_) return;
        std::vector<const QwenPagedKvSequence *> sequences;
        sequences.reserve(requests.size());
        for (const auto & request : requests) {
            sequences.push_back(&request->paged_kv);
        }
        paged_kv_->bind(sequences);
    }

    void release_paged_requests(
            const std::vector<std::shared_ptr<Request>> & requests) {
        if (!paged_kv_) return;
        for (const auto & request : requests) {
            paged_kv_->release(request->paged_kv);
        }
    }

    void detach_paged_kv() {
        if (paged_kv_) paged_kv_->detach();
    }

    void fail_requests(
            const std::vector<std::shared_ptr<Request>> & requests,
            std::exception_ptr error) {
        for (const auto & request : requests) {
            complete_request(request, error);
        }
    }

    std::vector<std::shared_ptr<Request>> take_pending(
            size_t admission_limit) {
        std::vector<std::shared_ptr<Request>> requests;
        std::lock_guard<std::mutex> lock(queue_mutex_);
        const size_t occupied = active_.size() + prefilling_.size();
        const size_t available = static_cast<size_t>(max_sequences_) > occupied
            ? static_cast<size_t>(max_sequences_) - occupied : 0;
        const size_t count = std::min(
            {available, pending_.size(), admission_limit});
        requests.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            requests.push_back(std::move(pending_.front()));
            pending_.pop_front();
            queued_.fetch_sub(1, std::memory_order_relaxed);
        }
        return requests;
    }

    void initialize_sampling(Request & request, const Tensor & prompt_ids) {
        const int primary = g_layer_placement.primary_device();
        const auto cuda_options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA, primary));
        auto random_host = mfq_tensor_backend::empty(
            {1}, mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCPU)
                .dtype(mfq_tensor_backend::kFloat32)
                .pinned_memory(true));
        auto random_cuda = mfq_tensor_backend::empty(
            {1}, cuda_options.dtype(mfq_tensor_backend::kFloat32));
        request.sampler.emplace(
            request.sampling,
            mfq::cuda::SamplingOps(
                std::move(random_host), std::move(random_cuda)));
        if (request.sampler->has_penalties()) {
            request.counts = mfq_tensor_backend::zeros(
                {model_.vocab_size()},
                cuda_options.dtype(mfq_tensor_backend::kInt32));
            sample_token_counts_add_cuda(request.counts, prompt_ids);
        }
    }

    void advance_prefills(
            const std::vector<std::shared_ptr<Request>> & incoming,
            bool yield_after_chunk) {
        for (const auto & request : incoming) {
            prefilling_.push_back(request);
        }
        prefilling_count_.store(
            static_cast<int64_t>(prefilling_.size()),
            std::memory_order_relaxed);
        if (prefilling_.empty()) return;
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        invalidate_decode_graph();
        const int primary = g_layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        std::vector<QwenBatchState> states;
        states.reserve(1 + prefilling_.size());
        if (!active_.empty()) {
            states.push_back(take_qwen_batch_state(
                model_, static_cast<int64_t>(active_.size()),
                paged_kv_.get()));
        }
        std::vector<std::shared_ptr<Request>> admitted;
        admitted.reserve(prefilling_.size());
        int64_t chunks_advanced = 0;
        while (!prefilling_.empty() &&
                !stopping_.load(std::memory_order_acquire) &&
                (!yield_after_chunk || chunks_advanced == 0)) {
            auto request = std::move(prefilling_.front());
            prefilling_.pop_front();
            prefilling_count_.store(
                static_cast<int64_t>(prefilling_.size()),
                std::memory_order_relaxed);
            if (request->cancel_requested.load(std::memory_order_acquire)) {
                try { mfq_cuda_synchronize(); } catch (...) {}
                if (paged_kv_) {
                    paged_kv_->release(request->paged_kv);
                    detach_paged_kv();
                }
                request->prefill_state.reset();
                request->prefill_timers.clear();
                request->prefill_ids = Tensor();
                complete_request(request);
                continue;
            }
            try {
                if (request->prefill_state.has_value()) {
                    std::vector<QwenBatchState> resumed;
                    resumed.push_back(std::move(*request->prefill_state));
                    request->prefill_state.reset();
                    restore_qwen_batch_states(
                        model_, resumed, request->prefill_offset,
                        paged_kv_.get());
                    bind_paged_requests({request});
                } else {
                    model_.reset(1);
                    if (paged_kv_) {
                        paged_kv_->ensure_tokens(
                            request->paged_kv,
                            static_cast<int64_t>(request->prompt.size()));
                        bind_paged_requests({request});
                    }
                    request->prefill_ids = mfq_tensor_backend::tensor(
                        request->prompt,
                        mfq_tensor_backend::TensorOptions()
                            .dtype(mfq_tensor_backend::kInt64)
                            .device(mfq_tensor_backend::Device(
                                mfq_tensor_backend::kCUDA, primary)))
                        .reshape({1, -1}).contiguous();
                    initialize_sampling(*request, request->prefill_ids);
                }
                Tensor hidden;
                std::unique_ptr<PrefillCudaTimer> final_timer;
                do {
                    const int64_t count = std::min(
                        prefill_chunk_size_,
                        request->prefill_ids.size(1) -
                            request->prefill_offset);
                    auto chunk_timer =
                        std::make_unique<PrefillCudaTimer>();
                    hidden = model_.hidden_forward(
                        request->prefill_ids.narrow(
                            1, request->prefill_offset, count).contiguous());
                    request->prefill_offset += count;
                    if (request->prefill_offset <
                            request->prefill_ids.size(1)) {
                        MFQ_CUDA_CHECK(cudaEventRecord(
                            chunk_timer->finished_event(),
                            mfq_get_current_cuda_stream()));
                        request->prefill_timers.push_back(
                            std::move(chunk_timer));
                    } else {
                        final_timer = std::move(chunk_timer);
                    }
                    ++prefill_chunks_;
                    ++chunks_advanced;
                } while (!yield_after_chunk &&
                    request->prefill_offset < request->prefill_ids.size(1) &&
                    !request->cancel_requested.load(
                        std::memory_order_acquire) &&
                    !stopping_.load(std::memory_order_acquire));
                if (request->cancel_requested.load(
                        std::memory_order_acquire)) {
                    try { mfq_cuda_synchronize(); } catch (...) {}
                    if (paged_kv_) {
                        paged_kv_->release(request->paged_kv);
                        detach_paged_kv();
                    }
                    request->prefill_timers.clear();
                    request->prefill_ids = Tensor();
                    model_.reset(1);
                    complete_request(request);
                    continue;
                }
                if (stopping_.load(std::memory_order_acquire)) {
                    throw std::runtime_error(
                        "continuous batching scheduler stopped during prefill");
                }
                if (request->prefill_offset < request->prefill_ids.size(1)) {
                    request->prefill_state = take_qwen_batch_state(
                        model_, 1, paged_kv_.get());
                    detach_paged_kv();
                    prefilling_.push_back(std::move(request));
                    ++prefill_yields_;
                    prefilling_count_.store(
                        static_cast<int64_t>(prefilling_.size()),
                        std::memory_order_relaxed);
                    continue;
                }
                auto logits = qwen_logits_from_last_hidden(
                    model_, std::move(hidden));
                MFQ_RUNTIME_CHECK(final_timer != nullptr,
                    "continuous batching lost the prefill timer");
                MFQ_CUDA_CHECK(cudaEventRecord(
                    final_timer->finished_event(),
                    mfq_get_current_cuda_stream()));
                request->prefill_timers.push_back(std::move(final_timer));
                auto next = mfq::cuda::sample_logits(
                    *request->sampler, std::move(logits), request->counts,
                    request->token_constraint);
                const int64_t token = next.item<int64_t>();
                double prefill_ms = 0.0;
                for (const auto & timer : request->prefill_timers) {
                    prefill_ms += timer->elapsed_ms();
                }
                request->prefill_timers.clear();
                request->prefill_ids = Tensor();
                publish_prefill(request, MfqPrefillTiming{
                    request->prompt.size(), prefill_ms, 0.0, prefill_ms});
                request->pending_token = token;
                request->cache_length =
                    static_cast<int64_t>(request->prompt.size());
                request->produced = 1;
                publish_token(request, token);
                if (request->cancel_requested.load(
                            std::memory_order_acquire) ||
                        request->produced >= request->generation_limit) {
                    if (paged_kv_) {
                        paged_kv_->release(request->paged_kv);
                        detach_paged_kv();
                    }
                    complete_request(request);
                    continue;
                }
                if (request->counts.defined()) {
                    sample_token_counts_add_cuda(
                        request->counts, next.contiguous());
                }
                states.push_back(take_qwen_batch_state(
                    model_, 1, paged_kv_.get()));
                admitted.push_back(request);
                ++admissions_;
                ++requests_;
            } catch (...) {
                auto error = std::current_exception();
                try { mfq_cuda_synchronize(); } catch (...) {}
                if (paged_kv_) {
                    try { paged_kv_->release(request->paged_kv); } catch (...) {}
                    detach_paged_kv();
                }
                request->prefill_state.reset();
                request->prefill_timers.clear();
                request->prefill_ids = Tensor();
                try { model_.reset(1); } catch (...) {}
                complete_request(request, error);
            }
        }
        active_.insert(active_.end(), admitted.begin(), admitted.end());
        if (!states.empty()) {
            int64_t max_cache_position = 0;
            for (const auto & request : active_) {
                max_cache_position = std::max(
                    max_cache_position, request->cache_length);
            }
            restore_qwen_batch_states(
                model_, states, max_cache_position, paged_kv_.get());
            bind_paged_requests(active_);
        } else {
            model_.reset(1);
            detach_paged_kv();
        }
        active_count_.store(
            static_cast<int64_t>(active_.size()),
            std::memory_order_relaxed);
        prefilling_count_.store(
            static_cast<int64_t>(prefilling_.size()),
            std::memory_order_relaxed);
        int64_t previous = max_batch_seen_.load();
        while (previous < static_cast<int64_t>(active_.size()) &&
                !max_batch_seen_.compare_exchange_weak(
                    previous, static_cast<int64_t>(active_.size()))) {}
    }

    void retire_cancelled_requests() {
        std::vector<std::shared_ptr<Request>> survivors;
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<int64_t> survivor_rows;
        int64_t survivor_max_position = 0;
        survivors.reserve(active_.size());
        survivor_rows.reserve(active_.size());
        for (size_t row = 0; row < active_.size(); ++row) {
            const auto & request = active_[row];
            if (request->cancel_requested.load(
                    std::memory_order_acquire)) {
                cancelled.push_back(request);
                continue;
            }
            survivors.push_back(request);
            survivor_rows.push_back(static_cast<int64_t>(row));
            survivor_max_position = std::max(
                survivor_max_position, request->cache_length);
        }
        if (survivors.size() == active_.size()) return;
        invalidate_decode_graph();
        if (survivors.empty()) {
            model_.reset(1);
            release_paged_requests(cancelled);
            detach_paged_kv();
        } else {
            compact_qwen_batch_state(
                model_, survivor_rows, survivor_max_position,
                paged_kv_.get());
            release_paged_requests(cancelled);
            bind_paged_requests(survivors);
            ++compactions_;
        }
        active_ = std::move(survivors);
        active_count_.store(
            static_cast<int64_t>(active_.size()),
            std::memory_order_relaxed);
        for (const auto & request : cancelled) {
            complete_request(request);
        }
    }

    void ensure_decode_metadata_buffers(int primary) {
        if (decode_metadata_host_.defined()) return;
        const int64_t capacity = 3 * static_cast<int64_t>(max_sequences_);
        decode_metadata_host_ = mfq_tensor_backend::empty(
            {capacity}, mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCPU)
                .pinned_memory(true));
        decode_metadata_cuda_ = mfq_tensor_backend::empty(
            {capacity}, mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, primary)));
    }

    void invalidate_decode_graph() {
        if (decode_graph_) decode_graph_->invalidate();
    }

    void decode_active() {
        if (active_.empty()) return;
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        const int primary = g_layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        retire_cancelled_requests();
        if (active_.empty()) return;
        if (paged_kv_) {
            bool page_table_changed = false;
            for (const auto & request : active_) {
                page_table_changed = paged_kv_->ensure_tokens(
                    request->paged_kv, request->cache_length + 1) ||
                    page_table_changed;
            }
            if (page_table_changed) bind_paged_requests(active_);
        }
        const int64_t batch = static_cast<int64_t>(active_.size());
        const bool batch_greedy = std::all_of(
            active_.begin(), active_.end(),
            [](const std::shared_ptr<Request> & request) {
                return request->sampler->greedy() &&
                    !request->sampler->has_penalties() &&
                    !request->token_constraint;
            });
        int64_t requested_len = 0;
        int64_t minimum_remaining =
            active_.front()->generation_limit - active_.front()->produced;
        for (const auto & request : active_) {
            const int64_t remaining =
                request->generation_limit - request->produced;
            minimum_remaining = std::min(minimum_remaining, remaining);
            requested_len = std::max(
                requested_len, request->cache_length + remaining);
        }
        const int64_t planned_len = decode_graph_bucket(
            requested_len, model_.max_position_embeddings());
        const int64_t graph_attention_parts = decode_graph_attention_parts(
            planned_len, FullBlock::kDecodeAttentionMaxParts);
        const char * graph_min_environment = std::getenv(
            "MFQ_CONTINUOUS_BATCH_CUDA_GRAPH_MIN_TOKENS");
        const int64_t graph_min_tokens = graph_min_environment != nullptr
            ? std::max<int64_t>(2, std::atoi(graph_min_environment)) : 16;
        std::vector<const void *> graph_state_addresses;
        bool graph_cache_hit = false;
        bool graph_decode = false;
        if (batch >= 2 && batch_greedy &&
                qwen_continuous_batch_greedy_enabled() &&
                qwen_continuous_batch_cuda_graph_enabled(model_)) {
            graph_state_addresses = qwen_decode_state_addresses(model_);
            graph_cache_hit = decode_graph_ && decode_graph_->matches(
                batch, planned_len, graph_state_addresses);
            graph_decode = graph_cache_hit ||
                minimum_remaining >= graph_min_tokens;
            if (graph_decode && !decode_graph_) {
                decode_graph_ =
                    std::make_unique<QwenContinuousDecodeGraph>();
            }
        }
        std::unique_ptr<MfqCudaStreamGuard> graph_stream_guard;
        std::vector<std::unique_ptr<MfqCudaStreamGuard>>
            graph_compute_stream_guards;
        if (graph_decode) {
            decode_graph_->ensure_compute_streams();
            graph_stream_guard = std::make_unique<MfqCudaStreamGuard>(
                decode_graph_->stream);
            graph_compute_stream_guards =
                activate_cuda_graph_compute_streams(
                    decode_graph_->compute_streams);
        }
        std::vector<int64_t> input_tokens;
        std::vector<int64_t> positions;
        std::vector<int64_t> sequence_lengths;
        const bool packed_metadata = graph_decode ||
            qwen_continuous_batch_packed_metadata_enabled();
        int64_t * packed_metadata_data = nullptr;
        if (packed_metadata) {
            ensure_decode_metadata_buffers(primary);
            packed_metadata_data =
                decode_metadata_host_.data_ptr<int64_t>();
        } else {
            input_tokens.reserve(active_.size());
            positions.reserve(active_.size());
            sequence_lengths.reserve(active_.size());
        }
        int64_t max_position = 0;
        int64_t max_sequence_length = 0;
        for (size_t row = 0; row < active_.size(); ++row) {
            const auto & request = active_[row];
            if (packed_metadata_data != nullptr) {
                packed_metadata_data[row] = request->pending_token;
                packed_metadata_data[batch + row] = request->cache_length;
                packed_metadata_data[2 * batch + row] =
                    request->cache_length + 1;
            } else {
                input_tokens.push_back(request->pending_token);
                positions.push_back(request->cache_length);
                sequence_lengths.push_back(request->cache_length + 1);
            }
            max_position = std::max(max_position, request->cache_length);
            max_sequence_length = std::max(
                max_sequence_length, request->cache_length + 1);
        }
        Tensor ids;
        Tensor pos;
        Tensor lengths;
        if (packed_metadata_data != nullptr) {
            const int64_t values = 3 * batch;
            auto metadata_host = decode_metadata_host_.narrow(0, 0, values);
            auto metadata_cuda = decode_metadata_cuda_.narrow(0, 0, values);
            metadata_cuda.copy_(metadata_host);
            auto matrix = metadata_cuda.reshape({3, batch});
            ids = matrix.narrow(0, 0, 1).reshape({batch, 1});
            pos = matrix.narrow(0, 1, 1).reshape({batch, 1});
            lengths = matrix.narrow(0, 2, 1).reshape({batch});
            ++packed_metadata_batches_;
        } else {
            const auto options = mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, primary));
            ids = mfq_tensor_backend::tensor(input_tokens, options)
                .reshape({batch, 1}).contiguous();
            pos = mfq_tensor_backend::tensor(positions, options)
                .reshape({batch, 1}).contiguous();
            lengths = mfq_tensor_backend::tensor(
                sequence_lengths, options).contiguous();
        }
        Tensor logits;
        Tensor graph_tokens;
        MoeContinuousBatchCacheScope moe_cache_scope(
            cached_moe_enabled_ && batch > 1);
        try {
            model_.cache_pos = max_position;
            if (graph_decode) {
                const auto invoke = [&]() {
                    auto hidden = model_.hidden_forward_static(
                        ids, pos, lengths, planned_len,
                        graph_attention_parts);
                    auto current_logits = qwen_logits_from_last_hidden(
                        model_, std::move(hidden));
                    return sample_greedy_cuda(
                        current_logits.contiguous().view({batch, -1}));
                };
                if (!graph_cache_hit) {
                    decode_graph_->invalidate();
                    mfq_cuda_empty_cache();
                    try {
                        DecodeGraphBranchScope branch_scope;
                        DecodeGraphTpProjectionScope tp_projection_scope;
                        decode_graph_->graph =
                            std::make_unique<MfqCudaGraph>();
                        prepare_decode_graph_memory(
                            model_, *decode_graph_->graph,
                            [&]() { (void)invoke(); },
                            decode_graph_->participant_streams());
                        decode_graph_->graph->capture_begin();
                        decode_graph_->static_next = invoke();
                        decode_graph_->graph->capture_end();
                        decode_graph_->set_key(
                            batch, planned_len,
                            qwen_decode_state_addresses(model_));
                        ++cuda_graph_captures_;
                    } catch (...) {
                        decode_graph_->invalidate();
                        throw;
                    }
                }
                decode_graph_->graph->replay();
                graph_tokens = decode_graph_->static_next;
                ++cuda_graph_replays_;
            } else {
                auto hidden = model_.hidden_forward(
                    ids, pos, lengths, nullptr, pos);
                logits = qwen_logits_from_last_hidden(
                    model_, std::move(hidden));
            }
            model_.cache_pos = max_sequence_length;
        } catch (...) {
            auto error = std::current_exception();
            invalidate_decode_graph();
            try { model_.reset(1); } catch (...) {}
            fail_requests(active_, error);
            release_paged_requests(active_);
            detach_paged_kv();
            active_.clear();
            active_count_.store(0);
            return;
        }
        ++decode_batches_;
        decode_tokens_.fetch_add(batch);
        Tensor batched_greedy_tokens;
        const int64_t * batched_greedy_data = nullptr;
        if (graph_tokens.defined()) {
            batched_greedy_tokens = graph_tokens
                .to(mfq_tensor_backend::kCPU).contiguous();
            MFQ_RUNTIME_CHECK(
                batched_greedy_tokens.scalar_type() ==
                    mfq_tensor_backend::kInt64 &&
                batched_greedy_tokens.numel() == batch,
                "continuous batching CUDA Graph sampler returned the wrong shape");
            batched_greedy_data =
                batched_greedy_tokens.data_ptr<int64_t>();
            ++batched_greedy_batches_;
        } else if (qwen_continuous_batch_greedy_enabled() && batch_greedy) {
            batched_greedy_tokens = sample_greedy_cuda(
                    logits.contiguous().view({batch, -1}))
                .to(mfq_tensor_backend::kCPU).contiguous();
            MFQ_RUNTIME_CHECK(
                batched_greedy_tokens.scalar_type() ==
                    mfq_tensor_backend::kInt64 &&
                batched_greedy_tokens.numel() == batch,
                "continuous batching greedy sampler returned the wrong shape");
            batched_greedy_data =
                batched_greedy_tokens.data_ptr<int64_t>();
            ++batched_greedy_batches_;
        }
        std::vector<std::shared_ptr<Request>> survivors;
        std::vector<std::pair<std::shared_ptr<Request>,
            std::exception_ptr>> completions;
        std::vector<int64_t> survivor_rows;
        survivors.reserve(active_.size());
        survivor_rows.reserve(active_.size());
        int64_t survivor_max_position = 0;
        for (size_t row = 0; row < active_.size(); ++row) {
            const auto & request = active_[row];
            request->cache_length += 1;
            if (request->cancel_requested.load(
                    std::memory_order_acquire)) {
                completions.emplace_back(request, std::exception_ptr{});
                continue;
            }
            try {
                Tensor next;
                int64_t token = 0;
                if (batched_greedy_data != nullptr) {
                    token = batched_greedy_data[row];
                } else {
                    next = mfq::cuda::sample_logits(
                        *request->sampler,
                        logits.narrow(0, static_cast<int64_t>(row), 1),
                        request->counts, request->token_constraint);
                    token = next.item<int64_t>();
                }
                request->pending_token = token;
                ++request->produced;
                publish_token(request, token);
                if (request->cancel_requested.load(
                            std::memory_order_acquire) ||
                        request->produced >= request->generation_limit) {
                    completions.emplace_back(
                        request, std::exception_ptr{});
                    continue;
                }
                if (request->counts.defined()) {
                    MFQ_RUNTIME_CHECK(next.defined(),
                        "batched greedy sampling cannot update token penalties");
                    sample_token_counts_add_cuda(
                        request->counts, next.contiguous());
                }
                survivors.push_back(request);
                survivor_rows.push_back(static_cast<int64_t>(row));
                survivor_max_position = std::max(
                    survivor_max_position, request->cache_length);
            } catch (...) {
                completions.emplace_back(
                    request, std::current_exception());
            }
        }
        if (survivors.empty()) {
            invalidate_decode_graph();
            model_.reset(1);
            release_paged_requests(active_);
            detach_paged_kv();
        } else if (survivors.size() != active_.size()) {
            invalidate_decode_graph();
            compact_qwen_batch_state(
                model_, survivor_rows, survivor_max_position,
                paged_kv_.get());
            std::vector<std::shared_ptr<Request>> retired;
            retired.reserve(completions.size());
            for (const auto & completion : completions) {
                retired.push_back(completion.first);
            }
            release_paged_requests(retired);
            bind_paged_requests(survivors);
            ++compactions_;
        } else {
            model_.cache_pos = survivor_max_position;
        }
        active_ = std::move(survivors);
        active_count_.store(
            static_cast<int64_t>(active_.size()),
            std::memory_order_relaxed);
        for (const auto & completion : completions) {
            complete_request(completion.first, completion.second);
        }
    }

    void worker_main() noexcept {
        for (;;) {
            try {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_ready_.wait(lock, [&] {
                        return stopping_ || !pending_.empty() ||
                            !active_.empty() || !prefilling_.empty();
                    });
                    if (stopping_) {
                        auto error = std::make_exception_ptr(
                            std::runtime_error(
                                "continuous batching scheduler stopped"));
                        std::vector<std::shared_ptr<Request>> pending;
                        while (!pending_.empty()) {
                            pending.push_back(std::move(pending_.front()));
                            pending_.pop_front();
                        }
                        std::vector<std::shared_ptr<Request>> prefilling(
                            prefilling_.begin(), prefilling_.end());
                        prefilling_.clear();
                        queued_.store(0);
                        prefilling_count_.store(0);
                        lock.unlock();
                        fail_requests(pending, error);
                        fail_requests(prefilling, error);
                        fail_requests(active_, error);
                        try {
                            std::lock_guard<std::mutex> model_lock(model_mutex_);
                            try { mfq_cuda_synchronize(); } catch (...) {}
                            release_paged_requests(prefilling);
                            release_paged_requests(active_);
                            model_.reset(1);
                            detach_paged_kv();
                        } catch (...) {}
                        active_.clear();
                        active_count_.store(0);
                        return;
                    }
                    if (active_.empty() && prefilling_.empty() &&
                            pending_.size() <
                                static_cast<size_t>(max_sequences_)) {
                        queue_ready_.wait_for(
                            lock, initial_batch_wait_, [&] {
                                return stopping_ ||
                                    pending_.size() >=
                                        static_cast<size_t>(max_sequences_);
                            });
                        if (stopping_) continue;
                    }
                }
                // CUDA kernels cannot be preempted. Service one decode step,
                // then advance at most one prompt chunk while decode remains
                // active. The partial Qwen state is detached until the next
                // scheduler turn, bounding each decode stall to one chunk.
                const bool decode_was_active = !active_.empty();
                if (decode_was_active) decode_active();
                const bool contended = !active_.empty();
                auto incoming = take_pending(contended
                    ? size_t{1}
                    : static_cast<size_t>(max_sequences_));
                if (contended && !incoming.empty()) {
                    ++interleaved_admissions_;
                }
                advance_prefills(incoming, contended);
                if (!decode_was_active) decode_active();
            } catch (...) {
                auto error = std::current_exception();
                std::vector<std::shared_ptr<Request>> prefilling(
                    prefilling_.begin(), prefilling_.end());
                fail_requests(active_, error);
                fail_requests(prefilling, error);
                try {
                    std::lock_guard<std::mutex> model_lock(model_mutex_);
                    try { mfq_cuda_synchronize(); } catch (...) {}
                    release_paged_requests(active_);
                    release_paged_requests(prefilling);
                    model_.reset(1);
                    detach_paged_kv();
                } catch (...) {}
                active_.clear();
                prefilling_.clear();
                active_count_.store(0);
                prefilling_count_.store(0);
            }
        }
    }

    mfq::cuda::Qwen35CausalLm & model_;
    std::mutex & model_mutex_;
    int32_t max_sequences_ = 0;
    int64_t prefill_chunk_size_ = 2048;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    std::chrono::microseconds initial_batch_wait_;
    std::thread worker_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::deque<std::shared_ptr<Request>> prefilling_;
    std::vector<std::shared_ptr<Request>> active_;
    std::atomic<bool> stopping_{false};
    std::atomic<int64_t> queued_{0};
    std::atomic<int64_t> active_count_{0};
    std::atomic<int64_t> prefilling_count_{0};
    std::atomic<int64_t> requests_{0};
    std::atomic<int64_t> decode_batches_{0};
    std::atomic<int64_t> decode_tokens_{0};
    std::atomic<int64_t> max_batch_seen_{0};
    std::atomic<int64_t> admissions_{0};
    std::atomic<int64_t> prefill_chunks_{0};
    std::atomic<int64_t> prefill_yields_{0};
    std::atomic<int64_t> interleaved_admissions_{0};
    std::atomic<int64_t> compactions_{0};
    std::atomic<int64_t> batched_greedy_batches_{0};
    std::atomic<int64_t> packed_metadata_batches_{0};
    std::atomic<int64_t> cuda_graph_captures_{0};
    std::atomic<int64_t> cuda_graph_replays_{0};
    std::atomic<int64_t> mtp_bypasses_{0};
    std::atomic<int64_t> prefix_cache_bypasses_{0};
    Tensor decode_metadata_host_;
    Tensor decode_metadata_cuda_;
    std::unique_ptr<QwenPagedKvArena> paged_kv_;
    std::unique_ptr<QwenContinuousDecodeGraph> decode_graph_;
};

static int run_qwen_continuous_batching_check(mfq::cuda::Qwen35CausalLm & model) {
    const auto incompatibility =
        qwen_continuous_batching_incompatibility(model);
    MFQ_RUNTIME_CHECK(incompatibility.empty(), incompatibility);
    MFQ_RUNTIME_CHECK(model.vocab_size() > 1024 &&
        model.max_position_embeddings() >= 208,
        "continuous batching check requires vocab>1024 and context>=208");

    MfqSamplingParams first_params;
    first_params.max_tokens = 20;
    first_params.temperature = 0.0;
    first_params.top_k = 1;
    first_params.top_p = 1.0;
    first_params.enable_mtp = false;
    first_params.seed = 20260907;
    auto second_params = first_params;
    second_params.max_tokens = 18;
    second_params.seed += 1;
    std::vector<int64_t> first_prompt(193);
    std::vector<int64_t> second_prompt(17);
    for (size_t index = 0; index < first_prompt.size(); ++index) {
        first_prompt[index] = 101 +
            static_cast<int64_t>((index * 37) % 900);
    }
    for (size_t index = 0; index < second_prompt.size(); ++index) {
        second_prompt[index] = 113 +
            static_cast<int64_t>((index * 53) % 880);
    }
    constexpr int64_t check_prefill_chunk_size = 64;

    auto serial = [&](const std::vector<int64_t> & prompt,
                      const MfqSamplingParams & params) {
        std::vector<int64_t> output;
        std::mutex mutex;
        DecodeGraphCache graph_cache(
            model.max_position_embeddings());
        TextSessionCache session_cache;
        const int32_t produced = generate_tokens(
            model, mutex, graph_cache, session_cache, prompt, params,
            [&](int64_t token) {
                output.push_back(token);
                return true;
            }, {}, {}, {}, nullptr, check_prefill_chunk_size);
        MFQ_RUNTIME_CHECK(
            produced == params.max_tokens &&
                output.size() == static_cast<size_t>(produced),
            "continuous batching serial oracle length mismatch");
        return output;
    };
    const auto first_reference = serial(first_prompt, first_params);
    const auto second_reference = serial(second_prompt, second_params);
    model.reset(1);

    std::mutex model_mutex;
    CudaContinuousBatcher batcher(
        model, model_mutex, 4, check_prefill_chunk_size,
        std::chrono::milliseconds(100));
    std::mutex gate_mutex;
    std::condition_variable gate_ready;
    bool first_prefilled = false;
    bool second_delivered = false;
    bool release_first = false;
    std::vector<int64_t> first_output;
    std::vector<int64_t> second_output;
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    int32_t first_produced = 0;
    int32_t second_produced = 0;

    std::thread first_thread([&] {
        try {
            first_produced = batcher.submit(
                first_prompt, first_params,
                [&](int64_t token) {
                    first_output.push_back(token);
                    if (first_output.size() == 1) {
                        std::unique_lock<std::mutex> lock(gate_mutex);
                        first_prefilled = true;
                        gate_ready.notify_one();
                        gate_ready.wait(lock, [&] { return release_first; });
                    }
                    return true;
                }, {}, {}, {});
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    bool first_queued = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
        if (batcher.queued_requests() > 0) {
            first_queued = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::thread second_thread([&] {
        try {
            second_produced = batcher.submit(
                second_prompt, second_params,
                [&](int64_t token) {
                    second_output.push_back(token);
                    if (second_output.size() == 1) {
                        std::lock_guard<std::mutex> lock(gate_mutex);
                        second_delivered = true;
                        gate_ready.notify_one();
                    }
                    return true;
                }, {}, {}, {});
        } catch (...) {
            second_error = std::current_exception();
        }
    });
    bool first_callback_started = false;
    bool callback_isolated = false;
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        first_callback_started = gate_ready.wait_for(
            lock, std::chrono::seconds(10), [&] {
                return first_prefilled;
            });
        callback_isolated = first_callback_started && gate_ready.wait_for(
            lock, std::chrono::seconds(10), [&] {
                return second_delivered;
            });
        release_first = true;
    }
    gate_ready.notify_one();
    first_thread.join();
    second_thread.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);
    MFQ_RUNTIME_CHECK(first_queued && first_callback_started &&
        callback_isolated,
        "a blocked response callback stalled the scheduler");
    MFQ_RUNTIME_CHECK(first_produced == first_params.max_tokens &&
        second_produced == second_params.max_tokens,
        "continuous batching generated token count mismatch");
    const auto print_mismatch = [](const char * name,
            const std::vector<int64_t> & reference,
            const std::vector<int64_t> & actual) {
        if (reference == actual) return;
        std::cerr << "continuous_batching_check mismatch " << name << " reference=";
        for (auto token : reference) std::cerr << token << ',';
        std::cerr << " actual=";
        for (auto token : actual) std::cerr << token << ',';
        std::cerr << '\n';
    };
    print_mismatch("first", first_reference, first_output);
    print_mismatch("second", second_reference, second_output);
    MFQ_RUNTIME_CHECK(first_output == first_reference,
        "continuous batching first request differs from serial greedy oracle");
    MFQ_RUNTIME_CHECK(second_output == second_reference,
        "continuous batching second request differs from serial greedy oracle");
    auto cancel_params = second_params;
    cancel_params.max_tokens = 12;
    int32_t cancellation_callbacks = 0;
    const int32_t cancellation_produced = batcher.submit(
        second_prompt, cancel_params,
        [&](int64_t) {
            ++cancellation_callbacks;
            return false;
        }, {}, {}, {});
    MFQ_RUNTIME_CHECK(cancellation_produced == 1 &&
        cancellation_callbacks == 1,
        "continuous batching callback cancellation did not stop at one token");
    const auto values = batcher.metrics();
    auto metric = [&](const std::string & name) {
        const auto found = std::find_if(
            values.begin(), values.end(), [&](const auto & item) {
                return item.first == name;
            });
        return found == values.end() ? 0.0 : found->second;
    };
    std::cout << "continuous_batching_check metrics max_batch="
              << metric("continuous_batching_max_batch")
              << " compactions="
              << metric("continuous_batching_compactions")
              << " batched_greedy_batches="
              << metric("continuous_batching_batched_greedy_batches")
              << " packed_metadata_batches="
              << metric("continuous_batching_packed_metadata_batches")
              << " cuda_graph_captures="
              << metric("continuous_batching_cuda_graph_captures")
              << " cuda_graph_replays="
              << metric("continuous_batching_cuda_graph_replays")
              << " paged_kv="
              << metric("continuous_batching_paged_kv")
              << " page_size="
              << metric("paged_kv_page_size")
              << " live_pages="
              << metric("paged_kv_live_pages")
              << " peak_pages="
              << metric("paged_kv_peak_live_pages")
              << " capacity_pages="
              << metric("paged_kv_capacity_pages")
              << " page_allocations="
              << metric("paged_kv_page_allocations")
              << " page_reuses="
              << metric("paged_kv_page_reuses")
              << " page_releases="
              << metric("paged_kv_page_releases")
              << " active="
              << metric("continuous_batching_active")
              << " prefilling="
              << metric("continuous_batching_prefilling")
              << " prefill_chunks="
              << metric("continuous_batching_prefill_chunks")
              << " prefill_yields="
              << metric("continuous_batching_prefill_yields")
              << " queued="
              << metric("continuous_batching_queued") << '\n';
    MFQ_RUNTIME_CHECK(metric("continuous_batching_max_batch") >= 2.0 &&
        metric("continuous_batching_compactions") >= 1.0 &&
        (!qwen_continuous_batch_greedy_enabled() ||
            metric("continuous_batching_batched_greedy_batches") >= 1.0) &&
        (!qwen_continuous_batch_packed_metadata_enabled() ||
            metric("continuous_batching_packed_metadata_batches") >= 1.0) &&
        (!qwen_continuous_batch_cuda_graph_enabled(model) ||
             (metric("continuous_batching_cuda_graph_captures") >= 1.0 &&
             metric("continuous_batching_cuda_graph_replays") >= 2.0)) &&
        (!qwen_continuous_paged_kv_enabled() ||
            (metric("continuous_batching_paged_kv") == 1.0 &&
             metric("paged_kv_page_size") ==
                static_cast<double>(QwenPagedKvArena::kPageSize) &&
             metric("paged_kv_live_pages") == 0.0 &&
             metric("paged_kv_peak_live_pages") > 0.0 &&
             metric("paged_kv_capacity_pages") >=
                metric("paged_kv_peak_live_pages") &&
             metric("paged_kv_page_allocations") ==
                metric("paged_kv_page_releases") &&
             metric("paged_kv_page_reuses") > 0.0)) &&
        metric("continuous_batching_prefill_chunks") >= 4.0 &&
        metric("continuous_batching_active") == 0.0 &&
        metric("continuous_batching_prefilling") == 0.0 &&
        metric("continuous_batching_queued") == 0.0,
        "continuous batching check did not exercise join and retire");
    std::cout << "continuous_batching_check PASS concurrent_requests=2"
              << " cancellation_tokens=1 max_batch="
              << metric("continuous_batching_max_batch")
              << " prompt_lengths=193,17 split_k=1"
              << " decode_batches="
              << metric("continuous_batching_decode_batches")
              << " compactions="
              << metric("continuous_batching_compactions") << '\n';
    return 0;
}

} // namespace mfq::cuda::continuous
