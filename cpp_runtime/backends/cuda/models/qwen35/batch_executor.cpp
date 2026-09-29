#include "batch_executor.h"

#include "models/qwen35/causal_lm.h"
#include "models/components.h"
#include "cuda_sampling.h"
#include "generation.h"
#include "text_session_cache.h"
#include "storage/moe_expert_cache.h"
#include "paged_kv.h"
#include "linear_attention.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Shared engine code owns queueing, worker lifetime, and request transitions.
// This adapter owns Qwen CUDA batch assembly and device state.

namespace mfq::cuda::qwen35 {

using internal::generate;
using internal::PrefillCudaTimer;
using internal::TextSessionCache;
using Tensor = mfq_tensor_backend::Tensor;
using LinearBlock = LinearAttentionBlock;
using mfq::cuda::continuous::QwenPagedKvArena;
using mfq::cuda::continuous::QwenPagedKvSequence;

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

bool qwen_continuous_batch_has_moe(
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

bool qwen_continuous_batch_has_cached_moe(
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
        : previous_(moe_continuous_batch_cache_serial()) {
        moe_continuous_batch_cache_serial() = enabled;
    }

    ~MoeContinuousBatchCacheScope() {
        moe_continuous_batch_cache_serial() = previous_;
    }

    MoeContinuousBatchCacheScope(
        const MoeContinuousBatchCacheScope &) = delete;
    MoeContinuousBatchCacheScope & operator=(
        const MoeContinuousBatchCacheScope &) = delete;

private:
    bool previous_ = false;
};

static std::string qwen_continuous_batching_incompatibility(
        const mfq::cuda::Qwen35CausalLm & model,
        const CudaExecutionContext& execution) {
    if (model.blocks.empty()) {
        return "continuous batching requires at least one model block";
    }
    if (execution.dense_cpu_layer_count != 0 ||
            !execution.dsv4_cpu_offload_layers.empty()) {
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
                        execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        if (const auto * linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.uses_moe_expert_cache() &&
                    linear->ffn.moe_top_k >
                        execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        return "continuous batching encountered an unsupported Qwen block";
    }
    return {};
}

static bool qwen_continuous_batch_cuda_graph_enabled(
        const mfq::cuda::Qwen35CausalLm& model,
        const CudaContinuousBatchConfig& config) {
    return config.cuda_graph &&
        !qwen_continuous_batch_has_cached_moe(model) &&
        mfq_cuda_graph_capture_supported() &&
        model_parallel_cuda_graph_enabled();
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
        model.lm_head.forward(*model.execution, last));
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

struct QwenBatchRequest : mfq::engine::ContinuousBatchRequest {
    QwenBatchRequest(
            const std::vector<int64_t>& input_prompt,
            const MfqSamplingParams& input_sampling,
            const MfqTokenConstraintPtr& input_constraint,
            const MfqCancellationCheck& input_cancelled)
        : ContinuousBatchRequest(
              input_prompt, input_sampling, input_constraint,
              input_cancelled) {}

    std::optional<mfq::cuda::Sampler> sampler;
    Tensor counts;
    Tensor prefill_ids;
    std::optional<QwenBatchState> prefill_state;
    std::vector<std::unique_ptr<PrefillCudaTimer>> prefill_timers;
    int64_t prefill_offset = 0;
    int64_t cache_length = 0;
    QwenPagedKvSequence paged_kv;
};

struct QwenBatchOperations {
    using Request = QwenBatchRequest;
    using State = mfq::engine::ContinuousBatchState<Request>;
    using Queue = mfq::engine::ContinuousBatchQueue<Request>;

    QwenBatchOperations(
            mfq::cuda::Qwen35CausalLm& model,
            CudaExecutionContext& execution,
            std::mutex& model_mutex,
            CudaContinuousBatchConfig config,
            mfq::engine::GenerationConfig generation)
        : model_(model), execution_(execution), model_mutex_(model_mutex),
          config_(std::move(config)),
          max_sequences_(static_cast<int32_t>(
              config_.scheduling.max_sequences)),
          prefill_chunk_size_(generation.prefill_chunk_size),
          moe_enabled_(qwen_continuous_batch_has_moe(model)),
          cached_moe_enabled_(
              qwen_continuous_batch_has_cached_moe(model)) {
        if (config_.scheduling.max_sequences >
                static_cast<std::size_t>(
                    std::numeric_limits<int32_t>::max()) ||
                max_sequences_ < 1) {
            throw std::invalid_argument(
                "continuous batching max sequences must be positive");
        }
        if (prefill_chunk_size_ < 1) {
            throw std::invalid_argument(
                "continuous batching prefill chunk size must be positive");
        }
        const auto incompatibility =
            qwen_continuous_batching_incompatibility(model_, execution_);
        if (!incompatibility.empty()) {
            throw std::runtime_error(incompatibility);
        }
        if (config_.paged_kv) {
            paged_kv_ = std::make_unique<QwenPagedKvArena>(
                model_, max_sequences_);
        }
    }

    QwenBatchOperations(const QwenBatchOperations&) = delete;
    QwenBatchOperations& operator=(const QwenBatchOperations&) = delete;

    std::vector<std::pair<std::string, double>> metrics() const {
        return {
            {"continuous_batching_requests",
                static_cast<double>(requests_.load())},
            {"continuous_batching_decode_batches",
                static_cast<double>(decode_batches_.load())},
            {"continuous_batching_decode_tokens",
                static_cast<double>(decode_tokens_.load())},

            {"continuous_batching_admissions",
                static_cast<double>(admissions_.load())},
            {"continuous_batching_prefill_chunks",
                static_cast<double>(prefill_chunks_.load())},
            {"continuous_batching_prefill_yields",
                static_cast<double>(prefill_yields_.load())},

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

    bool paged_kv_enabled() const noexcept { return paged_kv_ != nullptr; }
    int64_t paged_kv_page_size() const noexcept {
        return paged_kv_ ? paged_kv_->page_size() : 0;
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
            request->complete(error);
        }
    }

    void initialize_sampling(Request & request, const Tensor & prompt_ids) {
        const int primary = execution_.layer_placement.primary_device();
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
            const std::vector<std::shared_ptr<Request>>& incoming,
            bool yield_after_chunk,
            State& state,
            const Queue& queue) {
        CudaExecutionContextScope context_scope(execution_);
        for (const auto& request : incoming) {
            state.prefilling.push_back(request);
        }
        if (state.prefilling.empty()) return;
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        invalidate_decode_graph();
        const int primary = execution_.layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        std::vector<QwenBatchState> states;
        states.reserve(1 + state.prefilling.size());
        if (!state.active.empty()) {
            states.push_back(take_qwen_batch_state(
                model_, static_cast<int64_t>(state.active.size()),
                paged_kv_.get()));
        }
        std::vector<std::shared_ptr<Request>> admitted;
        admitted.reserve(state.prefilling.size());
        int64_t chunks_advanced = 0;
        while (!state.prefilling.empty() &&
                !queue.stopping() &&
                (!yield_after_chunk || chunks_advanced == 0)) {
            auto request = std::move(state.prefilling.front());
            state.prefilling.pop_front();
            if (request->cancel_requested.load(std::memory_order_acquire)) {
                try { mfq_cuda_synchronize(); } catch (...) {}
                if (paged_kv_) {
                    paged_kv_->release(request->paged_kv);
                    detach_paged_kv();
                }
                request->prefill_state.reset();
                request->prefill_timers.clear();
                request->prefill_ids = Tensor();
                request->complete();
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
                    const auto chunk = mfq::engine::next_prefill_chunk(
                        request->prefill_ids.size(1),
                        request->prefill_offset,
                        prefill_chunk_size_);
                    auto chunk_timer =
                        std::make_unique<PrefillCudaTimer>();
                    hidden = model_.hidden_forward(
                        request->prefill_ids.narrow(
                            1, chunk.offset, chunk.count).contiguous());
                    request->prefill_offset += chunk.count;
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
                    !queue.stopping());
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
                    request->complete();
                    continue;
                }
                if (queue.stopping()) {
                    throw std::runtime_error(
                        "continuous batching scheduler stopped during prefill");
                }
                if (request->prefill_offset < request->prefill_ids.size(1)) {
                    request->prefill_state = take_qwen_batch_state(
                        model_, 1, paged_kv_.get());
                    detach_paged_kv();
                    state.prefilling.push_back(std::move(request));
                    ++prefill_yields_;
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
                request->publish_prefill(MfqPrefillTiming{
                    request->prompt.size(), prefill_ms, 0.0, prefill_ms});
                request->pending_token = token;
                request->cache_length =
                    static_cast<int64_t>(request->prompt.size());
                request->produced = 1;
                request->publish_token(token);
                if (request->cancel_requested.load(
                            std::memory_order_acquire) ||
                        request->produced >= request->generation_limit) {
                    if (paged_kv_) {
                        paged_kv_->release(request->paged_kv);
                        detach_paged_kv();
                    }
                    request->complete();
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
                request->complete(error);
            }
        }
        state.active.insert(
            state.active.end(), admitted.begin(), admitted.end());
        if (!states.empty()) {
            int64_t max_cache_position = 0;
            for (const auto& request : state.active) {
                max_cache_position = std::max(
                    max_cache_position, request->cache_length);
            }
            restore_qwen_batch_states(
                model_, states, max_cache_position, paged_kv_.get());
            bind_paged_requests(state.active);
        } else {
            model_.reset(1);
            detach_paged_kv();
        }
    }

    void retire_cancelled_requests(State& state) {
        std::vector<std::shared_ptr<Request>> survivors;
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<int64_t> survivor_rows;
        int64_t survivor_max_position = 0;
        survivors.reserve(state.active.size());
        survivor_rows.reserve(state.active.size());
        for (size_t row = 0; row < state.active.size(); ++row) {
            const auto& request = state.active[row];
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
        if (survivors.size() == state.active.size()) return;
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
        state.active = std::move(survivors);
        for (const auto& request : cancelled) {
            request->complete();
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

    void decode_active(State& state) {
        CudaExecutionContextScope context_scope(execution_);
        if (state.active.empty()) return;
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        const int primary = execution_.layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        retire_cancelled_requests(state);
        if (state.active.empty()) return;
        if (paged_kv_) {
            bool page_table_changed = false;
            for (const auto& request : state.active) {
                page_table_changed = paged_kv_->ensure_tokens(
                    request->paged_kv, request->cache_length + 1) ||
                    page_table_changed;
            }
            if (page_table_changed) bind_paged_requests(state.active);
        }
        const int64_t batch = static_cast<int64_t>(state.active.size());
        const bool batch_greedy = std::all_of(
            state.active.begin(), state.active.end(),
            [](const std::shared_ptr<Request>& request) {
                return request->sampler->greedy() &&
                    !request->sampler->has_penalties() &&
                    !request->token_constraint;
            });
        int64_t requested_len = 0;
        int64_t minimum_remaining =
            state.active.front()->generation_limit -
            state.active.front()->produced;
        for (const auto& request : state.active) {
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
        std::vector<const void *> graph_state_addresses;
        bool graph_cache_hit = false;
        bool graph_decode = false;
        if (batch >= 2 && batch_greedy && config_.greedy &&
                qwen_continuous_batch_cuda_graph_enabled(
                    model_, config_)) {
            graph_state_addresses = qwen_decode_state_addresses(model_);
            graph_cache_hit = decode_graph_ && decode_graph_->matches(
                batch, planned_len, graph_state_addresses);
            graph_decode = graph_cache_hit || minimum_remaining >=
                config_.cuda_graph_minimum_tokens;
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
        const bool packed_metadata =
            graph_decode || config_.packed_metadata;
        int64_t * packed_metadata_data = nullptr;
        if (packed_metadata) {
            ensure_decode_metadata_buffers(primary);
            packed_metadata_data =
                decode_metadata_host_.data_ptr<int64_t>();
        } else {
            input_tokens.reserve(state.active.size());
            positions.reserve(state.active.size());
            sequence_lengths.reserve(state.active.size());
        }
        int64_t max_position = 0;
        int64_t max_sequence_length = 0;
        for (size_t row = 0; row < state.active.size(); ++row) {
            const auto& request = state.active[row];
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
            fail_requests(state.active, error);
            release_paged_requests(state.active);
            detach_paged_kv();
            state.active.clear();
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
        } else if (config_.greedy && batch_greedy) {
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
        survivors.reserve(state.active.size());
        survivor_rows.reserve(state.active.size());
        int64_t survivor_max_position = 0;
        for (size_t row = 0; row < state.active.size(); ++row) {
            const auto& request = state.active[row];
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
                request->publish_token(token);
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
            release_paged_requests(state.active);
            detach_paged_kv();
        } else if (survivors.size() != state.active.size()) {
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
        state.active = std::move(survivors);
        for (const auto& completion : completions) {
            completion.first->complete(completion.second);
        }
    }

    void shutdown(
            const std::vector<std::shared_ptr<Request>>& pending,
            State& state,
            std::exception_ptr error) {
        CudaExecutionContextScope context_scope(execution_);
        std::vector<std::shared_ptr<Request>> prefilling(
            state.prefilling.begin(), state.prefilling.end());
        fail_requests(pending, error);
        fail_requests(prefilling, error);
        fail_requests(state.active, error);
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        try { mfq_cuda_synchronize(); } catch (...) {}
        release_paged_requests(prefilling);
        release_paged_requests(state.active);
        model_.reset(1);
        detach_paged_kv();
    }

    void recover(State& state, std::exception_ptr error) {
        shutdown({}, state, error);
    }

    mfq::cuda::Qwen35CausalLm & model_;
    CudaExecutionContext& execution_;
    std::mutex & model_mutex_;
    const CudaContinuousBatchConfig config_;
    int32_t max_sequences_ = 0;
    int64_t prefill_chunk_size_ = 2048;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    std::atomic<int64_t> requests_{0};
    std::atomic<int64_t> decode_batches_{0};
    std::atomic<int64_t> decode_tokens_{0};
    std::atomic<int64_t> admissions_{0};
    std::atomic<int64_t> prefill_chunks_{0};
    std::atomic<int64_t> prefill_yields_{0};
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

struct QwenBatchExecutor::Impl {
    using Controller = mfq::engine::ContinuousBatchingController<
        QwenBatchRequest, QwenBatchOperations>;

    Impl(
            Qwen35CausalLm& model,
            CudaExecutionContext& execution,
            std::mutex& model_mutex,
            const CudaContinuousBatchConfig& config,
            mfq::engine::GenerationConfig generation)
        : controller_(
              config.scheduling,
              std::make_unique<QwenBatchOperations>(
                  model, execution, model_mutex, config, generation)) {}

    int32_t submit(
            const std::vector<int64_t>& prompt,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback& on_prefill,
            const MfqPromptCachePlan& cache_plan,
            const MfqTokenConstraintPtr& token_constraint,
            const MfqCancellationCheck& cancelled) {
        auto& operations = controller_.operations();
        const auto plan = mfq::engine::plan_generation(
            prompt, operations.model_.vocab_size(),
            operations.model_.max_position_embeddings(),
            sampling.max_tokens, cache_plan.stable_prefix_tokens);
        if (plan.generation_tokens == 0) return 0;
        auto request = std::make_shared<QwenBatchRequest>(
            prompt, sampling, token_constraint, cancelled);
        request->generation_limit = plan.generation_tokens;
        if (!cache_plan.session_id.empty() ||
                cache_plan.stable_prefix_tokens != 0) {
            ++operations.prefix_cache_bypasses_;
        }
        if (sampling.enable_mtp) ++operations.mtp_bypasses_;
        return controller_.submit(request, on_token, on_prefill);
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        const auto& operations = controller_.operations();
        auto result = controller_.metrics();
        auto backend = operations.metrics();
        result.insert(result.end(), backend.begin(), backend.end());
        return result;
    }

    int64_t queued_requests() const {
        return static_cast<int64_t>(controller_.queued());
    }

    bool paged_kv_enabled() const noexcept {
        return controller_.operations().paged_kv_enabled();
    }

    int64_t paged_kv_page_size() const noexcept {
        return controller_.operations().paged_kv_page_size();
    }

    Controller controller_;
};

QwenBatchExecutor::QwenBatchExecutor(
        Qwen35CausalLm& model, CudaExecutionContext& execution,
        std::mutex& model_mutex, CudaContinuousBatchConfig config,
        mfq::engine::GenerationConfig generation)
    : impl_(std::make_unique<Impl>(
          model, execution, model_mutex, std::move(config), generation)) {}

QwenBatchExecutor::~QwenBatchExecutor() = default;

std::int32_t QwenBatchExecutor::submit(
        const std::vector<std::int64_t>& prompt,
        const MfqSamplingParams& sampling,
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill,
        const MfqPromptCachePlan& cache_plan,
        const MfqTokenConstraintPtr& token_constraint,
        const MfqCancellationCheck& cancelled) {
    return impl_->submit(
        prompt, sampling, on_token, on_prefill, cache_plan,
        token_constraint, cancelled);
}

std::vector<std::pair<std::string, double>>
QwenBatchExecutor::metrics() const {
    return impl_->metrics();
}

std::int64_t QwenBatchExecutor::queued_requests() const {
    return impl_->queued_requests();
}

bool QwenBatchExecutor::paged_kv_enabled() const noexcept {
    return impl_->paged_kv_enabled();
}

std::int64_t QwenBatchExecutor::paged_kv_page_size() const noexcept {
    return impl_->paged_kv_page_size();
}

int run_qwen_continuous_batching_check(
        Qwen35CausalLm& model,
        const CudaRuntimeConfig& runtime_config) {
    auto check_config = runtime_config;
    check_config.generation.prefill_chunk_size = 64;
    check_config.continuous_batch.scheduling.max_sequences = 4;
    check_config.continuous_batch.scheduling.initial_batch_wait =
        std::chrono::milliseconds(100);
    auto& execution = *model.execution;
    const auto incompatibility =
        qwen_continuous_batching_incompatibility(model, execution);
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
    auto serial = [&](const std::vector<int64_t> & prompt,
                      const MfqSamplingParams & params) {
        std::vector<int64_t> output;
        std::mutex mutex;
        DecodeGraphCache graph_cache(
            model.max_position_embeddings());
        TextSessionCache session_cache(
            check_config.session_cache, check_config.prefix_cache);
        const int32_t produced = generate(
            model, mutex, graph_cache, session_cache, check_config,
            prompt, params,
            [&](int64_t token) {
                output.push_back(token);
                return true;
            }, {}, {}, {}, nullptr);
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
    QwenBatchExecutor batcher(
        model, execution, model_mutex, check_config.continuous_batch,
        check_config.generation);
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
                }, {}, {}, {}, {});
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
                }, {}, {}, {}, {});
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
        }, {}, {}, {}, {});
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
        (!check_config.continuous_batch.greedy ||
            metric("continuous_batching_batched_greedy_batches") >= 1.0) &&
        (!check_config.continuous_batch.packed_metadata ||
            metric("continuous_batching_packed_metadata_batches") >= 1.0) &&
        (!qwen_continuous_batch_cuda_graph_enabled(
                model, check_config.continuous_batch) ||
             (metric("continuous_batching_cuda_graph_captures") >= 1.0 &&
             metric("continuous_batching_cuda_graph_replays") >= 2.0)) &&
        (!check_config.continuous_batch.paged_kv ||
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

} // namespace mfq::cuda::qwen35

template <>
std::unique_ptr<mfq::engine::ContinuousBatching>
make_cuda_continuous_batching(
        mfq::cuda::Qwen35CausalLm& model,
        CudaExecutionContext& execution,
        std::mutex& model_mutex,
        const mfq::cuda::CudaContinuousBatchConfig& config,
        mfq::engine::GenerationConfig generation) {
    return std::make_unique<mfq::cuda::qwen35::QwenBatchExecutor>(
        model, execution, model_mutex, config, generation);
}
