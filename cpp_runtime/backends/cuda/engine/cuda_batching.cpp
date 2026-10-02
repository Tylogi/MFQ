#include "cuda_batching.h"

#include "decode_graph.h"
#include "models/qwen35/batch_state.h"
#include "models/qwen35/causal_lm.h"
#include "models/qwen35/paged_kv.h"
#include "cuda_execution.h"
#include "cuda_sampling.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <deque>
#include <unordered_map>

namespace mfq::cuda {

using internal::PrefillCudaTimer;
using Tensor = mfq_tensor_backend::Tensor;
using qwen35::Qwen35BatchStateAdapter;
using qwen35::QwenBatchState;
using continuous::QwenPagedKvSequence;

bool qwen_continuous_batch_cuda_graph_enabled(
        const Qwen35CausalLm& model,
        const CudaContinuousBatchConfig& config) {
    const auto& execution = *model.execution;
    return config.cuda_graph &&
        !Qwen35BatchStateAdapter::has_cached_moe(model) &&
        mfq_cuda_graph_capture_supported() &&
        model_parallel_cuda_graph_enabled(execution);
}

class MoeContinuousBatchCacheScope {
public:
    MoeContinuousBatchCacheScope(
            CudaExecutionContext& execution,
            bool enabled)
        : execution_(execution),
          previous_(execution.continuous_batch_cache_serial) {
        execution_.continuous_batch_cache_serial = enabled;
    }

    ~MoeContinuousBatchCacheScope() {
        execution_.continuous_batch_cache_serial = previous_;
    }

    MoeContinuousBatchCacheScope(
        const MoeContinuousBatchCacheScope &) = delete;
    MoeContinuousBatchCacheScope & operator=(
        const MoeContinuousBatchCacheScope &) = delete;

private:
    CudaExecutionContext& execution_;
    bool previous_ = false;
};

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

    void ensure_compute_streams(const ParallelConfig& parallel) {
        if (compute_streams.empty()) {
            compute_streams = make_cuda_graph_compute_streams(
                stream, parallel);
        }
    }

    std::vector<MfqCudaStream> participant_streams(
            const ModelParallelCollectiveRuntime& collectives) const {
        return cuda_graph_participant_streams(
            compute_streams, collectives);
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

struct QwenBatchRequest {
    QwenBatchRequest(std::string id, const mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output)
        : id(std::move(id)), prompt(input.prompt),
          sampling(input.sampling), token_constraint(input.token_constraint),
          output(output) {}
    std::string id;
    std::vector<int64_t> prompt;
    MfqSamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    mfq::engine::InferenceOutput& output;
    int32_t generation_limit = 0, produced = 0;
    int64_t pending_token = 0;
    bool eligible = false, done = false;
    std::exception_ptr error;
    std::vector<mfq::engine::EventData> events;
    bool stopped() const { return output.stopped(); }
    void publish_prefill(MfqPrefillTiming timing) {
        output.metrics.mark_prefill(timing);
        events.emplace_back(mfq::engine::PrefillProgress{timing});
    }
    void publish_token(int64_t token) {
        events.emplace_back(output.append(std::vector<int64_t>{token}));
    }
    void complete(std::exception_ptr failure = {}) { done = true; error = failure; }
    std::optional<mfq::cuda::Sampler> sampler;
    Tensor counts;
    Tensor prefill_ids;
    std::optional<QwenBatchState> prefill_state;
    std::vector<std::unique_ptr<PrefillCudaTimer>> prefill_timers;
    int64_t prefill_offset = 0;
    int64_t cache_length = 0;
    int32_t slot = -1;
    QwenPagedKvSequence paged_kv;
};

struct QwenBatchOperations {
    using Request = QwenBatchRequest;
    struct State {
        std::deque<std::shared_ptr<Request>> prefilling;
        std::vector<std::shared_ptr<Request>> active;
    };

    QwenBatchOperations(
            mfq::cuda::Qwen35CausalLm& model,
            CudaExecutionContext& execution,
            CudaContinuousBatchConfig config,
            int64_t prefill_chunk_size)
        : model_(model), state_adapter_(model), execution_(execution),
          config_(std::move(config)),
          max_sequences_(static_cast<int32_t>(
              config_.max_sequences)),
          prefill_chunk_size_(std::min(prefill_chunk_size, config_.prefill_token_budget)),
          moe_enabled_(state_adapter_.has_moe()),
          cached_moe_enabled_(state_adapter_.has_cached_moe()) {
        if (config_.max_sequences >
                static_cast<std::size_t>(
                    std::numeric_limits<int32_t>::max()) ||
                max_sequences_ < 1) {
            throw std::invalid_argument(
                "continuous batching max sequences must be positive");
        }
        if (prefill_chunk_size_ < 1) {
            throw std::invalid_argument(
                "continuous batching requires a positive prefill chunk");
        }
        const auto incompatibility =
            state_adapter_.incompatibility(execution_);
        if (!incompatibility.empty()) {
            throw std::runtime_error(incompatibility);
        }
        slots_.resize(static_cast<size_t>(max_sequences_));
        if (config_.paged_kv) {
            state_adapter_.enable_paged_kv(max_sequences_);
        }
    }

    QwenBatchOperations(const QwenBatchOperations&) = delete;
    QwenBatchOperations& operator=(const QwenBatchOperations&) = delete;

    std::vector<std::pair<std::string, double>> metrics() const {
        const auto paged = state_adapter_.paged_kv_stats();
        return {
            {"continuous_batching_requests",
                static_cast<double>(requests_)},
            {"continuous_batching_decode_batches",
                static_cast<double>(decode_batches_)},
            {"continuous_batching_decode_tokens",
                static_cast<double>(decode_tokens_)},

            {"continuous_batching_admissions",
                static_cast<double>(admissions_)},
            {"continuous_batching_prefill_chunks",
                static_cast<double>(prefill_chunks_)},
            {"continuous_batching_prefill_yields",
                static_cast<double>(prefill_yields_)},

            {"continuous_batching_compactions", 0.0},
            {"continuous_batching_stable_slot_releases",
                static_cast<double>(stable_slot_releases_)},
            {"continuous_batching_batched_greedy_batches",
                static_cast<double>(batched_greedy_batches_)},
            {"continuous_batching_packed_metadata_batches",
                static_cast<double>(packed_metadata_batches_)},
            {"continuous_batching_cuda_graph_captures",
                static_cast<double>(cuda_graph_captures_)},
            {"continuous_batching_cuda_graph_replays",
                static_cast<double>(cuda_graph_replays_)},
            {"continuous_batching_moe",
                moe_enabled_ ? 1.0 : 0.0},
            {"continuous_batching_moe_cached_row_serial",
                cached_moe_enabled_ ? 1.0 : 0.0},
            {"continuous_batching_paged_kv",
                state_adapter_.paged_kv_enabled() ? 1.0 : 0.0},
            {"paged_kv_page_size", static_cast<double>(paged.page_size)},
            {"paged_kv_live_pages", static_cast<double>(paged.live_pages)},
            {"paged_kv_peak_live_pages",
                static_cast<double>(paged.peak_live_pages)},
            {"paged_kv_capacity_pages",
                static_cast<double>(paged.capacity_pages)},
            {"paged_kv_reserved_bytes",
                static_cast<double>(paged.reserved_bytes)},
            {"paged_kv_page_allocations",
                static_cast<double>(paged.allocations)},
            {"paged_kv_page_reuses", static_cast<double>(paged.reuses)},
            {"paged_kv_page_releases",
                static_cast<double>(paged.releases)},
            {"paged_kv_table_updates",
                static_cast<double>(paged.table_updates)},
        };
    }

    bool paged_kv_enabled() const noexcept {
        return state_adapter_.paged_kv_enabled();
    }
    int64_t paged_kv_page_size() const noexcept {
        return state_adapter_.paged_kv_stats().page_size;
    }

    void bind_paged_requests(
            const std::vector<std::shared_ptr<Request>>& requests) {
        std::vector<const QwenPagedKvSequence*> sequences;
        sequences.reserve(requests.size());
        for (const auto& request : requests) {
            sequences.push_back(&request->paged_kv);
        }
        state_adapter_.bind_paged_requests(sequences);
    }

    int32_t acquire_slot(const std::shared_ptr<Request>& request) {
        const auto found = std::find(slots_.begin(), slots_.end(), nullptr);
        MFQ_RUNTIME_CHECK(found != slots_.end(),
            "continuous batching has no free stable slot");
        const auto slot = static_cast<int32_t>(found - slots_.begin());
        state_adapter_.move_paged_to_slot(slot, request->paged_kv);
        *found = request;
        request->slot = slot;
        return slot;
    }

    void release_slot(const std::shared_ptr<Request>& request) {
        if (request->slot < 0) {
            state_adapter_.release_paged(request->paged_kv);
            return;
        }
        const auto slot = static_cast<size_t>(request->slot);
        MFQ_RUNTIME_CHECK(slot < slots_.size() && slots_[slot] == request,
            "continuous batching stable slot ownership changed");
        slots_[slot].reset();
        state_adapter_.release_paged_slot(request->slot);
        request->slot = -1;
    }

    void release_paged_requests(
            const std::vector<std::shared_ptr<Request>> & requests) {
        for (const auto & request : requests) release_slot(request);
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

    void advance_prefills(State& state) {
        if (state.prefilling.empty()) return;
        // Swapping model state drops attention workspaces captured by decode.
        // A graph may only outlive those swaps once workspaces are slot-owned.
        invalidate_decode_graph();
        const int primary = execution_.layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        std::optional<QwenBatchState> slot_state;
        if (!state.active.empty()) {
            slot_state = state_adapter_.take(max_sequences_);
        }
        std::vector<std::shared_ptr<Request>> admitted;
        admitted.reserve(state.prefilling.size());
        int64_t tokens_advanced = 0;
        auto remaining = state.prefilling.size();
        while (remaining-- && !state.prefilling.empty() && tokens_advanced == 0) {
            auto request = std::move(state.prefilling.front());
            state.prefilling.pop_front();
            if (!request->eligible && !request->stopped()) {
                state.prefilling.push_back(std::move(request));
                continue;
            }
            if (request->stopped()) {
                try { mfq_cuda_synchronize(); } catch (...) {}
                if (state_adapter_.paged_kv_enabled()) {
                    state_adapter_.release_paged(request->paged_kv);
                    state_adapter_.detach_paged_kv();
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
                    state_adapter_.restore(
                        resumed, request->prefill_offset);
                    bind_paged_requests({request});
                } else {
                    model_.reset(1);
                    if (state_adapter_.paged_kv_enabled()) {
                        state_adapter_.ensure_request_tokens(
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
                    MfqOptional<Tensor> length = mfq_nullopt;
                    if (chunk.offset > 0 && chunk.count == 1)
                        length = mfq_tensor_backend::full({1}, chunk.offset + 1,
                            request->prefill_ids.options());
                    hidden = model_.hidden_forward(
                        request->prefill_ids.narrow(
                            1, chunk.offset, chunk.count).contiguous(), mfq_nullopt, length);
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
                    tokens_advanced += chunk.count;
                } while (false);
                if (request->stopped()) {
                    try { mfq_cuda_synchronize(); } catch (...) {}
                    if (state_adapter_.paged_kv_enabled()) {
                        state_adapter_.release_paged(request->paged_kv);
                        state_adapter_.detach_paged_kv();
                    }
                    request->prefill_timers.clear();
                    request->prefill_ids = Tensor();
                    model_.reset(1);
                    request->complete();
                    continue;
                }
                if (request->prefill_offset < request->prefill_ids.size(1)) {
                    request->prefill_state = state_adapter_.take(1);
                    state_adapter_.detach_paged_kv();
                    state.prefilling.push_back(std::move(request));
                    ++prefill_yields_;
                    request = state.prefilling.back();
                    request->publish_prefill({static_cast<std::size_t>(request->prefill_offset), 0.0, 0.0, 0.0});
                    continue;
                }
                auto logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
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
                if (request->stopped() ||
                        request->produced >= request->generation_limit) {
                    if (state_adapter_.paged_kv_enabled()) {
                        state_adapter_.release_paged(request->paged_kv);
                        state_adapter_.detach_paged_kv();
                    }
                    request->complete();
                    continue;
                }
                if (request->counts.defined()) {
                    sample_token_counts_add_cuda(
                        request->counts, next.contiguous());
                }
                auto admitted_state = state_adapter_.take(1);
                const auto slot = acquire_slot(request);
                if (!slot_state) {
                    slot_state = state_adapter_.make_slot_state(
                        admitted_state, max_sequences_);
                }
                state_adapter_.copy_to_slot(
                    *slot_state, admitted_state, slot);
                admitted.push_back(request);
                ++admissions_;
                ++requests_;
            } catch (...) {
                auto error = std::current_exception();
                try { mfq_cuda_synchronize(); } catch (...) {}
                try { release_slot(request); } catch (...) {}
                state_adapter_.detach_paged_kv();
                request->prefill_state.reset();
                request->prefill_timers.clear();
                request->prefill_ids = Tensor();
                try { model_.reset(1); } catch (...) {}
                request->complete(error);
            }
        }
        state.active.insert(
            state.active.end(), admitted.begin(), admitted.end());
        if (slot_state) {
            int64_t max_cache_position = 0;
            for (const auto& request : state.active) {
                max_cache_position = std::max(
                    max_cache_position, request->cache_length);
            }
            std::vector<QwenBatchState> states;
            states.push_back(std::move(*slot_state));
            state_adapter_.restore(
                states, max_cache_position);
            state_adapter_.bind_paged_slots();
        } else {
            model_.reset(1);
            state_adapter_.detach_paged_kv();
        }
    }

    void retire_cancelled_requests(State& state) {
        std::vector<std::shared_ptr<Request>> survivors;
        std::vector<std::shared_ptr<Request>> cancelled;
        int64_t survivor_max_position = 0;
        survivors.reserve(state.active.size());
        for (size_t row = 0; row < state.active.size(); ++row) {
            const auto& request = state.active[row];
            if (request->stopped()) {
                cancelled.push_back(request);
                continue;
            }
            survivors.push_back(request);
            survivor_max_position = std::max(
                survivor_max_position, request->cache_length);
        }
        if (survivors.size() == state.active.size()) return;
        release_paged_requests(cancelled);
        stable_slot_releases_ += static_cast<int64_t>(cancelled.size());
        if (survivors.empty()) {
            invalidate_decode_graph();
            model_.reset(1);
            state_adapter_.release_idle_paged_slots();
            state_adapter_.detach_paged_kv();
        } else {
            model_.cache_pos = survivor_max_position;
            state_adapter_.bind_paged_slots();
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
        for (auto& graph : decode_graphs_) graph->invalidate();
        decode_graphs_.clear();
    }

    void decode_active(State& state) {
        if (state.active.empty()) return;
        const int primary = execution_.layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        retire_cancelled_requests(state);
        if (state.active.empty()) return;
        if (state_adapter_.paged_kv_enabled()) {
            bool page_table_changed = false;
            for (const auto& request : state.active) {
                page_table_changed = state_adapter_.ensure_slot_tokens(
                    request->slot, request->cache_length + 1) ||
                    page_table_changed;
            }
            if (page_table_changed) state_adapter_.bind_paged_slots();
        }
        const int64_t batch = max_sequences_;
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
            graph_state_addresses = state_adapter_.decode_state_addresses();
            const auto found = std::find_if(
                decode_graphs_.begin(), decode_graphs_.end(),
                [&](const auto& graph) {
                    return graph->matches(
                        batch, planned_len, graph_state_addresses);
                });
            graph_cache_hit = found != decode_graphs_.end();
            graph_decode = graph_cache_hit || minimum_remaining >=
                config_.cuda_graph_minimum_tokens;
        }
        QwenContinuousDecodeGraph* decode_graph = nullptr;
        if (graph_cache_hit) {
            decode_graph = std::find_if(
                decode_graphs_.begin(), decode_graphs_.end(),
                [&](const auto& graph) {
                    return graph->matches(
                        batch, planned_len, graph_state_addresses);
                })->get();
        } else if (graph_decode) {
            if (decode_graphs_.size() >= 8) invalidate_decode_graph();
            decode_graphs_.push_back(
                std::make_unique<QwenContinuousDecodeGraph>());
            decode_graph = decode_graphs_.back().get();
        }
        std::unique_ptr<MfqCudaStreamGuard> graph_stream_guard;
        std::vector<std::unique_ptr<MfqCudaStreamGuard>>
            graph_compute_stream_guards;
        if (graph_decode) {
            const auto& parallel = execution_.tensor_parallel.enabled()
                ? execution_.tensor_parallel
                : execution_.expert_parallel;
            decode_graph->ensure_compute_streams(parallel);
            graph_stream_guard = std::make_unique<MfqCudaStreamGuard>(
                decode_graph->stream);
            graph_compute_stream_guards =
                activate_cuda_graph_compute_streams(
                    decode_graph->compute_streams);
        }
        ensure_decode_metadata_buffers(primary);
        auto* packed_metadata_data =
            decode_metadata_host_.data_ptr<int64_t>();
        std::fill_n(packed_metadata_data, batch, int64_t{0});
        std::fill_n(packed_metadata_data + batch, batch, int64_t{0});
        std::fill_n(packed_metadata_data + 2 * batch, batch, int64_t{1});
        int64_t max_position = 0;
        int64_t max_sequence_length = 0;
        for (const auto& request : state.active) {
            const auto slot = static_cast<int64_t>(request->slot);
            MFQ_RUNTIME_CHECK(slot >= 0 && slot < batch,
                "continuous batching request lost its stable slot");
            packed_metadata_data[slot] = request->pending_token;
            packed_metadata_data[batch + slot] = request->cache_length;
            packed_metadata_data[2 * batch + slot] =
                request->cache_length + 1;
            max_position = std::max(max_position, request->cache_length);
            max_sequence_length = std::max(
                max_sequence_length, request->cache_length + 1);
        }
        const int64_t values = 3 * batch;
        auto metadata_host = decode_metadata_host_.narrow(0, 0, values);
        auto metadata_cuda = decode_metadata_cuda_.narrow(0, 0, values);
        metadata_cuda.copy_(metadata_host);
        auto matrix = metadata_cuda.reshape({3, batch});
        Tensor ids = matrix.narrow(0, 0, 1).reshape({batch, 1});
        Tensor pos = matrix.narrow(0, 1, 1).reshape({batch, 1});
        Tensor lengths = matrix.narrow(0, 2, 1).reshape({batch});
        ++packed_metadata_batches_;
        std::vector<int32_t> paused_slots;
        for (const auto& request : state.active)
            if (!request->eligible) paused_slots.push_back(request->slot);
        // ponytail: masked slots still occupy GPU rows; restore recurrent state
        // and overwrite the uncommitted KV position when they become eligible.
        const auto paused_state = state_adapter_.capture_recurrent_slots(paused_slots);
        Tensor logits;
        Tensor graph_tokens;
        MoeContinuousBatchCacheScope moe_cache_scope(
            execution_, cached_moe_enabled_ && batch > 1);
        try {
            model_.cache_pos = max_position;
            if (graph_decode) {
                const auto invoke = [&]() {
                    auto hidden = model_.hidden_forward_static(
                        ids, pos, lengths, planned_len,
                        graph_attention_parts);
                    auto current_logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
                    return sample_greedy_cuda(
                        current_logits.contiguous().view({batch, -1}));
                };
                if (!graph_cache_hit) {
                    decode_graph->invalidate();
                    try {
                        DecodeGraphBranchScope branch_scope(
                            execution_.decode_graph_serial_branches);
                        DecodeGraphTpProjectionScope tp_projection_scope(
                            execution_.decode_graph_tp_projection_major);
                        decode_graph->graph =
                            std::make_unique<MfqCudaGraph>();
                        prepare_decode_graph_memory(
                            model_, *decode_graph->graph,
                            [&]() { (void)invoke(); },
                            decode_graph->participant_streams(
                                execution_.model_parallel_collectives));
                        decode_graph->graph->capture_begin();
                        decode_graph->static_next = invoke();
                        decode_graph->graph->capture_end();
                        decode_graph->set_key(
                            batch, planned_len,
                            state_adapter_.decode_state_addresses());
                        ++cuda_graph_captures_;
                    } catch (...) {
                        decode_graph->invalidate();
                        throw;
                    }
                }
                decode_graph->graph->replay();
                graph_tokens = decode_graph->static_next;
                ++cuda_graph_replays_;
            } else {
                auto hidden = model_.hidden_forward(
                    ids, pos, lengths, nullptr, pos);
                logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
            }
            model_.cache_pos = max_sequence_length;
        } catch (...) {
            auto error = std::current_exception();
            invalidate_decode_graph();
            try { model_.reset(1); } catch (...) {}
            fail_requests(state.active, error);
            release_paged_requests(state.active);
            state_adapter_.release_idle_paged_slots();
            state_adapter_.detach_paged_kv();
            state.active.clear();
            return;
        }
        state_adapter_.restore_recurrent_slots(paused_slots, paused_state);
        ++decode_batches_;
        decode_tokens_ += std::count_if(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->eligible; });
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
        survivors.reserve(state.active.size());
        int64_t survivor_max_position = 0;
        for (const auto& request : state.active) {
            if (!request->eligible) {
                survivors.push_back(request);
                survivor_max_position = std::max(survivor_max_position, request->cache_length);
                continue;
            }
            request->cache_length += 1;
            if (request->stopped()) {
                completions.emplace_back(request, std::exception_ptr{});
                continue;
            }
            try {
                Tensor next;
                int64_t token = 0;
                const auto slot = static_cast<int64_t>(request->slot);
                if (batched_greedy_data != nullptr) {
                    token = batched_greedy_data[slot];
                } else {
                    next = mfq::cuda::sample_logits(
                        *request->sampler,
                        logits.narrow(0, slot, 1),
                        request->counts, request->token_constraint);
                    token = next.item<int64_t>();
                }
                request->pending_token = token;
                ++request->produced;
                request->publish_token(token);
                if (request->stopped() ||
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
            stable_slot_releases_ += static_cast<int64_t>(state.active.size());
            state_adapter_.release_idle_paged_slots();
            state_adapter_.detach_paged_kv();
        } else {
            if (survivors.size() != state.active.size()) {
                std::vector<std::shared_ptr<Request>> retired;
                retired.reserve(completions.size());
                for (const auto& completion : completions) {
                    retired.push_back(completion.first);
                }
                release_paged_requests(retired);
                stable_slot_releases_ += static_cast<int64_t>(retired.size());
                state_adapter_.bind_paged_slots();
            }
            model_.cache_pos = survivor_max_position;
        }
        state.active = std::move(survivors);
        for (const auto& completion : completions) {
            completion.first->complete(completion.second);
        }
    }

    void recover(State& state, std::exception_ptr error) {
        std::vector<std::shared_ptr<Request>> prefilling(
            state.prefilling.begin(), state.prefilling.end());
        fail_requests(prefilling, error);
        fail_requests(state.active, error);
        try { mfq_cuda_synchronize(); } catch (...) {}
        release_paged_requests(prefilling);
        release_paged_requests(state.active);
        state_adapter_.release_idle_paged_slots();
        model_.reset(1);
        state_adapter_.detach_paged_kv();
    }

    Qwen35CausalLm& model_;
    Qwen35BatchStateAdapter state_adapter_;
    CudaExecutionContext& execution_;
    const CudaContinuousBatchConfig config_;
    int32_t max_sequences_ = 0;
    int64_t prefill_chunk_size_ = 2048;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    int64_t requests_{0};
    int64_t decode_batches_{0};
    int64_t decode_tokens_{0};
    int64_t admissions_{0};
    int64_t prefill_chunks_{0};
    int64_t prefill_yields_{0};
    int64_t stable_slot_releases_{0};
    int64_t batched_greedy_batches_{0};
    int64_t packed_metadata_batches_{0};
    int64_t cuda_graph_captures_{0};
    int64_t cuda_graph_replays_{0};
    Tensor decode_metadata_host_;
    Tensor decode_metadata_cuda_;
    std::vector<std::shared_ptr<Request>> slots_;
    std::vector<std::unique_ptr<QwenContinuousDecodeGraph>> decode_graphs_;
};

struct QwenBatchExecutor::Impl {
    QwenBatchOperations operations;
    QwenBatchOperations::State state;
    std::unordered_map<std::string, std::shared_ptr<QwenBatchRequest>> requests;
    bool decode_next = true;

    Impl(Qwen35CausalLm& model, CudaExecutionContext& execution,
         const CudaContinuousBatchConfig& config, int64_t chunk)
        : operations(model, execution, config, chunk) {}
};

QwenBatchExecutor::QwenBatchExecutor(Qwen35CausalLm& model, CudaExecutionContext& execution,
        const CudaContinuousBatchConfig& config, int64_t chunk)
    : impl_(std::make_unique<Impl>(model, execution, config, chunk)) {}
QwenBatchExecutor::~QwenBatchExecutor() = default;

void QwenBatchExecutor::admit(std::string id, const mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output) {
    auto& impl = *impl_;
    const auto plan = mfq::engine::plan_generation(input.prompt,
        impl.operations.model_.vocab_size(), impl.operations.model_.max_position_embeddings(),
        input.sampling.max_tokens);
    auto request = std::make_shared<QwenBatchRequest>(std::move(id), input, output);
    request->generation_limit = plan.generation_tokens;
    impl.requests.emplace(request->id, request);
    impl.state.prefilling.push_back(std::move(request));
}

mfq::engine::EngineStepResult QwenBatchExecutor::step(const std::vector<std::string>& eligible) {
    using namespace mfq::engine;
    auto& impl = *impl_;
    EngineStepResult result;
    for (auto& [id, request] : impl.requests)
        request->eligible = std::find(eligible.begin(), eligible.end(), id) != eligible.end();
    try {
        impl.operations.retire_cancelled_requests(impl.state);
        const bool decode = std::any_of(impl.state.active.begin(), impl.state.active.end(),
            [](const auto& request) { return request->eligible; });
        const bool prefill = std::any_of(impl.state.prefilling.begin(), impl.state.prefilling.end(),
            [](const auto& request) { return request->eligible || request->stopped(); });
        if (decode && (!prefill || impl.decode_next))
            impl.operations.decode_active(impl.state);
        else if (prefill) impl.operations.advance_prefills(impl.state);
        impl.decode_next = !impl.decode_next;
    } catch (...) {
        impl.operations.recover(impl.state, std::current_exception());
        impl.state = {};
    }
    for (auto it = impl.requests.begin(); it != impl.requests.end();) {
        auto& request = *it->second;
        if (!request.events.empty()) result.advanced.push_back(it->first);
        for (auto& event : request.events) result.events.push_back({it->first, std::move(event)});
        request.events.clear();
        if (!request.done) { ++it; continue; }
        if (request.error) {
            std::string message = "CUDA batch execution failed";
            try { std::rethrow_exception(request.error); }
            catch (const std::exception& e) { message = e.what(); } catch (...) {}
            result.events.push_back({it->first, Failed{"backend_failure", std::move(message)}});
        } else {
            auto delta = request.output.finish();
            if (!delta.diffs.empty()) result.events.push_back({it->first, std::move(delta)});
            if (request.output.result.cancelled)
                result.events.push_back({it->first, Cancelled{{request.prompt.size(),
                    static_cast<std::size_t>(request.output.result.completion_tokens)}, request.output.metrics}});
            else result.events.push_back({it->first, Completed{request.output.result.finish_reason,
                {request.prompt.size(), static_cast<std::size_t>(request.output.result.completion_tokens)}, request.output.metrics}});
        }
        it = impl.requests.erase(it);
    }
    result.status.available = impl.operations.max_sequences_ - impl.requests.size();
    result.has_work = !impl.requests.empty();
    return result;
}

std::vector<std::pair<std::string, double>> QwenBatchExecutor::metrics() const {
    auto result = impl_->operations.metrics();
    result.emplace_back("continuous_batching_active", impl_->state.active.size());
    result.emplace_back("continuous_batching_prefilling", impl_->state.prefilling.size());
    return result;
}
} // namespace mfq::cuda
