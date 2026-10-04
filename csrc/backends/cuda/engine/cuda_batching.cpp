#include "../kernels/mfq_cuda_sampling_ops.h"
#include "cuda_batching.h"

#include "../ops/cuda_execution.h"
#include "engine/decode_graph.h"
#include "models/common/full_block.h"
#include "models/qwen35/ops.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda {

using internal::PrefillCudaTimer;
using Tensor = mfq_tensor_backend::Tensor;
using qwen35::Qwen35BatchStateAdapter;
using qwen35::QwenBatchRequestState;

bool qwen_continuous_batch_cuda_graph_enabled(const Qwen35CausalLm &model,
                                              const CudaContinuousBatchConfig &config) {
    const auto &execution = *model.execution;
    return config.cuda_graph && !Qwen35BatchStateAdapter::has_cached_moe(model) &&
           mfq_cuda_graph_capture_supported() && model_parallel_cuda_graph_enabled(execution);
}

class MoeContinuousBatchCacheScope {
  public:
    MoeContinuousBatchCacheScope(CudaExecutionContext &execution, bool enabled)
        : execution_(execution), previous_(execution.continuous_batch_cache_serial) {
        execution_.continuous_batch_cache_serial = enabled;
    }

    ~MoeContinuousBatchCacheScope() { execution_.continuous_batch_cache_serial = previous_; }

    MoeContinuousBatchCacheScope(const MoeContinuousBatchCacheScope &) = delete;
    MoeContinuousBatchCacheScope &operator=(const MoeContinuousBatchCacheScope &) = delete;

  private:
    CudaExecutionContext &execution_;
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

    QwenContinuousDecodeGraph() : stream(mfq_get_stream_from_pool(false)) {}

    void ensure_compute_streams(const ParallelConfig &parallel) {
        if (compute_streams.empty()) {
            compute_streams = make_cuda_graph_compute_streams(stream, parallel);
        }
    }

    std::vector<MfqCudaStream>
    participant_streams(const ModelParallelCollectiveRuntime &collectives) const {
        return cuda_graph_participant_streams(compute_streams, collectives);
    }

    bool matches(int64_t candidate_batch, int64_t candidate_len,
                 const std::vector<const void *> &candidate_addresses) const {
        return valid && batch == candidate_batch && planned_len == candidate_len &&
               state_addresses == candidate_addresses;
    }

    void set_key(int64_t candidate_batch, int64_t candidate_len,
                 std::vector<const void *> candidate_addresses) {
        batch = candidate_batch;
        planned_len = candidate_len;
        state_addresses = std::move(candidate_addresses);
        valid = true;
    }

    void invalidate() {
        if (graph)
            graph->reset();
        graph.reset();
        static_next = Tensor();
        state_addresses.clear();
        valid = false;
    }
};

QwenBatchOperations::QwenBatchOperations(Qwen35CausalLm &model, CudaExecutionContext &execution,
                                         CudaContinuousBatchConfig config)
    : model_(model), state_adapter_(model, config.max_sequences, config.paged_kv),
      execution_(execution), config_(std::move(config)),
      max_sequences_(static_cast<int32_t>(config_.max_sequences)),
      moe_enabled_(state_adapter_.has_moe()),
      cached_moe_enabled_(state_adapter_.has_cached_moe()) {}

QwenBatchOperations::~QwenBatchOperations() = default;

int64_t QwenBatchOperations::vocab_size() const { return model_.vocab_size(); }

int64_t QwenBatchOperations::max_context() const { return model_.max_position_embeddings(); }

std::vector<std::pair<std::string, double>> QwenBatchOperations::metrics() const {
    const auto paged = state_adapter_.paged_kv_stats();
    return {
        {"continuous_batching_decode_batches", static_cast<double>(decode_batches_)},
        {"continuous_batching_decode_tokens", static_cast<double>(decode_tokens_)},

        {"continuous_batching_compactions", 0.0},
        {"continuous_batching_stable_slot_releases",
         static_cast<double>(state_adapter_.slot_releases())},
        {"continuous_batching_batched_greedy_batches",
         static_cast<double>(batched_greedy_batches_)},
        {"continuous_batching_packed_metadata_batches",
         static_cast<double>(packed_metadata_batches_)},
        {"continuous_batching_cuda_graph_captures", static_cast<double>(cuda_graph_captures_)},
        {"continuous_batching_cuda_graph_replays", static_cast<double>(cuda_graph_replays_)},
        {"continuous_batching_moe", moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_moe_cached_row_serial", cached_moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_paged_kv", state_adapter_.paged_kv_enabled() ? 1.0 : 0.0},
        {"paged_kv_page_size", static_cast<double>(paged.page_size)},
        {"paged_kv_live_pages", static_cast<double>(paged.live_pages)},
        {"paged_kv_peak_live_pages", static_cast<double>(paged.peak_live_pages)},
        {"paged_kv_capacity_pages", static_cast<double>(paged.capacity_pages)},
        {"paged_kv_reserved_bytes", static_cast<double>(paged.reserved_bytes)},
        {"paged_kv_page_allocations", static_cast<double>(paged.allocations)},
        {"paged_kv_page_reuses", static_cast<double>(paged.reuses)},
        {"paged_kv_page_releases", static_cast<double>(paged.releases)},
        {"paged_kv_table_updates", static_cast<double>(paged.table_updates)},
    };
}

void QwenBatchOperations::initialize_sampling(Request &request, const Tensor &prompt_ids) {
    request.sampler.emplace(request.sampling);
    if (request.sampler->has_penalties())
        request.counts = SamplingOps::token_counts(prompt_ids, model_.vocab_size());
}

void QwenBatchOperations::suspend_decode() {
    invalidate_decode_graph();
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    state_adapter_.suspend_decode();
}

QwenBatchOperations::Sample QwenBatchOperations::prefill(
    const std::shared_ptr<Request> &request, mfq::engine::PrefillChunk chunk) {
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard guard(primary);
    state_adapter_.prepare_prefill(request->cache, chunk.offset, request->prompt.size());
    if (chunk.offset == 0) {
        request->prefill_ids =
            mfq_tensor_backend::tensor(
                request->prompt,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, primary)))
                .reshape({1, -1})
                .contiguous();
        initialize_sampling(*request, request->prefill_ids);
    }
    Tensor hidden;
    std::unique_ptr<PrefillCudaTimer> final_timer;
    auto chunk_timer = std::make_unique<PrefillCudaTimer>();
    MfqOptional<Tensor> length = mfq_nullopt;
    if (chunk.offset > 0 && chunk.count == 1)
        length =
            mfq_tensor_backend::full({1}, chunk.offset + 1, request->prefill_ids.options());
    hidden = model_.hidden_forward(
        request->prefill_ids.narrow(1, chunk.offset, chunk.count).contiguous(), mfq_nullopt,
        length);
    const auto end = chunk.offset + chunk.count;
    if (end < request->prefill_ids.size(1)) {
        MFQ_CUDA_CHECK(
            cudaEventRecord(chunk_timer->finished_event(), mfq_get_current_cuda_stream()));
        request->prefill_timers.push_back(std::move(chunk_timer));
    } else {
        final_timer = std::move(chunk_timer);
    }
    if (end < static_cast<int64_t>(request->prompt.size())) {
        state_adapter_.pause_prefill(request->cache);
        return {};
    }
    auto logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
    MFQ_RUNTIME_CHECK(final_timer != nullptr, "continuous batching lost the prefill timer");
    MFQ_CUDA_CHECK(
        cudaEventRecord(final_timer->finished_event(), mfq_get_current_cuda_stream()));
    request->prefill_timers.push_back(std::move(final_timer));
    auto next = mfq::cuda::sample_logits(*request->sampler, std::move(logits), request->counts,
                                         request->token_constraint);
    const int64_t token = next.item<int64_t>();
    double prefill_ms = 0.0;
    for (const auto &timer : request->prefill_timers) {
        prefill_ms += timer->elapsed_ms();
    }
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
    return {std::move(next), token, {request->prompt.size(), prefill_ms, 0.0, prefill_ms}};
}

void QwenBatchOperations::activate(const std::shared_ptr<Request> &request, const Sample &sample) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    if (request->counts.defined())
        sample_token_counts_add_cuda(request->counts, sample.next.contiguous());
    state_adapter_.activate(request->cache);
}

void QwenBatchOperations::resume_decode(int64_t cache_position) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    state_adapter_.resume_decode(cache_position);
}

void QwenBatchOperations::discard_prefill(const std::shared_ptr<Request> &request) {
    state_adapter_.discard_prefill(request->cache);
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
}

void QwenBatchOperations::retire(const std::vector<std::shared_ptr<Request>> &retired,
                                 int64_t cache_position) {
    for (const auto &request : retired)
        state_adapter_.release(request->cache);
    if (cache_position == 0)
        invalidate_decode_graph();
    state_adapter_.finish_retire(cache_position);
}

void QwenBatchOperations::ensure_decode_metadata_buffers(int primary) {
    if (decode_metadata_host_.defined())
        return;
    const int64_t capacity = 3 * static_cast<int64_t>(max_sequences_);
    decode_metadata_host_ =
        mfq_tensor_backend::empty({capacity}, mfq_tensor_backend::TensorOptions()
                                                  .dtype(mfq_tensor_backend::kInt64)
                                                  .device(mfq_tensor_backend::kCPU)
                                                  .pinned_memory(true));
    decode_metadata_cuda_ = mfq_tensor_backend::empty(
        {capacity},
        mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64)
            .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, primary)));
}

void QwenBatchOperations::invalidate_decode_graph() {
    for (auto &graph : decode_graphs_)
        graph->invalidate();
    decode_graphs_.clear();
}

QwenBatchOperations::Decoded QwenBatchOperations::decode(State &state) {
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard primary_guard(primary);
    for (const auto &request : state.active)
        state_adapter_.ensure_decode_tokens(request->cache, request->cache_length + 1);
    int64_t max_position = 0;
    for (const auto &request : state.active)
        max_position = std::max(max_position, request->cache_length);
    state_adapter_.prepare_decode(max_position);
    const int64_t batch = max_sequences_;
    const bool batch_greedy = std::all_of(
        state.active.begin(), state.active.end(), [](const std::shared_ptr<Request> &request) {
            return request->sampler->greedy() && !request->sampler->has_penalties() &&
                   !request->token_constraint;
        });
    int64_t requested_len = 0;
    int64_t minimum_remaining =
        state.active.front()->generation_limit - state.active.front()->produced;
    for (const auto &request : state.active) {
        const int64_t remaining = request->generation_limit - request->produced;
        minimum_remaining = std::min(minimum_remaining, remaining);
        requested_len = std::max(requested_len, request->cache_length + remaining);
    }
    const int64_t planned_len =
        decode_graph_bucket(requested_len, model_.max_position_embeddings());
    const int64_t graph_attention_parts =
        decode_graph_attention_parts(planned_len, FullBlock::kDecodeAttentionMaxParts);
    std::vector<const void *> graph_state_addresses;
    bool graph_cache_hit = false;
    bool graph_decode = false;
    if (batch >= 2 && batch_greedy && config_.greedy &&
        qwen_continuous_batch_cuda_graph_enabled(model_, config_)) {
        graph_state_addresses = state_adapter_.decode_state_addresses();
        const auto found =
            std::find_if(decode_graphs_.begin(), decode_graphs_.end(), [&](const auto &graph) {
                return graph->matches(batch, planned_len, graph_state_addresses);
            });
        graph_cache_hit = found != decode_graphs_.end();
        graph_decode =
            graph_cache_hit || minimum_remaining >= config_.cuda_graph_minimum_tokens;
    }
    QwenContinuousDecodeGraph *decode_graph = nullptr;
    if (graph_cache_hit) {
        decode_graph =
            std::find_if(decode_graphs_.begin(), decode_graphs_.end(), [&](const auto &graph) {
                return graph->matches(batch, planned_len, graph_state_addresses);
            })->get();
    } else if (graph_decode) {
        if (decode_graphs_.size() >= 8)
            invalidate_decode_graph();
        decode_graphs_.push_back(std::make_unique<QwenContinuousDecodeGraph>());
        decode_graph = decode_graphs_.back().get();
    }
    std::unique_ptr<MfqCudaStreamGuard> graph_stream_guard;
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> graph_compute_stream_guards;
    if (graph_decode) {
        const auto &parallel = execution_.tensor_parallel.enabled()
                                   ? execution_.tensor_parallel
                                   : execution_.expert_parallel;
        decode_graph->ensure_compute_streams(parallel);
        graph_stream_guard = std::make_unique<MfqCudaStreamGuard>(decode_graph->stream);
        graph_compute_stream_guards =
            activate_cuda_graph_compute_streams(decode_graph->compute_streams);
    }
    ensure_decode_metadata_buffers(primary);
    auto *packed_metadata_data = decode_metadata_host_.data_ptr<int64_t>();
    std::fill_n(packed_metadata_data, batch, int64_t{0});
    std::fill_n(packed_metadata_data + batch, batch, int64_t{0});
    std::fill_n(packed_metadata_data + 2 * batch, batch, int64_t{1});
    int64_t max_sequence_length = 0;
    for (const auto &request : state.active) {
        const auto slot = static_cast<int64_t>(request->cache.slot());
        MFQ_RUNTIME_CHECK(slot >= 0 && slot < batch,
                          "continuous batching request lost its stable slot");
        packed_metadata_data[slot] = request->pending_token;
        packed_metadata_data[batch + slot] = request->cache_length;
        packed_metadata_data[2 * batch + slot] = request->cache_length + 1;
        max_sequence_length = std::max(max_sequence_length, request->cache_length + 1);
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
    for (const auto &request : state.active)
        if (!request->eligible)
            paused_slots.push_back(request->cache.slot());
    // ponytail: masked slots still occupy GPU rows; restore recurrent state
    // and overwrite the uncommitted KV position when they become eligible.
    const auto paused_state = state_adapter_.capture_recurrent_slots(paused_slots);
    Tensor logits;
    Tensor graph_tokens;
    MoeContinuousBatchCacheScope moe_cache_scope(execution_, cached_moe_enabled_ && batch > 1);
    try {
        if (graph_decode) {
            const auto invoke = [&]() {
                auto hidden = model_.hidden_forward_static(
                    ids, pos, lengths, {planned_len, graph_attention_parts});
                auto current_logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
                return sample_greedy_cuda(current_logits.contiguous().view({batch, -1}));
            };
            if (!graph_cache_hit) {
                decode_graph->invalidate();
                try {
                    DecodeGraphBranchScope branch_scope(
                        execution_.decode_graph_serial_branches);
                    DecodeGraphTpProjectionScope tp_projection_scope(
                        execution_.decode_graph_tp_projection_major);
                    decode_graph->graph = std::make_unique<MfqCudaGraph>();
                    prepare_decode_graph_memory(
                        model_, *decode_graph->graph, [&]() { (void)invoke(); },
                        decode_graph->participant_streams(
                            execution_.model_parallel_collectives));
                    decode_graph->graph->capture_begin();
                    decode_graph->static_next = invoke();
                    decode_graph->graph->capture_end();
                    decode_graph->set_key(batch, planned_len,
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
            auto hidden = model_.hidden_forward(ids, pos, lengths, nullptr, pos);
            logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
        }
        state_adapter_.finish_decode(max_sequence_length);
    } catch (...) {
        invalidate_decode_graph();
        throw;
    }
    state_adapter_.restore_recurrent_slots(paused_slots, paused_state);
    ++decode_batches_;
    decode_tokens_ += std::count_if(state.active.begin(), state.active.end(),
                                    [](const auto &request) { return request->eligible; });
    Tensor batched_greedy_tokens;
    if (graph_tokens.defined()) {
        batched_greedy_tokens = graph_tokens.to(mfq_tensor_backend::kCPU).contiguous();
        MFQ_RUNTIME_CHECK(batched_greedy_tokens.scalar_type() == mfq_tensor_backend::kInt64 &&
                              batched_greedy_tokens.numel() == batch,
                          "continuous batching CUDA Graph sampler returned the wrong shape");
        ++batched_greedy_batches_;
    } else if (config_.greedy && batch_greedy) {
        batched_greedy_tokens = sample_greedy_cuda(logits.contiguous().view({batch, -1}))
                                    .to(mfq_tensor_backend::kCPU)
                                    .contiguous();
        MFQ_RUNTIME_CHECK(batched_greedy_tokens.scalar_type() == mfq_tensor_backend::kInt64 &&
                              batched_greedy_tokens.numel() == batch,
                          "continuous batching greedy sampler returned the wrong shape");
        ++batched_greedy_batches_;
    }
    return {std::move(logits), std::move(batched_greedy_tokens)};
}

QwenBatchOperations::Sample QwenBatchOperations::sample(
    const std::shared_ptr<Request> &request, const Decoded &decoded) {
    const auto slot = static_cast<int64_t>(request->cache.slot());
    if (decoded.greedy_tokens.defined())
        return {{}, decoded.greedy_tokens.data_ptr<int64_t>()[slot], {}};
    auto next = mfq::cuda::sample_logits(*request->sampler, decoded.logits.narrow(0, slot, 1),
                                         request->counts, request->token_constraint);
    const auto token = next.item<int64_t>();
    return {std::move(next), token, {}};
}

void QwenBatchOperations::accept(const std::shared_ptr<Request> &request, const Sample &sample) {
    if (request->counts.defined()) {
        MFQ_RUNTIME_CHECK(sample.next.defined(),
                          "batched greedy sampling cannot update token penalties");
        sample_token_counts_add_cuda(request->counts, sample.next.contiguous());
    }
}

void QwenBatchOperations::recover(State &state) {
    std::vector<QwenBatchRequestState *> caches;
    for (const auto &request : state.prefilling)
        caches.push_back(&request->cache);
    for (const auto &request : state.active)
        caches.push_back(&request->cache);
    state_adapter_.recover(caches);
    invalidate_decode_graph();
}
} // namespace mfq::cuda
