#pragma once

#include "cuda_runtime_config.h"

#include "kernels/sampling_ops.h"
#include "ops/execution/cuda_execution.h"
#include "models/common/full_block.h"
#include "storage/state/batch_paged.h"
#include "driver.h"
#include "decode_graph.h"
#include "step.h"
#include "ops/execution/cuda_sampling.h"
#include <map>
#include <span>

namespace mfq::cuda {
namespace internal { class TextSessionCache; }

// Request compatibility is independent of the model's native state layout.
bool batch_requires_owned_generation(const mfq::engine::InferenceRequest&,
    const internal::TextSessionCache&, bool predictor);
bool batch_cuda_graph_enabled(const CudaExecutionContext&, const CudaContinuousBatchConfig&,
                              bool cached_moe);
void wait_for_batch_stream(const MfqCudaStream& destination, const MfqCudaStream& source);

struct BatchDecodeGraph {
    MfqCudaStream stream = mfq_get_stream_from_pool(false);
    std::vector<MfqCudaStream> compute_streams;
    std::unique_ptr<MfqCudaGraph> graph;
    mfq_tensor_backend::Tensor static_next;
    std::vector<const void*> state_addresses;
    int64_t batch = 0, planned_len = 0;
    bool valid = false;

    void ensure_compute_streams(const ParallelConfig&);
    std::vector<MfqCudaStream> participant_streams(const ModelParallelCollectiveRuntime&) const;
    bool matches(int64_t batch, int64_t planned_len, const std::vector<const void*>&) const;
    void set_key(int64_t batch, int64_t planned_len, std::vector<const void*>);
    void invalidate();
};

class MoeBatchCacheScope {
  public:
    MoeBatchCacheScope(CudaExecutionContext& execution, bool enabled)
        : execution_(execution), previous_(execution.continuous_batch_cache_serial) {
        execution_.continuous_batch_cache_serial = enabled;
    }
    ~MoeBatchCacheScope() { execution_.continuous_batch_cache_serial = previous_; }
    MoeBatchCacheScope(const MoeBatchCacheScope&) = delete;
    MoeBatchCacheScope& operator=(const MoeBatchCacheScope&) = delete;
  private:
    CudaExecutionContext& execution_;
    bool previous_;
};

struct BatchDecodeRow { int64_t token, position; };
struct BatchDecodeInputs {
    mfq_tensor_backend::Tensor ids, positions, lengths;
    int64_t max_length = 0;
};
class BatchDecodeMetadata {
  public:
    explicit BatchDecodeMetadata(int64_t capacity) : capacity_(capacity) {}
    BatchDecodeInputs pack(std::span<const BatchDecodeRow>, int64_t physical_rows, int device);
  private:
    int64_t capacity_;
    mfq_tensor_backend::Tensor host_, device_;
};

struct BatchSamplingRow {
    Sampler& sampler;
    mfq_tensor_backend::Tensor counts;
    const MfqTokenConstraintPtr& constraint;
};
struct BatchTokens {
    mfq_tensor_backend::Tensor device, host;
};
bool batch_sampling_is_greedy(std::span<const BatchSamplingRow>);
BatchTokens sample_cuda_batch(mfq_tensor_backend::Tensor logits,
    mfq_tensor_backend::Tensor greedy_tokens, std::span<const BatchSamplingRow>,
    bool greedy, int64_t& readbacks);

template <class Model> struct CudaBatchRequest : mfq::engine::BatchRequest {
    using BatchRequest::BatchRequest;
    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor pending, counts, prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    typename Model::BatchRequestState cache;
};

// Reusable CUDA numerical quanta. The model and its native state are bound at
// Engine composition; neither the shared batch policy nor this module selects
// a family. Other native executions can implement the quantum contract directly.
template <class Model> class CudaBatchExecution {
  public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Request = CudaBatchRequest<Model>;
    using Sample = mfq::engine::BatchSample;
    CudaBatchExecution(Model&, CudaExecutionContext&, CudaContinuousBatchConfig);
    CudaBatchExecution(const CudaBatchExecution&) = delete;
    CudaBatchExecution& operator=(const CudaBatchExecution&) = delete;
    mfq::engine::Metrics metrics() const;
    Sample execute(mfq::engine::PrefillWork<Request>);
    std::optional<std::vector<Sample>> execute(mfq::engine::DecodeWork<Request>);
    void accept(const std::shared_ptr<Request>&, const Sample&);
    void release(const std::shared_ptr<Request>&);
    void recover(std::span<const std::shared_ptr<Request>>);

  private:
    void initialize_sampling(Request&, const Tensor& prompt_ids);
    void invalidate_decode_graph();

    Model& model_;
    typename Model::BatchStorage native_;
    CudaExecutionContext& execution_;
    const CudaContinuousBatchConfig config_;
    int32_t max_sequences_ = 0;
    bool moe_enabled_ = false, cached_moe_enabled_ = false;
    int64_t decode_batches_ = 0, decode_tokens_ = 0, physical_decode_rows_ = 0;
    int64_t batched_greedy_batches_ = 0, packed_metadata_batches_ = 0, sampling_readbacks_ = 0;
    int64_t cuda_graph_captures_ = 0, cuda_graph_replays_ = 0, cuda_graph_prewarms_ = 0;
    std::map<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>> graph_bucket_counts_;
    BatchDecodeMetadata metadata_;
    std::vector<std::unique_ptr<BatchDecodeGraph>> decode_graphs_;
};
} // namespace mfq::cuda

namespace mfq::cuda {

template <class Model>
CudaBatchExecution<Model>::CudaBatchExecution(
        Model& model, CudaExecutionContext& execution, CudaContinuousBatchConfig config)
    : model_(model), native_(model, config.max_sequences, config.paged_kv),
      execution_(execution), config_(std::move(config)),
      max_sequences_(static_cast<int32_t>(config_.max_sequences)),
      moe_enabled_(model.num_experts() > 0),
      cached_moe_enabled_(moe_enabled_ && bool(execution.moe_expert_cache)),
      metadata_(max_sequences_) {}

template <class Model>
mfq::engine::Metrics CudaBatchExecution<Model>::metrics() const {
    const auto storage = native_.stats();
    const auto paged = storage.paged.value_or(PagedKvStats{});
    mfq::engine::Metrics result{
        {"continuous_batching_decode_batches", double(decode_batches_)},
        {"continuous_batching_decode_tokens", double(decode_tokens_)},
        {"continuous_batching_physical_decode_rows", double(physical_decode_rows_)},
        {"continuous_batching_compactions", 0.0},
        {"continuous_batching_sampling_readbacks", double(sampling_readbacks_)},
        {"continuous_batching_stable_slot_releases", double(storage.slot_releases)},
        {"continuous_batching_batched_greedy_batches", double(batched_greedy_batches_)},
        {"continuous_batching_packed_metadata_batches", double(packed_metadata_batches_)},
        {"continuous_batching_cuda_graph_enabled", batch_cuda_graph_enabled(execution_, config_, cached_moe_enabled_) ? 1.0 : 0.0},
        {"continuous_batching_cuda_graph_captures", double(cuda_graph_captures_)},
        {"continuous_batching_cuda_graph_replays", double(cuda_graph_replays_)},
        {"continuous_batching_cuda_graph_prewarms", double(cuda_graph_prewarms_)},
        {"continuous_batching_moe", moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_moe_cached_row_serial", cached_moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_paged_kv", storage.paged ? 1.0 : 0.0},
        {"paged_kv_page_size", double(paged.page_size)},
        {"paged_kv_live_pages", double(paged.live_pages)},
        {"paged_kv_peak_live_pages", double(paged.peak_live_pages)},
        {"paged_kv_capacity_pages", double(paged.capacity_pages)},
        {"paged_kv_reserved_bytes", double(paged.reserved_bytes)},
        {"paged_kv_page_allocations", double(paged.allocations)},
        {"paged_kv_page_reuses", double(paged.reuses)},
        {"paged_kv_page_releases", double(paged.releases)},
        {"paged_kv_table_updates", double(paged.table_updates)},
    };
    for (const auto& [bucket, counts] : graph_bucket_counts_) {
        const auto key = "continuous_batching_graph_b" + std::to_string(bucket.first) +
            "_ctx" + std::to_string(bucket.second);
        result.emplace_back(key + "_captures", counts.first);
        result.emplace_back(key + "_replays", counts.second);
    }
    return result;
}

template <class Model>
void CudaBatchExecution<Model>::initialize_sampling(Request& request, const Tensor& prompt_ids) {
    request.sampler.emplace(request.sampling);
    if (request.sampler->has_penalties())
        request.counts = SamplingOps::token_counts(prompt_ids, model_.vocab_size());
}

template <class Model>
typename CudaBatchExecution<Model>::Sample
CudaBatchExecution<Model>::execute(mfq::engine::PrefillWork<Request> work) {
    const auto& request = work.request;
    const auto chunk = work.chunk;
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard guard(primary);
    auto binding = native_.prefill(request->cache, chunk.offset, chunk.count, request->prompt.size());
    if (!chunk.offset) {
        request->prefill_ids = mfq_tensor_backend::tensor(request->prompt,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, primary)))
            .reshape({1, -1}).contiguous();
        initialize_sampling(*request, request->prefill_ids);
    }
    auto timer = std::make_unique<internal::PrefillCudaTimer>();
    MfqOptional<Tensor> length = mfq_nullopt;
    if (model_.uses_decode_sequence_length() && chunk.offset > 0 && chunk.count == 1)
        length = mfq_tensor_backend::full({1}, chunk.offset + 1, request->prefill_ids.options());
    auto hidden = model_.hidden_forward(
        request->prefill_ids.narrow(1, chunk.offset, chunk.count).contiguous(), mfq_nullopt, length);
    if (chunk.offset + chunk.count < static_cast<int64_t>(request->prompt.size())) {
        MFQ_CUDA_CHECK(cudaEventRecord(timer->finished_event(), mfq_get_current_cuda_stream()));
        request->prefill_timers.push_back(std::move(timer));
        binding.commit();
        return {};
    }
    auto logits = model_.logits_from_hidden(hidden.index({Slice(), -1, Slice()})
                                           .to(mfq_tensor_backend::kFloat16).contiguous());
    MFQ_CUDA_CHECK(cudaEventRecord(timer->finished_event(), mfq_get_current_cuda_stream()));
    request->prefill_timers.push_back(std::move(timer));
    Tensor next = sample_logits(*request->sampler, std::move(logits), request->counts, request->token_constraint);
    const auto token = next.template item<int64_t>();
    double prefill_ms = 0;
    for (const auto& elapsed : request->prefill_timers) prefill_ms += elapsed->elapsed_ms();
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
    request->pending = std::move(next);
    binding.commit();
    return {token, {request->prompt.size(), prefill_ms, 0.0, prefill_ms}};
}

template <class Model>
void CudaBatchExecution<Model>::release(const std::shared_ptr<Request>& request) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    native_.release(request->cache);
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
    request->pending = Tensor();
    if (!native_.stats().active_rows) invalidate_decode_graph();
}

template <class Model>
void CudaBatchExecution<Model>::invalidate_decode_graph() {
    for (auto& graph : decode_graphs_) graph->invalidate();
    decode_graphs_.clear();
}

template <class Model>
std::optional<std::vector<typename CudaBatchExecution<Model>::Sample>>
CudaBatchExecution<Model>::execute(mfq::engine::DecodeWork<Request> work) {
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard primary_guard(primary);
    const auto eligible = work.requests;
    MFQ_RUNTIME_CHECK(!eligible.empty(), "decode requires eligible requests");
    std::vector<std::pair<typename Model::BatchRequestState*, int64_t>> native_rows;
    std::vector<BatchSamplingRow> sampling;
    std::vector<BatchDecodeRow> rows;
    for (const auto& request : eligible) {
        native_rows.emplace_back(&request->cache, request->cache_length);
        sampling.push_back({*request->sampler, request->counts, request->token_constraint});
        rows.push_back({request->pending_token, request->cache_length});
    }
    // Power-of-two buckets bound padding and retained native storage.
    int64_t batch = 1;
    while (batch < eligible.size()) batch *= 2;
    batch = std::min<int64_t>(batch, max_sequences_);
    auto binding = native_.decode(native_rows, batch);
    const bool batch_greedy = batch_sampling_is_greedy(sampling);
    int64_t requested_len = 0;
    int64_t minimum_remaining = eligible.front()->generation_limit - eligible.front()->produced;
    for (const auto& request : eligible) {
        const int64_t remaining = request->generation_limit - request->produced;
        minimum_remaining = std::min(minimum_remaining, remaining);
        requested_len = std::max(requested_len, request->cache_length + remaining);
    }
    const auto planned_len = decode_graph_bucket(requested_len, model_.max_position_embeddings());
    const auto attention_parts = decode_graph_attention_parts(planned_len, FullBlock::kDecodeAttentionMaxParts);
    BatchDecodeGraph* decode_graph = nullptr;
    bool graph_cache_hit = false;
    if (batch_greedy && config_.greedy && batch_cuda_graph_enabled(execution_, config_, cached_moe_enabled_)) {
        const auto addresses = model_.batch_state_addresses(model_.blocks);
        const auto found = std::find_if(decode_graphs_.begin(), decode_graphs_.end(), [&](const auto& graph) {
            return graph->matches(batch, planned_len, addresses);
        });
        graph_cache_hit = found != decode_graphs_.end();
        if (graph_cache_hit) decode_graph = found->get();
        else if (minimum_remaining >= config_.cuda_graph_minimum_tokens) {
            if (decode_graphs_.size() >= 8) decode_graphs_.erase(decode_graphs_.begin());
            decode_graphs_.push_back(std::make_unique<BatchDecodeGraph>());
            decode_graph = decode_graphs_.back().get();
        }
    }
    std::vector<MfqCudaStream> previous_streams;
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> graph_guards;
    if (decode_graph) {
        decode_graph->ensure_compute_streams(execution_.tensor_parallel.enabled()
            ? execution_.tensor_parallel : execution_.expert_parallel);
        for (const auto& stream : decode_graph->compute_streams) {
            previous_streams.push_back(mfq_get_current_cuda_stream(stream.device_index()));
            wait_for_batch_stream(stream, previous_streams.back());
        }
        graph_guards = activate_cuda_graph_compute_streams(decode_graph->compute_streams);
    }
    const auto inputs = metadata_.pack(rows, batch, primary);
    ++packed_metadata_batches_;
    Tensor logits, graph_tokens;
    MoeBatchCacheScope moe_cache_scope(execution_, cached_moe_enabled_ && batch > 1);
    try {
        if (decode_graph) {
            const auto invoke = [&]() {
                auto hidden = model_.hidden_forward_static(inputs.ids, inputs.positions, inputs.lengths,
                                                          {planned_len, attention_parts});
                auto scores = model_.logits_from_hidden(hidden.index({Slice(), -1, Slice()})
                                                       .to(mfq_tensor_backend::kFloat16).contiguous());
                return sample_greedy_cuda(scores.contiguous().view({batch, -1}));
            };
            if (!graph_cache_hit) {
                decode_graph->invalidate();
                try {
                    DecodeGraphBranchScope branch_scope(execution_.decode_graph_serial_branches);
                    DecodeGraphTpProjectionScope tp_scope(execution_.decode_graph_tp_projection_major);
                    decode_graph->graph = std::make_unique<MfqCudaGraph>();
                    prepare_decode_graph_memory(model_, *decode_graph->graph, [&]() { (void)invoke(); },
                        decode_graph->participant_streams(execution_.model_parallel_collectives));
                    decode_graph->graph->capture_begin();
                    decode_graph->static_next = invoke();
                    decode_graph->graph->capture_end();
                    decode_graph->set_key(batch, planned_len, model_.batch_state_addresses(model_.blocks));
                    ++cuda_graph_captures_;
                    ++graph_bucket_counts_[{batch, planned_len}].first;
                } catch (...) { decode_graph->invalidate(); throw; }
                // Warmup restores native state; capture executes no rows.
                binding.abort();
                for (size_t i = 0; i < previous_streams.size(); ++i)
                    wait_for_batch_stream(previous_streams[i], decode_graph->compute_streams[i]);
                ++cuda_graph_prewarms_;
                return std::nullopt;
            }
            decode_graph->graph->replay();
            graph_tokens = decode_graph->static_next;
            ++cuda_graph_replays_;
            ++graph_bucket_counts_[{batch, planned_len}].second;
        } else {
            auto hidden = model_.hidden_forward(inputs.ids, inputs.positions, inputs.lengths, nullptr, inputs.positions);
            logits = model_.logits_from_hidden(hidden.index({Slice(), -1, Slice()})
                                              .to(mfq_tensor_backend::kFloat16).contiguous());
        }
        binding.commit();
    } catch (...) { invalidate_decode_graph(); throw; }
    ++decode_batches_;
    physical_decode_rows_ += batch;
    decode_tokens_ += eligible.size();
    if (graph_tokens.defined() || (config_.greedy && batch_greedy)) ++batched_greedy_batches_;
    auto tokens = sample_cuda_batch(std::move(logits), std::move(graph_tokens), sampling,
                                   config_.greedy && batch_greedy, sampling_readbacks_);
    if (decode_graph)
        for (size_t i = 0; i < previous_streams.size(); ++i)
            wait_for_batch_stream(previous_streams[i], decode_graph->compute_streams[i]);
    std::vector<Sample> result;
    for (size_t row = 0; row < eligible.size(); ++row) {
        eligible[row]->pending = tokens.device.narrow(0, row, 1);
        result.push_back({tokens.host.template data_ptr<int64_t>()[row], {}});
    }
    return result;
}

template <class Model>
void CudaBatchExecution<Model>::accept(const std::shared_ptr<Request>& request, const Sample&) {
    if (request->counts.defined()) {
        MFQ_RUNTIME_CHECK(request->pending.defined(), "batch sampler lost its pending token");
        sample_token_counts_add_cuda(request->counts, request->pending.contiguous());
    }
}

template <class Model>
void CudaBatchExecution<Model>::recover(std::span<const std::shared_ptr<Request>> requests) {
    std::vector<typename Model::BatchRequestState*> caches;
    for (const auto& request : requests) caches.push_back(&request->cache);
    try { native_.recover(caches); }
    catch (...) { invalidate_decode_graph(); throw; }
    invalidate_decode_graph();
}
} // namespace mfq::cuda
