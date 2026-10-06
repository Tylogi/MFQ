#pragma once

#include "cuda_runtime_config.h"
#include "continuous_batch.h"
#include "decode_graph.h"
#include "generation_step.h"
#include "../ops/cuda_sampling.h"
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

#include "batch_execution.tpp"
