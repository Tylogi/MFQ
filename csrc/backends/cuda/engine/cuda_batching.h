#pragma once

#include "continuous_batch.h"
#include <map>
#include "../ops/cuda_execution.h"
#include "cuda_runtime_config.h"
#include "../ops/cuda_sampling.h"
#include "models/qwen35/batch_state.h"

namespace mfq::cuda {

bool qwen_continuous_batch_cuda_graph_enabled(const Qwen35CausalLm &,
                                              const CudaContinuousBatchConfig &);
struct QwenContinuousDecodeGraph;

struct QwenBatchRequest : mfq::engine::BatchRequest {
    using BatchRequest::BatchRequest;
    std::optional<Sampler> sampler;
    int32_t decode_row = -1;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    qwen35::QwenBatchRequestState cache;
};

// Native operations consumed directly by engine::ContinuousBatch.
class QwenBatchOperations {
  public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Request = QwenBatchRequest;
    using State = mfq::engine::BatchState<Request>;
    struct Sample {
        Tensor next;
        int64_t token = 0;
        MfqPrefillTiming timing;
    };
    struct Decoded {
        Tensor tokens, host_tokens;
    };

    QwenBatchOperations(Qwen35CausalLm &, CudaExecutionContext &, CudaContinuousBatchConfig);
    ~QwenBatchOperations();
    QwenBatchOperations(const QwenBatchOperations &) = delete;
    QwenBatchOperations &operator=(const QwenBatchOperations &) = delete;
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    void suspend_decode();
    Sample prefill(const std::shared_ptr<Request> &, mfq::engine::PrefillChunk);
    void activate(const std::shared_ptr<Request> &, const Sample &);
    void resume_decode(int64_t cache_position);
    void discard_prefill(const std::shared_ptr<Request> &);
    void retire(const std::vector<std::shared_ptr<Request>> &, int64_t cache_position);
    Decoded decode(State &);
    Sample sample(const std::shared_ptr<Request> &, const Decoded &);
    void accept(const std::shared_ptr<Request> &, const Sample &);
    void recover(State &);

  private:
    void initialize_sampling(Request &, const Tensor &prompt_ids);
    void ensure_decode_metadata_buffers(int primary);
    void invalidate_decode_graph();

    Qwen35CausalLm &model_;
    qwen35::Qwen35BatchStateAdapter state_adapter_;
    CudaExecutionContext &execution_;
    const CudaContinuousBatchConfig config_;
    int32_t max_sequences_ = 0;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    int64_t decode_batches_{0};
    int64_t decode_tokens_{0};
    int64_t physical_decode_rows_{0};
    int64_t batched_greedy_batches_{0};
    int64_t packed_metadata_batches_{0};
    int64_t sampling_readbacks_{0};
    int64_t cuda_graph_captures_{0};
    int64_t cuda_graph_replays_{0};
    std::map<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>> graph_bucket_counts_;
    Tensor decode_metadata_host_;
    Tensor decode_metadata_cuda_;
    std::vector<std::unique_ptr<QwenContinuousDecodeGraph>> decode_graphs_;
};
} // namespace mfq::cuda
