#pragma once

#include "continuous_batching.h"
#include "cuda_runtime_config.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct CudaExecutionContext;

namespace mfq::cuda {

struct Qwen35Model;
template <typename Model>
struct CausalLm;
using Qwen35CausalLm = CausalLm<Qwen35Model>;

bool qwen_continuous_batch_cuda_graph_enabled(
    const Qwen35CausalLm& model,
    const CudaContinuousBatchConfig& config);

using QwenExclusiveGeneration = std::function<std::int32_t(
    const std::vector<std::int64_t>&,
    const MfqMultimodalInput*,
    const MfqSamplingParams&,
    const MfqTokenCallback&,
    const MfqPrefillCallback&,
    const MfqPromptCachePlan&,
    const MfqTokenConstraintPtr&,
    const MfqCancellationCheck&)>;

class QwenBatchExecutor final : public mfq::engine::ContinuousBatching {
public:
    QwenBatchExecutor(
        Qwen35CausalLm& model,
        CudaExecutionContext& execution,
        std::mutex& model_mutex,
        const CudaContinuousBatchConfig& config,
        std::int64_t prefill_chunk_size,
        QwenExclusiveGeneration exclusive_generation);
    ~QwenBatchExecutor() override;

    QwenBatchExecutor(const QwenBatchExecutor&) = delete;
    QwenBatchExecutor& operator=(const QwenBatchExecutor&) = delete;

    std::int32_t submit(
        const std::vector<std::int64_t>& prompt,
        const MfqSamplingParams& sampling,
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill,
        const MfqPromptCachePlan& cache_plan,
        const MfqTokenConstraintPtr& token_constraint,
        const MfqCancellationCheck& cancelled,
        const MfqMultimodalInput* media = nullptr) override;
    std::vector<std::pair<std::string, double>> metrics() const override;

    std::int64_t queued_requests() const;
    bool paged_kv_enabled() const noexcept;
    std::int64_t paged_kv_page_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::cuda
