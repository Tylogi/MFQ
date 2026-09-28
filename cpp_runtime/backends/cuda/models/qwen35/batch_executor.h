#pragma once

#include "continuous_batching.h"
#include "cuda_model_plan.h"
#include "engine/runtime_config.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct CudaExecutionContext;

namespace mfq::cuda {
template <CudaBackbone Backbone>
struct CausalLm;
using Qwen35CausalLm = CausalLm<CudaBackbone::generic_qwen>;
}

namespace mfq::cuda::qwen35 {

bool qwen_continuous_batch_has_moe(const Qwen35CausalLm& model);
bool qwen_continuous_batch_has_cached_moe(const Qwen35CausalLm& model);

// Scheduler admission is backend-neutral. This executor owns only physical
// CUDA batch packing and per-sequence device state.
class QwenBatchExecutor final : public mfq::engine::ContinuousBatchExecutor {
public:
    QwenBatchExecutor(
        Qwen35CausalLm& model,
        CudaExecutionContext& execution,
        std::mutex& model_mutex,
        CudaContinuousBatchConfig config,
        mfq::engine::GenerationConfig generation);
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
        const MfqCancellationCheck& cancelled) override;
    std::vector<std::pair<std::string, double>> metrics() const override;

    std::int64_t queued_requests() const;
    bool paged_kv_enabled() const noexcept;
    std::int64_t paged_kv_page_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

int run_qwen_continuous_batching_check(
    Qwen35CausalLm& model,
    const CudaRuntimeConfig& config);

} // namespace mfq::cuda::qwen35
