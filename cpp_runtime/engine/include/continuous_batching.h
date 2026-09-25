#pragma once

#include "mfq/runtime.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mfq::engine {

struct ContinuousBatchRequest {
    struct CancellationState {
        explicit CancellationState(MfqCancellationCheck external_check = {});

        bool load(std::memory_order order) const;
        void store(bool value, std::memory_order order);

    private:
        std::atomic<bool> requested_{false};
        MfqCancellationCheck external_check_;
    };

    ContinuousBatchRequest(
        std::vector<int64_t> prompt,
        MfqSamplingParams sampling,
        MfqTokenConstraintPtr token_constraint,
        MfqCancellationCheck cancelled = {});

    void publish_prefill(MfqPrefillTiming timing);
    void publish_token(int64_t token);
    void complete(std::exception_ptr error = {});
    int32_t consume(
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill,
        const std::function<void()>& wake_executor = {});

    std::vector<int64_t> prompt;
    MfqSamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    int32_t generation_limit = 0;
    int32_t produced = 0;
    int64_t pending_token = 0;
    CancellationState cancel_requested;

private:
    std::mutex output_mutex_;
    std::condition_variable output_ready_;
    std::optional<MfqPrefillTiming> prefill_timing_;
    std::deque<int64_t> output_tokens_;
    bool done_ = false;
    std::exception_ptr error_;
};

class ContinuousBatchExecutor {
public:
    virtual ~ContinuousBatchExecutor() = default;

    virtual int32_t submit(
        const std::vector<int64_t>& prompt,
        const MfqSamplingParams& sampling,
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill,
        const MfqPromptCachePlan& cache_plan,
        const MfqTokenConstraintPtr& token_constraint,
        const MfqCancellationCheck& cancelled) = 0;

    virtual std::vector<std::pair<std::string, double>> metrics() const = 0;
};

} // namespace mfq::engine
