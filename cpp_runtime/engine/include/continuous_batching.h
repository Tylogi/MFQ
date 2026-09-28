#pragma once

#include "mfq/runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::engine {

struct ContinuousBatchConfig {
    std::size_t max_sequences = 0;
    std::chrono::microseconds initial_batch_wait{1000};
};

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

template <typename Request>
class ContinuousBatchQueue {
public:
    explicit ContinuousBatchQueue(std::size_t capacity)
        : capacity_(capacity) {
        if (capacity_ == 0) {
            throw std::invalid_argument(
                "continuous batching capacity must be positive");
        }
    }

    void submit(std::shared_ptr<Request> request) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            throw std::runtime_error(
                "continuous batching scheduler is stopping");
        }
        pending_.push_back(std::move(request));
        ready_.notify_one();
    }

    bool wait_for_work(
            bool backend_has_work,
            std::chrono::microseconds initial_batch_wait) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] {
            return stopping_ || !pending_.empty() || backend_has_work;
        });
        if (!stopping_ && !backend_has_work &&
                pending_.size() < capacity_) {
            ready_.wait_for(lock, initial_batch_wait, [&] {
                return stopping_ || pending_.size() >= capacity_;
            });
        }
        return !stopping_;
    }

    std::vector<std::shared_ptr<Request>> take(
            std::size_t occupied,
            std::size_t admission_limit) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t available = capacity_ > occupied
            ? capacity_ - occupied : 0;
        const std::size_t count = std::min(
            {available, pending_.size(), admission_limit});
        std::vector<std::shared_ptr<Request>> requests;
        requests.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            requests.push_back(std::move(pending_.front()));
            pending_.pop_front();
        }
        return requests;
    }

    std::vector<std::shared_ptr<Request>> stop_and_drain() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        std::vector<std::shared_ptr<Request>> requests;
        requests.reserve(pending_.size());
        while (!pending_.empty()) {
            requests.push_back(std::move(pending_.front()));
            pending_.pop_front();
        }
        ready_.notify_all();
        return requests;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        ready_.notify_all();
    }

    void notify() { ready_.notify_one(); }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size();
    }

    bool stopping() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopping_;
    }

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<Request>> pending_;
    bool stopping_ = false;
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
