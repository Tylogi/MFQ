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
#include <thread>
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
    bool publish_token_sync(int64_t token);
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
    std::uint64_t published_tokens_ = 0;
    std::uint64_t consumed_tokens_ = 0;
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

template <typename Request>
struct ContinuousBatchState {
    std::deque<std::shared_ptr<Request>> prefilling;
    std::vector<std::shared_ptr<Request>> active;
};

// Owns queue, worker, and request-state transitions. Operations supplies only
// backend prefill/decode and device-state cleanup.
template <typename Request, typename Operations>
class ContinuousBatchingController {
public:
    using State = ContinuousBatchState<Request>;
    using Queue = ContinuousBatchQueue<Request>;
    using RequestPtr = std::shared_ptr<Request>;

    ContinuousBatchingController(
            ContinuousBatchConfig config,
            std::unique_ptr<Operations> operations)
        : config_(config),
          operations_(std::move(operations)),
          queue_(config.max_sequences) {
        if (!operations_) {
            throw std::invalid_argument(
                "continuous batching requires backend operations");
        }
        worker_ = std::thread([this] { worker_main(); });
    }

    ~ContinuousBatchingController() {
        queue_.stop();
        if (worker_.joinable()) worker_.join();
    }

    ContinuousBatchingController(const ContinuousBatchingController&) = delete;
    ContinuousBatchingController& operator=(
        const ContinuousBatchingController&) = delete;

    void submit(RequestPtr request) { queue_.submit(std::move(request)); }

    std::int32_t submit(
            RequestPtr request,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback& on_prefill) {
        queue_.submit(request);
        return request->consume(on_token, on_prefill, [this] {
            queue_.notify();
        });
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        return {
            {"continuous_batching_max_sequences",
                static_cast<double>(config_.max_sequences)},
            {"continuous_batching_active", static_cast<double>(active())},
            {"continuous_batching_prefilling",
                static_cast<double>(prefilling())},
            {"continuous_batching_queued", static_cast<double>(queued())},
            {"continuous_batching_max_batch",
                static_cast<double>(max_batch())},
            {"continuous_batching_interleaved_admissions",
                static_cast<double>(interleaved_admissions())},
        };
    }

    Operations& operations() noexcept { return *operations_; }
    const Operations& operations() const noexcept { return *operations_; }
    std::size_t queued() const { return queue_.size(); }
    std::int64_t active() const noexcept {
        return active_count_.load(std::memory_order_relaxed);
    }
    std::int64_t prefilling() const noexcept {
        return prefilling_count_.load(std::memory_order_relaxed);
    }
    std::int64_t max_batch() const noexcept {
        return max_batch_.load(std::memory_order_relaxed);
    }
    std::int64_t interleaved_admissions() const noexcept {
        return interleaved_admissions_.load(std::memory_order_relaxed);
    }

private:
    void publish_state() {
        active_count_.store(
            static_cast<std::int64_t>(state_.active.size()),
            std::memory_order_relaxed);
        prefilling_count_.store(
            static_cast<std::int64_t>(state_.prefilling.size()),
            std::memory_order_relaxed);
        auto previous = max_batch_.load(std::memory_order_relaxed);
        const auto current = static_cast<std::int64_t>(state_.active.size());
        while (previous < current && !max_batch_.compare_exchange_weak(
                previous, current, std::memory_order_relaxed)) {}
    }

    void clear_state() {
        state_.active.clear();
        state_.prefilling.clear();
        publish_state();
    }

    void worker_main() noexcept {
        for (;;) {
            try {
                if (!queue_.wait_for_work(
                        !state_.active.empty() || !state_.prefilling.empty(),
                        config_.initial_batch_wait)) {
                    auto error = std::make_exception_ptr(
                        std::runtime_error(
                            "continuous batching scheduler stopped"));
                    auto pending = queue_.stop_and_drain();
                    try {
                        operations_->shutdown(pending, state_, error);
                    } catch (...) {}
                    clear_state();
                    return;
                }
                // Service one decode step, then at most one prompt chunk
                // while decode remains active.
                const bool decode_was_active = !state_.active.empty();
                if (decode_was_active) {
                    operations_->decode_active(state_);
                    publish_state();
                }
                const bool contended = !state_.active.empty();
                auto incoming = queue_.take(
                    state_.active.size() + state_.prefilling.size(),
                    contended ? std::size_t{1} : config_.max_sequences);
                if (contended && !incoming.empty()) {
                    interleaved_admissions_.fetch_add(
                        1, std::memory_order_relaxed);
                }
                operations_->advance_prefills(
                    incoming, contended, state_, queue_);
                publish_state();
                if (!decode_was_active) {
                    operations_->decode_active(state_);
                    publish_state();
                }
            } catch (...) {
                try {
                    operations_->recover(state_, std::current_exception());
                } catch (...) {}
                clear_state();
            }
        }
    }

    const ContinuousBatchConfig config_;
    std::unique_ptr<Operations> operations_;
    Queue queue_;
    std::thread worker_;
    State state_;
    std::atomic<std::int64_t> active_count_{0};
    std::atomic<std::int64_t> prefilling_count_{0};
    std::atomic<std::int64_t> max_batch_{0};
    std::atomic<std::int64_t> interleaved_admissions_{0};
};

class ContinuousBatching {
public:
    virtual ~ContinuousBatching() = default;

    virtual int32_t submit(
        const std::vector<int64_t>& prompt,
        const MfqSamplingParams& sampling,
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill,
        const MfqPromptCachePlan& cache_plan,
        const MfqTokenConstraintPtr& token_constraint,
        const MfqCancellationCheck& cancelled,
        const MfqMultimodalInput* media = nullptr) = 0;

    virtual std::vector<std::pair<std::string, double>> metrics() const = 0;
};

} // namespace mfq::engine
