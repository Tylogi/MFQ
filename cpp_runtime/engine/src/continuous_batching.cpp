#include "continuous_batching.h"

namespace mfq::engine {

ContinuousBatchRequest::CancellationState::CancellationState(
        MfqCancellationCheck external_check)
    : external_check_(std::move(external_check)) {}

bool ContinuousBatchRequest::CancellationState::load(
        std::memory_order order) const {
    return requested_.load(order) ||
        (external_check_ && external_check_());
}

void ContinuousBatchRequest::CancellationState::store(
        bool value, std::memory_order order) {
    requested_.store(value, order);
}

ContinuousBatchRequest::ContinuousBatchRequest(
        std::vector<int64_t> input_prompt,
        MfqSamplingParams input_sampling,
        MfqTokenConstraintPtr input_constraint,
        MfqCancellationCheck cancelled)
    : prompt(std::move(input_prompt)),
      sampling(input_sampling),
      token_constraint(std::move(input_constraint)),
      cancel_requested(std::move(cancelled)) {}

void ContinuousBatchRequest::publish_prefill(MfqPrefillTiming timing) {
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        prefill_timing_ = std::move(timing);
    }
    output_ready_.notify_one();
}

void ContinuousBatchRequest::publish_token(int64_t token) {
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        output_tokens_.push_back(token);
        ++published_tokens_;
    }
    output_ready_.notify_one();
}

bool ContinuousBatchRequest::publish_token_sync(int64_t token) {
    std::unique_lock<std::mutex> lock(output_mutex_);
    output_tokens_.push_back(token);
    const auto published = ++published_tokens_;
    output_ready_.notify_one();
    output_ready_.wait(lock, [&] {
        return consumed_tokens_ >= published || done_;
    });
    return !cancel_requested.load(std::memory_order_acquire);
}

void ContinuousBatchRequest::complete(std::exception_ptr error) {
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        if (done_) return;
        error_ = error;
        done_ = true;
    }
    output_ready_.notify_one();
}

int32_t ContinuousBatchRequest::consume(
        const MfqTokenCallback& on_token,
        const MfqPrefillCallback& on_prefill,
        const std::function<void()>& wake_executor) {
    int32_t delivered = 0;
    bool callbacks_enabled = true;
    std::exception_ptr callback_error;
    std::exception_ptr producer_error;
    for (;;) {
        std::optional<MfqPrefillTiming> prefill;
        std::optional<int64_t> token;
        bool producer_done = false;
        {
            std::unique_lock<std::mutex> lock(output_mutex_);
            output_ready_.wait(lock, [&] {
                return prefill_timing_.has_value() ||
                    !output_tokens_.empty() || done_;
            });
            if (prefill_timing_.has_value()) {
                prefill = std::move(prefill_timing_);
                prefill_timing_.reset();
            } else if (!output_tokens_.empty()) {
                token = output_tokens_.front();
                output_tokens_.pop_front();
            } else {
                producer_done = done_;
                producer_error = error_;
            }
        }
        if (producer_done) break;
        if (!callbacks_enabled) continue;
        try {
            if (prefill.has_value()) {
                if (on_prefill) on_prefill(*prefill);
            } else if (token.has_value()) {
                ++delivered;
                if (on_token && !on_token(*token)) {
                    callbacks_enabled = false;
                    cancel_requested.store(true, std::memory_order_release);
                    if (wake_executor) wake_executor();
                }
            }
        } catch (...) {
            callback_error = std::current_exception();
            callbacks_enabled = false;
            cancel_requested.store(true, std::memory_order_release);
            if (wake_executor) wake_executor();
        }
        if (token.has_value()) {
            {
                std::lock_guard<std::mutex> lock(output_mutex_);
                ++consumed_tokens_;
            }
            output_ready_.notify_one();
        }
    }
    if (callback_error) std::rethrow_exception(callback_error);
    if (producer_error) std::rethrow_exception(producer_error);
    return delivered;
}

} // namespace mfq::engine
