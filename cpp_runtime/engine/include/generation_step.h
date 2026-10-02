#pragma once

#include "engine.h"
#include "generation_policy.h"

#include <coroutine>
#include <exception>
#include <utility>

namespace mfq::engine {

// One resume executes one quantum. Destroying the sequence also releases its
// suspended execution state; it must happen before publishing a terminal event.
class Generation {
  public:
    struct promise_type {
        std::optional<EventData> event;
        std::exception_ptr error;
        Generation get_return_object() {
            return Generation{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(EventData value) {
            event.emplace(std::move(value));
            return {};
        }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { error = std::current_exception(); }
    };
    Generation() = default;
    explicit Generation(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
    Generation(Generation &&other) noexcept : handle_(std::exchange(other.handle_, {})) {}
    Generation &operator=(Generation &&other) noexcept {
        if (this != &other) {
            if (handle_)
                handle_.destroy();
            handle_ = std::exchange(other.handle_, {});
        }
        return *this;
    }
    ~Generation() {
        if (handle_)
            handle_.destroy();
    }
    std::optional<EventData> next() {
        if (!handle_ || handle_.done())
            return {};
        auto &promise = handle_.promise();
        promise.event.reset();
        handle_.resume();
        if (promise.error)
            std::rethrow_exception(promise.error);
        return std::move(promise.event);
    }

  private:
    std::coroutine_handle<promise_type> handle_;
};

// Ops performs one device prefill/decode operation. Chunking, cancellation,
// token acceptance and publication order are independent of tensor storage.
template <class Ops>
Generation generate_sequence(Ops &ops, InferenceOutput &output, std::int64_t prompt_tokens,
    std::int64_t reused_tokens, std::int64_t stable_prefix_tokens, std::int64_t chunk_size) {
    double elapsed = 0.0;
    auto offset = reused_tokens;
    while (offset < prompt_tokens && !output.stopped()) {
        const auto end = offset < stable_prefix_tokens
                             ? std::min(prompt_tokens, stable_prefix_tokens)
                             : prompt_tokens;
        const auto chunk = next_prefill_chunk(end, offset, chunk_size);
        elapsed += ops.prefill(chunk);
        offset += chunk.count;
        co_yield PrefillProgress{
            {static_cast<std::size_t>(offset - reused_tokens), elapsed, 0.0, elapsed}};
    }
    if (output.stopped())
        co_return;
    auto token = ops.first_token();
    while (!output.stopped()) {
        auto delta = output.append(std::vector<std::int64_t>{token});
        ops.accept(token);
        co_yield std::move(delta);
        if (!output.stopped())
            token = ops.advance();
    }
}

} // namespace mfq::engine
