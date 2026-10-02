#pragma once

#include "engine.h"
#include "cuda_execution.h"
#include "decode_graph.h"
#include "cuda_runtime_config.h"

#include <coroutine>
#include <exception>
#include <optional>
#include <utility>

struct MtpModule;

namespace mfq::cuda::internal {
class TextSessionCache;

// Engine::step resumes one quantum. No worker, callback or consumer wait.
class Generation {
public:
    struct promise_type {
        std::optional<mfq::engine::EventData> event;
        std::exception_ptr error;
        Generation get_return_object() { return Generation{std::coroutine_handle<promise_type>::from_promise(*this)}; }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(mfq::engine::EventData value) { event.emplace(std::move(value)); return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { error = std::current_exception(); }
    };
    Generation() = default;
    explicit Generation(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
    Generation(Generation&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}
    Generation& operator=(Generation&& other) noexcept {
        if (this != &other) { if (handle_) handle_.destroy(); handle_ = std::exchange(other.handle_, {}); }
        return *this;
    }
    ~Generation() { if (handle_) handle_.destroy(); }
    std::optional<mfq::engine::EventData> next() {
        if (!handle_ || handle_.done()) return {};
        auto& promise = handle_.promise(); promise.event.reset();
        handle_.resume();
        if (promise.error) std::rethrow_exception(promise.error);
        return std::move(promise.event);
    }
private:
    std::coroutine_handle<promise_type> handle_;
};

template <typename Model>
Generation generate(
    Model& model, DecodeGraphCache& graph, TextSessionCache& cache,
    const CudaRuntimeConfig& config, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, MtpModule* mtp = nullptr,
    std::optional<CudaPreparedPrompt> prepared = {});
} // namespace mfq::cuda::internal
