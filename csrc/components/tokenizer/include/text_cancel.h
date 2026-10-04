#pragma once

#include <atomic>
#include <stdexcept>

namespace mfq::text {

// A preparation worker owns its stop flag. Nested tokenizer/template operations
// observe it without retaining callbacks or changing the public C tokenizer API.
inline thread_local const std::atomic<bool>* stop_flag = nullptr;

struct CancellationScope {
    const std::atomic<bool>* previous = stop_flag;
    explicit CancellationScope(const std::atomic<bool>& flag) { stop_flag = &flag; }
    ~CancellationScope() { stop_flag = previous; }
    CancellationScope(const CancellationScope&) = delete;
    CancellationScope& operator=(const CancellationScope&) = delete;
};

inline void check_cancelled() {
    if (stop_flag && stop_flag->load(std::memory_order_relaxed))
        throw std::runtime_error("text preparation cancelled");
}

} // namespace mfq::text
