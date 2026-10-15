#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace mfq::engine {

class PrefillActivity {
public:
    PrefillActivity() { active_.fetch_add(1, std::memory_order_relaxed); }
    PrefillActivity(const PrefillActivity&) = delete;
    PrefillActivity& operator=(const PrefillActivity&) = delete;
    ~PrefillActivity() {
        completed_at_.store(now(), std::memory_order_relaxed);
        active_.fetch_sub(1, std::memory_order_relaxed);
    }
    static bool recent() noexcept {
        return active_.load(std::memory_order_relaxed) > 0 ||
            now()-completed_at_.load(std::memory_order_relaxed) < 3000000000LL;
    }
private:
    static std::int64_t now() noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    inline static std::atomic<int> active_{0};
    inline static std::atomic<std::int64_t> completed_at_{0};
};

}
