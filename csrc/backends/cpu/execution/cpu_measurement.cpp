#include "mfq/moe/cpu_measurement.h"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mfq::cpu {
std::future<double> measure_work(std::function<void()> work,double preparation_ns) {
    if(!work || !std::isfinite(preparation_ns) || preparation_ns<0)
        throw std::invalid_argument("CPU measurement needs work and finite preparation time");
    return std::async(std::launch::async,[work=std::move(work),preparation_ns] {
        const auto started=std::chrono::steady_clock::now();
        work();
        const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        return preparation_ns+double(elapsed);
    });
}
} // namespace mfq::cpu
