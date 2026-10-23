#pragma once
#include <functional>
#include <future>

namespace mfq::cpu {
// The worker owns its captured host inputs until the measured work ends.
std::future<double> measure_work(std::function<void()> work,double preparation_ns=0);
} // namespace mfq::cpu
