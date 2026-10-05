#include "mfq/host_parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void coverage(int threads, int count) {
    std::vector<std::atomic<int>> visits(count);
    for (auto& visit : visits) visit.store(0);
    mfq::host_parallel_for(17, 17 + count, 3, threads,
        [&](std::int64_t begin, std::int64_t end) {
            for (auto i = begin; i < end; ++i) ++visits[i-17];
        });
    for (const auto& visit : visits) require(visit == 1, "host work omitted or duplicated");
}

void baseline(std::int64_t end, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& work) {
    std::atomic<std::int64_t> next{0};
    std::vector<std::thread> workers;
    for (int i=0; i<threads; ++i) workers.emplace_back([&] {
        for (;;) {
            const auto begin = next.fetch_add(1, std::memory_order_relaxed);
            if (begin >= end) return;
            work(begin, begin+1);
        }
    });
    for (auto& worker : workers) worker.join();
}

void benchmark() {
    const int threads = std::max(1u, std::thread::hardware_concurrency());
    constexpr int outputs = 640, width = 2560, operations = 144;
    std::vector<float> weights(outputs * width), input(width), result(outputs), reference;
    for (int i=0; i<outputs*width; ++i) weights[i] = float(i%29-14)*0.03125f;
    for (int i=0; i<width; ++i) input[i] = float(i%17-8)*0.0625f;
    auto work = [&](std::int64_t begin, std::int64_t end) {
        for (auto row=begin; row<end; ++row) {
            float total=0;
            for (int column=0; column<width; ++column)
                total += weights[row*width+column]*input[column];
            result[row]=total;
        }
    };
    baseline(outputs, threads, work);
    reference=result;
    mfq::host_parallel_for(0, outputs, 1, threads, work);
    require(result==reference, "dispatch changed expert projection results");
    for (int round=0; round<3; ++round) {
        auto start=std::chrono::steady_clock::now();
        for (int i=0; i<operations; ++i) baseline(outputs, threads, work);
        auto middle=std::chrono::steady_clock::now();
        for (int i=0; i<operations; ++i) mfq::host_parallel_for(0, outputs, 1, threads, work);
        auto end=std::chrono::steady_clock::now();
        const double old_ms=std::chrono::duration<double,std::milli>(middle-start).count();
        const double new_ms=std::chrono::duration<double,std::milli>(end-middle).count();
        require(result==reference, "benchmark result differs");
        std::cout << "{\"round\":" << round << ",\"threads\":" << threads
            << ",\"shape\":[640,2560],\"projections\":" << operations
            << ",\"baseline_ms\":" << old_ms << ",\"persistent_ms\":" << new_ms
            << ",\"speedup\":" << old_ms/new_ms << "}\n";
    }
}
} // namespace

int main(int argc, char**) {
    try {
        for (int repeat=0; repeat<100; ++repeat)
            for (int threads=1; threads<=12; ++threads) coverage(threads, 37+repeat);
        std::vector<std::thread> callers;
        for (int caller=0; caller<4; ++caller) callers.emplace_back([&] {
            for (int repeat=0; repeat<100; ++repeat) coverage(12, 640);
        });
        for (auto& caller : callers) caller.join();
        std::atomic<int> nested{0};
        mfq::host_parallel_for(0, 12, 1, 12, [&](auto begin, auto end) {
            for (auto i=begin; i<end; ++i)
                mfq::host_parallel_for(0, 7, 1, 12, [&](auto b, auto e) { nested += int(e-b); });
        });
        require(nested==84, "nested work deadlocked or was omitted");
        bool rejected=false;
        try {
            mfq::host_parallel_for(0, 640, 1, 12, [](auto, auto) { throw std::runtime_error("injected"); });
        } catch (const std::runtime_error&) { rejected=true; }
        require(rejected, "worker error did not propagate");
        coverage(12, 640);
        mfq::host_parallel_for(0, 0, 1, 12, [](auto, auto) { throw std::runtime_error("empty work invoked"); });
        std::cout << "Host pool: coverage, concurrency, nested work and failure recovery passed\n";
        if (argc>1) benchmark();
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
