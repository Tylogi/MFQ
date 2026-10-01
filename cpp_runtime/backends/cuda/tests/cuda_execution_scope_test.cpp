#include "cuda_engine.h"
#include "cuda_execution.h"

#include <atomic>
#include <thread>
#include <utility>
#include <vector>

int main() {
    CudaExecutionContext first;
    CudaExecutionContext second;
    first.tensor_parallel.devices = {0, 1};
    second.expert_parallel.devices = {2, 3};
    first.drop_file_cache = true;
    second.decode_graph_serial_branches = true;

    mfq::cuda::CudaEngine first_engine;
    mfq::cuda::CudaEngine second_engine;
    first_engine.metadata.architecture = "first";
    second_engine.metadata.architecture = "second";
    first_engine.runtime_metrics = [&first] {
        return std::vector<std::pair<std::string, double>>{
            {"calls", static_cast<double>(first.kl_mmq_dense_calls)}};
    };
    second_engine.runtime_metrics = [&second] {
        return std::vector<std::pair<std::string, double>>{
            {"calls", static_cast<double>(second.kl_mmq_dense_calls)}};
    };

    std::atomic<int> ready{0};
    std::atomic<bool> isolated{true};
    auto run = [&](CudaExecutionContext& context,
                   mfq::cuda::CudaEngine& engine,
                   bool first_context) {
        context.kl_mmq_dense_calls = first_context ? 11 : 29;
        context.continuous_batch_cache_serial = first_context;
        context.profiler.stats["engine"].calls = first_context ? 3 : 7;
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        const bool valid = first_context
            ? model_parallel_enabled(context) &&
                context.tensor_parallel.primary_device() == 0 &&
                !context.expert_parallel.enabled() &&
                context.kl_mmq_dense_calls == 11 &&
                context.profiler.stats.at("engine").calls == 3
            : model_parallel_enabled(context) &&
                !context.tensor_parallel.enabled() &&
                moe_parallel_config(context).primary_device() == 2 &&
                context.kl_mmq_dense_calls == 29 &&
                context.profiler.stats.at("engine").calls == 7;
        const auto metrics = engine.runtime_metrics();
        const auto expected = first_context ? 11.0 : 29.0;
        if (!valid || metrics.size() != 1 ||
                metrics.front().second != expected) {
            isolated.store(false, std::memory_order_relaxed);
        }
    };
    std::thread first_thread(
        run, std::ref(first), std::ref(first_engine), true);
    std::thread second_thread(
        run, std::ref(second), std::ref(second_engine), false);
    first_thread.join();
    second_thread.join();

    return isolated.load(std::memory_order_relaxed) &&
            first.drop_file_cache && !second.drop_file_cache &&
            !first.decode_graph_serial_branches &&
            second.decode_graph_serial_branches &&
            first.continuous_batch_cache_serial &&
            !second.continuous_batch_cache_serial
        ? 0 : 1;
}
