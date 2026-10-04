#include "../ops/cuda_execution.h"

#include <atomic>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

int main() {
#ifdef _WIN32
    _putenv_s("MFQ_DISABLE_NVQ_FUSION", "1");
#else
    setenv("MFQ_DISABLE_NVQ_FUSION", "1", 1);
#endif
    const auto disabled_config = load_cuda_execution_config();
#ifdef _WIN32
    _putenv_s("MFQ_DISABLE_NVQ_FUSION", "");
#else
    unsetenv("MFQ_DISABLE_NVQ_FUSION");
#endif
    const auto enabled_config = load_cuda_execution_config();
    if (disabled_config.nvq_fusion || !enabled_config.nvq_fusion) return 1;

    CudaExecutionContext first;
    CudaExecutionContext second;
    first.tensor_parallel.devices = {0, 1};
    second.expert_parallel.devices = {2, 3};
    first.drop_file_cache = true;
    first.config.nvq_fusion = false;
    second.decode_graph_serial_branches = true;

    std::atomic<int> ready{0};
    std::atomic<bool> isolated{true};
    auto run = [&](CudaExecutionContext& context,
                   bool first_context) {
        context.kl_mmq.dense_calls = first_context ? 11 : 29;
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
                context.kl_mmq.dense_calls == 11 &&
                context.profiler.stats.at("engine").calls == 3 &&
                !context.config.nvq_fusion
            : model_parallel_enabled(context) &&
                !context.tensor_parallel.enabled() &&
                moe_parallel_config(context).primary_device() == 2 &&
                context.kl_mmq.dense_calls == 29 &&
                context.profiler.stats.at("engine").calls == 7 &&
                context.config.nvq_fusion;
        if (!valid) {
            isolated.store(false, std::memory_order_relaxed);
        }
    };
    std::thread first_thread(
        run, std::ref(first), true);
    std::thread second_thread(
        run, std::ref(second), false);
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
