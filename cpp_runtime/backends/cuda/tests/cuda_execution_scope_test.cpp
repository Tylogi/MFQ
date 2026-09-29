#include "cuda_execution.h"

#include <atomic>
#include <functional>
#include <stdexcept>
#include <thread>

int main() {
    CudaExecutionContext first;
    CudaExecutionContext second;

    if (current_cuda_execution_context() != nullptr) return 1;
    {
        CudaExecutionContextScope first_scope(first);
        if (&cuda_execution_context() != &first) return 2;
        first.drop_file_cache = true;
        {
            CudaExecutionContextScope second_scope(second);
            if (&cuda_execution_context() != &second) return 3;
            if (second.drop_file_cache) return 4;
        }
        if (&cuda_execution_context() != &first || !first.drop_file_cache) {
            return 5;
        }
    }
    if (current_cuda_execution_context() != nullptr) return 6;

    std::atomic<int> ready{0};
    std::atomic<bool> isolated{true};
    auto run = [&](CudaExecutionContext& context, bool value) {
        CudaExecutionContextScope scope(context);
        context.decode_graph_serial_branches = value;
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        if (current_cuda_execution_context() != &context ||
                context.decode_graph_serial_branches != value) {
            isolated.store(false, std::memory_order_relaxed);
        }
    };
    std::thread first_thread(run, std::ref(first), true);
    std::thread second_thread(run, std::ref(second), false);
    first_thread.join();
    second_thread.join();
    if (!isolated.load(std::memory_order_relaxed)) return 7;

    try {
        (void)cuda_execution_context();
        return 8;
    } catch (const std::logic_error&) {
        return 0;
    }
}
