#include "mfq/host_parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace mfq {
namespace {

thread_local bool inside_host_work = false;

class WorkerScope {
public:
    WorkerScope() : previous_(inside_host_work) { inside_host_work = true; }
    ~WorkerScope() { inside_host_work = previous_; }
private:
    bool previous_;
};

class HostPool {
public:
    ~HostPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    void run(std::int64_t begin, std::int64_t end, std::int64_t grain,
             int threads, const std::function<void(std::int64_t, std::int64_t)>& function) {
        // Own the callback until every participant has acknowledged completion.
        std::unique_lock<std::mutex> dispatch(dispatch_mutex_);
        const int participants = static_cast<int>(std::min<std::int64_t>(
            threads, 1 + (end - begin - 1) / grain));
        const auto balanced = 1 + (end - begin - 1) / participants;
        const auto chunk = std::max(grain, balanced);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Each new worker starts before the current epoch is published.
            while (workers_.size() < static_cast<std::size_t>(participants - 1)) {
                const auto epoch = epoch_;
                const auto index = workers_.size();
                workers_.emplace_back([this, index, epoch] { worker_loop(index, epoch); });
            }
            function_ = &function;
            next_.store(begin, std::memory_order_relaxed);
            end_ = end;
            chunk_ = chunk;
            active_workers_ = participants - 1;
            remaining_ = active_workers_;
            failure_ = nullptr;
            cancelled_.store(false, std::memory_order_relaxed);
            ++epoch_;
        }
        ready_.notify_all();
        drain(); // The caller contributes compute instead of leaving one core idle.
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
        function_ = nullptr;
        if (failure_) std::rethrow_exception(failure_);
    }

private:
    void drain() {
        WorkerScope scope;
        try {
            while (!cancelled_.load(std::memory_order_relaxed)) {
                const auto begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
                if (begin >= end_) break;
                (*function_)(begin, begin + std::min(chunk_, end_ - begin));
            }
        } catch (...) {
            cancelled_.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
        }
    }

    void worker_loop(std::size_t index, std::uint64_t observed) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || observed != epoch_; });
            if (stopping_) return;
            observed = epoch_;
            if (index >= static_cast<std::size_t>(active_workers_)) continue;
            lock.unlock();
            drain();
            lock.lock();
            if (--remaining_ == 0) done_.notify_one();
        }
    }

    std::mutex dispatch_mutex_, mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> workers_;
    const std::function<void(std::int64_t, std::int64_t)>* function_ = nullptr;
    std::atomic<std::int64_t> next_{0};
    std::atomic<bool> cancelled_{false};
    std::int64_t end_ = 0, chunk_ = 1;
    std::uint64_t epoch_ = 0;
    int active_workers_ = 0, remaining_ = 0;
    bool stopping_ = false;
    std::exception_ptr failure_;
};

} // namespace

void host_parallel_for(std::int64_t begin, std::int64_t end,
    std::int64_t grain, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& function) {
    if (end <= begin) return;
    grain = std::max<std::int64_t>(grain, 1);
    if (threads <= 1 || inside_host_work || end - begin <= grain) {
        WorkerScope scope;
        function(begin, end);
        return;
    }
    static HostPool pool;
    pool.run(begin, end, grain, threads, function);
}

} // namespace mfq
