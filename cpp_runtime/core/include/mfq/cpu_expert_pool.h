#pragma once
#include "host_parallel.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace mfq::cpu {
enum class PoolAffinity { Auto, PCores, All };
enum class HostCore { First, Last };
struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0, p_threads = 0, e_cores = 0;
    std::vector<int> worker_cores;
    int host_core = -1;
};
CpuTopology detect_cpu_topology(bool reserve_caller, PoolAffinity = PoolAffinity::All,
                              HostCore = HostCore::First);
int planned_host_core();
std::vector<int> physical_cores(bool reserve_caller, PoolAffinity = PoolAffinity::All);
struct ThreadAffinity {
    std::vector<std::uint64_t> native;
    std::vector<unsigned long> cpu_sets;
    bool valid = false;
};
ThreadAffinity pin_current_thread(int core);
void restore_thread_affinity(const ThreadAffinity& previous);
class ScopedHostAffinity {
public:
    explicit ScopedHostAffinity(int core) : previous_(pin_current_thread(core)) {}
    ~ScopedHostAffinity() { restore_thread_affinity(previous_); }
    ScopedHostAffinity(const ScopedHostAffinity&) = delete;
    ScopedHostAffinity& operator=(const ScopedHostAffinity&) = delete;
private:
    ThreadAffinity previous_;
};

// Each projection contributes rows to one batch on MFQ's persistent workers.
class ExpertPool {
public:
    explicit ExpertPool(int workers = 0, bool pin = true);
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;
    void rows(std::int64_t count,
              const std::function<void(std::int64_t, std::int64_t)>& callback,
              mfq::HostParallelProfile* profile=nullptr);
    int participants() const { return participants_; }
private:
    CpuTopology topology_;
    int participants_ = 1;
    bool pin_ = true;
    std::unique_ptr<mfq::HostParallelPool> pool_;
};
ExpertPool& expert_pool();
} // namespace mfq::cpu
