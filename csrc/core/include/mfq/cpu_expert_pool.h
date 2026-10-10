#pragma once
#include "host_parallel.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace mfq::cpu {
enum class PoolAffinity { Auto, PCores, All };
struct CpuTopology {
    bool is_hybrid = false;
    int p_cores = 0, p_threads = 0, e_cores = 0;
    std::vector<int> worker_cores;
    int host_core = -1;
};
CpuTopology detect_cpu_topology(bool reserve_caller, PoolAffinity = PoolAffinity::All);
std::vector<int> physical_cores(bool reserve_caller, PoolAffinity = PoolAffinity::All);
struct ThreadAffinity {
    std::vector<std::uint64_t> native;
    bool valid = false;
};
ThreadAffinity pin_current_thread(int core);
void restore_thread_affinity(const ThreadAffinity& previous);

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
