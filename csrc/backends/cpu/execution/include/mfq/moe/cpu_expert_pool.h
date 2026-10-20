#pragma once
#include "mfq/host_parallel.h"
#include "mfq/cpu_topology.h"
#include <cstdint>
#include <functional>
#include <memory>

namespace mfq::cpu {

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
