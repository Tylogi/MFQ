#pragma once
#include <cstdint>
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
HostCore host_core_setting();
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
} // namespace mfq::cpu
