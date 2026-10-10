#include "mfq/cpu_expert_pool.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mfq::cpu {
namespace {
struct Core { int first = -1, threads = 0, performance = 0; };
std::vector<Core> os_cores() {
    std::vector<Core> result;
#ifdef _WIN32
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (!bytes) throw std::runtime_error("cannot query CPU cores");
    std::vector<unsigned char> data(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data()), &bytes))
        throw std::runtime_error("cannot read CPU core topology");
    GROUP_AFFINITY current{};
    GetThreadGroupAffinity(GetCurrentThread(), &current);
    DWORD_PTR process_mask = 0, system_mask = 0;
    const bool restricted = GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) != 0;
    for (DWORD at = 0; at < bytes;) {
        auto* item = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(data.data() + at);
        if (!item->Size || item->Size > bytes - at) throw std::runtime_error("invalid CPU topology record");
        Core core;
        core.performance = item->Processor.EfficiencyClass;
        for (WORD group = 0; group < item->Processor.GroupCount; ++group) {
            const auto& entry = item->Processor.GroupMask[group];
            auto mask = entry.Mask;
            if (restricted && entry.Group == current.Group) mask &= process_mask;
            for (int bit = 0; bit < int(sizeof(KAFFINITY) * 8); ++bit) {
                if (!(mask & (KAFFINITY(1) << bit))) continue;
                if (core.first < 0) core.first = entry.Group * 64 + bit;
                ++core.threads;
            }
        }
        if (core.first >= 0) result.push_back(core);
        at += item->Size;
    }
#elif defined(__linux__)
    const auto count = std::max<long>(1, sysconf(_SC_NPROCESSORS_CONF));
    const auto bytes = CPU_ALLOC_SIZE(count);
    cpu_set_t* allowed = CPU_ALLOC(count);
    if (!allowed) throw std::bad_alloc();
    CPU_ZERO_S(bytes, allowed);
    const bool known = sched_getaffinity(0, bytes, allowed) == 0;
    std::map<std::pair<int, int>, Core> grouped;
    for (int id = 0; id < count; ++id) {
        if (known && !CPU_ISSET_S(id, bytes, allowed)) continue;
        int package = 0, physical = id;
        const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(id) + "/topology/";
        std::ifstream(base + "physical_package_id") >> package;
        std::ifstream(base + "core_id") >> physical;
        auto& core = grouped[{package, physical}];
        if (core.first < 0) core.first = id;
        ++core.threads;
    }
    CPU_FREE(allowed);
    for (const auto& entry : grouped) result.push_back(entry.second);
#elif defined(__APPLE__)
    unsigned physical = 0;
    std::size_t bytes = sizeof(physical);
    if (sysctlbyname("hw.physicalcpu", &physical, &bytes, nullptr, 0) != 0 || !physical)
        throw std::runtime_error("cannot query physical CPU count");
    for (unsigned id = 0; id < physical; ++id) result.push_back({static_cast<int>(id), 1, 0});
#else
    const auto physical = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned id = 0; id < physical; ++id) result.push_back({static_cast<int>(id), 1, 0});
#endif
    if (result.empty()) throw std::runtime_error("CPU affinity contains no usable core");
    return result;
}
}

CpuTopology detect_cpu_topology(bool reserve_caller, PoolAffinity affinity) {
    const auto cores = os_cores();
    const auto maximum = std::max_element(cores.begin(), cores.end(),
        [](const Core& a, const Core& b) { return a.performance < b.performance; })->performance;
    CpuTopology result;
    for (const auto& core : cores) {
        const bool fast = core.performance == maximum;
        if (fast) { ++result.p_cores; result.p_threads += core.threads; }
        else ++result.e_cores;
        if (affinity == PoolAffinity::PCores && !fast) continue;
        result.worker_cores.push_back(core.first);
    }
    result.is_hybrid = result.e_cores != 0;
    if (reserve_caller && !result.worker_cores.empty()) {
        result.host_core = result.worker_cores.front();
        result.worker_cores.erase(result.worker_cores.begin());
    }
    return result;
}
std::vector<int> physical_cores(bool reserve_caller, PoolAffinity affinity) {
    return detect_cpu_topology(reserve_caller, affinity).worker_cores;
}

ThreadAffinity pin_current_thread(int core) {
    ThreadAffinity previous;
    if (core < 0) return previous;
#ifdef _WIN32
    GROUP_AFFINITY saved{}, selected{};
    selected.Group = static_cast<WORD>(core / 64);
    selected.Mask = KAFFINITY(1) << (core % 64);
    previous.native.resize(2);
    if (SetThreadGroupAffinity(GetCurrentThread(), &selected, &saved)) {
        previous.native[0] = saved.Group;
        previous.native[1] = saved.Mask;
        previous.valid = true;
    }
#elif defined(__linux__)
    const auto count = std::max<long>(sysconf(_SC_NPROCESSORS_CONF), core + 1);
    const auto bytes = CPU_ALLOC_SIZE(count);
    previous.native.resize((bytes + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t));
    auto* saved = reinterpret_cast<cpu_set_t*>(previous.native.data());
    auto* selected = CPU_ALLOC(count);
    if (!selected) throw std::bad_alloc();
    CPU_ZERO_S(bytes, selected);
    CPU_SET_S(core, bytes, selected);
    previous.valid = sched_getaffinity(0, bytes, saved) == 0 && sched_setaffinity(0, bytes, selected) == 0;
    CPU_FREE(selected);
#endif
    return previous;
}
void restore_thread_affinity(const ThreadAffinity& previous) {
    if (!previous.valid) return;
#ifdef _WIN32
    GROUP_AFFINITY saved{};
    saved.Group = static_cast<WORD>(previous.native[0]);
    saved.Mask = static_cast<KAFFINITY>(previous.native[1]);
    SetThreadGroupAffinity(GetCurrentThread(), &saved, nullptr);
#elif defined(__linux__)
    sched_setaffinity(0, previous.native.size() * sizeof(std::uint64_t),
        reinterpret_cast<const cpu_set_t*>(previous.native.data()));
#endif
}

ExpertPool::ExpertPool(int workers, bool pin)
    : topology_(detect_cpu_topology(true)), pin_(pin) {
    if (workers < 0) throw std::invalid_argument("negative CPU worker count");
    if (!workers) workers = static_cast<int>(topology_.worker_cores.size());
    participants_ = workers + 1;
    const auto placements = topology_.worker_cores;
    pool_ = std::make_unique<mfq::HostParallelPool>([placements, pin](std::size_t index) {
        if (pin && index < placements.size()) {
            try { (void)pin_current_thread(placements[index]); }
            catch (const std::bad_alloc&) {} // Optional worker placement.
        }
    },false);
}
ExpertPool::~ExpertPool() = default;
void ExpertPool::rows(std::int64_t count,
        const std::function<void(std::int64_t, std::int64_t)>& callback,
        mfq::HostParallelProfile* profile) {
    if (count <= 0) { if(profile)*profile={}; return; }
    struct Restore {
        ThreadAffinity saved;
        ~Restore() { restore_thread_affinity(saved); }
    } placement{pin_ ? pin_current_thread(topology_.host_core) : ThreadAffinity{}};
    pool_->run(0, count, 1, participants_, callback, profile);
}
ExpertPool& expert_pool() {
    static ExpertPool pool([] {
        const auto* text = std::getenv("MFQ_CPU_EXPERT_THREADS");
        return text ? std::max(1, std::atoi(text)) - 1 : 0;
    }());
    return pool;
}
} // namespace mfq::cpu
