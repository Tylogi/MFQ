#include "mfq/moe/cpu_expert_pool.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace mfq::cpu {
namespace {
void pin_worker_thread(int core) {
#ifdef _WIN32
    GROUP_AFFINITY target{};target.Group=static_cast<WORD>(core/64);
    target.Mask=KAFFINITY(1)<<(core&63);
    (void)SetThreadGroupAffinity(GetCurrentThread(),&target,nullptr);
#else
    (void)pin_current_thread(core);
#endif
}
}

ExpertPool::ExpertPool(int workers, bool pin)
    : topology_(detect_cpu_topology(true,PoolAffinity::All,host_core_setting())), pin_(pin) {
    if (workers < 0) throw std::invalid_argument("negative CPU worker count");
    if (!workers) workers = static_cast<int>(topology_.worker_cores.size());
    participants_ = workers + 1;
    const auto placements = topology_.worker_cores;
    pool_ = std::make_unique<mfq::HostParallelPool>([placements, pin](std::size_t index) {
        if (pin && index < placements.size()) {
            try { pin_worker_thread(placements[index]); }
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
