#include "mfq/cpu_topology.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace {
void verify_host_placement() {
    const auto first=mfq::cpu::detect_cpu_topology(true);
    const auto last=mfq::cpu::detect_cpu_topology(true,mfq::cpu::PoolAffinity::All,mfq::cpu::HostCore::Last);
    assert(first.worker_cores.size()==last.worker_cores.size());
    if(first.is_hybrid)assert(first.host_core==last.host_core);
#ifdef _WIN32
    const auto selected=[] {
        ULONG count=0;
        if(!GetThreadSelectedCpuSets(GetCurrentThread(),nullptr,0,&count) &&
                GetLastError()!=ERROR_INSUFFICIENT_BUFFER)throw std::runtime_error("cannot inspect caller CPU Sets");
        std::vector<ULONG> ids(count);
        if(count && !GetThreadSelectedCpuSets(GetCurrentThread(),ids.data(),count,&count))
            throw std::runtime_error("cannot read caller CPU Sets");
        ids.resize(count);return ids;
    };
    const auto original=selected();GROUP_AFFINITY original_group{};
    assert(GetThreadGroupAffinity(GetCurrentThread(),&original_group));
    {
        mfq::cpu::ScopedHostAffinity outer(last.host_core);
        const auto outer_sets=selected();assert(outer_sets.size()==1);
        {mfq::cpu::ScopedHostAffinity inner(first.host_core);assert(selected().size()==1);}
        assert(selected()==outer_sets);
        {mfq::cpu::ScopedHostAffinity noop(-1);assert(selected()==outer_sets);}
        assert(!mfq::cpu::pin_current_thread(100000).valid);assert(selected()==outer_sets);
        GROUP_AFFINITY current{};assert(GetThreadGroupAffinity(GetCurrentThread(),&current));
        assert(current.Group==original_group.Group && current.Mask==original_group.Mask);
    }
    assert(selected()==original);
    try {mfq::cpu::ScopedHostAffinity guard(last.host_core);throw std::runtime_error("placement unwind");}
    catch(const std::runtime_error&) {}
    assert(selected()==original);
    GROUP_AFFINITY restored{};assert(GetThreadGroupAffinity(GetCurrentThread(),&restored));
    assert(restored.Group==original_group.Group && restored.Mask==original_group.Mask);
    std::cout<<"Windows caller CPU Sets: nested/default/invalid/exception restoration and hard group mask unchanged PASS\n";
#elif defined(__linux__)
    const auto selected=[] {
        const auto bytes=CPU_ALLOC_SIZE(std::max<long>(1,sysconf(_SC_NPROCESSORS_CONF)));
        std::vector<std::uint64_t> mask((bytes+sizeof(std::uint64_t)-1)/sizeof(std::uint64_t));
        assert(sched_getaffinity(0,bytes,reinterpret_cast<cpu_set_t*>(mask.data()))==0);
        return mask;
    };
    const auto original=selected();
    {
        mfq::cpu::ScopedHostAffinity outer(last.host_core);
        const auto outer_mask=selected();
        assert(CPU_COUNT_S(outer_mask.size()*sizeof(std::uint64_t),
            reinterpret_cast<const cpu_set_t*>(outer_mask.data()))==1);
        assert(CPU_ISSET_S(last.host_core,outer_mask.size()*sizeof(std::uint64_t),
            reinterpret_cast<const cpu_set_t*>(outer_mask.data())));
        {
            mfq::cpu::ScopedHostAffinity inner(first.host_core);
            const auto inner_mask=selected();
            assert(CPU_ISSET_S(first.host_core,inner_mask.size()*sizeof(std::uint64_t),
                reinterpret_cast<const cpu_set_t*>(inner_mask.data())));
        }
        assert(selected()==outer_mask);
        {mfq::cpu::ScopedHostAffinity noop(-1);assert(selected()==outer_mask);}
        assert(!mfq::cpu::pin_current_thread(100000).valid);assert(selected()==outer_mask);
    }
    assert(selected()==original);
    try {mfq::cpu::ScopedHostAffinity guard(last.host_core);throw std::runtime_error("placement unwind");}
    catch(const std::runtime_error&) {}
    assert(selected()==original);
    std::cout<<"Linux caller affinity: nested/noop/invalid/exception restoration PASS\n";
#endif
}

void set_host_core(const char* value) {
#ifdef _WIN32
    assert(_putenv_s("MFQ_CPU_HOST_CORE",value ? value : "")==0);
#else
    assert((value ? setenv("MFQ_CPU_HOST_CORE",value,1) : unsetenv("MFQ_CPU_HOST_CORE"))==0);
#endif
}
}

int main() {
    verify_host_placement();
    const auto all=mfq::cpu::detect_cpu_topology(false);
    assert(!all.worker_cores.empty());
    assert(all.host_core==-1);
    assert(mfq::cpu::physical_cores(false)==all.worker_cores);
    const auto* previous=std::getenv("MFQ_CPU_HOST_CORE");
    const bool had_previous=previous!=nullptr;
    const std::string saved=previous ? previous : "";
    set_host_core(nullptr);
    assert(mfq::cpu::host_core_setting()==mfq::cpu::HostCore::First);
    for(const auto setting : {mfq::cpu::HostCore::First,mfq::cpu::HostCore::Last,mfq::cpu::HostCore::First}) {
        set_host_core(setting==mfq::cpu::HostCore::First ? "first" : "last");
        const auto expected=mfq::cpu::detect_cpu_topology(true,mfq::cpu::PoolAffinity::All,setting).host_core;
        assert(mfq::cpu::planned_host_core()==expected);
        assert(mfq::cpu::planned_host_core()==expected);
    }
    set_host_core("invalid");
    bool rejected=false;
    try { (void)mfq::cpu::planned_host_core(); }
    catch(const std::invalid_argument& error) {
        rejected=std::string(error.what())=="MFQ_CPU_HOST_CORE must be first or last";
    }
    assert(rejected);
    set_host_core(had_previous ? saved.c_str() : nullptr);
    std::cout<<"CPU topology and host placement settings PASS\n";
}
