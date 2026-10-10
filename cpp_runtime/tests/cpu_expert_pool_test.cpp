#include "mfq/cpu_expert_pool.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>
#include <iostream>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
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
#endif
}
}

int main() {
    verify_host_placement();
    // Alternate tiny and large batches to catch stale callback publication.
#ifdef _WIN32
    _putenv_s("MFQ_EXPERT_POOL_SPIN_US","0");
#else
    setenv("MFQ_EXPERT_POOL_SPIN_US","0",1);
#endif
    mfq::cpu::ExpertPool pool(5,false);
    std::vector<std::atomic<int>> seen(8192);
    for(int epoch=0;epoch<2000;++epoch) {
        const int n=epoch%3==0 ? 1 : epoch%3==1 ? 7 : 8192;
        for(int i=0;i<n;++i) seen[i]=0;
        pool.rows(n,[&](auto begin,auto end) {
            for(auto i=begin;i<end;++i) seen[i].fetch_add(1);
        });
        for(int i=0;i<n;++i) assert(seen[i]==1);
    }
    bool threw=false;
    try { pool.rows(100,[](auto,auto){throw std::runtime_error("callback failed");}); }
    catch(const std::runtime_error&) { threw=true; }
    assert(threw);
    std::atomic<int> completed{0};
    auto client=[&] { for(int i=0;i<100;++i) pool.rows(17,[&](auto b,auto e){completed+=static_cast<int>(e-b);}); };
    std::thread a(client),b(client); a.join();b.join();
    assert(completed==3400);
    mfq::HostParallelProfile profile;
    pool.rows(8192,[&](auto begin,auto end){completed+=static_cast<int>(end-begin);},&profile);
    if(profile.rows!=8192 || profile.participants!=6 || profile.worker_arrivals>5 ||
            profile.failed || completed!=3400+8192)
        throw std::runtime_error("expert pool observations changed work");
    pool.rows(0,[](auto,auto){throw std::runtime_error("empty profiled expert work");},&profile);
    if(profile.rows || profile.wall_ns || profile.participants)
        throw std::runtime_error("empty expert profile retained previous dispatch");
    pool.rows(0,[](auto,auto){assert(false);});
}
