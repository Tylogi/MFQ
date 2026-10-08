#include "mfq/cpu_expert_pool.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
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
