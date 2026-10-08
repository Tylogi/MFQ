#include "runtime/moe_host_expert_cache.h"

#include <chrono>
#include <array>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace tb=mfq_tensor_backend;
namespace {
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
MixedMoePool weight(std::size_t bytes,int marker) {
    MixedMoePool result;
    result.local_experts=1;
    result.nint.q_packed=tb::full({1,static_cast<std::int64_t>(bytes)},marker,
        tb::TensorOptions().dtype(tb::kUInt8));
    return result;
}
mfq::MoeCacheKey key(int expert,int source=0) { return {source,0,expert}; }
void bytes_equal(MoeHostExpertCache::Lease value,int marker) {
    check(value && value->weights.nint.q_packed.numel()==value->bytes,"host lease shape mismatch");
    const auto* data=value->weights.nint.q_packed.data_ptr<std::uint8_t>();
    for (std::size_t i=0; i<value->bytes; ++i) check(data[i]==marker,"host lease payload changed");
}
void wait_coalesced(const MoeHostExpertCache& cache,std::promise<void>& release) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while (!cache.stats().coalesced && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
    release.set_value();
    check(cache.stats().coalesced==1,"same-key host read did not coalesce");
}
}

int main() try {
    {
        MoeHostExpertCache cache(1000);
        cache.acquire(key(0),400,[] { return weight(400,1); });
        cache.acquire(key(1),300,[] { return weight(300,2); });
        cache.acquire(key(2),300,[] { return weight(300,3); });
        auto pinned=cache.acquire(key(0),400,[] { throw std::runtime_error("unexpected reread"); return weight(400,0); });
        cache.acquire(key(3),500,[] { return weight(500,4); });
        check(cache.contains(key(0)) && cache.contains(key(3)) &&
            !cache.contains(key(1)) && !cache.contains(key(2)),"byte LRU evicted the wrong entries");
        bytes_equal(pinned,1);
        cache.promote(key(0));
        check(cache.stats().managed_bytes==900,"promoted pinned lease was no longer charged");
        auto temporary=cache.acquire(key(4),800,[] { return weight(800,5); });
        check(!cache.contains(key(4)) && cache.stats().managed_bytes==400 &&
            cache.stats().transient_bytes==800,"pinned budget pressure exceeded managed capacity");
        pinned.reset(); temporary.reset();
        check(!cache.stats().managed_bytes && !cache.stats().transient_bytes,"host leases did not release their reservations");
        check(cache.stats().managed_peak_bytes<=1000,"managed RAM exceeded byte budget");
    }
    {
        MoeHostExpertCache cache(512);
        int copies=0;
        check(cache.demote(key(0),512,[&] { ++copies; return weight(512,9); }),"GPU victim was not admitted");
        auto lease=cache.acquire(key(0),512,[] { throw std::runtime_error("demoted weight reread SSD"); return weight(512,0); });
        bytes_equal(lease,9);
        cache.promote(key(0));
        check(!cache.contains(key(0)) && cache.stats().managed_bytes==512,"GPU promotion retained a persistent RAM duplicate");
        check(!cache.demote(key(1),512,[&] { ++copies; return weight(512,1); }) && copies==1,
            "pinned budget overflow performed an unnecessary GPU readback");
        lease.reset();
        check(cache.demote(key(1),512,[&] { ++copies; return weight(512,1); }),"released capacity was not reused");
        auto distinct=cache.acquire(key(1,1),512,[] { return weight(512,2); });
        bytes_equal(distinct,2);
        check(!cache.contains(key(1)) && cache.contains(key(1,1)),"source IDs shared a host cache entry");
    }
    {
        for (std::size_t budget:{0,128}) {
            MoeHostExpertCache cache(budget);
            auto lease=cache.acquire(key(0),512,[] { return weight(512,6); });
            check(!cache.stats().resident_entries && !cache.stats().managed_bytes &&
                cache.stats().transient_bytes==512,"oversized/disabled cache did not use bounded-route scratch");
            lease.reset();
            check(!cache.stats().transient_bytes,"uncached scratch reservation leaked");
        }
    }
    for (bool fail:{false,true}) {
        MoeHostExpertCache cache(64);
        std::promise<void> started,release;
        auto gate=release.get_future().share();
        int reads=0;
        auto first=std::async(std::launch::async,[&] {
            return cache.acquire(key(0),64,[&] {
                ++reads; started.set_value(); gate.wait();
                if (fail) throw std::runtime_error("injected host source failure");
                return weight(64,7);
            });
        });
        started.get_future().wait();
        auto second=std::async(std::launch::async,[&] {
            return cache.acquire(key(0),64,[] { throw std::runtime_error("duplicate host source read"); return weight(64,0); });
        });
        wait_coalesced(cache,release);
        if (fail) {
            for (auto* future:{&first,&second}) {
                bool rejected=false;
                try { future->get(); } catch (const std::runtime_error& error) {
                    rejected=std::string(error.what())=="injected host source failure";
                }
                check(rejected,"coalesced read failure was not propagated");
            }
            check(!cache.stats().managed_bytes && !cache.stats().resident_entries,"failed read left a reservation or resident entry");
            bytes_equal(cache.acquire(key(0),64,[] { return weight(64,8); }),8);
        } else {
            auto a=first.get(),b=second.get();
            check(a.get()==b.get() && reads==1,"coalesced readers did not share immutable data");
            bytes_equal(a,7);
        }
    }
    {
        MoeHostExpertCache cache(64);
        std::promise<void> started,release;
        auto gate=release.get_future().share();
        auto pending=std::async(std::launch::async,[&] {
            return cache.acquire(key(0),64,[&] { started.set_value(); gate.wait(); return weight(64,3); });
        });
        started.get_future().wait();
        cache.promote(key(0));
        release.set_value();
        auto lease=pending.get();
        check(!cache.contains(key(0)) && cache.stats().managed_bytes==64,"inflight promotion published a stale RAM duplicate");
        lease.reset();
        check(!cache.stats().managed_bytes,"inflight promotion lost its reservation owner");
    }
    {
        MoeHostExpertCache::Lease retained;
        {
            MoeHostExpertCache cache(64);
            retained=cache.acquire(key(0),64,[] { return weight(64,4); });
        }
        bytes_equal(retained,4);
    }
    std::cout<<"host expert byte LRU, leases, promotion/demotion, concurrent coalescing and failed-read recovery passed\n";
    {
        MoeHostExpertCache cache(128);
        cache.acquire(key(0),64,[] { return weight(64,1); });
        cache.acquire(key(1),64,[] { return weight(64,2); });
        cache.preserve_resident_experts();
        auto scratch=cache.acquire(key(2),64,[] { return weight(64,3); });
        check(cache.contains(key(0)) && cache.contains(key(1)) && !cache.contains(key(2)) &&
            cache.stats().evictions==0,"complete residency evicted the last RAM expert copy");
        scratch.reset();
        cache.promote(key(0));
        check(cache.demote(key(2),64,[] { return weight(64,3); }),"complete residency did not reuse promoted capacity");
    }
    {
        MoeHostExpertCache cache(64);cache.preserve_resident_experts();
        auto incoming=cache.acquire(key(0),64,[]{return weight(64,1);});
        auto reader=incoming;int copies=0;
        auto exchange=[&](const MixedMoePool& source){
            ++copies;auto value=source;
            std::memset(value.nint.q_packed.data_ptr(),9,64);return value;
        };
        check(!cache.exchange(key(0),key(1),incoming,exchange) && copies==0,"tier exchange overwrote a live CPU lease");
        reader.reset();const auto* before=incoming->weights.nint.q_packed.data_ptr();
        bool failed=false;
        try{cache.exchange(key(0),key(1),incoming,[](const auto&)->MixedMoePool{throw std::runtime_error("copy failed");});}
        catch(const std::runtime_error&){failed=true;}
        check(failed && cache.contains(key(0)) && !cache.contains(key(1)),"failed tier exchange changed RAM ownership");
        check(cache.exchange(key(0),key(1),incoming,exchange),"tier exchange failed without other readers");
        incoming.reset();auto outgoing=cache.acquire(key(1),64,[]{throw std::runtime_error("exchange reread SSD");return weight(64,0);});
        bytes_equal(outgoing,9);
        check(outgoing->weights.nint.q_packed.data_ptr()==before && cache.stats().managed_bytes==64 &&
            cache.stats().managed_peak_bytes==64 && !cache.contains(key(0)),"tier exchange grew or copied the RAM allocation");
    }
    int batch_cases=0;
    for(bool bulk:{false,true})for(int fail_at:{-2,0,1,2,3,-1}) {
        if(!bulk && fail_at==-2)continue;
        MoeHostExpertCache cache(192);cache.preserve_resident_experts();
        std::array<MoeHostExpertCache::Lease,3> held;
        for(int i=0;i<3;++i)held[i]=cache.acquire(key(i),64,[i]{return weight(64,i+1);});
        std::vector<MoeHostExpertCache::Exchange> changes;
        for(int i=0;i<3;++i)changes.push_back({key(i),key(i+3),&held[i],
            [i,fail_at,bulk](const MixedMoePool& original) {
                auto result=original;if(!bulk)std::memset(result.nint.q_packed.data_ptr(),i+7,64);
                if(fail_at==i)throw std::runtime_error("partial RAM reuse failure");return result;
            },[i](const MixedMoePool& original){std::memset(original.nint.q_packed.data_ptr(),i+1,64);}});
        int copied=0;
        std::function<void()> prepare;
        if(bulk)prepare=[&] {
            ++copied;
            for(int i=0;i<3;++i) {
                std::memset(held[i]->weights.nint.q_packed.data_ptr(),i+7,64);
                if(fail_at==-2 && i==1)throw std::runtime_error("partial bulk RAM reuse failure");
            }
        };
        auto reader=held[1];
        check(!cache.exchange_batch(changes,[&]{++copied;},prepare),"batch reused a CPU reader's fields");
        check(!copied,"rejected batch published a GPU mapping");reader.reset();
        bool failed=false;
        try {check(cache.exchange_batch(changes,[&]{if(fail_at==3)throw std::runtime_error("publication failure");},prepare),"batch was rejected without readers");}
        catch(const std::runtime_error&){failed=true;}
        check(failed==(fail_at>=0 || fail_at==-2),"batch failure did not propagate");
        for(int i=0;i<3;++i) {
            if(failed){check(cache.contains(key(i)) && !cache.contains(key(i+3)),"batch rollback changed an entry");bytes_equal(held[i],i+1);}
            else {
                held[i].reset();auto value=cache.acquire(key(i+3),64,[]{throw std::runtime_error("batch reread SSD");return weight(64,0);});
                bytes_equal(value,i+7);check(!cache.contains(key(i)),"batch retained the promoted key");
            }
        }
        check(cache.stats().managed_bytes==192 && cache.stats().managed_peak_bytes==192,"batch duplicated RAM payloads");
        ++batch_cases;
    }
    std::cout<<"atomic RAM exchange rollback, reader exclusion and publication checks passed batch_cases="<<batch_cases<<"\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
