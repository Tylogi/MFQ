#pragma once

#include "moe.h"
#include "moe_cache_policy.h"

#include <functional>
#include <memory>
#include <vector>

struct MoeHostExpertReservation;

// The reservation is destroyed after the last immutable weight reference.
// Leases keep an evicted/promoted payload charged until its CPU work or
// pinned-stage copy completes. Shared pool tables are accounted separately.
struct MoeHostExpert {
    std::shared_ptr<MoeHostExpertReservation> reservation;
    MixedMoePool weights;
    std::size_t bytes=0;
};

struct MoeHostExpertCacheStats {
    std::size_t budget_bytes=0, resident_bytes=0, resident_entries=0;
    std::size_t managed_bytes=0, managed_peak_bytes=0;
    std::size_t transient_bytes=0, transient_peak_bytes=0;
    std::uint64_t hits=0, misses=0, coalesced=0, evictions=0;
    std::uint64_t promotions=0, gpu_demotions=0, bypasses=0;
};

// Engine-wide, byte-bounded LRU for immutable CPU expert fields. CPU leases
// and GPU-upload leases share one source key. GPU residents are removed from
// this cache; their eviction can publish a copy without rereading the source.
class MoeHostExpertCache {
public:
    using Lease=std::shared_ptr<const MoeHostExpert>;
    using Load=std::function<MixedMoePool()>;
    explicit MoeHostExpertCache(std::size_t bytes);
    ~MoeHostExpertCache();
    MoeHostExpertCache(const MoeHostExpertCache&)=delete;
    MoeHostExpertCache& operator=(const MoeHostExpertCache&)=delete;
    Lease acquire(mfq::MoeCacheKey key,std::size_t bytes,const Load& load,bool retain=true);
    // Returns false without invoking copy when the managed budget cannot fit.
    bool demote(mfq::MoeCacheKey key,std::size_t bytes,const Load& copy);
    void promote(mfq::MoeCacheKey key);
    void erase(mfq::MoeCacheKey key);
    bool contains(mfq::MoeCacheKey key) const;
    MoeHostExpertCacheStats stats() const;
    // Complete-residency mode cannot evict the last copy of an expert.
    void preserve_resident_experts();
    bool exchange(mfq::MoeCacheKey incoming,mfq::MoeCacheKey outgoing,
        const Lease& expected,const std::function<MixedMoePool(const MixedMoePool&)>& copy);
    struct Exchange {
        mfq::MoeCacheKey incoming, outgoing;
        const Lease* expected = nullptr;
        std::function<MixedMoePool(const MixedMoePool&)> copy;
        std::function<void(const MixedMoePool&)> restore;
    };
    // Validate all readers and allocate every lookup entry before reusing RAM.
    // The publication callback must undo its own changes if it throws.
    // Optional prepare runs under reader exclusion and may modify every reused
    // payload. Any later failure restores all payloads before releasing the lock.
    bool exchange_batch(const std::vector<Exchange>& changes,
                        const std::function<void()>& publish = {},
                        const std::function<void()>& prepare = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
