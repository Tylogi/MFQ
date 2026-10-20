#pragma once

#include "ops/moe/moe.h"
#include "slots.h"

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
    std::size_t optional_bytes=0;
    std::uint64_t optional_reclaimed_bytes=0;
    std::size_t managed_bytes=0, managed_peak_bytes=0;
    std::size_t transient_bytes=0, transient_peak_bytes=0;
    std::uint64_t hits=0, misses=0, coalesced=0, evictions=0;
    std::uint64_t promotions=0, gpu_demotions=0, bypasses=0;
    bool reuse_priority=false;
    std::size_t reuse_protected_bytes=0;
    std::uint64_t reuse_promotions=0;
    std::uint64_t scan_admissions=0,scan_admitted_bytes=0;
    std::uint64_t device_duplicate_evictions=0,device_duplicate_evicted_bytes=0;
    std::uint64_t device_reuse_promotions=0;
    std::uint64_t speculative_windows=0,speculative_accepted_keys=0,speculative_cancels=0;
    std::uint64_t prediction_used_jobs=0,prediction_used_bytes=0;
    std::uint64_t prediction_started_jobs=0,prediction_started_bytes=0;
    std::size_t ghost_entries=0,ghost_bytes=0;
    std::uint64_t ghost_promotions=0;
};

// Engine-wide, byte-bounded LRU for immutable CPU expert fields. CPU leases
// and GPU-upload leases share one source key. GPU residents are removed from
// this cache; their eviction can publish a copy without rereading the source.
// Negative source IDs hold optional prefill backups, evictable even when the
// last copies of ordinary experts must remain resident.
class MoeHostExpertCache {
public:
    using Lease=std::shared_ptr<const MoeHostExpert>;
    using Load=std::function<MixedMoePool()>;
    explicit MoeHostExpertCache(std::size_t bytes);
    ~MoeHostExpertCache();
    MoeHostExpertCache(const MoeHostExpertCache&)=delete;
    MoeHostExpertCache& operator=(const MoeHostExpertCache&)=delete;
    // An unconfirmed prediction (touch=false) cannot evict reused images.
    // A later demand can admit that same payload without reading it again.
    Lease acquire(mfq::engine::MoeCacheKey key,std::size_t bytes,const Load& load,bool retain=true,bool touch=true);
    // A prefill scan may fill unused capacity, without evicting existing
    // experts or promoting scan hits. Returns empty without loading if full.
    Lease acquire_scan(mfq::engine::MoeCacheKey key,std::size_t bytes,const Load& load);
    // Returns false without invoking copy when the managed budget cannot fit.
    bool demote(mfq::engine::MoeCacheKey key,std::size_t bytes,const Load& copy);
    void promote(mfq::engine::MoeCacheKey key);
    void erase(mfq::engine::MoeCacheKey key);
    // Drop only idle negative-key backup images, oldest first. Active leases
    // and the last CPU copies of routed experts are never reclaimed here.
    std::size_t reclaim_optional(std::size_t bytes);
    bool contains(mfq::engine::MoeCacheKey key) const;
    // Read an already cached immutable image without invoking a loader.
    Lease find(mfq::engine::MoeCacheKey key,bool touch=true);
    // A dispatch preflight must not wait for another key's RAM publication.
    // Busy keys remain eligible for the normal parallel acquisition path.
    Lease try_find(mfq::engine::MoeCacheKey key,bool touch=true,bool consume=true);
    MoeHostExpertCacheStats stats() const;
    // A streaming scan must not evict repeatedly consumed decoded images.
    void enable_reuse_priority();
    void set_reuse_learning(bool enabled);
    // A prefill scan may replace idle entries from inactive layers. Existing
    // active-layer experts and live CPU/GPU leases remain protected.
    std::vector<int> set_scan_low_priority_sources(std::vector<int> sources);
    // Verification can read rejected draft rows. Keep their immutable images,
    // but learn recency/reuse only from the accepted route prefix. Entries
    // are ordered by token and layer; repeated keys represent real reuse by
    // separate accepted tokens and must not be deduplicated across tokens.
    void begin_speculative_learning();
    void commit_speculative_learning(const std::vector<mfq::engine::MoeCacheKey>& accepted);
    void discard_speculative_learning() noexcept;
    // Complete-residency mode cannot evict the last copy of an expert.
    void preserve_resident_experts();
    // Notifications take a separate lock and never call back into a device
    // cache. Each registered key has one bounded, coalesced pending update.
    class DeviceCacheHints {
    public:
        // Reuse-only GPU observers also confirm reuse of an existing RAM
        // backup, even when GPU hits never read that backup directly.
        explicit DeviceCacheHints(std::vector<mfq::engine::MoeCacheKey> keys,bool confirmed_reuse=false);
        ~DeviceCacheHints();
        void publish(std::size_t index,bool resident);
    private:
        friend class MoeHostExpertCache;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    // Under RAM pressure, discard idle one-use duplicates before unique
    // one-use images. Keep repeatedly consumed RAM backups of revocable GPU
    // storage. Hints affect placement only; the checkpoint is authoritative.
    void set_device_cache_hints(std::shared_ptr<DeviceCacheHints> hints);
    bool exchange(mfq::engine::MoeCacheKey incoming,mfq::engine::MoeCacheKey outgoing,
        const Lease& expected,const std::function<MixedMoePool(const MixedMoePool&)>& copy);
    struct Exchange {
        mfq::engine::MoeCacheKey incoming, outgoing;
        const Lease* expected = nullptr;
        std::function<MixedMoePool(const MixedMoePool&)> copy;
        std::function<void(const MixedMoePool&)> restore;
    };
    // Validate all readers and allocate every lookup entry before reusing RAM.
    // The publication callback must undo its own changes if it throws.
    // Optional prepare runs under reader exclusion and may modify every reused
    // payload. Failure restores every payload before releasing reader exclusion.
    // MFQ_MOE_KEYED_EXCHANGE=1 limits exclusion to these incoming/outgoing keys;
    // unrelated immutable acquisitions can proceed during copying/publication.
    bool exchange_batch(const std::vector<Exchange>& changes,
                        const std::function<void()>& publish = {},
                        const std::function<void()>& prepare = {},bool independent_readers=false);
private:
    Lease load(mfq::engine::MoeCacheKey key,std::size_t bytes,const Load& loader,bool retain,bool scan,bool touch);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
