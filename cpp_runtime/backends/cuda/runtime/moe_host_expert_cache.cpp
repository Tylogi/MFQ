#include "moe_host_expert_cache.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {
struct Ledger {
    std::atomic<std::size_t> managed{0},managed_peak{0},transient{0},transient_peak{0};
};
void charge(std::atomic<std::size_t>& used,std::atomic<std::size_t>& peak,std::size_t bytes) {
    const auto value=used.fetch_add(bytes)+bytes;
    auto old=peak.load();
    while (old<value && !peak.compare_exchange_weak(old,value)) {}
}
}

struct MoeHostExpertReservation {
    std::shared_ptr<Ledger> ledger;
    std::size_t bytes;
    bool managed;
    bool demand_admission=false;
    bool unconsumed_prediction=false;
    MoeHostExpertReservation(std::shared_ptr<Ledger> value,std::size_t size,bool retained)
        :ledger(std::move(value)),bytes(size),managed(retained) {
        if (managed) charge(ledger->managed,ledger->managed_peak,bytes);
        else charge(ledger->transient,ledger->transient_peak,bytes);
    }
    ~MoeHostExpertReservation() {
        (managed ? ledger->managed : ledger->transient).fetch_sub(bytes);
    }
};

struct MoeHostExpertCache::DeviceCacheHints::Impl {
    std::mutex mutex;
    std::vector<mfq::MoeCacheKey> keys;
    std::vector<std::uint8_t> flags;
    std::vector<std::size_t> pending;
    const bool confirmed_reuse;
    explicit Impl(std::vector<mfq::MoeCacheKey> value,bool reuse)
        :keys(std::move(value)),flags(keys.size()),confirmed_reuse(reuse) {
        pending.reserve(keys.size());
    }
};
MoeHostExpertCache::DeviceCacheHints::DeviceCacheHints(std::vector<mfq::MoeCacheKey> keys,bool confirmed_reuse)
    :impl_(std::make_unique<Impl>(std::move(keys),confirmed_reuse)) {}
MoeHostExpertCache::DeviceCacheHints::~DeviceCacheHints()=default;
void MoeHostExpertCache::DeviceCacheHints::publish(std::size_t index,bool resident) {
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    auto& flags=state.flags.at(index);
    if(bool(flags&1)==resident)return;
    if(!(flags&2))state.pending.push_back(index);
    flags=std::uint8_t(2|(resident?1:0));
}

struct MoeHostExpertCache::Impl {
    struct Entry {
        Lease value;std::list<mfq::MoeCacheKey>::iterator position;bool reused=false;
        Entry *device_prev=nullptr,*device_next=nullptr;bool device_duplicate=false;
        std::uint64_t birth_epoch=0;
        std::uint8_t reuse_credit=0;
        bool ghost_eligible=true;
        bool unconfirmed_prediction=false;
    };
    struct Flight {
        bool ready=false;
        bool retain=true;
        Lease value;
        std::exception_ptr error;
        std::condition_variable condition;
    };
    const std::size_t budget;
    std::shared_ptr<Ledger> ledger=std::make_shared<Ledger>();
    mutable std::mutex mutex;
    std::list<mfq::MoeCacheKey> lru;
    std::list<mfq::MoeCacheKey> reused;
    std::unordered_map<mfq::MoeCacheKey,Entry,mfq::MoeCacheKeyHash> entries;
    struct Ghost {std::size_t bytes;std::list<mfq::MoeCacheKey>::iterator position;};
    std::list<mfq::MoeCacheKey> ghost_lru;
    std::unordered_map<mfq::MoeCacheKey,Ghost,mfq::MoeCacheKeyHash> ghosts;
    std::size_t ghost_bytes=0;
    std::unordered_map<mfq::MoeCacheKey,std::shared_ptr<Flight>,mfq::MoeCacheKeyHash> flights;
    // Bypassed payloads still have an active consumer. Reuse that immutable
    // lease even when the RAM budget cannot retain it in the LRU.
    std::unordered_map<mfq::MoeCacheKey,std::weak_ptr<const MoeHostExpert>,mfq::MoeCacheKeyHash> transient;
    std::unordered_set<mfq::MoeCacheKey,mfq::MoeCacheKeyHash> exchanging;
    std::condition_variable exchange_complete;
    std::size_t exchange_placeholders=0;
    MoeHostExpertCacheStats counters;
    std::vector<int> scan_low_priority_sources;
    std::vector<mfq::MoeCacheKey> scan_victims;
    std::size_t scan_victim_cursor=0;
    bool preserve=false,reuse_learning=true;
    bool speculative_learning=false;
    std::uint64_t learning_epoch=0;
    std::shared_ptr<DeviceCacheHints> device_hints;
    std::unordered_set<mfq::MoeCacheKey,mfq::MoeCacheKeyHash> device_cached;
    Entry *device_head=nullptr,*device_tail=nullptr;

    explicit Impl(std::size_t bytes):budget(bytes) {}
    void forget_ghost(mfq::MoeCacheKey key) {
        const auto found=ghosts.find(key);if(found==ghosts.end())return;
        ghost_bytes-=found->second.bytes;ghost_lru.erase(found->second.position);ghosts.erase(found);
    }
    void remember_eviction(mfq::MoeCacheKey key,const Entry& entry) {
        // GPU duplicates and unconsumed predictions are not evidence that a
        // unique RAM expert will be needed again. Remember only probation
        // payloads actually acquired by a consumer, without keeping data.
        if(key.source<0 || entry.reused || !entry.ghost_eligible || device_cached.contains(key) ||
                entry.value->reservation->unconsumed_prediction)return;
        forget_ghost(key);ghost_lru.push_front(key);
        try {ghosts.emplace(key,Ghost{entry.value->bytes,ghost_lru.begin()});}
        catch(...) {ghost_lru.pop_front();throw;}
        ghost_bytes+=entry.value->bytes;
        while(ghost_bytes>budget || ghosts.size()>65536)forget_ghost(ghost_lru.back());
    }
    void demand(const Lease& value) {
        if(!value->reservation->unconsumed_prediction)return;
        value->reservation->unconsumed_prediction=false;
        ++counters.prediction_used_jobs;counters.prediction_used_bytes+=value->bytes;
    }
    void wait_exchange(std::unique_lock<std::mutex>& lock,mfq::MoeCacheKey key) {
        if(exchanging.contains(key))exchange_complete.wait(lock,[&]{return !exchanging.contains(key);});
    }
    void device_duplicate(Entry& entry,bool valid) noexcept {
        // Borrowed GPU storage can be revoked by prefill or KV growth. Keep
        // repeatedly consumed RAM backups ahead of one-use unique images.
        valid=valid && !entry.reused;
        if(entry.device_duplicate==valid)return;
        if(valid) {
            entry.device_prev=nullptr;entry.device_next=device_head;
            if(device_head)device_head->device_prev=&entry;else device_tail=&entry;
            device_head=&entry;
        }else {
            if(entry.device_prev)entry.device_prev->device_next=entry.device_next;else device_head=entry.device_next;
            if(entry.device_next)entry.device_next->device_prev=entry.device_prev;else device_tail=entry.device_prev;
            entry.device_prev=entry.device_next=nullptr;
        }
        entry.device_duplicate=valid;
    }
    void refresh_device_hints() {
        if(!device_hints)return;
        auto& hints=*device_hints->impl_;std::lock_guard lock(hints.mutex);
        for(const auto index:hints.pending) {
            const auto key=hints.keys[index];const bool resident=hints.flags[index]&1;
            if(resident)device_cached.insert(key);else device_cached.erase(key);
            if(!exchanging.contains(key))if(auto found=entries.find(key);found!=entries.end()) {
                device_duplicate(found->second,resident);
                if(resident)learn_device_reuse(key,found->second);
            }
            hints.flags[index]&=1;
        }
        hints.pending.clear();
    }
    void remove(decltype(entries)::iterator found) {
        device_duplicate(found->second,false);
        counters.resident_bytes-=found->second.value->bytes;
        if(found->first.source<0)counters.optional_bytes-=found->second.value->bytes;
        if(found->second.reused)counters.reuse_protected_bytes-=found->second.value->bytes;
        (found->second.reused?reused:lru).erase(found->second.position);
        entries.erase(found);
    }
    bool confirm_prediction_first_use(Entry& entry) {
        if(!entry.unconfirmed_prediction)return false;
        entry.unconfirmed_prediction=false;
        if(counters.reuse_priority && !counters.prediction_first_uses++)std::fprintf(stderr,
            "moe_ram_prediction_first_use scope=first_confirmation enabled=1 reused=0\n");
        return true;
    }
    void touch(mfq::MoeCacheKey key,Entry& entry) {
        if(!reuse_learning || speculative_learning)return;
        entry.ghost_eligible=true;
        const bool refault=ghosts.contains(key);
        if(refault){forget_ghost(key);++counters.ghost_promotions;}
        if(entry.device_duplicate){device_duplicate(entry,false);device_duplicate(entry,true);}
        if(!refault && confirm_prediction_first_use(entry)) {
            // The forecast read was an admission, not a previous consumer.
            // One accepted use earns recency; only another use earns reuse.
            lru.splice(lru.begin(),lru,entry.position);return;
        }
        entry.unconfirmed_prediction=false;
        if(!counters.reuse_priority || key.source<0) {
            lru.splice(lru.begin(),lru,entry.position);return;
        }
        if(!entry.reused) {
            reused.splice(reused.begin(),lru,entry.position);entry.reuse_credit=0;
            entry.reused=true;counters.reuse_protected_bytes+=entry.value->bytes;++counters.reuse_promotions;
            device_duplicate(entry,false);
        }else entry.reuse_credit=std::min<std::uint8_t>(3,entry.reuse_credit+1);
        // Leave a probation window for new routes without allowing one-use
        // expert scans to consume a quarter of the byte-bounded RAM cache.
        // Repeated demand and confirmed refaults retain most of the payloads.
        const auto limit=budget-budget/10;
        while(counters.reuse_protected_bytes>limit && !reused.empty()) {
            auto candidate=reused.end();
            do {--candidate;}while(candidate!=reused.begin() && exchanging.contains(*candidate));
            if(exchanging.contains(*candidate))break;
            auto& old=entries.at(*candidate);
            // Repeated consumers earn bounded second chances. A stream of
            // lightly reused experts cannot age out a frequent route solely
            // by touching it later. Credits decay at replacement, without
            // timers, a history heap or additional payload storage.
            if(old.reuse_credit) {
                --old.reuse_credit;reused.splice(reused.begin(),reused,old.position);
                if(!counters.reuse_clock_rotations++)std::fprintf(stderr,
                    "moe_ram_reuse_clock scope=first_rotation enabled=1 maximum_credit=3\n");
                continue;
            }
            counters.reuse_protected_bytes-=old.value->bytes;old.reused=false;
            lru.splice(lru.begin(),reused,old.position);
            device_duplicate(old,device_cached.contains(*old.position));
        }
    }
    void learn_device_reuse(mfq::MoeCacheKey key,Entry& entry) {
        if(!counters.reuse_priority || entry.reused || !device_hints || !device_hints->impl_->confirmed_reuse)return;
        if(!reuse_learning || speculative_learning)return;
        // Repeated GPU hits must not leave a hot expert's revocable RAM
        // backup permanently classified as a one-use image. Warmup and
        // uncommitted speculation retain the normal learning exclusions.
        const auto before=counters.reuse_promotions;
        entry.unconfirmed_prediction=false; // The GPU has confirmed multiple consumers.
        touch(key,entry);
        if(counters.reuse_promotions!=before && ++counters.device_reuse_promotions==1)
            std::fprintf(stderr,"moe_ram_gpu_reuse_dispatch scope=first_promotion enabled=1\n");
    }
    bool make_scan_room(mfq::MoeCacheKey incoming,std::size_t bytes) {
        if(!bytes || bytes>budget)return false;
        if(ledger->managed.load()<=budget-bytes)return true;
        if(preserve || std::binary_search(scan_low_priority_sources.begin(),
                                          scan_low_priority_sources.end(),incoming.source))return false;
        // This bounded snapshot is built once at the phase boundary, rather
        // than scanning all cached experts for every prefill miss.
        while(ledger->managed.load()>budget-bytes && scan_victim_cursor<scan_victims.size()) {
            const auto key=scan_victims[scan_victim_cursor++];
            const auto found=entries.find(key);
            if(found==entries.end() || exchanging.contains(key) || found->second.value.use_count()!=1)continue;
            remember_eviction(key,found->second);
            remove(found);++counters.evictions;
        }
        return ledger->managed.load()<=budget-bytes;
    }
    bool make_room(std::size_t bytes,bool prediction=false) {
        if (!bytes || bytes>budget) return false;
        refresh_device_hints();
        // Keep the ordinary two-queue policy among unique RAM images. A
        // separate intrusive queue avoids scanning the whole RAM cache on
        // every miss, and skips leases and transactions just like normal LRU.
        for(auto* candidate=device_tail;ledger->managed.load()>budget-bytes && candidate;) {
            auto* previous=candidate->device_prev;const auto key=*candidate->position;
            if(!exchanging.contains(key) && !(preserve && key.source>=0) && candidate->value.use_count()==1) {
                counters.device_duplicate_evicted_bytes+=candidate->value->bytes;
                remove(entries.find(key));++counters.evictions;
                if(++counters.device_duplicate_evictions==1)std::fprintf(stderr,
                    "moe_ram_device_duplicate_dispatch scope=first_eviction enabled=1\n");
            }
            candidate=previous;
        }
        if(ledger->managed.load()<=budget-bytes)return true;
        for(auto* queue:{&lru,&reused}) {
          if(prediction && queue==&reused)break;
          auto candidate=queue->end();
          while (ledger->managed.load()>budget-bytes && candidate!=queue->begin()) {
            --candidate;
            if(exchanging.contains(*candidate))continue;
            auto found=entries.find(*candidate);
            // Negative source IDs are reusable prefill backup images. Their
            // GPU copy still exists outside the loan; discard them before a
            // request needs more RAM, even in complete-residency mode.
            if (preserve && candidate->source>=0) continue;
            // Externally held payloads stay charged even after removal. Keep
            // their lookup entry rather than evicting useful pinned data.
            if (found->second.value.use_count()!=1) continue;
            remember_eviction(*candidate,found->second);
            device_duplicate(found->second,false);
            candidate=queue->erase(candidate);
            counters.resident_bytes-=found->second.value->bytes;
            if(found->first.source<0)counters.optional_bytes-=found->second.value->bytes;
            if(found->second.reused)counters.reuse_protected_bytes-=found->second.value->bytes;
            entries.erase(found);
            ++counters.evictions;
          }
          if(ledger->managed.load()<=budget-bytes)return true;
        }
        return false;
    }
    void publish(mfq::MoeCacheKey key,Lease value) {
        lru.push_front(key);
        try {
            const auto [found,inserted]=entries.emplace(key,Entry{value,lru.begin()});
            if (!inserted)
                throw std::logic_error("duplicate host expert publication");
            found->second.birth_epoch=speculative_learning?learning_epoch:0;
            found->second.unconfirmed_prediction=value->reservation->unconsumed_prediction;
            found->second.ghost_eligible=reuse_learning && !speculative_learning && !found->second.unconfirmed_prediction;
            device_duplicate(found->second,device_cached.contains(key));
        }
        catch (...) { lru.pop_front(); throw; }
        counters.resident_bytes+=value->bytes;
        if(key.source<0)counters.optional_bytes+=value->bytes;
        if(device_cached.contains(key))learn_device_reuse(key,entries.at(key));
    }
    void admit_prediction(mfq::MoeCacheKey key,const Lease& value) {
        if(const auto found=entries.find(key);found!=entries.end()) {
            touch(key,found->second);return;
        }
        auto& reservation=*value->reservation;
        if(!reservation.demand_admission || !make_room(value->bytes))return;
        // The prediction already owns the immutable payload. Admit its first
        // confirmed use without another read or a second allocation.
        publish(key,value);
        charge(ledger->managed,ledger->managed_peak,value->bytes);
        ledger->transient.fetch_sub(value->bytes);
        reservation.managed=true;reservation.demand_admission=false;
        transient.erase(key);
    }
};

MoeHostExpertCache::MoeHostExpertCache(std::size_t bytes):impl_(std::make_unique<Impl>(bytes)) {}
MoeHostExpertCache::~MoeHostExpertCache()=default;
void MoeHostExpertCache::enable_reuse_priority() {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->counters.reuse_priority)std::fprintf(stderr,
        "moe_ram_reuse_dispatch enabled=1 protected_budget_bytes=%zu probation_budget_bytes=%zu\n",
        impl_->budget-impl_->budget/10,impl_->budget/10);
    impl_->counters.reuse_priority=true;
}
void MoeHostExpertCache::set_reuse_learning(bool enabled) {
    std::lock_guard lock(impl_->mutex);impl_->reuse_learning=enabled;
}
std::vector<int> MoeHostExpertCache::set_scan_low_priority_sources(std::vector<int> sources) {
    std::sort(sources.begin(),sources.end());
    sources.erase(std::unique(sources.begin(),sources.end()),sources.end());
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    std::vector<mfq::MoeCacheKey> victims;
    if(!sources.empty()) {
        victims.reserve(state.entries.size());
        for(auto* queue:{&state.lru,&state.reused})
            for(auto it=queue->rbegin();it!=queue->rend();++it)
                if(std::binary_search(sources.begin(),sources.end(),it->source))victims.push_back(*it);
    }
    auto previous=std::move(state.scan_low_priority_sources);
    state.scan_low_priority_sources=std::move(sources);
    state.scan_victims=std::move(victims);state.scan_victim_cursor=0;
    return previous;
}
void MoeHostExpertCache::begin_speculative_learning() {
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    if(state.speculative_learning)throw std::logic_error("nested RAM expert cache speculation");
    if(++state.learning_epoch==0) {
        for(auto& [key,entry]:state.entries)entry.birth_epoch=0;
        state.learning_epoch=1;
    }
    state.speculative_learning=true;
}
void MoeHostExpertCache::commit_speculative_learning(const std::vector<mfq::MoeCacheKey>& accepted) {
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    if(!state.speculative_learning)return;
    state.speculative_learning=false;
    state.counters.speculative_accepted_keys+=accepted.size();
    if(++state.counters.speculative_windows==1)std::fprintf(stderr,
        "moe_ram_speculative_learning scope=first_commit enabled=1 accepted_keys=%zu\n",accepted.size());
    if(!state.reuse_learning)return;
    for(const auto key:accepted)if(!state.exchanging.contains(key)) {
        const auto found=state.entries.find(key);if(found==state.entries.end())continue;
        auto& entry=found->second;
        if(entry.birth_epoch!=state.learning_epoch || state.ghosts.contains(key))state.touch(key,entry);
        else {
            // The first accepted token confirms admission. A later accepted
            // token can promote this image just as a second ordinary decode
            // would; rejected rows never appear in this ordered sequence.
            state.lru.splice(state.lru.begin(),state.lru,entry.position);
            entry.birth_epoch=0;
            entry.ghost_eligible=true;
            state.confirm_prediction_first_use(entry);
            if(entry.device_duplicate){state.device_duplicate(entry,false);state.device_duplicate(entry,true);}
        }
    }
}
void MoeHostExpertCache::discard_speculative_learning() noexcept {
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    if(state.speculative_learning)++state.counters.speculative_cancels;
    state.speculative_learning=false;
}
void MoeHostExpertCache::set_device_cache_hints(std::shared_ptr<DeviceCacheHints> hints) {
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    while(state.device_head)state.device_duplicate(*state.device_head,false);
    state.device_cached.clear();state.device_hints=std::move(hints);
}

MoeHostExpertCache::Lease MoeHostExpertCache::find(mfq::MoeCacheKey key,bool touch) {
    std::unique_lock lock(impl_->mutex);impl_->wait_exchange(lock,key);
    const auto found=impl_->entries.find(key);
    if(found==impl_->entries.end())return {};
    impl_->demand(found->second.value);
    if(touch)impl_->touch(key,found->second);
    ++impl_->counters.hits;
    return found->second.value;
}

MoeHostExpertCache::Lease MoeHostExpertCache::try_find(mfq::MoeCacheKey key,bool touch,bool consume) {
    std::unique_lock lock(impl_->mutex,std::try_to_lock);
    if(!lock.owns_lock() || impl_->exchanging.contains(key))return {};
    const auto found=impl_->entries.find(key);
    if(found==impl_->entries.end())return {};
    if(consume)impl_->demand(found->second.value);
    if(touch)impl_->touch(key,found->second);
    ++impl_->counters.hits;
    return found->second.value;
}

void MoeHostExpertCache::preserve_resident_experts() {
    std::lock_guard lock(impl_->mutex);
    impl_->preserve=true;
}

MoeHostExpertCache::Lease MoeHostExpertCache::acquire(mfq::MoeCacheKey key,std::size_t bytes,
        const Load& load,bool retain,bool touch) {
    return this->load(key,bytes,load,retain,false,touch);
}

MoeHostExpertCache::Lease MoeHostExpertCache::acquire_scan(mfq::MoeCacheKey key,std::size_t bytes,
        const Load& load) {
    return this->load(key,bytes,load,true,true,false);
}

MoeHostExpertCache::Lease MoeHostExpertCache::load(mfq::MoeCacheKey key,std::size_t bytes,
        const Load& load,bool retain,bool scan,bool touch) {
    if (!load || !bytes) throw std::invalid_argument("host expert needs a loader and exact byte size");
    auto& state=*impl_;
    std::shared_ptr<Impl::Flight> flight;
    std::shared_ptr<MoeHostExpertReservation> reservation;
    bool managed=false;
    {
        std::unique_lock lock(state.mutex);
        state.wait_exchange(lock,key);
        if (retain) {
            const auto hit=state.entries.find(key);
            if (hit!=state.entries.end()) {
                if(touch){state.demand(hit->second.value);state.touch(key,hit->second);}
                ++state.counters.hits;
                return hit->second.value;
            }
            if(const auto active=state.transient.find(key);active!=state.transient.end()) {
                if(auto value=active->second.lock()) {
                    ++state.counters.coalesced;
                    if(touch && !scan){state.demand(value);state.admit_prediction(key,value);}
                    return value;
                }
                state.transient.erase(active);
            }
            const auto pending=state.flights.find(key);
            if (pending!=state.flights.end()) {
                flight=pending->second;
                ++state.counters.coalesced;
                flight->condition.wait(lock,[&] { return flight->ready; });
                if (flight->error) std::rethrow_exception(flight->error);
                if(touch && !scan){state.demand(flight->value);state.admit_prediction(key,flight->value);}
                return flight->value;
            }
        }
        if(scan && !state.make_scan_room(key,bytes))return {};
        ++state.counters.misses;
        // Predictions may replace idle probation entries, never the reused
        // hot set. If those entries are leased, keep a bounded transient.
        managed=retain && (scan || state.make_room(bytes,!touch));
        if (!managed) ++state.counters.bypasses;
        reservation=std::make_shared<MoeHostExpertReservation>(state.ledger,bytes,managed);
        reservation->demand_admission=retain && !scan && !touch && !managed;
        reservation->unconsumed_prediction=retain && !scan && !touch;
        if(reservation->unconsumed_prediction) {
            ++state.counters.prediction_started_jobs;state.counters.prediction_started_bytes+=bytes;
        }
        if (retain) {
            flight=std::make_shared<Impl::Flight>();
            state.flights.emplace(key,flight);
        }
    }
    try {
        auto value=std::make_shared<MoeHostExpert>();
        value->reservation=std::move(reservation);
        value->bytes=bytes;
        value->weights=load();
        std::lock_guard lock(state.mutex);
        if (managed && (!flight || flight->retain)) {
            state.publish(key,value);
            if(touch && !scan && state.ghosts.contains(key))state.touch(key,state.entries.at(key));
            if(scan){++state.counters.scan_admissions;state.counters.scan_admitted_bytes+=bytes;}
        }
        else if(retain) {
            if(flight && !flight->retain)value->reservation->demand_admission=false;
            state.transient.insert_or_assign(key,value);
        }
        if (flight) {
            flight->value=value;
            flight->ready=true;
            state.flights.erase(key);
            flight->condition.notify_all();
        }
        return value;
    } catch (...) {
        if (flight) {
            std::lock_guard lock(state.mutex);
            flight->error=std::current_exception();
            flight->ready=true;
            state.flights.erase(key);
            flight->condition.notify_all();
        }
        throw;
    }
}

bool MoeHostExpertCache::demote(mfq::MoeCacheKey key,std::size_t bytes,const Load& copy) {
    if (!copy || !bytes) throw std::invalid_argument("GPU expert demotion needs exact fields");
    auto& state=*impl_;
    std::shared_ptr<MoeHostExpertReservation> reservation;
    std::shared_ptr<Impl::Flight> flight;
    {
        std::unique_lock lock(state.mutex);state.wait_exchange(lock,key);
        if (state.entries.contains(key)) return true;
        if (state.flights.contains(key) || !state.make_room(bytes)) return false;
        reservation=std::make_shared<MoeHostExpertReservation>(state.ledger,bytes,true);
        flight=std::make_shared<Impl::Flight>();
        state.flights.emplace(key,flight);
    }
    try {
        auto value=std::make_shared<MoeHostExpert>();
        value->reservation=std::move(reservation);
        value->bytes=bytes;
        value->weights=copy();
        std::lock_guard lock(state.mutex);
        if (flight->retain) {
            state.publish(key,value);
            ++state.counters.gpu_demotions;
        }
        flight->value=value;
        flight->ready=true;
        state.flights.erase(key);
        flight->condition.notify_all();
        return flight->retain;
    } catch (...) {
        std::lock_guard lock(state.mutex);
        flight->error=std::current_exception();
        flight->ready=true;
        state.flights.erase(key);
        flight->condition.notify_all();
        throw;
    }
}

void MoeHostExpertCache::promote(mfq::MoeCacheKey key) {
    auto& state=*impl_;
    std::unique_lock lock(state.mutex);state.wait_exchange(lock,key);
    const auto found=state.entries.find(key);
    if (found!=state.entries.end()) { state.remove(found); ++state.counters.promotions; }
    const auto flight=state.flights.find(key);
    if (flight!=state.flights.end()) flight->second->retain=false;
}

void MoeHostExpertCache::erase(mfq::MoeCacheKey key) {
    auto& state=*impl_;
    std::unique_lock lock(state.mutex);state.wait_exchange(lock,key);
    const auto found=state.entries.find(key);
    if (found!=state.entries.end()) state.remove(found);
    const auto flight=state.flights.find(key);
    if (flight!=state.flights.end()) flight->second->retain=false;
}

bool MoeHostExpertCache::contains(mfq::MoeCacheKey key) const {
    std::unique_lock lock(impl_->mutex);impl_->wait_exchange(lock,key);
    return impl_->entries.contains(key);
}

std::size_t MoeHostExpertCache::reclaim_optional(std::size_t bytes) {
    auto& state=*impl_;std::lock_guard lock(state.mutex);
    std::size_t released=0;
    auto candidate=state.lru.end();
    while(candidate!=state.lru.begin() && released<bytes && state.counters.optional_bytes) {
        --candidate;
        if(candidate->source>=0)continue;
        if(state.exchanging.contains(*candidate))continue;
        const auto found=state.entries.find(*candidate);
        if(found->second.value.use_count()!=1)continue;
        released+=found->second.value->bytes;
        state.device_duplicate(found->second,false);
        candidate=state.lru.erase(candidate);
        state.counters.resident_bytes-=found->second.value->bytes;
        state.counters.optional_bytes-=found->second.value->bytes;
        state.entries.erase(found);
        ++state.counters.evictions;
    }
    state.counters.optional_reclaimed_bytes+=released;
    return released;
}
bool MoeHostExpertCache::exchange(mfq::MoeCacheKey incoming,mfq::MoeCacheKey outgoing,
        const Lease& expected,const std::function<MixedMoePool(const MixedMoePool&)>& copy) {
    if(!expected || !copy || incoming==outgoing)return false;
    auto& state=*impl_;std::unique_lock lock(state.mutex);
    state.wait_exchange(lock,incoming);state.wait_exchange(lock,outgoing);
    auto found=state.entries.find(incoming);
    if(found==state.entries.end() || found->second.value!=expected || expected.use_count()!=2 ||
        state.entries.contains(outgoing) || state.flights.contains(incoming) || state.flights.contains(outgoing))return false;
    auto value=std::make_shared<MoeHostExpert>();
    value->reservation=expected->reservation;value->bytes=expected->bytes;
    const auto position=found->second.position;
    const bool was_reused=found->second.reused;
    // Allocate the new lookup entry before copying into the reused fields.
    state.entries.emplace(outgoing,Impl::Entry{value,position});
    state.entries.at(outgoing).birth_epoch=state.speculative_learning?state.learning_epoch:0;
    try {value->weights=copy(expected->weights);}
    catch(...) {state.entries.erase(outgoing);throw;}
    state.device_duplicate(state.entries.at(incoming),false);
    state.entries.erase(incoming);*position=outgoing;
    state.device_duplicate(state.entries.at(outgoing),state.device_cached.contains(outgoing));
    if(incoming.source<0)state.counters.optional_bytes-=value->bytes;
    if(outgoing.source<0)state.counters.optional_bytes+=value->bytes;
    if(was_reused)state.counters.reuse_protected_bytes-=value->bytes;
    state.lru.splice(state.lru.begin(),was_reused?state.reused:state.lru,position);
    ++state.counters.promotions;++state.counters.gpu_demotions;
    return true;
}

bool MoeHostExpertCache::exchange_batch(const std::vector<Exchange>& changes,
        const std::function<void()>& publish,const std::function<void()>& prepare,bool independent_readers) {
    if(changes.empty())return false;
    auto& state=*impl_;std::unique_lock lock(state.mutex);
    const auto* option=std::getenv("MFQ_MOE_KEYED_EXCHANGE");
    const bool keyed=option?option[0]=='1':independent_readers;
    std::unordered_set<mfq::MoeCacheKey,mfq::MoeCacheKeyHash> keys;
    struct Prepared {std::shared_ptr<MoeHostExpert> value;std::list<mfq::MoeCacheKey>::iterator position;bool reused;};
    std::vector<Prepared> prepared;prepared.reserve(changes.size());
    for(const auto& change:changes) {
        if(!change.expected || !*change.expected || !change.copy || !change.restore ||
            !keys.insert(change.incoming).second || !keys.insert(change.outgoing).second)return false;
        const auto found=state.entries.find(change.incoming);
        if(found==state.entries.end() || found->second.value!=*change.expected ||
            change.expected->use_count()!=2 || state.entries.contains(change.outgoing) ||
            state.exchanging.contains(change.incoming) || state.exchanging.contains(change.outgoing) ||
            state.flights.contains(change.incoming) || state.flights.contains(change.outgoing))return false;
        auto value=std::make_shared<MoeHostExpert>();
        value->reservation=(*change.expected)->reservation;value->bytes=(*change.expected)->bytes;
        prepared.push_back({std::move(value),found->second.position,found->second.reused});
    }
    // The optional keyed path excludes readers of both names while DMA and
    // publication run, allowing unrelated immutable experts to be acquired.
    std::size_t allocated=0,touched=0;
    bool reserved=false;
    const auto unreserve=[&] {
        if(keyed)for(const auto& key:keys)state.exchanging.erase(key);
        if(reserved)state.exchange_placeholders-=changes.size();
    };
    try {
        for(;allocated<changes.size();++allocated) {
            state.entries.emplace(changes[allocated].outgoing,
                Impl::Entry{prepared[allocated].value,prepared[allocated].position});
            state.entries.at(changes[allocated].outgoing).birth_epoch=state.speculative_learning?state.learning_epoch:0;
        }
        if(keyed) {
            state.exchanging.reserve(state.exchanging.size()+keys.size());
            for(const auto& key:keys)state.exchanging.insert(key);
            reserved=true;state.exchange_placeholders+=changes.size();
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "moe_host_keyed_exchange scope=first_transaction enabled=1 changes=%zu\n",changes.size());
            lock.unlock();
        }
        if(prepare) {
            touched=changes.size();
            prepare();
        }
        for(std::size_t i=0;i<changes.size();++i) {
            if(!prepare)touched=i+1; // A callback can fail after changing part of a field.
            prepared[i].value->weights=changes[i].copy((*changes[i].expected)->weights);
        }
        if(publish)publish();
    } catch(...) {
        if(!lock.owns_lock())lock.lock();
        const auto failure=std::current_exception();std::exception_ptr recovery;
        while(touched) {
            --touched;
            try {changes[touched].restore((*changes[touched].expected)->weights);}
            catch(...) {if(!recovery)recovery=std::current_exception();}
        }
        for(std::size_t i=0;i<allocated;++i)state.entries.erase(changes[i].outgoing);
        unreserve();lock.unlock();state.exchange_complete.notify_all();
        if(recovery)std::rethrow_exception(recovery);
        std::rethrow_exception(failure);
    }
    if(!lock.owns_lock())lock.lock();
    for(std::size_t i=0;i<changes.size();++i) {
        state.device_duplicate(state.entries.at(changes[i].incoming),false);
        state.entries.erase(changes[i].incoming);*prepared[i].position=changes[i].outgoing;
        state.device_duplicate(state.entries.at(changes[i].outgoing),state.device_cached.contains(changes[i].outgoing));
        if(changes[i].incoming.source<0)state.counters.optional_bytes-=prepared[i].value->bytes;
        if(changes[i].outgoing.source<0)state.counters.optional_bytes+=prepared[i].value->bytes;
        if(prepared[i].reused)state.counters.reuse_protected_bytes-=prepared[i].value->bytes;
        state.lru.splice(state.lru.begin(),prepared[i].reused?state.reused:state.lru,prepared[i].position);
    }
    state.counters.promotions+=changes.size();state.counters.gpu_demotions+=changes.size();
    unreserve();lock.unlock();state.exchange_complete.notify_all();
    return true;
}

MoeHostExpertCacheStats MoeHostExpertCache::stats() const {
    const auto& state=*impl_;
    std::lock_guard lock(state.mutex);
    auto result=state.counters;
    result.budget_bytes=state.budget;
    result.resident_entries=state.entries.size()-state.exchange_placeholders;
    result.managed_bytes=state.ledger->managed.load();
    result.managed_peak_bytes=state.ledger->managed_peak.load();
    result.transient_bytes=state.ledger->transient.load();
    result.transient_peak_bytes=state.ledger->transient_peak.load();
    result.ghost_entries=state.ghosts.size();result.ghost_bytes=state.ghost_bytes;
    return result;
}
