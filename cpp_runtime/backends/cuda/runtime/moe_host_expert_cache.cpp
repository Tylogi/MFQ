#include "moe_host_expert_cache.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

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
    MoeHostExpertReservation(std::shared_ptr<Ledger> value,std::size_t size,bool retained)
        :ledger(std::move(value)),bytes(size),managed(retained) {
        if (managed) charge(ledger->managed,ledger->managed_peak,bytes);
        else charge(ledger->transient,ledger->transient_peak,bytes);
    }
    ~MoeHostExpertReservation() {
        (managed ? ledger->managed : ledger->transient).fetch_sub(bytes);
    }
};

struct MoeHostExpertCache::Impl {
    struct Entry { Lease value; std::list<mfq::MoeCacheKey>::iterator position; };
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
    std::unordered_map<mfq::MoeCacheKey,Entry,mfq::MoeCacheKeyHash> entries;
    std::unordered_map<mfq::MoeCacheKey,std::shared_ptr<Flight>,mfq::MoeCacheKeyHash> flights;
    MoeHostExpertCacheStats counters;

    explicit Impl(std::size_t bytes):budget(bytes) {}
    void remove(decltype(entries)::iterator found) {
        counters.resident_bytes-=found->second.value->bytes;
        lru.erase(found->second.position);
        entries.erase(found);
    }
    bool make_room(std::size_t bytes) {
        if (!bytes || bytes>budget) return false;
        auto candidate=lru.end();
        while (ledger->managed.load()>budget-bytes) {
            if (candidate==lru.begin()) return false;
            --candidate;
            auto found=entries.find(*candidate);
            // Externally held payloads stay charged even after removal. Keep
            // their lookup entry rather than evicting useful pinned data.
            if (found->second.value.use_count()!=1) continue;
            candidate=lru.erase(candidate);
            counters.resident_bytes-=found->second.value->bytes;
            entries.erase(found);
            ++counters.evictions;
        }
        return true;
    }
    void publish(mfq::MoeCacheKey key,Lease value) {
        lru.push_front(key);
        try {
            if (!entries.emplace(key,Entry{value,lru.begin()}).second)
                throw std::logic_error("duplicate host expert publication");
        }
        catch (...) { lru.pop_front(); throw; }
        counters.resident_bytes+=value->bytes;
    }
};

MoeHostExpertCache::MoeHostExpertCache(std::size_t bytes):impl_(std::make_unique<Impl>(bytes)) {}
MoeHostExpertCache::~MoeHostExpertCache()=default;

MoeHostExpertCache::Lease MoeHostExpertCache::acquire(mfq::MoeCacheKey key,std::size_t bytes,
        const Load& load,bool retain) {
    if (!load || !bytes) throw std::invalid_argument("host expert needs a loader and exact byte size");
    auto& state=*impl_;
    std::shared_ptr<Impl::Flight> flight;
    std::shared_ptr<MoeHostExpertReservation> reservation;
    bool managed=false;
    {
        std::unique_lock lock(state.mutex);
        if (retain) {
            const auto hit=state.entries.find(key);
            if (hit!=state.entries.end()) {
                state.lru.splice(state.lru.begin(),state.lru,hit->second.position);
                ++state.counters.hits;
                return hit->second.value;
            }
            const auto pending=state.flights.find(key);
            if (pending!=state.flights.end()) {
                flight=pending->second;
                ++state.counters.coalesced;
                flight->condition.wait(lock,[&] { return flight->ready; });
                if (flight->error) std::rethrow_exception(flight->error);
                return flight->value;
            }
        }
        ++state.counters.misses;
        managed=retain && state.make_room(bytes);
        if (!managed) ++state.counters.bypasses;
        reservation=std::make_shared<MoeHostExpertReservation>(state.ledger,bytes,managed);
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
        if (managed && (!flight || flight->retain)) state.publish(key,value);
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
        std::lock_guard lock(state.mutex);
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
    std::lock_guard lock(state.mutex);
    const auto found=state.entries.find(key);
    if (found!=state.entries.end()) { state.remove(found); ++state.counters.promotions; }
    const auto flight=state.flights.find(key);
    if (flight!=state.flights.end()) flight->second->retain=false;
}

void MoeHostExpertCache::erase(mfq::MoeCacheKey key) {
    auto& state=*impl_;
    std::lock_guard lock(state.mutex);
    const auto found=state.entries.find(key);
    if (found!=state.entries.end()) state.remove(found);
    const auto flight=state.flights.find(key);
    if (flight!=state.flights.end()) flight->second->retain=false;
}

bool MoeHostExpertCache::contains(mfq::MoeCacheKey key) const {
    std::lock_guard lock(impl_->mutex);
    return impl_->entries.contains(key);
}

MoeHostExpertCacheStats MoeHostExpertCache::stats() const {
    const auto& state=*impl_;
    std::lock_guard lock(state.mutex);
    auto result=state.counters;
    result.budget_bytes=state.budget;
    result.resident_entries=state.entries.size();
    result.managed_bytes=state.ledger->managed.load();
    result.managed_peak_bytes=state.ledger->managed_peak.load();
    result.transient_bytes=state.ledger->transient.load();
    result.transient_peak_bytes=state.ledger->transient_peak.load();
    return result;
}
