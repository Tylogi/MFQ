#pragma once
#include "nint_rows.h"
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace mfq {
// Canonical encoded rows stay immutable while a consumer holds their lease.
class NintRowCache {
public:
    explicit NintRowCache(std::size_t capacity) : capacity_(capacity) {}
    std::shared_ptr<const NintRowBatch> find(std::uint64_t key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) return {};
        recency_.splice(recency_.begin(), recency_, found->second);
        return found->second->value;
    }
    void insert(std::uint64_t key, std::shared_ptr<const NintRowBatch> value) {
        if (!capacity_ || !value) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (const auto found = entries_.find(key); found != entries_.end()) {
            recency_.splice(recency_.begin(), recency_, found->second);
            return;
        }
        recency_.push_front({key, std::move(value)});
        try { entries_.emplace(key, recency_.begin()); }
        catch (...) { recency_.pop_front(); throw; }
        if (entries_.size() > capacity_) {
            entries_.erase(recency_.back().key);
            recency_.pop_back();
        }
    }
private:
    struct Entry { std::uint64_t key; std::shared_ptr<const NintRowBatch> value; };
    std::size_t capacity_;
    mutable std::list<Entry> recency_;
    mutable std::unordered_map<std::uint64_t, std::list<Entry>::iterator> entries_;
    mutable std::mutex mutex_;
};
}
