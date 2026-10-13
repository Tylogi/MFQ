#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mfq {
struct MoeCacheKey {
    int source = -1, cohort = -1, expert = -1;
    bool operator==(const MoeCacheKey& other) const noexcept {
        return source == other.source && cohort == other.cohort && expert == other.expert;
    }
};
struct MoeCacheKeyHash {
    size_t operator()(const MoeCacheKey& key) const noexcept;
};
struct MoeCacheSlotLease {
    int slot = -1;
    uint64_t generation = 0;
    bool hit = false;
    std::optional<MoeCacheKey> replaced;
};
class MoeCacheSlotBook {
public:
    explicit MoeCacheSlotBook(int capacity);
    int capacity() const noexcept;
    int size() const noexcept;
    int slot_for(const MoeCacheKey& key) const noexcept;
    const std::optional<MoeCacheKey>& key_for(int slot) const;
    MoeCacheSlotLease acquire(const MoeCacheKey& key);
    void touch(int slot);
    void mark_inflight(int slot);
    void clear_inflight(int slot);
    bool discard(const MoeCacheKey& key, int slot, uint64_t generation);
    bool inflight(int slot) const;
    bool replace(int slot, const MoeCacheKey& expected, const MoeCacheKey& replacement);
    struct Replacement {
        int slot = -1;
        MoeCacheKey expected{}, replacement{};
        uint64_t generation = 0;
        bool committed = false;
        std::unordered_map<MoeCacheKey, int, MoeCacheKeyHash>::node_type incoming, original;
    };
    Replacement prepare_replace(
        int slot, const MoeCacheKey& expected, const MoeCacheKey& replacement);
    bool replacement_valid(const Replacement& change) const;
    bool commit_replace(Replacement& change);
    void rollback_replace(Replacement& change);
private:
    struct Slot {
        std::optional<MoeCacheKey> key;
        uint64_t generation = 0;
        bool inflight = false, listed = false;
        std::list<int>::iterator position;
    };
    Slot& checked(int slot);
    const Slot& checked(int slot) const;
    void erase_lru(int slot);
    void append_lru(int slot);
    std::vector<Slot> slots_;
    std::vector<int> free_;
    std::list<int> lru_;
    std::unordered_map<MoeCacheKey, int, MoeCacheKeyHash> by_key_;
};
struct MoeArenaDemand {
    std::string signature;
    int64_t slot_bytes = 0;
    int minimum_slots = 0, registered_experts = 0;
};
std::unordered_map<std::string, int> plan_moe_arena_slots(int64_t budget_bytes, const std::vector<MoeArenaDemand>& demands);
} // namespace mfq
