#include "moe_cache_policy.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mfq {
size_t MoeCacheKeyHash::operator()(const MoeCacheKey& key) const noexcept {
    size_t value = static_cast<uint32_t>(key.source);
    value = value * 1315423911u + static_cast<uint32_t>(key.cohort);
    return value * 1315423911u + static_cast<uint32_t>(key.expert);
}
MoeCacheSlotBook::MoeCacheSlotBook(int capacity) : slots_(static_cast<size_t>(std::max(capacity, 0))) {
    if (capacity <= 0) throw std::invalid_argument("MoE cache capacity must be positive");
    free_.reserve(capacity);
    for (int slot = capacity - 1; slot >= 0; --slot) free_.push_back(slot);
}
int MoeCacheSlotBook::capacity() const noexcept { return static_cast<int>(slots_.size()); }
int MoeCacheSlotBook::size() const noexcept { return static_cast<int>(by_key_.size()); }
int MoeCacheSlotBook::slot_for(const MoeCacheKey& key) const noexcept {
    const auto found = by_key_.find(key);
    return found == by_key_.end() ? -1 : found->second;
}
const std::optional<MoeCacheKey>& MoeCacheSlotBook::key_for(int slot) const { return checked(slot).key; }
MoeCacheSlotLease MoeCacheSlotBook::acquire(const MoeCacheKey& key) {
    if (const auto found = by_key_.find(key); found != by_key_.end()) {
        auto& state = checked(found->second);
        touch(found->second);
        return {found->second, state.generation, true, std::nullopt};
    }
    int slot = -1;
    std::optional<MoeCacheKey> replaced;
    if (!free_.empty()) { slot = free_.back(); free_.pop_back(); }
    else {
        for (int candidate : lru_) if (!slots_[candidate].inflight) { slot = candidate; break; }
        if (slot < 0) throw std::runtime_error("all MoE cache slots are in-flight");
        auto& state = checked(slot);
        replaced = state.key;
        if (state.key) by_key_.erase(*state.key);
        erase_lru(slot);
    }
    auto& state = checked(slot);
    ++state.generation;
    state.key = key;
    state.inflight = false;
    by_key_.emplace(key, slot);
    append_lru(slot);
    return {slot, state.generation, false, replaced};
}
void MoeCacheSlotBook::touch(int slot) {
    auto& state = checked(slot);
    if (!state.key) throw std::runtime_error("cannot touch an empty MoE cache slot");
    if (state.listed) lru_.splice(lru_.end(), lru_, state.position);
    else append_lru(slot);
}
void MoeCacheSlotBook::mark_inflight(int slot) {
    auto& state = checked(slot);
    if (!state.key) throw std::runtime_error("cannot mark an empty MoE cache slot in-flight");
    state.inflight = true;
}
void MoeCacheSlotBook::clear_inflight(int slot) { checked(slot).inflight = false; }
bool MoeCacheSlotBook::inflight(int slot) const { return checked(slot).inflight; }
bool MoeCacheSlotBook::replace(
        int slot, const MoeCacheKey& expected, const MoeCacheKey& replacement) {
    auto& state = checked(slot);
    if (!state.key || !(*state.key == expected) || by_key_.count(replacement)) return false;
    by_key_.emplace(replacement, slot);
    by_key_.erase(expected);
    state.key = replacement;
    ++state.generation;
    state.inflight = false;
    if (state.listed) lru_.splice(lru_.end(), lru_, state.position);
    return true;
}
MoeCacheSlotBook::Replacement MoeCacheSlotBook::prepare_replace(
        int slot, const MoeCacheKey& expected, const MoeCacheKey& replacement) {
    const auto& state = checked(slot);
    if (!state.key || !(*state.key == expected) || by_key_.count(replacement))
        throw std::runtime_error("MoE replacement no longer owns its slot");
    by_key_.reserve(slots_.size() * 2);
    std::unordered_map<MoeCacheKey, int, MoeCacheKeyHash> detached;
    detached.emplace(replacement, slot);
    Replacement result;
    result.slot = slot;
    result.expected = expected;
    result.replacement = replacement;
    result.generation = state.generation;
    result.incoming = detached.extract(replacement);
    return result;
}
bool MoeCacheSlotBook::replacement_valid(const Replacement& change) const {
    const auto& state = checked(change.slot);
    return !change.committed && state.key && *state.key == change.expected &&
        state.generation == change.generation && !by_key_.count(change.replacement);
}
bool MoeCacheSlotBook::commit_replace(Replacement& change) {
    if (!replacement_valid(change)) return false;
    auto inserted = by_key_.insert(std::move(change.incoming));
    if (!inserted.inserted) {
        change.incoming = std::move(inserted.node);
        return false;
    }
    change.original = by_key_.extract(change.expected);
    auto& state = checked(change.slot);
    state.key = change.replacement;
    ++state.generation;
    state.inflight = false;
    if (state.listed) lru_.splice(lru_.end(), lru_, state.position);
    change.committed = true;
    return true;
}
void MoeCacheSlotBook::rollback_replace(Replacement& change) {
    auto& state = checked(change.slot);
    if (change.committed) {
        if (!state.key || !(*state.key == change.replacement) ||
                state.generation != change.generation + 1)
            throw std::runtime_error("MoE replacement changed during rollback");
        change.incoming = by_key_.extract(change.replacement);
        by_key_.insert(std::move(change.original));
        state.key = change.expected;
        state.generation = change.generation;
        change.committed = false;
    } else if (!state.key || !(*state.key == change.expected) ||
            state.generation != change.generation) {
        return;
    }
    state.inflight = false;
}
bool MoeCacheSlotBook::discard(const MoeCacheKey& key, int slot, uint64_t generation) {
    auto& state = checked(slot);
    if (!state.key || !(*state.key == key) || state.generation != generation) return false;
    by_key_.erase(key);
    state.key.reset(); state.inflight = false;
    erase_lru(slot); free_.push_back(slot);
    return true;
}
MoeCacheSlotBook::Slot& MoeCacheSlotBook::checked(int slot) {
    if (slot < 0 || slot >= capacity()) throw std::out_of_range("MoE cache slot is out of range");
    return slots_[slot];
}
const MoeCacheSlotBook::Slot& MoeCacheSlotBook::checked(int slot) const {
    if (slot < 0 || slot >= capacity()) throw std::out_of_range("MoE cache slot is out of range");
    return slots_[slot];
}
void MoeCacheSlotBook::erase_lru(int slot) {
    auto& state = checked(slot);
    if (!state.listed) return;
    lru_.erase(state.position); state.listed = false;
}
void MoeCacheSlotBook::append_lru(int slot) {
    lru_.push_back(slot);
    auto& state = checked(slot);
    state.position = std::prev(lru_.end()); state.listed = true;
}
std::unordered_map<std::string, int> plan_moe_arena_slots(int64_t budget, const std::vector<MoeArenaDemand>& demands) {
    if (budget <= 0) throw std::invalid_argument("MoE cache budget must be positive");
    if (demands.empty()) throw std::invalid_argument("MoE cache planner requires at least one arena");
    std::unordered_set<std::string> signatures;
    std::unordered_map<std::string, int> result;
    int64_t minimum_bytes = 0;
    long double registered_bytes = 0;
    for (const auto& demand : demands) {
        if (demand.signature.empty() || !signatures.insert(demand.signature).second)
            throw std::invalid_argument("MoE cache arena signatures must be unique and non-empty");
        if (demand.slot_bytes <= 0 || demand.minimum_slots <= 0 || demand.registered_experts < demand.minimum_slots)
            throw std::invalid_argument("invalid MoE cache arena demand");
        if (demand.slot_bytes > (std::numeric_limits<int64_t>::max() - minimum_bytes) / demand.minimum_slots)
            throw std::overflow_error("MoE cache minimum byte count overflows int64");
        minimum_bytes += demand.slot_bytes * demand.minimum_slots;
        registered_bytes += static_cast<long double>(demand.slot_bytes) * demand.registered_experts;
        result.emplace(demand.signature, demand.minimum_slots);
    }
    if (minimum_bytes > budget) throw std::runtime_error("MoE cache budget is below the minimum decode working set");
    int64_t used_bytes = minimum_bytes;
    for (const auto& demand : demands) {
        const auto target_bytes = static_cast<long double>(budget) * demand.slot_bytes * demand.registered_experts / registered_bytes;
        const auto target_slots = static_cast<int>(std::min<long double>(demand.registered_experts,
            std::max<long double>(demand.minimum_slots, target_bytes / demand.slot_bytes)));
        const int64_t additional = int64_t(target_slots - result.at(demand.signature)) * demand.slot_bytes;
        if (additional <= budget - used_bytes) { result[demand.signature] = target_slots; used_bytes += additional; }
    }
    while (true) {
        const MoeArenaDemand* best = nullptr;
        long double best_deficit = -1;
        for (const auto& demand : demands) {
            const int current = result.at(demand.signature);
            if (current >= demand.registered_experts || demand.slot_bytes > budget - used_bytes) continue;
            const auto deficit = static_cast<long double>(demand.registered_experts - current) / demand.registered_experts;
            if (deficit > best_deficit) { best = &demand; best_deficit = deficit; }
        }
        if (!best) break;
        ++result.at(best->signature); used_bytes += best->slot_bytes;
    }
    return result;
}
} // namespace mfq
