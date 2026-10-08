#include "moe_cache_policy.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_lru_replaces_oldest_non_inflight_slot() {
    mfq::MoeCacheSlotBook book(2);
    const mfq::MoeCacheKey first{1, 0, 3};
    const mfq::MoeCacheKey second{1, 0, 4};
    const mfq::MoeCacheKey third{2, 0, 9};

    const auto first_lease = book.acquire(first);
    const auto second_lease = book.acquire(second);
    require(!first_lease.replaced.has_value(), "first lease replaced a key");
    require(!second_lease.replaced.has_value(), "second lease replaced a key");

    book.touch(first_lease.slot);
    const auto third_lease = book.acquire(third);
    require(
        third_lease.replaced == std::optional<mfq::MoeCacheKey>(second),
        "LRU did not replace the oldest key");
    require(book.slot_for(first) == first_lease.slot, "touched key was evicted");
    require(book.slot_for(third) == third_lease.slot, "new key was not installed");
}

void test_inflight_slot_is_not_replaced() {
    mfq::MoeCacheSlotBook book(2);
    const mfq::MoeCacheKey first{0, 0, 0};
    const mfq::MoeCacheKey second{0, 0, 1};
    const mfq::MoeCacheKey third{0, 0, 2};

    const auto first_lease = book.acquire(first);
    const auto second_lease = book.acquire(second);
    book.mark_inflight(first_lease.slot);
    book.touch(second_lease.slot);

    const auto third_lease = book.acquire(third);
    require(
        third_lease.replaced == std::optional<mfq::MoeCacheKey>(second),
        "in-flight slot was selected for replacement");
    require(book.slot_for(first) == first_lease.slot, "in-flight key was evicted");
}

void test_all_inflight_slots_reject_replacement() {
    mfq::MoeCacheSlotBook book(1);
    const auto lease = book.acquire({0, 0, 0});
    book.mark_inflight(lease.slot);
    bool rejected = false;
    try {
        (void)book.acquire({0, 0, 1});
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "all-inflight cache accepted a replacement");
}

void test_failed_lease_can_be_discarded() {
    mfq::MoeCacheSlotBook book(1);
    const mfq::MoeCacheKey failed{0, 1, 2};
    const auto lease = book.acquire(failed);
    book.mark_inflight(lease.slot);

    require(
        book.discard(failed, lease.slot, lease.generation),
        "failed lease was not discarded");
    require(book.size() == 0, "discarded lease remained resident");
    require(book.slot_for(failed) == -1, "discarded key remained mapped");
    require(
        !book.discard(failed, lease.slot, lease.generation),
        "stale lease discarded a free slot");

    const mfq::MoeCacheKey replacement{0, 1, 3};
    const auto next = book.acquire(replacement);
    require(next.slot == lease.slot, "discarded slot was not reusable");
    require(!next.hit, "replacement unexpectedly hit the cache");
    require(
        !book.discard(failed, next.slot, lease.generation),
        "stale lease discarded a newer generation");
    require(
        book.slot_for(replacement) == next.slot,
        "newer cache generation was damaged");
}

void test_budget_planner_honors_minimums_and_hard_limit() {
    const std::vector<mfq::MoeArenaDemand> demands{
        {"nint4-gu", 256, 8, 64},
        {"nint4-down", 128, 8, 64},
    };
    const auto plan = mfq::plan_moe_arena_slots(4096, demands);
    require(plan.at("nint4-gu") >= 8, "gate/up minimum was not satisfied");
    require(plan.at("nint4-down") >= 8, "down minimum was not satisfied");
    const int64_t used =
        static_cast<int64_t>(plan.at("nint4-gu")) * 256 +
        static_cast<int64_t>(plan.at("nint4-down")) * 128;
    require(used <= 4096, "planner exceeded its hard byte budget");
}

void test_budget_planner_rejects_insufficient_budget() {
    const std::vector<mfq::MoeArenaDemand> demands{
        {"nint4-gu", 256, 8, 64},
        {"nint4-down", 128, 8, 64},
    };
    bool rejected = false;
    try {
        (void)mfq::plan_moe_arena_slots(3071, demands);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "planner accepted less than the minimum working set");
}

void test_budget_planner_caps_registered_experts() {
    const std::vector<mfq::MoeArenaDemand> demands{
        {"small", 64, 2, 3},
        {"large", 128, 2, 4},
    };
    const auto plan = mfq::plan_moe_arena_slots(1 << 20, demands);
    require(plan.at("small") == 3, "planner exceeded small registered count");
    require(plan.at("large") == 4, "planner exceeded large registered count");
}

}  // namespace

int main() {
    try {
        {
            mfq::MoeCacheSlotBook book(1);
            const mfq::MoeCacheKey old{0,0,0},fresh{0,0,1};
            const auto lease=book.acquire(old);
            book.mark_inflight(lease.slot);
            if(book.replace(lease.slot,{0,0,9},fresh))throw std::runtime_error("stale GPU exchange committed");
            if(!book.replace(lease.slot,old,fresh) || book.slot_for(old)!=-1 || book.slot_for(fresh)!=0 ||
                book.inflight(0) || book.discard(old,lease.slot,lease.generation))
                throw std::runtime_error("GPU exchange ownership or generation is invalid");
        }
        {
            mfq::MoeCacheSlotBook book(2);
            const mfq::MoeCacheKey a{0,0,0},b{0,0,1},c{1,0,0},d{1,0,1};
            auto first=book.acquire(a),second=book.acquire(b);
            auto left=book.prepare_replace(first.slot,a,c),right=book.prepare_replace(second.slot,b,d);
            book.mark_inflight(first.slot);book.mark_inflight(second.slot);
            require(book.slot_for(c)<0 && book.slot_for(d)<0,"prepared replacement was published early");
            require(book.commit_replace(left) && book.commit_replace(right),"prepared replacements failed to commit");
            book.rollback_replace(right);book.rollback_replace(left);
            require(book.slot_for(a)==first.slot && book.slot_for(b)==second.slot &&
                book.slot_for(c)<0 && book.slot_for(d)<0 && !book.inflight(first.slot) &&
                !book.inflight(second.slot),"batch rollback did not restore original GPU keys");
            auto stale=book.prepare_replace(first.slot,a,c);
            require(book.replace(first.slot,a,d) && !book.commit_replace(stale),"stale prepared generation committed");
            book.mark_inflight(first.slot);book.rollback_replace(stale);
            require(book.inflight(first.slot) && book.slot_for(d)==first.slot,"stale rollback changed another owner's slot");
        }
        test_lru_replaces_oldest_non_inflight_slot();
        test_inflight_slot_is_not_replaced();
        test_all_inflight_slots_reject_replacement();
        test_failed_lease_can_be_discarded();
        test_budget_planner_honors_minimums_and_hard_limit();
        test_budget_planner_rejects_insufficient_budget();
        test_budget_planner_caps_registered_experts();
        std::cout << "moe_cache_policy_tests=9 passed=9\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "moe_cache_policy_test failure=" << error.what() << "\n";
        return 1;
    }
}
