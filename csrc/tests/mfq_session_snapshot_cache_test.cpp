#include "session_snapshot_cache.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct State {
    std::vector<std::int64_t> tokens;
    std::string input_key;
    std::size_t bytes = 0;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

const auto same_snapshot = [](const State& left, const State& right) {
    return left.tokens == right.tokens && left.input_key == right.input_key;
};

} // namespace

int main() {
    mfq::engine::SessionSnapshotCache<State> cache({2, 2, 100});
    cache.store("a", {{1}, "", 10}, same_snapshot);
    cache.store("a", {{1, 2}, "", 20}, same_snapshot);
    cache.store("b", {{1}, "", 10}, same_snapshot);

    auto match = cache.find_best(
        "b", {1, 9}, 2, [](const State&) { return true; });
    require(match && match->session_id == "b", "requested session must win ties");
    cache.record_hit(*match);
    require(cache.metrics().hits == 1, "cache hit must be recorded");

    cache.store("a", {{1, 2, 3}, "", 30}, same_snapshot);
    match = cache.find_best(
        "a", {1, 9}, 2, [](const State&) { return true; });
    require(
        match && match->session_id == "b",
        "oldest per-session snapshot must be evicted");
    require(cache.fork("a", "c") == 2, "fork must copy retained snapshots");
    require(cache.metrics().sessions == 2, "fork must respect session limit");

    mfq::engine::SessionSnapshotCache<State> budget({4, 4, 25});
    budget.store("a", {{1}, "", 10}, same_snapshot);
    budget.store("b", {{2}, "", 20}, same_snapshot);
    require(budget.metrics().sessions == 1, "byte budget must evict LRU state");
    require(budget.metrics().bytes == 20, "byte accounting must stay exact");
    return 0;
}
