#include "scheduler.h"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <map>
#include <thread>
using namespace mfq::engine;
using namespace std::chrono_literals;

struct FakeEngine final : Engine {
    struct Work { EngineRequest request; int steps = 0; bool cancelled = false; };
    std::map<std::string, Work> active;
    std::array<std::atomic<int>, 16> advances{};
    std::atomic<int> released{0}, reloads{0};
    std::thread::id owner;
    bool fail = false;
    std::atomic<bool> defer{false};
    std::atomic<const float*> media{nullptr};
    EngineInfo info() const override { return {2, 128, 64, false, false, true, {}}; }
    void check_thread() {
        if (owner == std::thread::id{}) owner = std::this_thread::get_id();
        assert(owner == std::this_thread::get_id());
    }
    Admission admit(EngineRequest&& request) override {
        check_thread();
        if (defer || active.size() == 2) return Admission::deferred;
        if (request.input.media) media = request.input.media->pixel_values.data();
        auto id = request.id; active.emplace(id, Work{std::move(request)});
        return Admission::accepted;
    }
    void cancel(const RequestId& id) override {
        check_thread(); if (active.contains(id)) active.at(id).cancelled = true;
    }
    EngineStatus status() const override { return {2 - active.size(), true}; }
    EngineStepResult step(const std::vector<RequestId>& eligible) override {
        check_thread();
        EngineStepResult result;
        for (auto it = active.begin(); it != active.end();) {
            auto& work = it->second;
            const auto id = it->first;
            if (work.cancelled) {
                it = active.erase(it); ++released;
                result.events.push_back({id, Cancelled{}}); continue;
            }
            if (std::find(eligible.begin(), eligible.end(), id) == eligible.end()) { ++it; continue; }
            if (work.request.token_ids == std::vector<int64_t>{63}) throw std::runtime_error("device failure");
            ++advances.at(std::stoi(id));
            ++work.steps;
            result.advanced.push_back(id);
            if (work.steps <= int(work.request.token_ids.size()))
                result.events.push_back({id, PrefillProgress{{std::size_t(work.steps), 1, 0, 1}}});
            else if (work.steps <= int(work.request.token_ids.size()) + work.request.input.sampling.max_tokens) {
                OutputDelta delta{{1, 2, 3}, {}};
                if (work.request.token_ids == std::vector<int64_t>{60} ||
                        work.request.token_ids == std::vector<int64_t>{61} ||
                        work.request.token_ids == std::vector<int64_t>{62}) {
                    common_chat_msg_diff diff;
                    const auto marker = work.request.token_ids.front();
                    diff.content_delta.assign(marker == 60 ? (work.steps == 2 ? 2000 : 6000)
                        : (marker == 61 ? 1024 : 8192), 'x');
                    delta.diffs.push_back(std::move(diff));
                }
                result.events.push_back({id, std::move(delta)});
            }
            if (work.steps > int(work.request.token_ids.size()) + work.request.input.sampling.max_tokens ||
                    (work.request.token_ids == std::vector<int64_t>{60} &&
                     work.steps == int(work.request.token_ids.size()) + work.request.input.sampling.max_tokens)) {
                it = active.erase(it); ++released;
                InferenceMetrics metrics;
                metrics.mtp.available = true;
                metrics.mtp.cycles = std::stoi(id);
                result.events.push_back({id, Completed{{}, {}, metrics}}); continue;
            }
            ++it;
        }
        result.status = status(); result.has_work = !active.empty(); return result;
    }
    SessionResult session(const SessionCommand&) override { check_thread(); return {7, {}}; }
    int64_t reload(int64_t context) override {
        check_thread(); assert(active.empty());
        if (context == 13) throw std::runtime_error("load failed");
        ++reloads; return context;
    }
    void shutdown() override { check_thread(); assert(active.empty()); }
    ControlResult control(ControlRequest) override { check_thread(); return Metrics{}; }
};
EngineRequest request(int id, int tokens = 12) {
    EngineRequest result;
    result.id = std::to_string(id); result.token_ids = {1, 2};
    result.input.sampling.max_tokens = tokens; return result;
}
int drain(const std::shared_ptr<MfqScheduledRequest>& handle, bool cancelled = false, bool failed = false) {
    int terminals = 0;
    while (!handle->done()) for (const auto& event : handle->wait()) {
        if (const auto* delta = std::get_if<OutputDelta>(&event.data))
            assert(delta->token_ids == std::vector<int64_t>({1, 2, 3}));
        if (terminal(event.data)) {
            ++terminals;
            assert(std::holds_alternative<Cancelled>(event.data) == cancelled);
            assert(std::holds_alternative<Failed>(event.data) == failed);
            if (const auto* completed = std::get_if<Completed>(&event.data)) {
                assert(completed->metrics.mtp.available);
                assert(completed->metrics.mtp.cycles == std::stoul(event.id));
            }
        }
    }
    assert(terminals == 1);
    assert(handle->wait().empty());
    return terminals;
}
static void check_input_ownership_and_budget() {
    FakeEngine engine;
    engine.defer = true;
    MfqScheduler scheduler(engine, {8, 4096, 2, 65536});
    auto input = request(0, 1000);
    input.input.cache_plan.session_id = "media-session";
    input.input.media.emplace().pixel_values.resize(10000, 0.5f);
    const auto* pixels = input.input.media->pixel_values.data();
    auto pending = scheduler.submit(std::move(input));
    auto too_large = request(1);
    too_large.input.media.emplace().audio_features.resize(10000);
    bool rejected = false;
    try { scheduler.submit(std::move(too_large)); }
    catch (const MfqSchedulerOverloaded&) { rejected = true; }
    assert(rejected);
    auto second = scheduler.submit(request(1, 1000));
    rejected = false;
    try { scheduler.submit(request(2)); }
    catch (const MfqSchedulerOverloaded&) { rejected = true; }
    assert(rejected);
    engine.defer = false;
    while (!engine.media.load()) std::this_thread::yield();
    assert(engine.media == pixels); // Deferred retries and acceptance never clone the media buffer.
    assert(scheduler.cancel_session("media-session")); // Logical identity survives moving the payload.
    drain(pending, true);
    scheduler.cancel_request("1");
    drain(second, true);
    (void)scheduler.status();
    drain(scheduler.submit(request(2, 1))); // Both count and byte reservations were returned.
    auto invalid = request(3); invalid.input.media.emplace().pixel_values.reserve(65536);
    rejected = false;
    try { scheduler.submit(std::move(invalid)); }
    catch (const MfqSchedulerOverloaded&) { rejected = true; }
    assert(rejected); // Account reserved capacity, even when the vector is empty.
}

int main() {
    check_input_ownership_and_budget();
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {8, 4096});
        auto slow = scheduler.submit(request(0));
        auto fast = scheduler.submit(request(1));
        drain(fast);
        const auto paused = engine.advances[0].load();
        assert(paused > 0 && paused < 12);
        auto another = scheduler.submit(request(2, 3));
        drain(another);
        assert(engine.advances[0] == paused);
        drain(slow);
        auto blocked = scheduler.submit(request(3, 100));
        while (engine.advances[3] == 0) std::this_thread::yield();
        assert(scheduler.cancel_request("3"));
        assert(!scheduler.cancel_request("unknown"));
        drain(blocked, true);
        auto session = request(4, 100); session.input.cache_plan.session_id = "session";
        auto handle = scheduler.submit(session);
        session.id = "5";
        bool conflict = false;
        try { scheduler.submit(session); } catch (const std::invalid_argument&) { conflict = true; }
        assert(conflict);
        conflict = false;
        try { scheduler.session({SessionCommand::Kind::close, "session"}); } catch (const std::runtime_error&) { conflict = true; }
        assert(conflict);
        assert(scheduler.cancel_session("session"));
        drain(handle, true);
        assert(scheduler.session({SessionCommand::Kind::close, "session"}).count == 7);
        auto deadline = request(6, 100); deadline.deadline = Clock::now() + 15ms;
        auto expiring = scheduler.submit(deadline);
        std::this_thread::sleep_for(30ms);
        drain(expiring, true);
        auto active = scheduler.submit(request(7, 100));
        auto queued = scheduler.submit(request(8, 100));
        auto waiting = scheduler.submit(request(9, 100));
        assert(scheduler.cancel_request("9"));
        drain(waiting, true);
        assert(scheduler.reload(64) == 64 && engine.reloads == 1);
        drain(active, true); drain(queued, true);
        auto shutdown = scheduler.submit(request(10, 100));
        scheduler.shutdown();
        drain(shutdown, true);
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine);
        auto first = scheduler.submit(request(0, 100));
        auto bad = request(1); bad.token_ids = {63};
        auto second = scheduler.submit(bad);
        drain(first, false, true); drain(second, false, true);
        assert(!scheduler.status().healthy);
        assert(engine.active.empty());
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 4096}); // bytes, not event count, limits progress
        auto large = request(0, 100); large.token_ids = {61};
        auto slow = scheduler.submit(large);
        drain(scheduler.submit(request(1, 4)));
        const auto paused = engine.advances[0].load();
        assert(paused >= 2 && paused <= 3);
        drain(scheduler.submit(request(2, 4)));
        assert(engine.advances[0] == paused);
        scheduler.cancel_request("0");
        drain(slow, true); // reserved terminal space remains available
        auto oversized = request(3); oversized.token_ids = {62};
        auto bad = scheduler.submit(oversized);
        drain(scheduler.submit(request(4, 4)));
        drain(bad, false, true);
        assert(engine.released == 5);
        bool failed = false;
        try { scheduler.reload(13); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
        failed = false;
        try { scheduler.submit(request(5)); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
        assert(scheduler.reload(64) == 64);
        assert(scheduler.status().healthy);
        drain(scheduler.submit(request(6, 4)));
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 8192});
        auto variable = request(0, 4); variable.token_ids = {60};
        auto slow = scheduler.submit(variable);
        drain(scheduler.submit(request(1, 4)));
        // Both deltas fit individually. Their sum pauses the producer instead
        // of reporting output_limit or losing the second delta.
        int outputs = 0, terminals = 0;
        while (!slow->done()) for (const auto& event : slow->wait()) {
            outputs += std::holds_alternative<OutputDelta>(event.data);
            if (terminal(event.data)) {
                assert(std::holds_alternative<Completed>(event.data));
                ++terminals;
            }
        }
        assert(outputs == 4 && terminals == 1);
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 8192});
        auto variable = request(0, 2); variable.token_ids = {60};
        auto slow = scheduler.submit(variable);
        // The final delta and terminal arrive together, with the delta blocked.
        while (engine.released == 0) std::this_thread::yield();
        scheduler.shutdown();
        drain(slow); // shutdown must not wait for the consumer to drain
    }
}
