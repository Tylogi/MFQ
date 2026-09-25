#include "transport.h"

#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class TestTransport final : public MfqTransport {
public:
    int run(const MfqScheduler& scheduler) override {
        assert(scheduler.supports_generation());
        auto request = scheduler.activate_request("request-1");
        assert(request);
        assert(!scheduler.activate_request("request-1"));
        assert(!scheduler.cancel_session(""));
        request->set_session_id("session-1");
        assert(scheduler.cancel_session("session-1"));
        assert(request->cancel_flag()->load());
        request->finish();
        assert(!scheduler.cancel_request("request-1"));

        bool saw_token = false;
        bool saw_prefill = false;
        const int result = scheduler.generate(
            {1, 2}, {},
            [&](int64_t token) {
                saw_token = token == 3;
                return true;
            },
            [&](const MfqPrefillTiming& timing) {
                saw_prefill = timing.prompt_tokens == 2;
            },
            {}, {});
        assert(saw_token);
        assert(saw_prefill);
        return result;
    }
};

int main() {
    static_assert(std::is_abstract_v<MfqTransport>);
    MfqInferenceEngine engine;
    engine.generate = [](
            const std::vector<int64_t>& prompt,
            const MfqSamplingParams&,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback& on_prefill,
            const MfqPromptCachePlan&,
            const MfqTokenConstraintPtr&) {
        on_prefill({prompt.size(), 0.0, 0.0, 0.0});
        on_token(3);
        return 1;
    };
    MfqRuntime runtime(
        std::move(engine), std::make_unique<TestTransport>());
    assert(runtime.run() == 1);

    MfqInferenceEngine queued_engine;
    queued_engine.max_concurrent_requests = 1;
    std::mutex gate;
    std::condition_variable changed;
    bool first_running = false;
    bool release_first = false;
    int engine_calls = 0;
    queued_engine.generate = [&](
            const std::vector<int64_t>&,
            const MfqSamplingParams&,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback&,
            const MfqPromptCachePlan&,
            const MfqTokenConstraintPtr&) {
        std::unique_lock<std::mutex> lock(gate);
        ++engine_calls;
        first_running = true;
        changed.notify_all();
        changed.wait(lock, [&] { return release_first; });
        lock.unlock();
        on_token(3);
        return 1;
    };
    MfqScheduler scheduler(queued_engine);
    auto previous = scheduler.activate_request("previous", "session", false);
    auto replacement = scheduler.activate_request(
        "replacement", "session", true);
    assert(previous && replacement && previous->cancelled());
    previous->finish();
    replacement->finish();

    auto first = scheduler.activate_request("first");
    auto second = scheduler.activate_request("second");
    assert(first && second);
    int first_result = 0;
    int second_result = -1;
    std::thread first_thread([&] {
        first_result = scheduler.generate(
            *first, {1}, {}, [](int64_t) { return true; }, {}, {}, {});
    });
    {
        std::unique_lock<std::mutex> lock(gate);
        changed.wait(lock, [&] { return first_running; });
    }
    std::thread second_thread([&] {
        second_result = scheduler.generate(
            *second, {2}, {}, [](int64_t) { return true; }, {}, {}, {});
    });
    const bool cancelled = scheduler.cancel_request("second");
    assert(cancelled);
    second_thread.join();
    assert(second_result == 0 && engine_calls == 1);
    {
        std::lock_guard<std::mutex> lock(gate);
        release_first = true;
    }
    changed.notify_all();
    first_thread.join();
    assert(first_result == 1);
}
