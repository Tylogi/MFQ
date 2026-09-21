#include "transport.h"

#include <cassert>
#include <memory>
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
}
