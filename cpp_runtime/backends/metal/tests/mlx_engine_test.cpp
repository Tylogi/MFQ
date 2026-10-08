#include "mlx_engine.h"
#include "grammar_fixture.h"

#include <atomic>
#include <chrono>
#include <iostream>

using namespace mfq::engine;
using namespace mfq::metal;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class F> static void rejects(F run) {
    bool failed = false;
    try { run(); } catch (const std::exception&) { failed = true; }
    require(failed, "operation should have been rejected");
}
struct FakeModel final : MlxEngineModel {
    std::atomic<int> produced{0};
    bool fail = false, duplex = false;
    void generate(InferenceRequest& request, MlxGenerationJob& job) override {
        job.prefill({request.prompt.size(), 2.0, 0.0, 2.0});
        for (int index = 0; index < request.sampling.max_tokens; ++index) {
            if (!job.token(index % 3)) break;
            ++produced;
        }
        if (fail) throw std::runtime_error("native cleanup failed");
        job.mtp.available = true;
        job.mtp.used = request.sampling.enable_mtp;
        job.mtp.accepted_tokens = 2;
    }
    std::int64_t reload(std::int64_t context) override { return context; }
    SessionResult session(const SessionCommand& command) override {
        return {command.kind == SessionCommand::Kind::fork ? 1u : 0u, {{"sessions", 1}}};
    }
    ControlResult control(ControlRequest request) override {
        if (std::holds_alternative<RuntimeMetrics>(request)) return Metrics{{"healthy", 1}};
        if (std::holds_alternative<MfqDuplexSessionParams>(request)) duplex = true;
        else if (std::holds_alternative<MfqDuplexStepInput>(request)) return MfqDuplexStepResult{};
        else if (std::holds_alternative<StopDuplex>(request)) duplex = false;
        else throw std::invalid_argument("unexpected text control");
        return std::monostate{};
    }
};
static EngineRequest request(int tokens = 12) {
    EngineRequest result;
    result.id = "test";
    result.token_ids = {0, 1, 2};
    result.input.sampling.max_tokens = tokens;
    return result;
}
static std::unique_ptr<MlxEngine> engine(std::unique_ptr<FakeModel> model) {
    EngineInfo info;
    info.vocab_size = 4;
    info.max_context = 1024;
    info.capabilities.mtp = true;
    auto text = std::make_unique<TextProcessor>(test::tokenizer_bytes(4), 4, "fixture");
    return std::make_unique<MlxEngine>(std::move(model), std::move(text), info);
}
static EventData finish(MlxEngine& engine, std::vector<std::int64_t>* tokens = nullptr) {
    const auto limit = Clock::now() + std::chrono::seconds(5);
    while (Clock::now() < limit) {
        auto step = engine.step({"test"});
        for (auto& event : step.events) {
            if (tokens) if (auto* output = std::get_if<OutputDelta>(&event.data))
                tokens->insert(tokens->end(), output->token_ids.begin(), output->token_ids.end());
            if (terminal(event.data)) return std::move(event.data);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("native worker did not finish");
}
int main() {
    try {
        {
            auto loaded = engine(std::make_unique<FakeModel>());
            require(loaded->admit(request()) == Admission::accepted, "request admission");
            require(loaded->admit(EngineRequest{"other"}) == Admission::deferred, "serial capacity");
            std::vector<std::int64_t> tokens;
            auto event = finish(*loaded, &tokens);
            const auto& completed = std::get<Completed>(event);
            require(completed.usage.completion_tokens == 12 && tokens.size() == 12, "token accounting");
            for (std::size_t index = 0; index < tokens.size(); ++index)
                require(tokens[index] == static_cast<std::int64_t>(index % 3), "token ordering");
            require(completed.metrics.prefill_tokens == 3 && completed.metrics.prefill_ms == 2.0,
                    "prefill accounting");
            require(completed.metrics.mtp.used && completed.metrics.mtp.accepted_tokens == 2, "MTP metrics");
            require(loaded->status().healthy && loaded->status().available == 1, "released capacity");
            require(loaded->reload(512) == 512 && loaded->info().max_context == 512, "reload context");
            require(loaded->session({SessionCommand::Kind::fork, "a", "b"}).count == 1, "session fork");
            loaded->control(MfqDuplexSessionParams{});
            require(loaded->status().available == 0, "duplex exclusivity");
            require(std::holds_alternative<MfqDuplexStepResult>(loaded->control(MfqDuplexStepInput{})), "duplex step");
            rejects([&] { loaded->reload(256); });
            loaded->control(StopDuplex{});
            require(loaded->status().available == 1, "duplex release");
            require(std::get<std::string>(loaded->control(DecodeTokens{{0, 1}, {}})) == "ab", "text control");
        }
        for (bool cleanup_failure : {false, true}) {
            auto model = std::make_unique<FakeModel>();
            auto* native = model.get();
            native->fail = cleanup_failure;
            auto loaded = engine(std::move(model));
            loaded->admit(request(1000));
            loaded->step({"test"}); // Start the worker, then abandon its output.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            require(native->produced <= 4, "unbounded output while scheduler is paused");
            require(std::get<Metrics>(loaded->control(RuntimeMetrics{})).size() == 1, "responsive metrics");
            for (const auto kind : {SessionCommand::Kind::metrics, SessionCommand::Kind::trim,
                    SessionCommand::Kind::budget})
                require(loaded->session({kind}).metrics.size() == 1, "concurrent cache maintenance");
            rejects([&] { loaded->reload(128); });
            rejects([&] { loaded->session({SessionCommand::Kind::clear}); });
            rejects([&] { loaded->session({SessionCommand::Kind::memory_budget}); });
            rejects([&] { loaded->session({SessionCommand::Kind::refresh}); });
            loaded->cancel("test");
            auto event = finish(*loaded);
            require(cleanup_failure ? std::holds_alternative<Failed>(event) :
                    std::holds_alternative<Cancelled>(event), "cancelled worker cleanup");
        }
        {
            auto loaded = engine(std::make_unique<FakeModel>());
            loaded->admit(request(1000));
            loaded->step({"test"});
            loaded->shutdown(); // Destroy a coroutine whose producer can be blocked.
            require(!loaded->status().healthy, "shutdown state");
        }
        {
            auto model = std::make_unique<FakeModel>();
            auto* native = model.get();
            auto loaded = engine(std::move(model));
            loaded->admit(request());
            loaded->cancel("test");
            require(std::holds_alternative<Cancelled>(finish(*loaded)), "cancel before execution");
            require(native->produced == 0, "cancelled request started native generation");
        }
        {
            auto model = std::make_unique<FakeModel>();
            model->fail = true;
            auto loaded = engine(std::move(model));
            loaded->admit(request(1000));
            loaded->step({"test"});
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            rejects([&] { loaded->shutdown(); });
            require(!loaded->status().healthy, "failed shutdown state");
        }
        std::cout << "Metal engine ownership, backpressure, cancellation and controls passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
