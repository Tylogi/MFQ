#include "request_executor.h"

#include <cassert>
#include <map>

using namespace mfq::engine;

struct TestOps {
    bool batch = false, reset_fails = false;
    int failure = 0, live = 0, resets = 0, advances = 0, accepted = 0;
    std::vector<PrefillChunk> chunks;
    std::map<RequestId, ExecutionRequest *> batched;
    bool can_batch(const EngineRequest &) const { return batch; }
    bool exclusive() const { return false; }
    bool mtp_available() const { return false; }
    void admit_batch(const RequestId &id, ExecutionRequest &request) {
        batched.emplace(id, &request);
    }
    void step_batch(const std::vector<RequestId> &eligible) {
        for (auto it = batched.begin(); it != batched.end();) {
            auto &request = *it->second;
            if (!request.output.stopped() &&
                std::find(eligible.begin(), eligible.end(), it->first) != eligible.end())
                request.append(10);
            if (request.output.stopped()) {
                request.complete();
                it = batched.erase(it);
            } else
                ++it;
        }
    }
    void reset() {
        ++resets;
        if (reset_fails)
            throw std::runtime_error("cleanup failed");
    }
    double prefill(PrefillChunk chunk) {
        chunks.push_back(chunk);
        return 1.0;
    }
    std::int64_t first_token() { return 10; }
    std::int64_t advance() { return 10 + ++advances; }
    void accept(std::int64_t) { ++accepted; }
    Generation generate(InferenceRequest &input, InferenceOutput &output) {
        struct Storage {
            int &live;
            explicit Storage(int &live) : live(live) { ++live; }
            ~Storage() { --live; }
        } storage(live);
        if (failure == 1)
            throw InferenceInputError(InferenceInputErrorCode::Invalid, "invalid media");
        if (failure == 2)
            throw std::runtime_error("device failed");
        auto sequence = generate_sequence(
            *this, output, input.prompt.size(), 0, input.cache_plan.stable_prefix_tokens, 2);
        while (auto event = sequence.next())
            co_yield std::move(*event);
    }
};

static EngineRequest request(std::string id = "one") {
    EngineRequest result;
    result.id = std::move(id);
    result.token_ids = {1, 2, 3, 4, 5};
    result.input.sampling.max_tokens = 2;
    return result;
}

static EngineInfo info() {
    EngineInfo result;
    result.max_requests = 2;
    result.max_context = 32;
    result.vocab_size = 32;
    return result;
}

static const EventData &terminal_event(const EngineStepResult &result) {
    assert(std::count_if(result.events.begin(), result.events.end(), [](const auto &event) {
        return terminal(event.data);
    }) == 1);
    assert(terminal(result.events.back().data));
    return result.events.back().data;
}

int main() {
    // A restored prefix and snapshot boundary must not enlarge a prefill tick.
    {
        TestOps ops;
        InferenceRequest input;
        input.prompt.resize(7, 1);
        input.sampling.max_tokens = 3;
        InferenceOutput output(input, nullptr, "chunks");
        auto sequence = generate_sequence(ops, output, 7, 1, 4, 2);
        std::vector<std::int64_t> tokens;
        std::size_t prefilled = 0;
        while (auto event = sequence.next()) {
            if (auto *progress = std::get_if<PrefillProgress>(&*event))
                prefilled = progress->timing.prompt_tokens;
            if (auto *delta = std::get_if<OutputDelta>(&*event))
                tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        }
        assert(prefilled == 6 && ops.chunks.size() == 4);
        assert(ops.chunks[0].offset == 1 && ops.chunks[0].count == 2);
        assert(ops.chunks[1].offset == 3 && ops.chunks[1].count == 1);
        assert(ops.chunks[2].offset == 4 && ops.chunks[2].count == 2);
        assert(ops.chunks[3].offset == 6 && ops.chunks[3].count == 1);
        assert((tokens == std::vector<std::int64_t>{10, 11, 12}));
        assert(ops.accepted == 3 && ops.advances == 2);
    }
    // Serial and batch use identical usage/terminal rules, including a zero
    // generation budget and cancellation while output-blocked.
    for (bool batch : {false, true}) {
        TestOps ops;
        ops.batch = batch;
        RequestExecutor executor(2);
        assert(executor.admit(request(), nullptr, info(), ops) == Admission::accepted);
        assert(executor.step({}, ops).events.empty());
        std::size_t tokens = 0, terminals = 0;
        for (int tick = 0; !executor.empty() && tick < 12; ++tick) {
            auto result = executor.step({"one"}, ops);
            for (const auto &event : result.events) {
                if (auto *delta = std::get_if<OutputDelta>(&event.data))
                    tokens += delta->token_ids.size();
                if (auto *done = std::get_if<Completed>(&event.data)) {
                    ++terminals;
                    assert(ops.live == 0 && ops.batched.empty());
                    assert(done->usage.prompt_tokens == 5 && done->usage.completion_tokens == 2);
                    assert(std::holds_alternative<UsageUpdate>(
                        result.events[result.events.size() - 2].data));
                    terminal_event(result);
                }
            }
        }
        assert(executor.empty() && tokens == 2 && terminals == 1);
        assert(executor.step({"one"}, ops).events.empty());
        auto zero = request();
        zero.input.sampling.max_tokens = 0;
        assert(executor.admit(std::move(zero), nullptr, info(), ops) == Admission::accepted);
        assert(std::get<Completed>(terminal_event(executor.step({"one"}, ops)))
                   .usage.completion_tokens == 0);
        for (bool started : {false, true}) {
            assert(executor.admit(request(), nullptr, info(), ops) == Admission::accepted);
            if (started)
                executor.step({"one"}, ops);
            const auto chunks = ops.chunks.size();
            executor.cancel("one");
            executor.cancel("one");
            assert(std::holds_alternative<Cancelled>(terminal_event(executor.step({}, ops))));
            assert(executor.empty() && ops.live == 0 && ops.batched.empty());
            assert(ops.chunks.size() == chunks);
        }
    }
    // An ineligible peer must not execute when another batched request cancels.
    {
        TestOps ops;
        ops.batch = true;
        RequestExecutor executor(2);
        assert(executor.admit(request("a"), nullptr, info(), ops) == Admission::accepted);
        assert(executor.admit(request("b"), nullptr, info(), ops) == Admission::accepted);
        assert(executor.admit(request("c"), nullptr, info(), ops) == Admission::deferred);
        executor.cancel("a");
        auto cancelled = executor.step({}, ops);
        assert(std::holds_alternative<Cancelled>(terminal_event(cancelled)));
        assert(cancelled.events.back().id == "a" && executor.status().available == 1);
        assert(ops.batched.at("b")->output.result.completion_tokens == 0);
        executor.cancel("b");
        executor.step({}, ops);
    }
    // Request errors are recoverable; device or cleanup failures stop admission.
    for (int failure : {1, 2, 3}) {
        TestOps ops;
        ops.failure = failure == 3 ? 1 : failure;
        ops.reset_fails = failure == 3;
        RequestExecutor executor;
        executor.admit(request(), nullptr, info(), ops);
        auto result = executor.step({"one"}, ops);
        const auto &error = std::get<Failed>(terminal_event(result));
        assert(error.code == (failure == 1 ? "invalid_request" : "backend_failure"));
        assert(ops.live == 0 && ops.resets == 1 && executor.empty());
        assert(result.status.healthy == (failure == 1));
        ops.failure = 0;
        assert(executor.admit(request(), nullptr, info(), ops) ==
               (failure == 1 ? Admission::accepted : Admission::deferred));
    }
}
