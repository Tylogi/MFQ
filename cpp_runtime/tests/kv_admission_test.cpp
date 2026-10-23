#include "continuous_batch.h"
#include "runtime_config.h"

#include <cassert>
#include <memory>

using namespace mfq::engine;

static void arithmetic_test() {
    KvAdmissionBudget budget({{0, 100}, {1, 30}});
    assert(!budget.reserve({{0, 40}, {1, 31}}));
    assert(budget.reserve({{0, 100}, {1, 30}}));
    assert(!budget.reserve({{0, 1}}));
    assert(!budget.reserve({{2, 1}}));
    const auto maximum = std::numeric_limits<std::size_t>::max();
    assert(kv_admission_multiply(maximum, 2) == maximum);
    assert(prefill_budget_add(maximum, 1) == maximum);
    assert(kv_admission_multiply(0, maximum) == 0);
    assert(kv_admission_round_up(16, 16) == 16);
    assert(kv_admission_round_up(17, 16) == 32);
    assert(kv_admission_round_up(maximum, 16) == maximum);
    assert(kv_admission_mapping_bytes(2, 8, 1, 4, 16) == 32);
    assert(kv_admission_mapping_bytes(4, 2, 1, 4, 16) == 32);
    assert(kv_admission_mapping_bytes(2, 8, 5, 4, 16) == 64);
    assert(kv_admission_mapping_bytes(2, 8, 0, 4, 16) == 0);

    NewTokenRatioTracker ratio;
    assert(std::abs(ratio.current() - .7) < 1e-12);
    assert(ratio.expected(100, 20) == 56);
    assert(ratio.expected(100000, 0) == 2868);
    assert(ratio.expected(10, 10) == 0);
    for (int step = 0; step < 300; ++step) ratio.decay_step();
    assert(std::abs(ratio.current() - (.7 + .098) / 2) < 1e-12);
    for (int step = 0; step < 1000; ++step) ratio.decay_step();
    assert(std::abs(ratio.current() - .098) < 1e-12);
    ratio.reset();
    assert(std::abs(ratio.current() - .7) < 1e-12);
    InferenceRequest input;
    input.prompt = {1, 2};
    input.sampling.max_tokens = 100;
    auto a = std::make_shared<BatchRequest>("a", input);
    auto b = std::make_shared<BatchRequest>("b", input);
    a->produced = 5;
    b->produced = 10;
    ratio.after_retraction(std::vector{a, b});
    assert(std::abs(ratio.current() - 55.0 / 201) < 1e-12);
    a->generation_limit = 100000;
    a->produced = 1;
    ratio.after_retraction(std::vector{a});
    assert(std::abs(ratio.current() - 21.0 / 100001) < 1e-12);
    ratio.decay_step();
    assert(std::abs(ratio.current() - .098) < 1e-12);
    a->generation_limit = 1;
    ratio.after_retraction(std::vector{a});
    assert(ratio.current() == 1);
    KvAdmissionConfig clipped;
    clipped.decode_token_clip = 10;
    assert(NewTokenRatioTracker(clipped).expected(100000, 1) == 7);
    b->prompt.resize(4);
    a->produced = b->produced;
    auto ordered = retraction_order(std::vector{a, b});
    assert(ordered.front() == b);
    a->produced = 1;
    ordered = retraction_order(std::vector{b, a});
    assert(ordered.front() == a);
    for (const auto bad : {0.0, -1.0, 1.1, std::numeric_limits<double>::infinity()}) {
        KvAdmissionConfig config;
        config.initial_ratio = bad;
        try { (void)NewTokenRatioTracker(config); assert(false); } catch (const std::invalid_argument&) {}
    }
}

struct AdmissionOps {
    using Request = BatchRequest;
    using State = BatchState<Request>;
    struct Sample { int64_t token = 10; MfqPrefillTiming timing; };
    struct Memory {
        std::size_t capacity = 22, used = 0, prefills = 0, retracts = 0, replays = 0, peak_rows = 0;
        std::vector<std::string> victims;
    };
    Memory& memory;
    explicit AdmissionOps(Memory& memory) : memory(memory) {}
    KvAdmissionConfig admission_config() const {
        KvAdmissionConfig config;
        config.initial_ratio = .1;
        return config;
    }
    KvAdmissionBytes admission_available() const { return {{0, memory.capacity - memory.used}}; }
    KvAdmissionBytes admission_cost(const std::shared_ptr<Request>& request, int64_t expected) const {
        const auto offset = request->replay_offset >= 0 ? request->replay_offset : request->prefill_offset;
        const auto length = request->replay_offset >= 0 ? request->replay_length() : int64_t(request->prompt.size());
        return {{0, length - offset + expected}};
    }
    PrefillChunk prefill_chunk(const std::shared_ptr<Request>&, PrefillChunk chunk) { return chunk; }
    void suspend_decode() {}
    void resume_decode(int64_t) {}
    Sample prefill(const std::shared_ptr<Request>&, PrefillChunk chunk) {
        memory.used += chunk.count;
        ++memory.prefills;
        assert(memory.used <= memory.capacity);
        return {};
    }
    void activate(const std::shared_ptr<Request>&, const Sample&) {}
    bool decode_has_capacity(const State& state) const {
        return memory.used + std::count_if(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->eligible; }) <= memory.capacity;
    }
    int decode(State& state) {
        assert(!state.active.empty() && decode_has_capacity(state));
        memory.peak_rows = std::max(memory.peak_rows, state.active.size());
        for (const auto& request : state.active) if (request->eligible) ++memory.used;
        return 0;
    }
    Sample sample(const std::shared_ptr<Request>& request, int) { return {request->pending_token + 1, {}}; }
    void accept(const std::shared_ptr<Request>&, const Sample&) {}
    void retract(const std::shared_ptr<Request>& request) {
        assert(memory.used >= std::size_t(request->cache_length));
        memory.used -= request->cache_length;
        ++memory.retracts;
        memory.victims.push_back(request->id);
    }
    void replay_prefill(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        assert(request->accepted_tokens.size() == std::size_t(request->produced));
        assert(request->accepted_tokens.back() == request->pending_token);
        memory.used += chunk.count;
        assert(memory.used <= memory.capacity);
        ++memory.replays;
    }
    void discard_prefill(const std::shared_ptr<Request>& request) {
        memory.used -= request->replay_offset >= 0 ? request->replay_offset : request->prefill_offset;
    }
    void retire(const std::vector<std::shared_ptr<Request>>& requests, int64_t) {
        for (const auto& request : requests) memory.used -= request->cache_length;
    }
    void recover(State&) { assert(false); }
    Metrics metrics() const { return {}; }
};

static double metric(const Metrics& metrics, std::string_view name) {
    for (const auto& [key, value] : metrics) if (key == name) return value;
    assert(false);
    return 0;
}

static void queue_and_retraction_test() {
    AdmissionOps::Memory memory;
    ContinuousBatch<AdmissionOps> batch(2, 2, memory);
    std::vector<InferenceRequest> inputs(3);
    std::vector<std::unique_ptr<InferenceOutput>> outputs;
    std::vector<Generation> sequences;
    std::vector<std::vector<int64_t>> tokens(3);
    std::vector<bool> done(3);
    for (int row = 0; row < 3; ++row) {
        inputs[row].prompt.assign(row + 3, 1);
        inputs[row].sampling.max_tokens = 10;
        outputs.push_back(std::make_unique<InferenceOutput>(inputs[row], nullptr, std::to_string(row)));
        sequences.push_back(batch.generate(std::to_string(row), inputs[row], *outputs.back()));
    }
    for (int tick = 0; tick < 1000 && !std::all_of(done.begin(), done.end(), [](bool x) { return x; }); ++tick) {
        std::vector<std::string> eligible;
        for (int row = 0; row < 3; ++row) if (!done[row]) eligible.push_back(std::to_string(row));
        batch.step(eligible);
        for (int row = 0; row < 3; ++row) if (!done[row]) {
            auto step = sequences[row].next();
            if (!step) { done[row] = true; continue; }
            if (step.value) if (const auto* delta = std::get_if<OutputDelta>(&*step.value))
                tokens[row].insert(tokens[row].end(), delta->token_ids.begin(), delta->token_ids.end());
        }
    }
    assert(batch.empty() && memory.used == 0);
    assert(memory.peak_rows >= 2 && memory.retracts > 0 && memory.replays > 0);
    const auto metrics = batch.metrics();
    assert(metric(metrics, "continuous_batching_admission_deferred") > 0);
    assert(metric(metrics, "continuous_batching_retractions") == double(memory.retracts));
    assert(memory.victims.front() == "2");
    for (const auto& output : tokens) {
        assert(output.size() == 10);
        for (int i = 0; i < 10; ++i) assert(output[i] == 10 + i);
    }
    // A deferred request can be cancelled without allocating physical state.
    memory.capacity = 0;
    InferenceRequest input;
    input.prompt.assign(4, 1);
    input.sampling.max_tokens = 2;
    InferenceOutput output(input, nullptr, "cancel");
    auto sequence = batch.generate("cancel", input, output);
    assert(sequence.next());
    batch.step({"cancel"});
    assert(memory.used == 0);
    sequence = {};
    assert(batch.empty() && memory.used == 0);
}

static void prefix_and_single_row_test() {
    struct CachedOps : AdmissionOps {
        using AdmissionOps::AdmissionOps;
        int64_t prepare(const std::shared_ptr<Request>&) { memory.used += 2; return 2; }
    };
    AdmissionOps::Memory memory;
    memory.capacity = 5;
    ContinuousBatch<CachedOps> cached(2, 2, memory);
    InferenceRequest input;
    input.prompt.assign(4, 1);
    input.sampling.max_tokens = 2;
    InferenceOutput output(input, nullptr, "cached");
    auto sequence = cached.generate("cached", input, output);
    for (int tick = 0; tick < 30; ++tick) {
        cached.step({"cached"});
        if (!sequence.next()) break;
    }
    assert(cached.empty() && memory.used == 0 && memory.prefills == 1);

    struct OneRowOps : AdmissionOps {
        using AdmissionOps::AdmissionOps;
        bool decode_has_capacity(const State&) const { return false; }
    };
    memory = {};
    ContinuousBatch<OneRowOps> solitary(2, 2, memory);
    InferenceOutput single_output(input, nullptr, "single");
    sequence = solitary.generate("single", input, single_output);
    for (int tick = 0; tick < 30; ++tick) {
        solitary.step({"single"});
        if (!sequence.next()) break;
    }
    assert(solitary.empty() && memory.retracts == 0 && memory.peak_rows == 1);
}

static void adaptive_chunk_test() {
    struct AdaptiveOps : AdmissionOps {
        using AdmissionOps::AdmissionOps;
        PrefillChunk prefill_chunk(const std::shared_ptr<Request>&, PrefillChunk chunk) {
            chunk.count = 1;
            return chunk;
        }
        Sample prefill(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
            assert(chunk.offset == request->prefill_offset && chunk.count == 1);
            return AdmissionOps::prefill(request, chunk);
        }
    };
    AdmissionOps::Memory memory;
    ContinuousBatch<AdaptiveOps> batch(2, 2, memory);
    InferenceRequest input;
    input.prompt.assign(4, 1);
    input.sampling.max_tokens = 2;
    InferenceOutput output(input, nullptr, "adaptive");
    auto sequence = batch.generate("adaptive", input, output);
    for (int tick = 0; tick < 40; ++tick) {
        batch.step({"adaptive"});
        if (!sequence.next()) break;
    }
    assert(batch.empty() && memory.used == 0 && memory.prefills == 4);
}

int main() {
    arithmetic_test();
    queue_and_retraction_test();
    prefix_and_single_row_test();
    adaptive_chunk_test();
}
