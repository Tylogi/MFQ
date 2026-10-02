#include "continuous_batch.h"
#include "generation_policy.h"
#include "models/common/attention.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include "models/common/gated_mlp.h"
#include "models/common/grid_vision_model.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "models/deepseek_v4/causal_lm.h"
#include "models/deepseek_v41/causal_lm.h"
#include "models/deepseek_v41/engram.h"
#include "models/gemma4/causal_lm.h"
#include "models/glm5_next/causal_lm.h"
#include "models/glm_dsa/causal_lm.h"
#include "models/minicpmo45/causal_lm.h"
#include "models/qwen35/causal_lm.h"
#include "models/qwen4_exp/causal_lm.h"
#include "sampling.h"
#include <array>
#include <cassert>
#include <numeric>

using namespace mfq::engine;

struct BatchOps {
    using Request = BatchRequest;
    using State = BatchState<Request>;
    struct Model {
        int vocab_size() const { return 100; }
        int max_position_embeddings() const { return 100; }
    } model_;
    int64_t vocab_size() const { return model_.vocab_size(); }
    int64_t max_context() const { return model_.max_position_embeddings(); }
    struct Sample {
        int64_t token;
        MfqPrefillTiming timing;
    };
    void suspend_decode() {}
    Sample prefill(const std::shared_ptr<Request> &request, PrefillChunk chunk) {
        assert(chunk.offset == request->prefill_offset && chunk.count <= 2);
        return {10, {request->prompt.size(), 1, 0, 1}};
    }
    void activate(const std::shared_ptr<Request> &, const Sample &) {}
    void resume_decode(int64_t) {}
    void discard_prefill(const std::shared_ptr<Request> &) {}
    void retire(const std::vector<std::shared_ptr<Request>> &, int64_t) {}
    int decode(State &) { return 0; }
    Sample sample(const std::shared_ptr<Request> &request, int) {
        return {request->pending_token + 1, {}};
    }
    void accept(const std::shared_ptr<Request> &, const Sample &) {}
    void recover(State &) {}
    Metrics metrics() const { return {}; }
};

void batch_test() {
    InferenceRequest input;
    input.prompt = {1, 2, 3, 4, 5};
    input.sampling.max_tokens = 3;
    ExecutionRequest a(input, nullptr, "a"), b(input, nullptr, "b");
    ContinuousBatch<BatchOps> batch(8, 2);
    batch.admit("a", a);
    batch.admit("b", b);
    for (int i = 0; i < 3; ++i)
        batch.step({"a"});
    assert(!a.done && !b.done && b.events.empty());
    assert(a.output.result.completion_tokens == 1);
    batch.step({"a", "b"});
    batch.step({"a", "b"});
    assert(a.output.result.completion_tokens == 2);
    const auto paused = a.output.result.completion_tokens;
    for (int i = 0; i < 2; ++i)
        batch.step({"b"});
    assert(a.output.result.completion_tokens == paused && !a.done);
    assert(b.output.result.completion_tokens == 1);
    // Cancellation retires a paused row; the other row continues to its limit.
    a.output.result.cancelled = true;
    for (int i = 0; i < 8 && !b.done; ++i)
        batch.step({"b"});
    assert(a.done && b.done && b.output.result.completion_tokens == 3);
    std::vector<int64_t> tokens;
    for (const auto &event : b.events)
        if (const auto *delta = std::get_if<OutputDelta>(&event))
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
    assert((tokens == std::vector<int64_t>{10, 11, 12}));
    ExecutionRequest c(input, nullptr, "c");
    batch.admit("c", c);
    c.output.result.cancelled = true;
    batch.step({});
    assert(c.done && c.output.result.completion_tokens == 0);
}

void batch_failure_test() {
    enum Stage { suspend, prefill, activate, resume };
    struct Fault {
        Stage stage;
        bool armed = false;
        size_t recovered = 0;
    };
    struct FailingOps : BatchOps {
        Fault &fault;
        explicit FailingOps(Fault &state) : fault(state) {}
        void check(Stage stage) {
            if (fault.armed && fault.stage == stage)
                throw std::runtime_error("device failure");
        }
        void suspend_decode() { check(suspend); }
        Sample prefill(const std::shared_ptr<Request> &request, PrefillChunk chunk) {
            check(Stage::prefill);
            return BatchOps::prefill(request, chunk);
        }
        void activate(const std::shared_ptr<Request> &, const Sample &) { check(Stage::activate); }
        void resume_decode(int64_t) { check(resume); }
        void recover(State &state) { fault.recovered = state.prefilling.size(); }
    };
    InferenceRequest input;
    input.prompt = {1, 2};
    input.sampling.max_tokens = 3;
    for (auto stage : {suspend, prefill, activate, resume}) {
        ExecutionRequest a(input, nullptr, "a"), b(input, nullptr, "b");
        Fault fault{stage};
        ContinuousBatch<FailingOps> batch(8, 2, fault);
        batch.admit("a", a);
        batch.step({"a"});
        assert(!a.done && a.output.result.completion_tokens == 1);
        batch.admit("b", b);
        fault.armed = true;
        batch.step({"b"});
        // Covers a popped prefill row and failure after adding it to active.
        assert(a.done && a.failure && b.done && b.failure && fault.recovered == 2);
        for (const auto &[key, value] : batch.metrics())
            if (key == "continuous_batching_active" || key == "continuous_batching_prefilling")
                assert(value == 0);
        batch.step({});
        fault.armed = false;
        ExecutionRequest c(input, nullptr, "c");
        batch.admit("c", c);
        for (int i = 0; i < 5 && !c.done; ++i)
            batch.step({"c"});
        assert(c.done && !c.failure && c.output.result.completion_tokens == 3);
    }
    for (const auto limits : {std::pair{0, 2}, std::pair{2, 0}}) {
        try {
            ContinuousBatch<BatchOps> batch(limits.first, limits.second);
            assert(false);
        } catch (const std::invalid_argument &) {
        }
    }
}

struct Block {
    int kept = -1, begun = 0, committed = 0;
    bool fail = false;
    void reset(int) { kept = 0; }
    void begin_speculative(int) {
        if (fail)
            throw std::runtime_error("begin failed");
        ++begun;
    }
    void commit_speculative() { ++committed; }
    void rollback_speculative(int64_t keep) { kept = keep; }
};
struct Model {
    int64_t cache_pos = 10, decode_position_delta = 2, speculative_start = -1,
            speculative_confirmed = 0;
    bool speculative_suffix_forward = false;
    std::vector<std::unique_ptr<Block>> blocks;
    bool supports_suffix_speculation() const { return true; }
    int max_position_embeddings() const { return 100; }
    void adapter_begin_speculative() {}
    void adapter_commit_speculative() {}
    void adapter_rollback_speculative(int64_t) {}
    void adapter_reset(int64_t) {}
};
void model_test() {
    using namespace mfq::models;
    Model model;
    for (int i = 0; i < 2; ++i)
        model.blocks.push_back(std::make_unique<Block>());
    auto guard = [](const auto &) { return 0; };
    begin_speculative_suffix(model, 4, guard);
    model.cache_pos = 14;
    finish_speculative(model, false, 2, guard);
    assert(model.cache_pos == 12 && model.speculative_start == -1);
    for (const auto &block : model.blocks)
        assert(block->kept == 12);
    model.blocks.back()->fail = true;
    try {
        begin_speculative_suffix(model, 3, guard);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(model.speculative_start == -1 && model.blocks.front()->committed == 1);
    reset_model(model, 1, guard);
    assert(model.cache_pos == 0 && model.decode_position_delta == 0);

    // Noncommutative scalar operators detect residual/norm ordering changes.
    auto fused = [](int normalized, int residual) -> std::optional<int> {
        return residual + normalized * 3;
    };
    auto fallback = [](int, int) { return std::optional<int>{}; };
    auto ffn = [](int x) { return x * 3; };
    auto add = [](int a, int b) { return a + b; };
    assert(feed_forward(7, 2, fused, ffn, add) == feed_forward(7, 2, fallback, ffn, add));
    auto result = attention_layer(
        5, [](int x) { return x - 2; }, [](int x) { return x * 4; },
        [&](int residual, int attention) {
            return residual_ffn(
                residual, attention,
                [](int a, int b) { return std::array<int, 2>{a + b, (a + b) / 2}; },
                [&](int r, int n) { return feed_forward(r, n, fallback, ffn, add); });
        });
    assert(result == 41);
    int commits = 0;
    assert(hyperconnection_layer(
               2, [](int x) { return x + 1; }, [](int x) { return x * 2; },
               [](int x, int residual, int) { return x + residual; },
               [](int x, int previous) { return x + previous; }, [](int x) { return x * 3; },
               [](int x, int residual, int) { return x + residual; },
               [&](int mix) {
                   assert(mix == 11);
                   ++commits;
               }) == 41);
    assert(commits == 1);
    int expert_calls = 0;
    auto experts = [&] {
        ++expert_calls;
        return 10;
    };
    auto dense = [] { return 2; };
    auto post = [](int x, gemma4::FfnNorm) { return x * 2; };
    auto scale = [](int x) { return x * 2; };
    auto fused_gemma = [](int d, int e, int r) { return ((d * 2 + e * 2) * 2 + r) * 2; };
    assert(gemma4::feed_forward(false, true, false, 1, dense, experts, post, add, add, scale,
                                fused_gemma) == 10 &&
           expert_calls == 0);
    for (bool fusion : {false, true})
        assert(gemma4::feed_forward(true, true, fusion, 1, dense, experts, post, add, add, scale,
                                    fused_gemma) == 98);
    assert(expert_calls == 2);
    assert(pre_norm_layer(
               4, [](int x, int stage) { return x - stage; }, [](int x) { return x * 2; }, ffn,
               add) == 45);
}
void inner_model_flow_test() {
    using namespace mfq::models;
    using Heads = AttentionHeads<int>;
    // All three native fusion choices must implement the same attention graph.
    for (int fusion = 0; fusion < 3; ++fusion) {
        std::string order;
        int cached = 0;
        const auto result = full_attention(
            2,
            [&](int x) {
                order += 'P';
                return Heads{x, 3, 5, 7};
            },
            [&](Heads &h) {
                if (fusion != 2)
                    return false;
                order += 'F';
                cached = h.query = h.query * 2 + 3;
                return true;
            },
            [&](Heads &h) {
                order += 'N';
                h.query *= 2;
            },
            [&](Heads &h) {
                if (fusion != 1)
                    return false;
                order += 'F';
                cached = h.query += 3;
                return true;
            },
            [&](Heads &h) {
                order += 'R';
                h.query += 3;
            },
            [&](const Heads &h) {
                order += 'C';
                cached = h.query;
            },
            [&](const Heads &h) {
                order += 'A';
                assert(cached == h.query);
                return h.query * h.key + h.value;
            },
            [&](int h, std::optional<int> gate) {
                order += 'O';
                return h * gate.value();
            });
        assert(result == 182 && cached == 7);
        assert(order == (fusion == 0 ? "PNRCAO" : fusion == 1 ? "PNFAO" : "PFAO"));
    }
    std::string order;
    auto result = qwen35::linear_attention(
        2,
        [&](int x) {
            order += 'P';
            return qwen35::LinearProjections<int>{x, 3, 4, 5, 6, 7};
        },
        [&](const auto &p) {
            order += 'G';
            return std::array<int, 2>{p.alpha + 1, p.beta - 1};
        },
        [&](const auto &p) {
            order += 'C';
            return std::array<int, 3>{p.qkv + 1, p.qk * 2, p.value - 1};
        },
        [&](auto qkv, auto gates) {
            order += 'R';
            return qkv[0] * gates[0] + qkv[1] * gates[1] + qkv[2];
        },
        [&](int h) {
            order += 'N';
            return h - 2;
        },
        [&](int h, int gate) {
            order += 'O';
            return h * gate;
        });
    assert(result == 290 && order == "PGCRNO");

    for (bool gelu : {false, true})
        for (double limit : {0.0, 2.0}) {
            for (int fusion = 0; fusion < 4; ++fusion) {
                const double effective_limit = gelu ? 0.0 : limit;
                auto activate = [](double gate, double up, GatedActivation mode, double limit) {
                    return (gate + (mode == GatedActivation::gelu ? 10 : 1) + limit) * up;
                };
                auto optional = [](bool selected, double value) {
                    return selected ? std::optional<double>(value) : std::nullopt;
                };
                const double hidden = (3 + (gelu ? 10 : 1) + effective_limit) * 4;
                int down_calls = 0;
                auto output = gated_mlp(
                    2.0, gelu, limit,
                    [&](double x, auto a, double l) {
                        return optional(fusion == 1, activate(x + 1, x * 2, a, l) - 3);
                    },
                    [&](double x, auto a, double l) {
                        return optional(fusion == 2, activate(x + 1, x * 2, a, l));
                    },
                    [](double x) { return std::array<double, 2>{x + 1, x * 2}; }, activate,
                    [&](double h) {
                        ++down_calls;
                        return h - 3;
                    },
                    [&](double g, double u, auto a, double l) {
                        return optional(fusion == 3, activate(g, u, a, l) - 3);
                    });
                assert(output == hidden - 3 && down_calls == (fusion == 0 || fusion == 2));
            }
        }
}

struct PredictorOps {
    struct Tensor {
        std::vector<int64_t> shape;
        int value = 0;
    };
    struct Model {
        struct Config {
            int hidden_size = 4, max_position_embeddings = 8;
            std::vector<int> mrope_sections;
        } config;
        int64_t cache_pos = 0;
        std::array<int, 2> blocks{1, 2};
    } model;
    std::vector<int> visited;
    std::optional<Tensor> seen_lengths, seen_cache;
    static int rank(const Tensor &t) { return t.shape.size(); }
    static int64_t size(const Tensor &t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size() + axis : axis);
    }
    static int64_t elements(const Tensor &t) {
        int64_t count = 1;
        for (auto n : t.shape)
            count *= n;
        return count;
    }
    static bool defined(const Tensor &t) { return !t.shape.empty(); }
    static Tensor embed(Tensor ids, const Tensor &like) { return {like.shape, ids.value + 2}; }
    static Tensor normalize(Tensor t, mfq::models::qwen35::PredictorNorm role) {
        t.value += static_cast<int>(role) + 1;
        return t;
    }
    static void trace(const char *, const Tensor &) {}
    static Tensor fuse(Tensor e, Tensor h) {
        h.value += e.value * 10;
        return h;
    }
    static Tensor positions(Tensor t, int64_t start, int64_t tokens) {
        return defined(t) ? t : Tensor{{tokens}, static_cast<int>(start)};
    }
    static Tensor lengths(int64_t batch, int64_t end, const Tensor &) {
        return {{batch}, static_cast<int>(end)};
    }
    static Tensor cache_positions(const Tensor &, int64_t start, int64_t tokens) {
        return {{tokens}, static_cast<int>(start)};
    }
    Tensor layer(int block, Tensor h, const Tensor &, const std::optional<Tensor> &lengths,
                 const std::optional<Tensor> &cache) {
        visited.push_back(block);
        seen_lengths = lengths;
        seen_cache = cache;
        h.value = h.value * 2 + block;
        return h;
    }
};

void predictor_flow_test() {
    using mfq::models::qwen35::mtp_predictor;
    PredictorOps ops;
    auto run = [&](int64_t tokens, PredictorOps::Tensor pos = {}) {
        return mtp_predictor(ops, {{1, tokens, 4}, 3}, {{1, tokens}, 5}, pos);
    };
    assert(run(2).value == 347 && ops.model.cache_pos == 2);
    assert((ops.visited == std::vector<int>{1, 2}));
    assert(!ops.seen_lengths && !ops.seen_cache);
    assert(run(1, {{1}, 90}).value == 347 && ops.model.cache_pos == 3);
    assert(ops.seen_lengths->value == 3 && ops.seen_cache->value == 2);
    ops.model.config.mrope_sections = {1, 1, 1};
    run(1, {{3, 1}, 20});
    assert(ops.model.cache_pos == 4 && ops.seen_cache->value == 3);
    const auto layer_calls = ops.visited.size();
    for (int invalid = 0; invalid < 3; ++invalid) {
        try {
            if (invalid == 0)
                run(5); // Context overflow.
            if (invalid == 1)
                run(1, {{2, 1}, 0}); // Invalid position shape.
            if (invalid == 2)
                mtp_predictor(ops, {{1, 1, 4}, 3}, {{2, 1}, 5}, {});
            assert(false);
        } catch (const std::runtime_error &) {
        }
        assert(ops.model.cache_pos == 4 && ops.visited.size() == layer_calls);
    }
}

// A non-CUDA tensor exercises the actual shared CausalLm class end to end.
struct CausalTestOps {
    using Tensor = PredictorOps::Tensor;
    struct ForwardPlan {
        int tag = 0;
    };
    struct SessionState {
        int64_t cache_pos = 0, decode_position_delta = 0;
    };
    enum class SessionStateKind { Unsupported, Full };
    struct SessionCodec {
        template <class Model> static auto kind(const Model &) { return SessionStateKind::Full; }
        template <class Model> static bool supports_paged(const Model &) { return true; }
        template <class Model>
        static SessionState capture(const Model &m, const std::vector<int64_t> &) {
            return {m.cache_pos, 0};
        }
        template <class Model> static void restore(Model &m, const SessionState &s) {
            m.cache_pos = s.cache_pos;
        }
    };
    struct Block {
        int id;
        int64_t kept = -1;
        bool supports_speculation() const { return true; }
        void reset(int64_t) {}
        void begin_speculative(int64_t) {}
        void commit_speculative() {}
        void rollback_speculative(int64_t value) { kept = value; }
    };
    mfq::models::CausalLmMetadata metadata;
    std::vector<std::unique_ptr<Block>> blocks;
    Tensor output_norm;
    int embed = 0, lm_head = 0;
    bool fail_layer = false;
    int seen_position = -1, seen_cache = -1, seen_plan = -1;
    std::optional<Tensor> seen_lengths;
    CausalTestOps() {
        metadata.hidden_size = 4;
        metadata.max_position_embeddings = 32;
        blocks.push_back(std::make_unique<Block>(Block{1}));
        blocks.push_back(std::make_unique<Block>(Block{2}));
    }
    static bool mask_all_ones(const Tensor &value) { return value.value == 1; }
    static int execution_scope() { return 0; }
    static int block_scope(const std::unique_ptr<Block> &) { return 0; }
    static int64_t rank(const Tensor &t) { return t.shape.size(); }
    static int64_t size(const Tensor &t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size() + axis : axis);
    }
    static Tensor batch_ids(Tensor t) {
        t.shape.insert(t.shape.begin(), 1);
        return t;
    }
    static Tensor device_ids(Tensor t) { return t; }
    static Tensor embed_tokens(Tensor t) {
        t.shape.push_back(4);
        t.value += 4;
        return t;
    }
    static Tensor sequence_lengths(int64_t batch, int64_t end) { return {{batch}, int(end)}; }
    static Tensor last_hidden(Tensor t) {
        t.shape.erase(t.shape.begin() + 1);
        return t;
    }
    static Tensor softcap(Tensor t, double) {
        t.value = -t.value;
        return t;
    }
    static Tensor adapter_finalize_hidden(Tensor t, const Tensor &, int64_t, int64_t) {
        t.value += 3;
        return t;
    }
    static Tensor adapter_logits(int, Tensor t) {
        t.value *= 5;
        return t;
    }
    static Tensor adapter_last_logits(int, Tensor t) {
        t.value *= 7;
        return t;
    }
    static Tensor adapter_next_token(int, Tensor t) {
        t.value %= 11;
        return t;
    }
    static void adapter_reset(int64_t) {}
    static bool adapter_requires_batch_reset(int64_t) { return false; }
    static void adapter_begin_forward(bool) {}
    static void adapter_finish_forward(const Tensor &, int64_t, int64_t) {}
    static void adapter_begin_speculative() {}
    static void adapter_commit_speculative() {}
    static void adapter_rollback_speculative(int64_t) {}
    template <class Model> static auto forward_ops(Model &model, typename Model::Inputs input) {
        struct Ops : Model::Inputs {
            Model &model;
            Tensor pos, full_positions, cache_positions;
            std::optional<Tensor> effective_attention_mask;
            Ops(Model &m, typename Model::Inputs i) : Model::Inputs(std::move(i)), model(m) {}
            static int64_t rank(const Tensor &t) { return CausalTestOps::rank(t); }
            static int64_t size(const Tensor &t, int axis) { return CausalTestOps::size(t, axis); }
            static int64_t elements(const Tensor &t) { return PredictorOps::elements(t); }
            static Tensor device_ids(Tensor t) { return t; }
            static Tensor position_range(int64_t start, int64_t n) { return {{n}, int(start)}; }
            static Tensor offset_positions(Tensor t, int64_t delta) {
                t.value += delta;
                return t;
            }
            static bool has_mrope() { return false; }
            void prepare_inputs() {}
            Tensor prepare_hidden(int64_t, int64_t) {
                model.seen_position = pos.value;
                model.seen_cache = cache_positions.value;
                model.seen_plan = this->plan.tag;
                model.seen_lengths = this->seq_len;
                return this->input_embeddings;
            }
            Tensor layer(const std::unique_ptr<Block> &block, Tensor t) {
                if (model.fail_layer)
                    throw std::runtime_error("injected layer failure");
                t.value = t.value * 2 + block->id;
                return t;
            }
            void trace(const Tensor &t) {
                if (this->block_trace)
                    this->block_trace->push_back(t);
            }
            Tensor finish(Tensor t, int64_t b, int64_t n) {
                if (this->raw_hidden)
                    *this->raw_hidden = t;
                return model.finalize_hidden(t, b, n);
            }
        };
        return Ops(model, std::move(input));
    }
};

void causal_lm_test() {
    using Tensor = CausalTestOps::Tensor;
    struct TestModel : mfq::models::CausalModelBase<CausalTestOps, TestModel> {
        bool adapter_supports_speculation() const { return true; }
        bool adapter_supports_suffix_speculation() const { return true; }
    } model;
    Tensor ids{{1, 2}, 2}, raw;
    std::vector<Tensor> trace;
    auto hidden = model.hidden_forward(ids, {}, {}, &trace, {}, &raw);
    assert(hidden.value == 31 && raw.value == 28 && model.cache_pos == 2);
    assert(trace.size() == 3 && trace[0].value == 6 && trace[1].value == 13 &&
           trace[2].value == 28);
    assert(model.forward(Tensor{{1}, 2}).value == 155 && model.cache_pos == 3);
    assert(model.last_logits(Tensor{{1}, 2}).value == 217 && model.cache_pos == 4);
    assert(model.seen_lengths->value == 4);
    assert(model.next_token(Tensor{{1}, 2}).value == 9 && model.cache_pos == 5);
    model.decode_position_delta = 10;
    model.hidden_forward(Tensor{{1}, 2});
    assert(model.seen_position == 15 && model.seen_cache == 5 && model.cache_pos == 6);
    model.next_token_static(Tensor{{1}, 2}, Tensor{{1}, 20}, Tensor{{1}, 21}, {17});
    assert(model.seen_position == 20 && model.seen_cache == 20 && model.seen_plan == 17 &&
           model.cache_pos == 6);
    auto state = model.capture_text_session_state({1, 2, 3, 4, 5, 6});
    model.reset(1);
    assert(model.cache_pos == 0 && model.decode_position_delta == 0);
    model.restore_text_session_state(state);
    assert(model.cache_pos == 6 && model.decode_position_delta == 10);
    model.begin_speculative_suffix(2);
    model.hidden_forward_speculative_suffix(Tensor{{2}, 2});
    model.rollback_speculative(1);
    assert(model.cache_pos == 7 && model.blocks[0]->kept == 7 && !model.speculative_suffix_forward);
    model.begin_speculative_suffix(1);
    model.fail_layer = true;
    try {
        model.hidden_forward_speculative_suffix(Tensor{{1}, 2});
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(!model.speculative_suffix_forward && model.cache_pos == 7);
    model.fail_layer = false;
    model.rollback_speculative();
    for (Tensor invalid : {Tensor{{}, 0}, Tensor{{1, 1, 1}, 0}, Tensor{{0}, 0}}) {
        try {
            model.hidden_forward(invalid);
            assert(false);
        } catch (const std::runtime_error &) {
        }
        assert(model.cache_pos == 7);
    }
    // Semantic positions do not advance the logical cache unless requested.
    model.hidden_forward_inputs(ids, Tensor{{1, 2, 4}, 6}, Tensor{{2}, 100}, {}, nullptr, {}, true);
    assert(model.cache_pos == 9 && model.seen_position == 100 && model.seen_cache == 7);
    try {
        model.hidden_forward_inputs(ids, Tensor{{1, 2, 3}, 0});
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(model.cache_pos == 9);
}

struct FinalCausalTestOps : CausalTestOps {
    Tensor collapse_hidden(Tensor t, int64_t, int64_t) const {
        t.value = t.value * 2 + 1;
        return t;
    }
    Tensor normalize_hidden(Tensor t, const Tensor &, int64_t, int64_t) const {
        t.value *= 3;
        return t;
    }
};

void family_model_test() {
    using namespace mfq::models;
    struct QwenOps : CausalTestOps {
        Tensor positions;
        static bool defined(const Tensor &t) { return !t.shape.empty(); }
        static Tensor position_axes(Tensor t, int64_t start, int64_t count) {
            assert(start == 1 && count == 3);
            t.shape[0] = count;
            return t;
        }
        static Tensor concat_positions(Tensor first, Tensor second) {
            second.shape.back() += first.shape.back();
            return second;
        }
        qwen4_exp::Config config{};
        int batch = 1;
        static bool adapter_supports_speculation() { return false; }
    };
    qwen4_exp::CausalLm<QwenOps> qwen;
    // Dispatch must reach the family's rules, including from inherited entry points.
    assert(qwen.supports_speculation());
    qwen.hidden_forward_inputs({{1, 2}, 2}, {{1, 2, 4}, 6}, CausalTestOps::Tensor{{2}, 10});
    assert(qwen.cache_pos == 2); // Qwen4 advances even with explicit semantic positions.
    qwen.positions = {{3, 2}, 0};
    auto prepared = qwen.adapter_prepare_positions({{4, 1}, 9}, 1, 1);
    assert(prepared.positions.shape == std::vector<int64_t>({3, 1}));
    assert(prepared.full_positions.shape == std::vector<int64_t>({3, 3}));
    glm5_next::CausalLm<FinalCausalTestOps> glm;
    try {
        glm.hidden_forward({{1, 2}, 2}, CausalTestOps::Tensor{{2}, 10});
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(glm.cache_pos == 0); // GLM accepts only contiguous implicit positions.
    minicpmo45::CausalLm<CausalTestOps> mini;
    mini.next_token({{1}, 2});
    mini.next_token({{1}, 2});
    assert(!mini.seen_lengths); // Its BF16 attention uses a different decode contract.
    const CausalTestOps::Tensor ones{{1, 2}, 1}, masked{{1, 2}, 0};
    assert(!mini.adapter_attention_mask(ones, 2, 0));
    assert(!mini.adapter_attention_mask(ones, 1, 3));
    assert(mini.adapter_attention_mask(ones, 2, 3));
    assert(mini.adapter_attention_mask(masked, 2, 0));
}

void remaining_family_flows_test() {
    using namespace mfq::models;
    int state = 0, saved = -1;
    auto run = [&](int64_t start, int64_t count) {
        state += count;
        return int(start + count);
    };
    auto checkpoint = [&] { saved = state; };
    auto rollback = [&] { state = saved; };
    auto concat = [](int a, int b) { return a * 10 + b; };
    assert(recurrent_window(5, true, 2, false, run, checkpoint, rollback, concat) == 25);
    assert(saved == 2 && state == 5);
    state = 0;
    try {
        recurrent_window(
            5, true, 2, false,
            [&](int64_t start, int64_t count) {
                state += count;
                if (start)
                    throw std::runtime_error("suffix failed");
                return int(count);
            },
            checkpoint, rollback, concat);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(state == 2);

    assert(deepseek_v4::select_compressed(
               4, 513, [] { return 1; }, [] { return 2; }, [] { return 3; }) == 1);
    assert(deepseek_v4::select_compressed(
               128, 513, [] { return 1; }, [] { return 2; }, [] { return 3; }) == 2);
    assert(deepseek_v4::select_compressed(
               4, 0, [] { return 1; }, [] { return 2; }, [] { return 3; }) == 3);
    assert(glm_dsa::indexed_attention(
               true, 5, 2, 3, [] { return 9; }, [] { return 2; }, [] { return 3; }, concat) == 23);
    try {
        glm_dsa::indexed_attention(
            true, 5, 1, 3, [] { return 9; }, [] { return 2; }, [] { return 3; }, concat);
        assert(false);
    } catch (const std::runtime_error &) {
    }

    // Compression must emit the same complete groups across an unaligned chunk boundary.
    struct Compression {
        int64_t partial_length = 0;
        std::array<int, 4> values{};
    };
    auto compress = [](Compression &state, const std::vector<int> &input, int64_t position) {
        return deepseek_v41::compress(
            state, input.size(), position, 4, [&] { return input; }, [] {},
            [&](int64_t complete, int64_t) {
                std::vector<int> result;
                for (int64_t i = 0; i < complete; ++i)
                    result.push_back(
                        std::accumulate(input.begin() + i * 4, input.begin() + (i + 1) * 4, 0));
                return result;
            },
            [&](int64_t start, int64_t count) {
                std::copy_n(input.begin() + start, count, state.values.begin());
            },
            [&](int64_t token, int64_t slot) { state.values[slot] = input[token]; },
            [&] {
                return std::vector<int>{
                    std::accumulate(state.values.begin(), state.values.end(), 0)};
            },
            [](const auto &parts) {
                std::vector<int> result;
                for (const auto &part : parts)
                    result.insert(result.end(), part.begin(), part.end());
                return result;
            });
    };
    Compression full, chunks;
    const auto expected = compress(full, {1, 2, 3, 4, 5, 6, 7, 8}, 0);
    assert(!compress(chunks, {1, 2, 3}, 0));
    assert(compress(chunks, {4, 5, 6, 7, 8}, 3) == expected);
    assert(full.partial_length == 0 && chunks.partial_length == 0);

    const int64_t sizes[]{2, 2};
    const bool mask[]{true, false, true, true, true};
    auto positions = minicpmo45::patch_positions(1, 5, 70, sizes, mask);
    assert(!positions.all_active && (positions.ids == std::vector<int64_t>{0, 0, 35, 2450, 2485}));

    int64_t tts_position = 0;
    int tts_resets = 0, tts_layers = 0;
    std::array<int, 2> decoder_layers{2, 3};
    auto tts = [&](int64_t tokens) {
        return minicpmo45::tts_forward(
            1, 1, tokens, 4, tts_position, decoder_layers, [&] { ++tts_resets; },
            [](int hidden) { return hidden + 1; }, [&] { return tts_position; },
            [&](int block, int hidden, int64_t position) {
                ++tts_layers;
                assert(position == tts_position);
                return hidden * block;
            },
            [](int hidden) { return hidden + 1; });
    };
    assert(tts(2) == 13 && tts_position == 2 && tts_resets == 1);
    assert(tts(1) == 13 && tts_position == 3 && tts_resets == 1);
    try {
        tts(2);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(tts_position == 3 && tts_layers == 4);

    deepseek_v41::EngramHashState hashes(8, 3, 2, 0, {1}, {36}, {5, 7, 11, 13}, {0, 5, 12, 23},
                                         {3, 5, 7}, {0, 1, 2, 3, 4, 5, 6, 7});
    const int64_t ids[]{1, 2, 3, 4};
    const auto all = hashes.forward(ids, 1, 4, 0).values;
    assert((all == std::vector<int64_t>{3, 8, 15, 26, 3, 8, 15, 26, 3, 8, 16, 27, 3, 8, 14, 23}));
    auto first = hashes.forward(ids, 1, 2, 0).values;
    hashes.begin_speculative();
    auto second = hashes.forward(ids + 2, 1, 2, 2).values;
    hashes.rollback_speculative();
    assert(hashes.forward(ids + 2, 1, 2, 2).values == second);
    first.insert(first.end(), second.begin(), second.end());
    assert(first == all);

    auto reduce = [](int x) { return x * 2; };
    auto gate = [] { return 3; };
    auto add = [](int x, int y) { return x + y; };
    auto gated = [](int x, int y, int g) { return x + y * g; };
    const auto fallback = [](int, int) { return std::optional<int>{}; };
    const auto fused = [](int x, int y) { return std::optional<int>{x * 2 + y * 3}; };
    assert(combine_experts(5, 7, false, false, fallback, reduce, gate, add, gated) == 31);
    assert(combine_experts(5, 7, false, false, fused, reduce, gate, add, gated) == 31);
    assert(combine_experts(10, 7, true, true, fallback, reduce, gate, add, gated) == 17);
}

void boundary_model_test() {
    using namespace mfq::models;
    for (bool ple : {false, true})
        for (bool linear : {false, true}) {
            int selected = 0, embedded = 0;
            auto result = qwen4_exp::decoder_layer(
                2, ple, linear,
                [&](int x) {
                    ++embedded;
                    return x * 3;
                },
                [](int x, int y) { return x + y; },
                [](int x) { return std::array<int, 2>{x + 2, x}; },
                [&](int x) {
                    selected = 1;
                    return x * 3;
                },
                [&](int x) {
                    selected = 2;
                    return x * 5;
                },
                [](int x, auto mix) { return x + mix[1]; },
                [](int x) { return std::array<int, 2>{x - 1, x}; }, [](int x) { return x * 7; },
                [](int x, auto mix) { return x + mix[1]; });
            const int residual = ple ? 8 : 2;
            const int attended = (residual + 2) * (linear ? 3 : 5) + residual;
            assert(result == (attended - 1) * 7 + attended);
            assert(embedded == int(ple) && selected == (linear ? 1 : 2));
        }
    for (bool linear : {false, true}) {
        std::vector<int> norms;
        auto pre = [](int x) { return std::array<int, 3>{1, 2, x}; };
        auto result = glm5_next::decoder_layer(
            3, linear, pre,
            [&](int x, int role) {
                norms.push_back(role);
                return x + role + 1;
            },
            [](int x) { return x * 2; }, [](int x) { return x * 3; },
            [](int branch, int residual, const auto &) { return branch + residual; }, pre,
            [](int x) { return x * 5; });
        const int attended = 3 + 4 * (linear ? 2 : 3);
        assert(result == attended + (attended + 2) * 5);
        assert((norms == std::vector<int>{0, 1}));
    }
    struct Mix {
        int branch, next_pre;
    };
    int previous = 1, captured = 0;
    auto collapse = [](int hidden, int pre, int role) { return Mix{hidden + pre, pre + role + 1}; };
    auto expand = [](int branch, int residual, const Mix &, int) { return residual + branch; };
    auto result = deepseek_v41::decoder_layer(
        2, true, previous, [](int x) { return x * 3; }, [&](int x) { captured = x; }, collapse,
        [](int x) { return x * 2; }, expand, [](int x) { return x * 3; });
    assert(captured == 6 && result == 86 && previous == 4);
    previous = 1;
    try {
        deepseek_v41::mega_layer(
            2, previous, collapse, [](int x) { return x; }, expand,
            [](int) -> int { throw std::runtime_error("ffn failed"); });
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(previous == 1); // Never commit pre-weights from a failed layer.

    glm5_next::CausalLm<FinalCausalTestOps> glm;
    assert(glm.adapter_finalize_hidden({{1, 1, 4}, 2}, {}, 1, 1).value == 15);
    deepseek_v4::CausalLm<FinalCausalTestOps> v4;
    deepseek_v41::CausalLm<FinalCausalTestOps> v41;
    assert(v4.adapter_finalize_hidden({{1, 1, 4}, 2}, {}, 1, 1).value == 15);
    assert(v41.adapter_finalize_hidden({{1, 1, 4}, 2}, {}, 1, 1).value == 15);

    for (bool parallel : {false, true}) {
        std::array<int, 2> calls{};
        auto result = additive_branches(
            [&](int i) {
                ++calls[i];
                return i == 0 ? 3 : 5;
            },
            [&](auto &run, auto &outputs) {
                if (parallel)
                    outputs = {run(0), run(1)};
                return parallel;
            },
            [](int low, int high) { return low * 10 + high; });
        assert(result == 35 && calls[0] == 1 && calls[1] == 1);
    }
    try {
        additive_branches([](int) { return 1; }, [](auto &, auto &) { return true; },
                          [](int a, int b) { return a + b; });
        assert(false);
    } catch (const std::runtime_error &) {
    }
}

struct TtsTraceOps {
    using Tensor = int;
    std::string order;
    int temperature(int value, double) {
        order += 'T';
        return value;
    }
    int penalties(int value, std::span<const int> history, double) {
        order += 'P';
        assert(history.size() == 16 && history.front() == 4);
        return value;
    }
    void mask_eos(int &, int64_t) { order += 'E'; }
    int reference(int value, int64_t, double, int64_t keep) {
        order += 'R';
        assert(keep == 3);
        return value;
    }
    int top_k(int value, int64_t) {
        order += 'K';
        return value;
    }
    int top_p(int value, double) {
        order += 'N';
        return value;
    }
    int min_p(int value, double) {
        order += 'M';
        return value;
    }
    int sample(int value) {
        order += 'S';
        return value;
    }
};
void token_generation_test() {
    for (int limit : {0, 1, 4})
        for (bool eos : {false, true}) {
            std::vector<int> accepted;
            int samples = 0, advances = 0;
            const auto result = generate_tokens(
                limit,
                [&](int64_t step) {
                    ++samples;
                    return int(step);
                },
                [&](int token, int64_t) { accepted.push_back(token); }, [&](int) { return eos; },
                [&](int) { ++advances; });
            const int count = eos && limit ? 1 : limit;
            assert(samples == count && accepted.size() == size_t(count) && result.tokens == count);
            assert(advances == std::max(0, count - 1) && result.hit_eos == (eos && limit > 0));
        }
    int accepted = 0;
    try {
        generate_tokens(
            3, [](int64_t step) { return step; }, [&](int, int64_t) { ++accepted; },
            [](int) { return false; }, [](int) { throw std::runtime_error("decode failed"); });
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(accepted == 1);
    std::array<int, 20> history{};
    std::iota(history.begin(), history.end(), 0);
    mfq::models::minicpmo45::TtsSampling options;
    options.minimum_keep = 3;
    options.min_p = 0.1;
    options.validate(3, 6562);
    TtsTraceOps ops;
    mfq::models::minicpmo45::sample_tts(ops, 0, options, history, 0, false);
    assert(ops.order == "PEKNMTS");
    ops.order.clear();
    mfq::models::minicpmo45::sample_tts(ops, 0, options, history, 0, true);
    assert(ops.order == "TPER");
    ops.order.clear();
    options.minimum_steps = 0;
    options.repetition_penalty = 1;
    mfq::models::minicpmo45::sample_tts(ops, 0, options, {}, 0, false);
    assert(ops.order == "KNMTS");
    std::mt19937 rng(42);
    const float scores[]{0, 1, 5, -1};
    for (int i = 0; i < 10; ++i)
        assert(sample_top_k_top_p(scores, 3, 0.01, 0, rng) == 2);
    const float invalid[]{-INFINITY, -INFINITY};
    try {
        sample_top_k_top_p(invalid, 1, 1, 0, rng);
        assert(false);
    } catch (const std::invalid_argument &) {
    }
}

struct LoadTestModel : mfq::models::CausalModelBase<CausalTestOps, LoadTestModel> {
    static bool accepts_backbone(std::string_view value) { return value == "fixture"; }
    void adapter_set_max_position_embeddings(int64_t) {}
    template <class Graph, class Source>
    void adapter_load_config(std::string_view, const Graph &, const Source &) {
        metadata.num_hidden_layers = 2;
        metadata.layer_types = {"first", "second"};
    }
};
void model_loading_test() {
    struct Graph {
        std::string backbone = "fixture";
        struct {
            int64_t text_layers = 2;
        } topology;
    } graph;
    struct Loader {
        bool has_output = true;
        std::vector<std::string> calls;
        int embedding(const std::string &name) {
            calls.push_back(name);
            return 11;
        }
        int output(const std::string &name) {
            calls.push_back(name);
            return 22;
        }
        int tied_output(int value, const std::string &name) {
            calls.push_back("tie:" + name);
            return value;
        }
        bool has_weight(const std::string &) { return has_output; }
        void final_state(CausalTestOps::Tensor &) { calls.push_back("norm"); }
        void prepare_blocks() { calls.push_back("prepare"); }
        auto block(int layer, const std::string &type) {
            calls.push_back(type);
            return std::make_unique<CausalTestOps::Block>(CausalTestOps::Block{layer});
        }
    };
    for (int mode = 0; mode < 3; ++mode) {
        LoadTestModel model;
        model.blocks.clear();
        model.load_definition("", graph, 0, 16);
        assert(model.max_position_embeddings() == 16);
        model.metadata.tie_word_embeddings = mode == 1;
        Loader loader;
        loader.has_output = mode != 2;
        model.load_weights(loader);
        assert(model.embed == 11 && model.lm_head == (mode ? 11 : 22) && model.blocks.size() == 2);
        assert(loader.calls == std::vector<std::string>({"model.token_embedding.weight", "norm",
                                                         mode ? "tie:model.token_embedding.weight"
                                                              : "model.output.weight",
                                                         "prepare", "first", "second"}));
    }
    LoadTestModel model;
    model.blocks.clear();
    model.load_definition("", graph, 0);
    Loader loader;
    model.load_weights(loader, false);
    assert(model.blocks.empty() && loader.calls.size() == 3);
    for (int invalid = 0; invalid < 3; ++invalid) {
        auto broken = graph;
        if (invalid == 0)
            broken.backbone = "other";
        if (invalid == 1)
            broken.topology.text_layers = 3;
        try {
            model.load_definition("", broken, 0, invalid == 2 ? 33 : 0);
            assert(false);
        } catch (const std::runtime_error &) {
        }
    }
}

struct MediaTestOps {
    using Tensor = CausalTestOps::Tensor;
    std::string calls;
    static int64_t rank(const Tensor &t) { return t.shape.size(); }
    static int64_t size(const Tensor &t, int axis) { return t.shape.at(axis); }
    static bool defined(const Tensor &t) { return !t.shape.empty(); }
    static Tensor batch_ids(Tensor t) {
        t.shape.insert(t.shape.begin(), 1);
        return t;
    }
    static Tensor device_ids(Tensor t) { return t; }
    Tensor embed(Tensor ids) {
        calls += 'E';
        return {{ids.shape[0], ids.shape[1], 4}, 4};
    }
    Tensor vision(Tensor, Tensor, Tensor) {
        calls += 'V';
        return {{1, 4, 3}, 7};
    }
    Tensor resample(Tensor t, Tensor) {
        calls += 'R';
        assert(t.value == 7);
        return {{1, 2, 4}, 10};
    }
    void reset_audio() { calls += 'A'; }
    Tensor audio(Tensor, Tensor) {
        calls += 'U';
        return {{1, 3, 4}, 20};
    }
    auto audio_lengths(Tensor) {
        calls += 'L';
        return std::vector<int64_t>{3};
    }
    void scatter(Tensor &target, const Tensor &source,
                 const mfq::models::minicpmo45::MediaBound &bound) {
        calls += 'S';
        assert(bound.end - bound.begin == source.shape[1]);
        target.value += source.value;
    }
};

void composition_boundary_test() {
    using namespace mfq::models;
    using Tensor = CausalTestOps::Tensor;
    MediaTestOps ops;
    minicpmo45::MultimodalInputs<Tensor> input{{{6}, 1}, {{1}, 1}, {{1}, 1},       {{1}, 1},
                                               {{1}, 1}, {{1}, 1}, {{0, 0, 1, 3}}, {{0, 0, 3, 6}}};
    auto result = minicpmo45::encode(ops, input);
    assert(ops.calls == "EVRSAULS" && result.input_embeddings.value == 34);
    assert(input.ids.shape == std::vector<int64_t>({1, 6}));
    minicpmo45::CausalLm<CausalTestOps> model;
    auto forwarded = minicpmo45::multimodal_forward(
        model, input.ids, result, std::optional<Tensor>{}, std::optional<Tensor>{});
    assert(model.cache_pos == 6 && forwarded.hidden_states.value == 143 &&
           forwarded.logits.value == 715);
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto broken = input;
        if (invalid == 0)
            broken.images[0].end = 4;
        if (invalid == 1)
            broken.images[0].source = 1;
        if (invalid == 2)
            broken.audios[0].source = -1;
        if (invalid == 3)
            broken.pixels = {};
        try {
            minicpmo45::encode(ops, broken);
            assert(false);
        } catch (const std::runtime_error &) {
        }
    }
    auto text = input;
    text.images.clear();
    text.audios.clear();
    ops.calls.clear();
    assert(minicpmo45::encode(ops, text).input_embeddings.value == 4 && ops.calls == "E");
    try {
        minicpmo45::media_bounds(std::array<int64_t, 4>{0, 0, 2, 1});
        assert(false);
    } catch (const std::runtime_error &) {
    }

    mfq::GridVisionConfig config;
    config.spatial_merge_size = 2;
    config.hidden_size = 3;
    config.num_position_embeddings = 4;
    std::array<int, 2> layers{1, 2};
    std::string order;
    const auto encoded = grid_vision::encode(
        1, config, {{1, 2, 2}}, layers,
        [&](int x, int64_t count) {
            order += 'P';
            assert(count == 4);
            return x + 2;
        },
        [&](const mfq::LearnedPositionInterpolation &p) {
            order += 'I';
            assert(p.patch_count == 4);
            return 5;
        },
        [&](int x, int y) {
            order += 'A';
            return x + y;
        },
        [&](int layer, int x, const mfq::GridVisionLayout &layout) {
            order += 'L';
            assert(layout.patch_count == 4);
            return x * layer;
        });
    assert(encoded == 16 && order == "PIALL");
    order.clear();
    auto merge = [&](int64_t count) {
        return grid_vision::merge(
            encoded, count, config,
            [&](int x) {
                order += 'N';
                return x + 1;
            },
            [&](int x, int64_t rows, int64_t width) {
                order += 'R';
                assert(rows == 1 && width == 12);
                return x;
            },
            [&](int x) {
                order += 'U';
                return x * 2;
            },
            [&](int x) {
                order += 'G';
                return x + 3;
            },
            [&](int x) {
                order += 'D';
                return x * 5;
            });
    };
    assert(merge(4) == 185 && order == "NRUGD");
    try {
        merge(3);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    int scatters = 0;
    auto prepare = [&](std::vector<int64_t> ids) {
        return grid_vision::prepare(
            ids, 185, 1, 99, 98, config, {{1, 2, 2}}, [](const auto &) { return 7; },
            [&](int &embedding, int vision, const auto &indices) {
                ++scatters;
                assert(indices == std::vector<int64_t>{1});
                embedding += vision;
            },
            [](const mfq::GridMropePositions &positions) {
                assert(positions.token_count == 3);
                return 1;
            });
    };
    assert(prepare({1, 99, 2}).embeddings == 192 && scatters == 1);
    for (const auto &ids : {std::vector<int64_t>{1, 2, 3}, std::vector<int64_t>{1, 98, 3}}) {
        try {
            prepare(ids);
            assert(false);
        } catch (const std::invalid_argument &) {
        }
    }
    assert(scatters == 1);

    gemma4::Config gemma;
    gemma.num_attention_heads = 8;
    gemma.num_key_value_heads = 2;
    gemma.num_global_key_value_heads = 1;
    gemma.head_dim = 4;
    gemma.global_head_dim = 8;
    gemma.sliding_window = 16;
    gemma.full_rotary_factor = 0.5;
    gemma.attention_key_equals_value = true;
    auto local = gemma4::layer_spec(gemma, "sliding_attention");
    auto global = gemma4::layer_spec(gemma, "full_attention");
    assert(local.head_dim == 4 && local.kv_heads == 2 && local.window == 16 &&
           !local.value_equals_key);
    assert(global.head_dim == 8 && global.kv_heads == 1 && global.window == 0 &&
           global.value_equals_key && global.rotary_pairs == 2);
    try {
        gemma4::layer_spec(gemma, "unknown");
        assert(false);
    } catch (const std::runtime_error &) {
    }

    // Model head policy is shared even for a backend whose projection is just an integer operation.
    model.metadata.final_logit_softcapping = 1;
    assert(model.logits_from_hidden({{1, 4}, 3}).value == -15);
    assert(model.last_logits({{1}, 2}).value == -217);
}

int main() {
    model_loading_test();
    composition_boundary_test();
    boundary_model_test();
    token_generation_test();
    family_model_test();
    remaining_family_flows_test();
    causal_lm_test();
    inner_model_flow_test();
    predictor_flow_test();
    batch_test();
    batch_failure_test();
    model_test();
}
