#include "speculative_sequence.h"

#include <iostream>
#include <thread>

namespace {
using namespace mfq::engine;

struct Tensor {
    std::vector<int64_t> shape;
    bool proposal = false;
};

struct Model {
    int64_t cache_pos = 0, decode_position_delta = 0, speculative_start = 0;
    int vocab_size() const { return 3; }
    int max_position_embeddings() const { return 1024; }
    void begin_speculative_suffix(int) { speculative_start = cache_pos-1; }
    void commit_speculative() {}
    void rollback_speculative(int keep) { cache_pos = speculative_start+1+keep; }
};

struct Predictor {
    mtp::GenerationStats last_stats;
    mtp::PolicyState policy_state;
    uint64_t last_cycles = 0, last_accepted = 0, last_rejected = 0;
    int64_t position = 0;
    bool legacy = false;
    int maximum_draft_depth() const { return 3; }
    bool dspark_policy() const { return legacy; }
    bool blockwise_drafting() const { return false; }
    bool supports_session_state() const { return true; }
    bool teacher_forced_prompt_prime() const { return true; }
    bool target_bootstrap_decode() const { return false; }
    bool split_target_verification() const { return false; }
    bool retains_partial_target_prefix() const { return true; }
    int64_t cache_position() const { return position; }
    void trim_cache_to(int64_t value) { position = value; }
    void append_target_context(Tensor, int64_t) { throw std::logic_error("not blockwise"); }
    Tensor draft_block(int, Tensor, int) { throw std::logic_error("not blockwise"); }
    Tensor draft_next(Tensor, Tensor) { throw std::logic_error("not blockwise"); }
};

struct Prepared {
    std::vector<int64_t> token_ids;
    Tensor embeddings, positions;
    int decode_position_delta = 0;
    bool transformed() const { return false; }
};

struct Sampler {
    bool greedy() const { return true; }
    double next_uniform() { return 0.5; }
};

struct Ops {
    using Tensor = ::Tensor;
    Model& model;
    Predictor& mtp;
    const Prepared* prepared = nullptr;
    int target = 0;
    Tensor input_ids, counts;
    bool penalties = false;
    Sampler sampler_;
    void initialize(const std::vector<int64_t>& prompt, const MfqSamplingParams&) {
        input_ids = ids_for(prompt);
    }
    Sampler& sampler() { return sampler_; }
    static bool defined(const Tensor& t) { return !t.shape.empty(); }
    static int rank(const Tensor& t) { return t.shape.size(); }
    static int64_t size(const Tensor& t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size()+axis : axis);
    }
    static Tensor clone(const Tensor& t) { return t; }
    static Tensor slice(Tensor t, int axis, int64_t, int64_t length) {
        t.shape.at(axis < 0 ? t.shape.size()+axis : axis) = length;
        return t;
    }
    static Tensor reshape(Tensor t, std::initializer_list<int64_t> shape) {
        int64_t total = 1, fixed = 1;
        for (int64_t dim : t.shape) total *= dim;
        t.shape = shape;
        for (int64_t dim : t.shape) if (dim > 0) fixed *= dim;
        for (auto& dim : t.shape) if (dim < 0) dim = total/fixed;
        return t;
    }
    static Tensor concatenate(const std::vector<Tensor>& values, int axis) {
        auto result = values.at(0);
        result.shape.at(axis) = 0;
        for (const auto& value : values) result.shape.at(axis) += value.shape.at(axis);
        return result;
    }
    static Tensor ids_for(const std::vector<int64_t>& ids) {
        return {{1, static_cast<int64_t>(ids.size())}};
    }
    static void add_counts(Tensor, Tensor) {}
    static Tensor decode_positions(int64_t, int64_t count) { return {{count}}; }
    Tensor target_forward(Tensor ids, Tensor* raw = nullptr, int64_t confirmed = 0) {
        const auto count = size(ids, 1);
        if (confirmed) model.speculative_start = model.cache_pos;
        model.cache_pos += count;
        Tensor hidden{{1, count, 1}};
        if (raw) *raw = hidden;
        std::this_thread::sleep_for(std::chrono::milliseconds(count > 1 ? 12 : 1));
        return hidden;
    }
    Tensor target_suffix(Tensor ids, Tensor* raw) { return target_forward(ids, raw); }
    struct Prefill { Tensor hidden, raw; double elapsed; };
    Prefill prefill(PrefillChunk chunk) {
        auto hidden = target_forward({{1, chunk.count}});
        return {hidden, hidden, 1.0};
    }
    struct Head { Tensor sample_hidden, chain_hidden; };
    Head predictor_step(Tensor hidden, Tensor ids, Tensor) {
        mtp.position += size(ids, 1);
        hidden.proposal = true;
        return {hidden, hidden};
    }
    static Tensor logits_for(Tensor hidden) {
        hidden.shape.back() = 3;
        return hidden;
    }
    static int32_t sample_normal(Tensor logits, Tensor) { return logits.proposal ? 1 : 0; }
    static int32_t sample_constrained(Tensor logits, Tensor counts, const MfqTokenConstraintPtr&) {
        return sample_normal(logits, counts);
    }
    static std::vector<float> probabilities(Tensor logits, Tensor, const MfqSamplingParams&) {
        return logits.proposal ? std::vector<float>{0, 1, 0} : std::vector<float>{1, 0, 0};
    }
    static mtp::CompactDistribution compact_probabilities(Tensor logits, Tensor counts,
        const MfqSamplingParams& sampling) { return {{0, 1, 2}, probabilities(logits, counts, sampling)}; }
    static std::vector<mtp::CompactDistribution> compact_probability_rows(Tensor logits,
        const MfqSamplingParams& sampling) {
        return std::vector<mtp::CompactDistribution>(size(logits, 0), compact_probabilities(logits, {}, sampling));
    }
};

void run(bool dspark) {
    Model model;
    Predictor mtp;
    mtp.legacy = dspark;
    InferenceRequest request;
    request.chat = false;
    request.prompt = {0, 0, 0};
    request.sampling.max_tokens = 256;
    request.sampling.mtp_max_draft_tokens = 3;
    InferenceOutput output(request, nullptr, "mtp-test");
    Tensor last_hidden;
    auto generation = speculative_sequence(Ops{model, mtp}, request, output, 512, 0, Tensor{}, &last_hidden);
    std::vector<int64_t> tokens;
    while (auto step = generation.next()) {
        if (!step.value) continue;
        if (const auto* delta = std::get_if<TokenOutput>(&*step.value))
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
    }
    if (tokens != std::vector<int64_t>(256, 0) || model.cache_pos != 258 || !Ops::defined(last_hidden))
        throw std::runtime_error("shared MTP output/cache boundary mismatch");
    const auto& stats = output.metrics.mtp;
    if (dspark ? stats.park_count != 0 || stats.standard_tokens != 0 :
        stats.park_count == 0 || stats.standard_tokens == 0)
        throw std::runtime_error("shared MTP policy dispatch mismatch");
}
}

int main() {
    run(false);
    run(true);
    std::cout << "shared speculative sequence output/cache/dispatch tests passed\n";
}
