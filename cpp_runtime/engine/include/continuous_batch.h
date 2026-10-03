#pragma once
#include "generation_step.h"
#include <deque>
#include <memory>
#include <unordered_map>

namespace mfq::engine {

// Physical row metadata only. Request lifecycle and output belong to the executor.
struct BatchRequest {
    BatchRequest(std::string id, const InferenceRequest& input)
        : id(std::move(id)), prompt(input.prompt), sampling(input.sampling),
          token_constraint(input.token_constraint), generation_limit(input.sampling.max_tokens) {}
    std::string id;
    std::vector<int64_t> prompt;
    MfqSamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    int32_t generation_limit = 0, produced = 0;
    int64_t pending_token = 0;
    bool eligible = false;
    int64_t prefill_offset = 0, cache_length = 0;
};

template <class Request> struct BatchState {
    std::deque<std::shared_ptr<Request>> prefilling;
    std::vector<std::shared_ptr<Request>> active;
};

// Groups pending numerical operations. All rows run generate_sequence(), also
// used by B=1 execution; this component never accepts tokens or publishes events.
template <class Ops> class ContinuousBatch {
    using Request = typename Ops::Request;
    using Sample = typename Ops::Sample;
    using State = BatchState<Request>;
    struct Work {
        std::shared_ptr<Request> request;
        std::optional<PrefillChunk> prefill;
        std::optional<Sample> result;
        std::exception_ptr failure;
        bool decode = false, active = false;
    };
    int64_t prefill_chunk_size;
    Ops operations;
    State state;
    std::unordered_map<std::string, std::shared_ptr<Work>> requests;
    bool decode_next = true;
    int64_t admissions = 0, prefill_chunks = 0, prefill_yields = 0;

    int64_t cache_position() const {
        int64_t position = 0;
        for (const auto& request : state.active)
            position = std::max(position, request->cache_length);
        return position;
    }
    void recover(std::exception_ptr error) {
        State failed;
        for (const auto& [id, work] : requests) {
            work->failure = error;
            work->prefill.reset();
            work->decode = work->active = false;
            failed.prefilling.push_back(work->request);
        }
        state = {};
        operations.recover(failed);
    }
    void release(Work& work) {
        if (!requests.contains(work.request->id)) return;
        try {
            if (work.active) {
                std::erase(state.active, work.request);
                operations.retire({work.request}, cache_position());
            } else if (!work.failure) {
                operations.suspend_decode();
                operations.discard_prefill(work.request);
                operations.resume_decode(cache_position());
            }
        } catch (...) {
            auto error = std::current_exception();
            try { recover(error); } catch (...) { error = std::current_exception(); }
            requests.erase(work.request->id);
            std::rethrow_exception(error);
        }
        requests.erase(work.request->id);
    }
    struct SequenceOps {
        ContinuousBatch& batch;
        Work& work;
        void schedule_prefill(PrefillChunk chunk) { work.result.reset(); work.prefill = chunk; }
        void schedule_decode() { work.result.reset(); work.decode = true; }
        bool ready() const { return work.result.has_value() || work.failure; }
        const Sample& sample() const {
            if (work.failure) std::rethrow_exception(work.failure);
            return work.result.value();
        }
        double prefill(PrefillChunk) { return sample().timing.llm_ms; }
        int64_t first_token() { return sample().token; }
        int64_t advance() { return sample().token; }
        void accept(int64_t token) {
            auto& request = *work.request;
            // activate() initializes counts for the first sample.
            if (request.produced) batch.operations.accept(work.request, sample());
            request.pending_token = token;
            ++request.produced;
        }
    };

  public:
    template <class... Args>
    explicit ContinuousBatch(int64_t chunk_size, int64_t token_budget, Args&&... args)
        : prefill_chunk_size(std::min(chunk_size, token_budget)),
          operations(std::forward<Args>(args)...) {
        if (prefill_chunk_size <= 0)
            throw std::invalid_argument("continuous batching requires a positive prefill chunk");
    }

    Generation generate(std::string id, InferenceRequest& input, InferenceOutput& output) {
        if (output.stopped()) co_return;
        auto work = std::make_shared<Work>();
        work->request = std::make_shared<Request>(std::move(id), input);
        if (!requests.emplace(work->request->id, work).second)
            throw std::invalid_argument("duplicate batch request id");
        ExecutionCleanup cleanup{output.cleanup_failure, [&] { release(*work); }};
        SequenceOps ops{*this, *work};
        auto sequence = generate_sequence(ops, output, input.prompt.size(), 0, 0, prefill_chunk_size);
        while (auto step = sequence.next()) co_yield std::move(step);
        cleanup.finish();
    }

    void step(const std::vector<std::string>& eligible) {
        for (const auto& request : state.active) {
            const auto& work = *requests.at(request->id);
            request->eligible = work.decode &&
                std::find(eligible.begin(), eligible.end(), request->id) != eligible.end();
        }
        auto prefill = requests.end();
        for (const auto& id : eligible) {
            auto it = requests.find(id);
            if (it != requests.end() && it->second->prefill) { prefill = it; break; }
        }
        const bool has_decode = std::any_of(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->eligible; });
        try {
            if (has_decode && (prefill == requests.end() || decode_next)) {
                auto decoded = operations.decode(state);
                for (const auto& request : state.active) if (request->eligible) {
                    auto& work = *requests.at(request->id);
                    work.result = operations.sample(request, decoded);
                    ++request->cache_length;
                    work.decode = false;
                }
            } else if (prefill != requests.end()) {
                auto& work = *prefill->second;
                auto& request = work.request;
                const auto chunk = *work.prefill;
                operations.suspend_decode();
                auto sample = operations.prefill(request, chunk);
                request->prefill_offset = chunk.offset + chunk.count;
                ++prefill_chunks;
                if (request->prefill_offset == static_cast<int64_t>(request->prompt.size())) {
                    request->cache_length = request->prompt.size();
                    operations.activate(request, sample);
                    state.active.push_back(request);
                    work.active = true;
                    ++admissions;
                } else ++prefill_yields;
                operations.resume_decode(cache_position());
                work.result = std::move(sample);
                work.prefill.reset();
            }
            decode_next = !decode_next;
        } catch (...) {
            recover(std::current_exception());
            throw;
        }
    }

    Metrics metrics() const {
        auto result = operations.metrics();
        result.emplace_back("continuous_batching_active", state.active.size());
        result.emplace_back("continuous_batching_prefilling", requests.size() - state.active.size());
        result.emplace_back("continuous_batching_requests", admissions);
        result.emplace_back("continuous_batching_admissions", admissions);
        result.emplace_back("continuous_batching_prefill_chunks", prefill_chunks);
        result.emplace_back("continuous_batching_prefill_yields", prefill_yields);
        return result;
    }
};
} // namespace mfq::engine
