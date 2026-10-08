#pragma once
#include "generation_step.h"
#include "prefill_activity.h"
#include <memory>
#include <span>
#include <unordered_map>

namespace mfq::engine {

// Logical row metadata. Native request storage is associated with the execution
// implementation; no slot, tensor, KV or recurrent layout is prescribed here.
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
    int64_t prefill_offset = 0, cache_length = 0;
};

struct BatchSample {
    int64_t token = 0;
    MfqPrefillTiming timing;
};

template <class Request> struct PrefillWork {
    const std::shared_ptr<Request>& request;
    PrefillChunk chunk;
};
template <class Request> struct DecodeWork {
    std::span<const std::shared_ptr<Request>> requests;
};

// One seam: execute a selected numerical quantum and return its results.
// execute(PrefillWork) completes that chunk, including native state switching.
// execute(DecodeWork) returns ordered samples, or nullopt for preparation with
// no committed rows. accept(), release() and recover() own native state only;
// output, cancellation and publication remain in the shared request lifecycle.
template <class Execution> class ContinuousBatch {
    using Request = typename Execution::Request;
    struct Work {
        std::shared_ptr<Request> request;
        std::optional<PrefillChunk> prefill;
        std::optional<BatchSample> result;
        std::exception_ptr failure;
        bool decode = false;
    };
    int64_t prefill_chunk_size;
    Execution execution;
    std::vector<std::shared_ptr<Request>> active;
    std::unordered_map<std::string, std::shared_ptr<Work>> requests;
    bool decode_next = true;
    int64_t admissions = 0, prefill_chunks = 0, prefill_yields = 0;

    void recover(std::exception_ptr error) {
        std::vector<std::shared_ptr<Request>> failed;
        for (const auto& [id, work] : requests) {
            work->failure = error;
            work->prefill.reset();
            work->decode = false;
            failed.push_back(work->request);
        }
        active.clear();
        execution.recover(failed);
    }
    void release(Work& work) {
        if (!requests.contains(work.request->id)) return;
        try {
            std::erase(active, work.request);
            if (!work.failure) execution.release(work.request);
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
        const BatchSample& sample() const {
            if (work.failure) std::rethrow_exception(work.failure);
            return work.result.value();
        }
        double prefill(PrefillChunk) { return sample().timing.llm_ms; }
        int64_t first_token() { return sample().token; }
        int64_t advance() { return sample().token; }
        void accept(int64_t token) {
            auto& request = *work.request;
            batch.execution.accept(work.request, sample());
            request.pending_token = token;
            ++request.produced;
        }
    };

  public:
    template <class... Args>
    explicit ContinuousBatch(int64_t chunk_size, int64_t token_budget, Args&&... args)
        : prefill_chunk_size(std::min(chunk_size, token_budget)),
          execution(std::forward<Args>(args)...) {
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
        std::vector<std::shared_ptr<Request>> decoding;
        for (const auto& request : active) {
            const auto& work = *requests.at(request->id);
            if (work.decode && std::find(eligible.begin(), eligible.end(), request->id) != eligible.end())
                decoding.push_back(request);
        }
        auto prefill = requests.end();
        for (const auto& id : eligible) {
            auto it = requests.find(id);
            if (it != requests.end() && it->second->prefill) { prefill = it; break; }
        }
        try {
            if (!decoding.empty() && (prefill == requests.end() || decode_next)) {
                auto decoded = execution.execute(DecodeWork<Request>{decoding});
                if (!decoded) return; // Preparation committed no row.
                if (decoded->size() != decoding.size())
                    throw std::logic_error("batch decode result count disagrees with selected rows");
                for (size_t row = 0; row < decoding.size(); ++row) {
                    auto& request = decoding[row];
                    auto& work = *requests.at(request->id);
                    work.result = std::move((*decoded)[row]);
                    ++request->cache_length;
                    work.decode = false;
                }
            } else if (prefill != requests.end()) {
                auto& work = *prefill->second;
                auto& request = work.request;
                const auto chunk = *work.prefill;
                PrefillActivity prefill_activity;
                auto sample = execution.execute(PrefillWork<Request>{request, chunk});
                request->prefill_offset = chunk.offset + chunk.count;
                ++prefill_chunks;
                if (request->prefill_offset == static_cast<int64_t>(request->prompt.size())) {
                    request->cache_length = request->prompt.size();
                    active.push_back(request);
                    ++admissions;
                } else ++prefill_yields;
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
        auto result = execution.metrics();
        result.emplace_back("continuous_batching_active", active.size());
        result.emplace_back("continuous_batching_prefilling", requests.size() - active.size());
        result.emplace_back("continuous_batching_requests", admissions);
        result.emplace_back("continuous_batching_admissions", admissions);
        result.emplace_back("continuous_batching_prefill_chunks", prefill_chunks);
        result.emplace_back("continuous_batching_prefill_yields", prefill_yields);
        return result;
    }
};
} // namespace mfq::engine
