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

template <class Sample> struct PrefillBatchResult {
    Sample sample;
    std::exception_ptr failure;
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
    int64_t prefill_chunk_size, prefill_token_budget;
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
        : prefill_chunk_size(std::min(chunk_size, token_budget)), prefill_token_budget(token_budget),
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
        const bool rows_ready = std::all_of(state.active.begin(), state.active.end(),
            [&](const auto& request) {
                return std::find(eligible.begin(), eligible.end(), request->id) == eligible.end() ||
                    requests.at(request->id)->decode;
            });
        // Let eligible peers consume their previous sample and schedule the
        // next decode before executing it. Otherwise staggered coroutines
        // alternate single-row decodes and never form a stable batch.
        const bool has_decode = rows_ready && std::any_of(state.active.begin(), state.active.end(),
            [](const auto& request) { return request->eligible; });
        bool prefilling_operation = false;
        bool decoding_operation = false;
        std::vector<std::shared_ptr<Work>> prefill_work;
        try {
            if (has_decode && (prefill == requests.end() || decode_next)) {
                decoding_operation = true;
                auto decoded = operations.decode(state);
                decoding_operation = false;
                for (const auto& request : state.active) if (request->eligible) {
                    auto& work = *requests.at(request->id);
                    work.result = operations.sample(request, decoded);
                    ++request->cache_length;
                    work.decode = false;
                }
            } else if (prefill != requests.end()) {
                prefill_work.push_back(prefill->second);
                using Item = std::pair<std::shared_ptr<Request>, PrefillChunk>;
                std::vector<Item> items{{prefill->second->request, *prefill->second->prefill}};
                if constexpr (requires { operations.prefill_batch_limit(); operations.prefill_batch(items); }) {
                    auto budget=prefill_token_budget;
                    if constexpr(requires {operations.prefill_batch_token_budget();})
                        budget=std::min(budget,operations.prefill_batch_token_budget());
                    auto remaining = budget - items.front().second.count;
                    const auto limit = operations.prefill_batch_limit();
                    for (const auto& id : eligible) {
                        auto it = requests.find(id);
                        if (it == requests.end() || !it->second->prefill ||
                            std::find(prefill_work.begin(),prefill_work.end(),it->second) != prefill_work.end()) continue;
                        const auto chunk = *it->second->prefill;
                        if (items.size() >= limit || chunk.count > remaining) continue;
                        remaining -= chunk.count;
                        prefill_work.push_back(it->second);
                        items.emplace_back(it->second->request,chunk);
                    }
                }
                operations.suspend_decode();
                PrefillActivity prefill_activity;
                prefilling_operation = true;
                std::vector<PrefillBatchResult<Sample>> samples;
                if constexpr (requires { operations.prefill_batch(items); }) {
                    if (items.size() > 1) samples = operations.prefill_batch(items);
                }
                if (samples.empty()) {
                    const auto& [request,chunk]=items.front();
                    samples.push_back({operations.prefill(request, chunk),{}});
                }
                prefilling_operation = false;
                if (samples.size() != items.size()) throw std::runtime_error("invalid packed prefill result count");
                for (std::size_t row = 0; row < items.size(); ++row) {
                    auto& work = *prefill_work[row];
                    auto& request = work.request;
                    auto& result = samples[row];
                    const auto chunk = items[row].second;
                    if (result.failure) {
                        work.failure = result.failure;
                        operations.discard_prefill(request);
                    } else {
                        request->prefill_offset = chunk.offset + chunk.count;
                        ++prefill_chunks;
                        if (request->prefill_offset == static_cast<int64_t>(request->prompt.size())) {
                            request->cache_length = request->prompt.size();
                            operations.activate(request,result.sample);
                            state.active.push_back(request);
                            work.active = true;
                            ++admissions;
                        } else ++prefill_yields;
                        work.result = std::move(result.sample);
                    }
                    work.prefill.reset();
                }
                operations.resume_decode(cache_position());
            }
            decode_next = !decode_next;
        } catch (...) {
            if constexpr (requires { Ops::isolate_context_growth_failures; }) {
                bool capacity_failure = false;
                try { throw; }
                catch (const InferenceInputError& error) {
                    capacity_failure = error.code == InferenceInputErrorCode::ContextLength;
                } catch (...) {}
                if (Ops::isolate_context_growth_failures && decoding_operation && capacity_failure) {
                    // Capacity checks run before numerical decode. Release the
                    // newest eligible request while preserving established rows.
                    const auto victim = std::find_if(state.active.rbegin(),state.active.rend(),
                        [](const auto& request) { return request->eligible; });
                    if (victim != state.active.rend()) {
                        auto request = *victim;
                        auto& work = *requests.at(request->id);
                        work.failure = std::current_exception(); work.decode = false;
                        std::erase(state.active,request);
                        try {
                            operations.retire({request},cache_position());
                            work.active = false;
                            return;
                        } catch (...) { recover(std::current_exception()); throw; }
                    }
                }
            }
            if constexpr (requires { Ops::isolate_prefill_failures; }) {
                if (Ops::isolate_prefill_failures && prefilling_operation && prefill != requests.end()) {
                    const auto error = std::current_exception();
                    try {
                        for (const auto& work : prefill_work) {
                            work->failure = error;
                            work->prefill.reset();
                            work->decode = false;
                            operations.discard_prefill(work->request);
                        }
                        operations.resume_decode(cache_position());
                        return;
                    } catch (...) { recover(std::current_exception()); throw; }
                }
            }
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
