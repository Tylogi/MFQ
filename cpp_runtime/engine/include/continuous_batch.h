#pragma once
#include "generation_policy.h"
#include "request_executor.h"
#include <algorithm>
#include <deque>
#include <memory>
#include <unordered_map>

namespace mfq::engine {

struct BatchRequest {
    BatchRequest(std::string id, mfq::engine::ExecutionRequest &execution)
        : id(std::move(id)), prompt(execution.input.prompt), sampling(execution.input.sampling),
          token_constraint(execution.input.token_constraint), output(execution.output),
          execution(execution) {}
    std::string id;
    std::vector<int64_t> prompt;
    MfqSamplingParams sampling;
    MfqTokenConstraintPtr token_constraint;
    mfq::engine::InferenceOutput &output;
    mfq::engine::ExecutionRequest &execution;
    int32_t generation_limit = 0, produced = 0;
    int64_t pending_token = 0;
    bool eligible = false;
    bool stopped() const { return output.stopped(); }
    void publish_prefill(MfqPrefillTiming timing) { execution.prefill(timing); }
    void publish_token(int64_t token) { execution.append(token); }
    void complete(std::exception_ptr failure = {}) { execution.complete(failure); }
    int64_t prefill_offset = 0;
    int64_t cache_length = 0;
};

template <class Request> struct BatchState {
    std::deque<std::shared_ptr<Request>> prefilling;
    std::vector<std::shared_ptr<Request>> active;
};

// The engine chooses work and accepts tokens; Ops owns device state and kernels.
template <class Ops> class ContinuousBatch {
    using Request = typename Ops::Request;
    using State = BatchState<Request>;
    int64_t prefill_chunk_size;
    Ops operations;
    State state;
    std::unordered_map<std::string, std::shared_ptr<Request>> requests;
    bool decode_next = true;
    int64_t admissions = 0, prefill_chunks = 0, prefill_yields = 0;

    static int64_t cache_position(const std::vector<std::shared_ptr<Request>> &active) {
        int64_t position = 0;
        for (const auto &request : active)
            position = std::max(position, request->cache_length);
        return position;
    }

    void retire_cancelled() {
        std::vector<std::shared_ptr<Request>> survivors, retired;
        for (const auto &request : state.active)
            (request->stopped() ? retired : survivors).push_back(request);
        if (retired.empty())
            return;
        operations.retire(retired, cache_position(survivors));
        state.active = std::move(survivors);
        for (const auto &request : retired)
            request->complete();
    }

    void prefill() {
        auto remaining = state.prefilling.size();
        while (remaining--) {
            auto request = state.prefilling.front();
            state.prefilling.pop_front();
            if (!request->eligible && !request->stopped()) {
                state.prefilling.push_back(request);
                continue;
            }
            operations.suspend_decode();
            if (!request->stopped()) {
                const auto chunk = next_prefill_chunk(request->prompt.size(),
                                                      request->prefill_offset, prefill_chunk_size);
                auto sample = operations.prefill(request, chunk);
                request->prefill_offset = chunk.offset + chunk.count;
                ++prefill_chunks;
                if (!request->stopped() &&
                    request->prefill_offset < static_cast<int64_t>(request->prompt.size())) {
                    state.prefilling.push_back(request);
                    ++prefill_yields;
                    request->publish_prefill(
                        {static_cast<size_t>(request->prefill_offset), 0.0, 0.0, 0.0});
                    operations.resume_decode(cache_position(state.active));
                    return;
                }
                if (!request->stopped()) {
                    request->publish_prefill(sample.timing);
                    request->cache_length = request->prompt.size();
                    request->pending_token = sample.token;
                    request->produced = 1;
                    request->publish_token(sample.token);
                    if (!request->stopped() && request->produced < request->generation_limit) {
                        operations.activate(request, sample);
                        state.active.push_back(request);
                        ++admissions;
                        operations.resume_decode(cache_position(state.active));
                        return;
                    }
                }
            }
            operations.discard_prefill(request);
            request->complete();
            operations.resume_decode(cache_position(state.active));
            return;
        }
    }

    void decode() {
        auto decoded = operations.decode(state);
        std::vector<std::shared_ptr<Request>> survivors, retired;
        std::vector<std::exception_ptr> failures;
        for (const auto &request : state.active) {
            if (!request->eligible) {
                survivors.push_back(request);
                continue;
            }
            ++request->cache_length;
            std::exception_ptr error;
            try {
                if (!request->stopped()) {
                    auto sample = operations.sample(request, decoded);
                    request->pending_token = sample.token;
                    ++request->produced;
                    request->publish_token(sample.token);
                    if (!request->stopped() && request->produced < request->generation_limit) {
                        operations.accept(request, sample);
                        survivors.push_back(request);
                        continue;
                    }
                }
            } catch (...) {
                error = std::current_exception();
            }
            retired.push_back(request);
            failures.push_back(error);
        }
        operations.retire(retired, cache_position(survivors));
        state.active = std::move(survivors);
        for (size_t i = 0; i < retired.size(); ++i)
            retired[i]->complete(failures[i]);
    }

  public:
    template <class... Args>
    explicit ContinuousBatch(int64_t chunk_size, int64_t token_budget, Args &&...args)
        : prefill_chunk_size(std::min(chunk_size, token_budget)),
          operations(std::forward<Args>(args)...) {
        if (prefill_chunk_size <= 0)
            throw std::invalid_argument("continuous batching requires a positive prefill chunk");
    }

    void admit(std::string id, ExecutionRequest &execution) {
        const auto &input = execution.input;
        const auto plan = plan_generation(input.prompt, operations.vocab_size(),
                                          operations.max_context(), input.sampling.max_tokens);
        auto request = std::make_shared<Request>(std::move(id), execution);
        request->generation_limit = plan.generation_tokens;
        auto [it, inserted] = requests.emplace(request->id, request);
        if (!inserted)
            throw std::invalid_argument("duplicate batch request id");
        try {
            state.prefilling.push_back(std::move(request));
        } catch (...) {
            requests.erase(it);
            throw;
        }
    }

    void step(const std::vector<std::string> &eligible) {
        for (auto &[id, request] : requests)
            request->eligible = std::find(eligible.begin(), eligible.end(), id) != eligible.end();
        try {
            retire_cancelled();
            const bool has_decode =
                std::any_of(state.active.begin(), state.active.end(),
                            [](const auto &request) { return request->eligible; });
            const bool has_prefill = std::any_of(
                state.prefilling.begin(), state.prefilling.end(),
                [](const auto &request) { return request->eligible || request->stopped(); });
            if (has_decode && (!has_prefill || decode_next))
                decode();
            else if (has_prefill)
                prefill();
            decode_next = !decode_next;
        } catch (...) {
            const auto error = std::current_exception();
            // Include a request removed from the prefill queue before a device failure.
            State failed;
            for (const auto &[id, request] : requests) {
                request->complete(error);
                failed.prefilling.push_back(request);
            }
            operations.recover(failed);
            state = {};
        }
        std::erase_if(requests, [](const auto &entry) { return entry.second->execution.done; });
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        auto result = operations.metrics();
        result.emplace_back("continuous_batching_active", state.active.size());
        result.emplace_back("continuous_batching_prefilling", state.prefilling.size());
        result.emplace_back("continuous_batching_requests", admissions);
        result.emplace_back("continuous_batching_admissions", admissions);
        result.emplace_back("continuous_batching_prefill_chunks", prefill_chunks);
        result.emplace_back("continuous_batching_prefill_yields", prefill_yields);
        return result;
    }
};
} // namespace mfq::engine
