#pragma once

#include "generation_step.h"

#include <memory>
#include <unordered_map>

namespace mfq::engine {

template <class Cache> SessionResult control_session(Cache &cache, const SessionCommand &command) {
    switch (command.kind) {
    case SessionCommand::Kind::fork:
        return {cache.fork_session(command.source, command.target), {}};
    case SessionCommand::Kind::close:
        return {cache.close_session(command.source), {}};
    case SessionCommand::Kind::clear:
        return {cache.clear(), {}};
    case SessionCommand::Kind::trim:
        return {cache.trim_hot(command.bytes), {}};
    case SessionCommand::Kind::metrics:
        return {0, cache.metrics()};
    }
    throw std::invalid_argument("unknown session command");
}

// Serial and batched execution write into the same request state. Only the
// executor drains events and finishes the request after device cleanup.
struct ExecutionRequest {
    InferenceRequest input;
    InferenceOutput output;
    Generation generation;
    std::vector<EventData> events;
    std::exception_ptr failure;
    bool started = false, batched = false, done = false;

    ExecutionRequest(InferenceRequest prepared, const MfqTokenizer *tokenizer, const RequestId &id)
        : input(std::move(prepared)), output(input, tokenizer, id) {}

    void prefill(MfqPrefillTiming timing) {
        output.metrics.mark_prefill(timing);
        events.emplace_back(PrefillProgress{timing});
    }
    void append(std::int64_t token) { events.emplace_back(output.append({token})); }
    void complete(std::exception_ptr error = {}) {
        done = true;
        failure = error;
    }
};

class RequestExecutor {
  public:
    explicit RequestExecutor(std::size_t capacity = 1) : capacity_(capacity) {}
    bool empty() const { return requests_.empty(); }
    EngineStatus status(bool exclusive = false) const {
        if (!healthy_)
            return {0, false};
        if (exclusive)
            return {0, true};
        for (const auto &[id, request] : requests_)
            if (!request->batched)
                return {0, true};
        return {capacity_ - requests_.size(), true};
    }

    template <class Ops>
    Admission admit(
        EngineRequest request, const TextProcessor *text, const EngineInfo &info, Ops &ops) {
        if (requests_.contains(request.id))
            throw std::invalid_argument("duplicate request ID");
        const bool batched = ops.can_batch(request);
        if (!status(ops.exclusive()).available || (!batched && !empty()))
            return Admission::deferred;
        const bool raw = !request.token_ids.empty();
        InferenceRequest input;
        if (raw) {
            input.prompt = std::move(request.token_ids);
            input.sampling = request.input.sampling;
            input.cache_plan = request.input.cache_plan;
        } else {
            if (!text)
                throw std::invalid_argument("text input requires a tokenizer");
            input = text->prepare(std::move(request.input), info.max_context);
        }
        const auto plan = plan_generation(input.prompt,
            info.vocab_size,
            info.max_context,
            input.sampling.max_tokens,
            input.cache_plan.stable_prefix_tokens);
        input.sampling.max_tokens = plan.generation_tokens;
        input.cache_plan.stable_prefix_tokens = plan.stable_prefix_tokens;
        auto current = std::make_unique<ExecutionRequest>(
            std::move(input), raw ? nullptr : &text->tokenizer(), request.id);
        current->output.metrics.mtp.available = ops.mtp_available();
        current->batched = batched;
        // Insert first so allocation failure cannot leave the backend holding
        // a reference to an unowned request.
        auto [it, inserted] = requests_.emplace(request.id, std::move(current));
        try {
            if (batched)
                ops.admit_batch(it->first, *it->second);
        } catch (...) {
            requests_.erase(it);
            throw;
        }
        return Admission::accepted;
    }

    void cancel(const RequestId &id) {
        if (auto it = requests_.find(id); it != requests_.end())
            it->second->output.result.cancelled = true;
    }

    template <class Ops> EngineStepResult step(const std::vector<RequestId> &eligible, Ops &ops) {
        const bool batched = std::any_of(requests_.begin(), requests_.end(), [](const auto &entry) {
            return entry.second->batched;
        });
        if (batched) {
            try {
                ops.step_batch(eligible);
            } catch (...) {
                healthy_ = false;
                throw;
            }
        }
        EngineStepResult result;
        for (auto it = requests_.begin(); it != requests_.end();) {
            auto &current = *it->second;
            if (!current.batched &&
                (current.output.result.cancelled ||
                    std::find(eligible.begin(), eligible.end(), it->first) != eligible.end())) {
                try {
                    if (!current.started) {
                        current.started = true;
                        current.generation = ops.generate(current.input, current.output);
                    }
                    if (auto event = current.generation.next()) {
                        if (auto *progress = std::get_if<PrefillProgress>(&*event))
                            current.output.metrics.mark_prefill(progress->timing);
                        current.events.push_back(std::move(*event));
                    } else
                        current.complete();
                } catch (...) {
                    current.generation = {};
                    current.complete(std::current_exception());
                    // A cleanup failure is a backend failure even when the
                    // original request was invalid.
                    try {
                        ops.reset();
                    } catch (...) {
                        current.failure = std::current_exception();
                        healthy_ = false;
                    }
                }
            }
            if (!current.events.empty())
                result.advanced.push_back(it->first);
            for (auto &event : current.events)
                result.events.push_back({it->first, std::move(event)});
            current.events.clear();
            if (!current.done) {
                ++it;
                continue;
            }
            current.generation = {};
            finish(it->first, current, result);
            it = requests_.erase(it);
        }
        result.status = status(ops.exclusive());
        result.has_work = !empty();
        return result;
    }

  private:
    void finish(const RequestId &id, ExecutionRequest &request, EngineStepResult &result);
    std::size_t capacity_;
    bool healthy_ = true;
    std::unordered_map<RequestId, std::unique_ptr<ExecutionRequest>> requests_;
};

} // namespace mfq::engine
