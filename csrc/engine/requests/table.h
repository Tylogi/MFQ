#pragma once

#include "step.h"
#include "processor.h"
#include "budget.h"
#include "text_cancel.h"

#include <memory>
#include <atomic>
#include <future>
#include <thread>
#include <unordered_map>

namespace mfq::engine {

// Serial, independent-state and physically batched execution share request
// state. Concurrent admission does not imply physical batching. Only the
// executor drains events and finishes the request after device cleanup.
// This is the RequestTable row: the loop that calls step() is the only writer,
// and complete() is the only place that ends a request, so a request reaches
// exactly one terminal event.
struct ExecutionRequest {
    std::optional<AdmissionBudget::Lease> memory;
    InferenceRequest input;
    InferenceOutput output;
    Generation generation;
    std::optional<InferenceInput> unprepared;
    std::shared_ptr<const TextProcessor> text;
    std::future<InferenceRequest> preparation;
    std::shared_ptr<std::atomic<bool>> stop = std::make_shared<std::atomic<bool>>(false);
    EngineInfo limits;
    std::exception_ptr failure;
    bool started = false, concurrent = false, done = false;

    ExecutionRequest(InferenceRequest prepared, const MfqTokenizer *tokenizer, const RequestId &id)
        : input(std::move(prepared)), output(input, tokenizer, id) {}

    void complete(std::exception_ptr error = {}) {
        stop->store(true, std::memory_order_relaxed);
        done = true;
        failure = error;
    }
};

class RequestExecutor {
  public:
    explicit RequestExecutor(std::size_t capacity = 1) : capacity_(capacity) {}
    ~RequestExecutor() { clear(); }
    bool empty() const { return requests_.empty(); }
    void reset(std::size_t capacity) {
        if (!empty()) throw std::logic_error("executor reset requires no requests");
        capacity_ = capacity;
        healthy_ = true;
    }
    EngineStatus status(bool exclusive = false) const {
        if (!healthy_)
            return {0, false};
        if (exclusive)
            return {0, true};
        for (const auto &[id, request] : requests_)
            if (!request->concurrent)
                return {0, true};
        return {capacity_ - requests_.size(), true};
    }

    template <class Ops>
    Admission admit(
        EngineRequest&& request, std::shared_ptr<const TextProcessor> text, const EngineInfo &info, Ops &ops) {
        if (requests_.contains(request.id))
            throw std::invalid_argument("duplicate request ID");
        // Concurrent admission is opt-in; backends without this capability
        // remain serial without changing their native interfaces.
        bool concurrent = false;
        if constexpr (requires { ops.can_overlap(request); })
            concurrent = ops.can_overlap(request);
        if (!status(ops.exclusive()).available || (!concurrent && !empty()))
            return Admission::deferred;
        std::optional<AdmissionBudget::Lease> memory;
        if constexpr (requires { ops.reserve(request); }) {
            auto reservation = ops.reserve(request);
            if (!reservation) return Admission::deferred;
            memory.emplace(std::move(*reservation));
        }
        const bool raw = request.diagnostic_input.has_value();
        if (raw && (request.diagnostic_input->tokens.empty() || !request.input.prompt.empty() ||
                request.input.chat_input || request.input.media))
            throw InferenceInputError(InferenceInputErrorCode::Invalid,
                "diagnostic tokens require a nonempty prompt and cannot be mixed with text or media");
        const MfqTokenizer *tokenizer = nullptr;
        InferenceRequest input;
        if (raw) {
            input.prompt = std::move(request.diagnostic_input->tokens);
            input.sampling = request.input.sampling;
            input.cache_plan = request.input.cache_plan;
            input.stop_token_ids = info.stop_token_ids;
        } else {
            if (!text)
                throw std::invalid_argument("text input requires a tokenizer");
            // Preparation starts from step(), after ownership and admission are settled.
            input.sampling = request.input.sampling;
        }
        if (raw) prepare_plan(input, info);
        auto current = std::make_unique<ExecutionRequest>(
            std::move(input), tokenizer, request.id);
        if (memory) current->memory.emplace(std::move(*memory));
        current->output.metrics.mtp.available = ops.mtp_available();
        current->concurrent = concurrent;
        if (!raw) {
            current->unprepared = std::move(request.input);
            current->text = std::move(text);
            current->limits = info;
        }
        requests_.emplace(request.id, std::move(current));
        return Admission::accepted;
    }

    void cancel(const RequestId &id) {
        if (auto it = requests_.find(id); it != requests_.end()) {
            it->second->stop->store(true, std::memory_order_relaxed);
            it->second->output.result.cancelled = true;
        }
    }

    std::exception_ptr clear() {
        std::exception_ptr failure;
        for (auto& [id, request] : requests_) {
            request->stop->store(true, std::memory_order_relaxed);
            request->generation = {};
            if (request->output.cleanup_failure) failure = request->output.cleanup_failure;
        }
        requests_.clear();
        return failure;
    }

    template <class Ops> EngineStepResult step(const std::vector<RequestId> &eligible, Ops &ops) {
        try {
            ops.execute(eligible);
        } catch (const std::bad_alloc&) {
            for (auto& [id, request] : requests_) request->complete(std::current_exception());
        } catch (const ResourceExhausted&) {
            for (auto& [id, request] : requests_) request->complete(std::current_exception());
        } catch (...) {
            healthy_ = false;
            for (auto& [id, request] : requests_)
                request->complete(std::current_exception());
        }
        EngineStepResult result;
        for (auto it = requests_.begin(); it != requests_.end();) {
            auto &current = *it->second;
            if (!current.done && (current.output.result.cancelled ||
                std::find(eligible.begin(), eligible.end(), it->first) != eligible.end())) {
                try {
                    if (current.unprepared || current.preparation.valid()) {
                        if (current.output.result.cancelled) {
                            current.unprepared.reset();
                            current.complete();
                        } else if (!advance_preparation(current)) {
                            result.wake_at = Clock::now() + std::chrono::milliseconds(1);
                            ++it;
                            continue;
                        }
                    }
                    if (!current.done && !current.started) {
                        current.started = true;
                        current.generation = ops.generate(it->first, current);
                    }
                    auto step = current.done ? mfq::StepResult<EventData>{} : current.generation.next();
                    if (step.state == StepState::advanced)
                        result.advanced.push_back(it->first);
                    else if (step.state == StepState::waiting)
                        result.wake_at = Clock::now() + std::chrono::milliseconds(1);
                    if (step.value) {
                        if (auto* progress = std::get_if<PrefillProgress>(&*step.value))
                            current.output.metrics.mark_prefill(progress->timing);
                        result.events.push_back({it->first, std::move(*step.value)});
                    }
                    if (!step) current.complete();
                } catch (...) {
                    current.complete(std::current_exception());
                }
            }
            if (!current.done) {
                ++it;
                continue;
            }
            current.generation = {};
            if (current.output.cleanup_failure) {
                healthy_ = false;
                current.complete(current.output.cleanup_failure);
            }
            finish(it->first, current, result);
            it = requests_.erase(it);
        }
        result.status = status(ops.exclusive());
        result.has_work = !empty();
        return result;
    }

  private:
    void finish(const RequestId &id, ExecutionRequest &request, EngineStepResult &result);
    static void prepare_plan(InferenceRequest& input, const EngineInfo& info) {
        const auto plan = plan_generation(input.prompt, info.vocab_size, info.max_context,
            input.sampling.max_tokens, input.cache_plan.stable_prefix_tokens);
        input.sampling.max_tokens = plan.generation_tokens;
        input.cache_plan.stable_prefix_tokens = plan.stable_prefix_tokens;
    }
    bool advance_preparation(ExecutionRequest& request) {
        if (request.unprepared) {
            if (preparing_->load()) return false;
            std::promise<InferenceRequest> result;
            request.preparation = result.get_future();
            preparing_->store(true);
            try {
                // Only one CPU preparation runs per executor, including across reload.
                // It owns the old tokenizer and input; cancellation never joins it on the scheduler thread.
                std::thread([busy = preparing_, stop = request.stop, text = request.text, context = request.limits.max_context,
                             input = std::move(*request.unprepared), result = std::move(result)]() mutable {
                    mfq::text::CancellationScope cancellation(*stop);
                    try {
                        mfq::text::check_cancelled();
                        result.set_value(text->prepare(std::move(input), context));
                    }
                    catch (...) { result.set_exception(std::current_exception()); }
                    busy->store(false);
                }).detach();
            } catch (...) {
                preparing_->store(false);
                throw;
            }
            request.unprepared.reset();
        }
        if (request.preparation.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        request.input = request.preparation.get();
        prepare_plan(request.input, request.limits);
        request.output.prepare(&request.text->tokenizer());
        return true;
    }
    std::shared_ptr<std::atomic<bool>> preparing_ = std::make_shared<std::atomic<bool>>(false);
    std::size_t capacity_;
    bool healthy_ = true;
    std::unordered_map<RequestId, std::unique_ptr<ExecutionRequest>> requests_;
};

} // namespace mfq::engine
