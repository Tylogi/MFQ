#pragma once

#include "generation_step.h"
#include "admission_budget.h"
#include "text_cancel.h"

#include <memory>
#include <atomic>
#include <future>
#include <thread>
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
    bool started = false, batched = false, done = false;

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
            if (!request->batched)
                return {0, true};
        return {capacity_ - requests_.size(), true};
    }

    template <class Ops>
    Admission admit(
        EngineRequest&& request, std::shared_ptr<const TextProcessor> text, const EngineInfo &info, Ops &ops) {
        if (requests_.contains(request.id))
            throw std::invalid_argument("duplicate request ID");
        const bool batched = ops.can_batch(request);
        if (!status(ops.exclusive()).available || (!batched && !empty()))
            return Admission::deferred;
        std::optional<AdmissionBudget::Lease> memory;
        if constexpr (requires { ops.reserve(request); }) {
            auto reservation = ops.reserve(request);
            if (!reservation) return Admission::deferred;
            memory.emplace(std::move(*reservation));
        }
        const bool raw = !request.token_ids.empty();
        const MfqTokenizer *tokenizer = nullptr;
        InferenceRequest input;
        if (raw) {
            input.prompt = std::move(request.token_ids);
            input.sampling = request.input.sampling;
            input.cache_plan = request.input.cache_plan;
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
        current->batched = batched;
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
                current.failure = current.output.cleanup_failure;
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

// Backend owns native resources and guards; this composition owns the common
// Engine lifecycle. Destroy request coroutines before their text/model resources.
template <class Backend> class EngineInstance final : public Engine {
  public:
    Backend backend;

    explicit EngineInstance(typename Backend::Options options) : backend{std::move(options)} {
        load();
    }
    ~EngineInstance() override { if (loaded_) { try { shutdown(); } catch (...) {} } }
    EngineInfo info() const override { return info_; }
    EngineStatus status() const override {
        auto result = loaded_ ? requests_.status(backend.exclusive()) : EngineStatus{0, false};
        if constexpr (requires { backend.available(); })
            result.available = std::min(result.available, backend.available());
        return result;
    }
    Admission admit(EngineRequest&& request) override {
        return backend.visit([&](auto& ops) {
            return requests_.admit(std::move(request), text_, info_, ops);
        });
    }
    void cancel(const RequestId& id) override { requests_.cancel(id); }
    EngineStepResult step(const std::vector<RequestId>& eligible) override {
        auto result = backend.visit([&](auto& ops) {
            auto result = requests_.step(eligible, ops);
            if constexpr (requires { ops.advance_control(result); }) ops.advance_control(result);
            advance_text_control(result);
            result.has_work |= result.has_control_work;
            return result;
        });
        if (result.control) control_pending_ = false;
        result.status = status();
        return result;
    }
    SessionResult session(const SessionCommand& command) override {
        return backend.visit([&](auto& ops) { return control_session(ops.cache, command); });
    }
    ControlResult control(ControlRequest request) override {
        if (!loaded_) throw std::runtime_error("engine is unloaded");
        auto result = std::visit([&](auto&& value) -> ControlResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (!std::is_same_v<T, RuntimeMetrics> && !std::is_same_v<T, StopDuplex>)
                if (control_pending_) throw std::runtime_error("engine control is already pending");
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams> || std::is_same_v<T, MfqDuplexStepInput>)
                if (!requests_.empty()) throw std::runtime_error("duplex conflicts with active generation");
            if constexpr (std::is_same_v<T, StopDuplex>) {
                if (text_control_.valid()) {
                    text_control_stop_->store(true);
                    if (!info_.duplex) return std::monostate{};
                }
            }
            if constexpr (std::is_same_v<T, DecodeTokens> ||
                          std::is_same_v<T, PrepareDuplex> || std::is_same_v<T, PrepareDuplexStep>) {
                if (!text_) throw std::invalid_argument("text control requires a tokenizer");
                if (text_control_.valid() || text_control_busy_->exchange(true))
                    throw std::runtime_error("text control is already pending");
                text_control_stop_ = std::make_shared<std::atomic<bool>>(false);
                std::promise<ControlCompletion> completion;
                text_control_ = completion.get_future();
                try {
                    std::thread([text = text_, busy = text_control_busy_, stop = text_control_stop_,
                                 value = std::move(value), completion = std::move(completion)]() mutable {
                        mfq::text::CancellationScope cancellation(*stop);
                        std::optional<ControlCompletion> result;
                        std::exception_ptr failure;
                        try {
                            mfq::text::check_cancelled();
                            if constexpr (std::is_same_v<T, DecodeTokens>)
                                result = text->decode_tokens(value.tokens, value.excluded);
                            else if constexpr (std::is_same_v<T, PrepareDuplex>) {
                                text->prepare_duplex_session(value.prompt, value.parameters);
                                result = std::move(value.parameters);
                            } else {
                                text->prepare_duplex_step(value.text, value.input);
                                result = std::move(value.input);
                            }
                        } catch (...) { failure = std::current_exception(); }
                        busy->store(false);
                        if (failure) completion.set_exception(failure);
                        else completion.set_value(std::move(*result));
                    }).detach();
                } catch (...) { text_control_ = {}; text_control_busy_->store(false); throw; }
                return ControlPending{};
            } else
                return backend.visit([&](auto& ops) { return ops.control(std::move(value)); });
        }, std::move(request));
        if (std::holds_alternative<ControlPending>(result)) control_pending_ = true;
        return result;
    }
    std::int64_t reload(std::int64_t context) override {
        if (context < 1) throw std::invalid_argument("reload context must be positive");
        if (!requests_.empty() || control_pending_) throw std::runtime_error("reload requires a quiescent engine");
        shutdown();
        backend.options.context_size = context;
        load();
        return info_.max_context;
    }
    void shutdown() override {
        loaded_ = false;
        if (text_control_stop_) text_control_stop_->store(true);
        text_control_ = {};
        control_pending_ = false;
        auto failure = requests_.clear();
        text_.reset();
        backend.unload();
        if (failure) std::rethrow_exception(failure);
    }

  private:
    void advance_text_control(EngineStepResult& result) {
        if (!text_control_.valid()) return;
        if (text_control_stop_->load()) {
            text_control_ = {};
            result.control = Cancelled{};
        } else if (text_control_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            try { result.control = text_control_.get(); }
            catch (const std::exception& error) { result.control = Failed{"invalid_request", error.what(), false}; }
        } else {
            result.has_control_work = true;
            result.wake_at = Clock::now() + std::chrono::milliseconds(1);
        }
    }
    bool control_pending_ = false;
    std::future<ControlCompletion> text_control_;
    std::shared_ptr<std::atomic<bool>> text_control_stop_;
    std::shared_ptr<std::atomic<bool>> text_control_busy_ = std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<TextProcessor> text_;
    EngineInfo info_;
    RequestExecutor requests_;
    bool loaded_ = false;

    void load() {
        try {
            auto [info, text] = backend.load();
            info_ = std::move(info);
            text_ = std::move(text);
            if (text_) info_.chat = text_->chat_template_capabilities();
            requests_.reset(info_.max_requests);
            loaded_ = true;
        } catch (...) {
            shutdown();
            throw;
        }
    }
};

} // namespace mfq::engine
