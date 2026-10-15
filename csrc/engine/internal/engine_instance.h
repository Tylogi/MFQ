#pragma once

#include "request_executor.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <variant>

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
    case SessionCommand::Kind::budget:
        if constexpr (requires { cache.set_hot_limit(command.bytes); cache.set_disk_limit(command.bytes); }) {
            const auto released = cache.set_hot_limit(command.bytes);
            if (command.disk_bytes) cache.set_disk_limit(*command.disk_bytes);
            return {released, {}};
        }
        else throw std::invalid_argument("prefix cache budgets are unavailable");
    case SessionCommand::Kind::refresh:
        if constexpr (requires { cache.refresh_disk_index(); }) return {cache.refresh_disk_index(), {}};
        else throw std::invalid_argument("prefix cache refresh is unavailable");
    case SessionCommand::Kind::memory_budget:
        throw std::invalid_argument("online resident memory budgets are unavailable for this backend");
    case SessionCommand::Kind::metrics:
        return {0, cache.metrics()};
    }
    throw std::invalid_argument("unknown session command");
}

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
        return backend.visit([&](auto& ops) { return control_session(ops.session_cache(), command); });
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
                return backend.visit([&](auto& ops) { return ops.control(ControlRequest{std::move(value)}); });
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
