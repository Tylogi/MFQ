#include "scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <limits>

using namespace mfq::engine;

namespace {
// Charge retained capacities, including preallocated media, before entering the mailbox.
std::size_t input_bytes(const EngineRequest& request) {
    std::size_t bytes = sizeof(request);
    const auto add = [&](std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max() - bytes)
            throw MfqSchedulerOverloaded("request input size overflows the queue budget");
        bytes += value;
    };
    const auto strings = [&](const auto&... values) { (add(values.capacity()), ...); };
    const auto vectors = [&](const auto&... values) { (add(values.capacity() * sizeof(typename std::decay_t<decltype(values)>::value_type)), ...); };
    const auto fields = [&](const auto& map) {
        for (const auto& [key, value] : map) { add(sizeof(key) + sizeof(value) + 4 * sizeof(void*)); strings(key, value); }
    };
    const auto& input = request.input;
    strings(request.id, input.prompt, input.cache_plan.session_id);
    vectors(request.token_ids, input.stops);
    for (const auto& stop : input.stops) strings(stop);
    if (input.chat_input) {
        const auto& chat = *input.chat_input;
        const auto& value = chat.template_inputs;
        if (chat.preformatted_prompt) strings(*chat.preformatted_prompt);
        strings(value.grammar, value.json_schema);
        vectors(value.messages, value.tools);
        fields(value.chat_template_kwargs);
        for (const auto& message : value.messages) {
            strings(message.role, message.content, message.reasoning_content, message.tool_name, message.tool_call_id);
            vectors(message.content_parts, message.tool_calls);
            fields(message.extra_fields);
            for (const auto& part : message.content_parts) strings(part.type, part.text);
            for (const auto& call : message.tool_calls) strings(call.name, call.arguments, call.id);
        }
        for (const auto& tool : value.tools) strings(tool.name, tool.description, tool.parameters);
    }
    if (input.media) {
        const auto& m = *input.media;
        strings(m.processor_name);
        vectors(m.pixel_values, m.pixel_shape, m.patch_mask, m.patch_mask_shape,
                m.target_sizes, m.target_sizes_shape, m.vision_grid, m.vision_grid_shape,
                m.vision_types, m.image_grid, m.image_grid_shape, m.video_grid, m.video_grid_shape,
                m.image_bounds, m.image_permutation, m.image_permutation_offsets,
                m.audio_features, m.audio_features_shape, m.audio_lengths, m.audio_bounds);
    }
    return bytes;
}
constexpr std::size_t terminal_reserve = sizeof(EngineEvent) + 128 + 64 + 512;
std::size_t event_bytes(const EventData& data) {
    return std::visit([](const auto& value) -> std::size_t {
        using T = std::decay_t<decltype(value)>;
        std::size_t bytes = sizeof(EngineEvent);
        if constexpr (std::is_same_v<T, OutputDelta>) {
            bytes += value.token_ids.size() * sizeof(std::int64_t) + value.diffs.size() * sizeof(common_chat_msg_diff);
            for (const auto& diff : value.diffs) {
                bytes += diff.content_delta.size() + diff.reasoning_content_delta.size();
                bytes += diff.tool_call_delta.name.size() + diff.tool_call_delta.arguments.size() + diff.tool_call_delta.id.size();
            }
        } else if constexpr (std::is_same_v<T, Completed>) {
            bytes += value.finish_reason.size();
        } else if constexpr (std::is_same_v<T, Failed>) {
            bytes += value.code.size() + value.message.size();
        }
        return bytes;
    }, data);
}
}

std::vector<EngineEvent> MfqScheduledRequest::wait() {
    std::unique_lock lock(mutex_);
    ready_.wait(lock, [&] { return terminal_ || !events_.empty(); });
    std::vector<EngineEvent> result;
    while (!events_.empty()) {
        result.push_back(std::move(events_.front()));
        events_.pop_front();
    }
    bytes_ = 0;
    lock.unlock();
    wake_->notify_one();
    return result;
}

bool MfqScheduledRequest::done() const {
    std::lock_guard lock(mutex_);
    return terminal_ && events_.empty();
}

MfqScheduler::MfqScheduler(Engine& engine) : MfqScheduler(engine, Limits{}) {}
MfqScheduler::MfqScheduler(Engine& engine, Limits limits)
    : engine_(engine), limits_(limits), info_(engine.info()) {
    if (limits.events < 4 || limits.bytes < 4096 || !limits.requests || !limits.input_bytes)
        throw std::invalid_argument("scheduler requires positive input limits and at least 4 outbox events and 4096 bytes");
    worker_ = std::thread([this] { loop(); });
}
MfqScheduler::~MfqScheduler() { shutdown(); }
void MfqScheduler::shutdown() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    wake_->notify_one();
    if (worker_.joinable()) worker_.join();
}
void MfqScheduler::enqueue(Command command) const {
    std::lock_guard lock(mutex_);
    if (stopping_) throw std::runtime_error("scheduler is stopping");
    const auto* submit = std::get_if<Submit>(&command);
    const auto bytes = submit ? submit->request.reserved_bytes : 0;
    if (submit && (reserved_requests_ >= limits_.requests || bytes > limits_.input_bytes - reserved_bytes_))
        throw MfqSchedulerOverloaded("scheduler input queue budget is exhausted");
    const bool submitting = submit != nullptr;
    mailbox_.push_back(std::move(command));
    if (submitting) { ++reserved_requests_; reserved_bytes_ += bytes; }
    wake_->notify_one();
}
EngineInfo MfqScheduler::info() const {
    std::lock_guard lock(mutex_);
    return info_;
}
EngineStatus MfqScheduler::status() const {
    Status command;
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
std::shared_ptr<MfqScheduledRequest> MfqScheduler::submit(EngineRequest input) const {
    if (input.id.empty() || input.id.size() > 128)
        throw std::invalid_argument("request ID must contain 1-128 bytes");
    auto outbox = std::make_shared<MfqScheduledRequest>();
    outbox->wake_ = wake_;
    const auto bytes = input_bytes(input);
    Submit command{{std::move(input), outbox}, {}};
    command.request.reserved_bytes = bytes;
    auto result = command.reply.get_future();
    enqueue(std::move(command));
    result.get();
    return outbox;
}
bool MfqScheduler::cancel_request(const std::string& id) const {
    Cancel command{id, false, false, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
bool MfqScheduler::cancel_session(const std::string& id) const {
    Cancel command{id, true, false, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
void MfqScheduler::cancel_all() const {
    Cancel command{{}, false, true, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); result.get();
}
ControlResult MfqScheduler::control(ControlRequest request) const {
    Control command{std::move(request), {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
SessionResult MfqScheduler::session(SessionCommand request) const {
    Session command{std::move(request), {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
std::int64_t MfqScheduler::reload(std::int64_t context) const {
    Reload command{context, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
void MfqScheduler::prepare_duplex_session(const std::string& prompt, MfqDuplexSessionParams& params) const {
    params = std::get<MfqDuplexSessionParams>(control(PrepareDuplex{prompt, params}));
}
void MfqScheduler::prepare_duplex_step(const std::string& text, MfqDuplexStepInput& input) const {
    input = std::get<MfqDuplexStepInput>(control(PrepareDuplexStep{text, input}));
}
std::string MfqScheduler::decode_tokens(const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded) const {
    return std::get<std::string>(control(DecodeTokens{tokens, excluded}));
}

void MfqScheduler::publish(Request& request, EngineEvent event) {
    auto& box = *request.outbox;
    std::unique_lock lock(box.mutex_);
    if (box.terminal_) return;
    const bool end = terminal(event.data);
    request.finished |= end;
    if (request.failure) {
        if (!end) return;
        event.data = *request.failure;
    }
    if (auto* failed = std::get_if<Failed>(&event.data)) {
        failed->code.resize(std::min<std::size_t>(failed->code.size(), 64));
        failed->message.resize(std::min<std::size_t>(failed->message.size(), 512));
    }
    if (auto* completed = std::get_if<Completed>(&event.data))
        completed->finish_reason.resize(std::min<std::size_t>(completed->finish_reason.size(), 64));
    const auto bytes = event_bytes(event.data) + event.id.size();
    if (!end && bytes > limits_.bytes - terminal_reserve) {
        request.failure = Failed{"output_limit", "engine output exceeds the request outbox budget"};
        lock.unlock();
        cancel(request);
        return;
    }
    if (!request.pending.empty() || (!end &&
            (box.events_.size() >= limits_.events - 1 || box.bytes_ + bytes > limits_.bytes - terminal_reserve))) {
        request.pending.push_back(std::move(event));
        return;
    }
    box.bytes_ += bytes;
    box.terminal_ = end;
    box.events_.push_back(std::move(event));
    box.ready_.notify_all();
}
void MfqScheduler::flush(Request& request) {
    auto& box = *request.outbox;
    std::lock_guard lock(box.mutex_);
    while (!request.pending.empty()) {
        auto& event = request.pending.front();
        const bool end = terminal(event.data);
        const auto bytes = event_bytes(event.data) + event.id.size();
        if (!end && (box.events_.size() >= limits_.events - 1 ||
                box.bytes_ + bytes > limits_.bytes - terminal_reserve)) break;
        box.bytes_ += bytes;
        box.terminal_ = end;
        box.events_.push_back(std::move(event));
        request.pending.pop_front();
        box.ready_.notify_all();
    }
}
void MfqScheduler::cancel(Request& request) {
    if (request.finished) {
        if (request.pending.empty()) return;
        auto terminal = std::move(request.pending.back());
        request.pending.clear();
        publish(request, std::move(terminal));
        return;
    }
    if (request.cancelling) return;
    request.cancelling = true;
    request.pending.clear();
    if (request.admitted) engine_.cancel(request.id);
    else {
        publish(request, {request.id, Cancelled{}});
    }
}

void MfqScheduler::release_input_budget(const Request& request) {
    // ponytail: reserve through terminal; release earlier when Engine exposes input retirement.
    std::lock_guard lock(mutex_);
    --reserved_requests_;
    reserved_bytes_ -= request.reserved_bytes;
}

void MfqScheduler::loop() noexcept {
    std::optional<Reload> reload;
    bool healthy = true;
    bool duplex_active = false;
    for (;;) {
        std::deque<Command> commands;
        bool stopping;
        { std::lock_guard lock(mutex_); commands.swap(mailbox_); stopping = stopping_; }
        for (auto& command : commands) std::visit([&](auto& value) {
            using T = std::decay_t<decltype(value)>;
            try {
                if constexpr (std::is_same_v<T, Submit>) {
                    auto& request = value.request;
                    const auto id = request.id;
                    if (stopping || !healthy || reload || duplex_active)
                        throw std::runtime_error("scheduler is not accepting requests");
                    if (requests_.contains(id)) throw std::invalid_argument("request ID is already active");
                    const auto& session = request.session;
                    if (!session.empty()) for (const auto& [other_id, other] : requests_)
                        if (other.session == session)
                            throw std::invalid_argument("session already has an active request");
                    order_.push_back(id); requests_.emplace(id, std::move(request));
                    value.reply.set_value();
                } else if constexpr (std::is_same_v<T, Cancel>) {
                    bool found = false;
                    for (auto& [id, request] : requests_) {
                        if (value.all || (value.session ? request.session == value.id : id == value.id)) {
                            cancel(request); found = true;
                        }
                    }
                    value.reply.set_value(found);
                } else if constexpr (std::is_same_v<T, Control>) {
                    const bool start_duplex = std::holds_alternative<MfqDuplexSessionParams>(value.request);
                    const bool stop_duplex = std::holds_alternative<StopDuplex>(value.request);
                    if (reload) throw std::runtime_error("reload is pending");
                    if ((std::holds_alternative<MfqDuplexSessionParams>(value.request) ||
                         std::holds_alternative<MfqDuplexStepInput>(value.request) ||
                         std::holds_alternative<StopDuplex>(value.request)) && !requests_.empty())
                        throw std::runtime_error("duplex conflicts with active generation");
                    auto result = engine_.control(std::move(value.request));
                    if (start_duplex) duplex_active = true;
                    if (stop_duplex) duplex_active = false;
                    value.reply.set_value(std::move(result));
                } else if constexpr (std::is_same_v<T, Session>) {
                    if (value.request.kind != SessionCommand::Kind::metrics) {
                        if (duplex_active || reload) throw std::runtime_error("session operation conflicts with runtime control");
                        for (const auto& [id, request] : requests_) {
                            const auto& session = request.session;
                            if (value.request.kind == SessionCommand::Kind::clear ||
                                value.request.kind == SessionCommand::Kind::trim ||
                                session == value.request.source || session == value.request.target)
                                throw std::runtime_error("session operation conflicts with active generation");
                        }
                    }
                    value.reply.set_value(engine_.session(value.request));
                } else if constexpr (std::is_same_v<T, Status>) {
                    auto status = engine_.status();
                    status.healthy &= healthy;
                    if (!status.healthy || reload || duplex_active || stopping) status.available = 0;
                    value.reply.set_value(status);
                } else {
                    if (reload) throw std::runtime_error("reload already pending");
                    if (duplex_active) {
                        (void)engine_.control(StopDuplex{});
                        duplex_active = false;
                    }
                    for (auto& [id, request] : requests_) cancel(request);
                    reload.emplace(std::move(value));
                }
            } catch (...) {
                if constexpr (std::is_same_v<T, Submit>) release_input_budget(value.request);
                value.reply.set_exception(std::current_exception());
            }
        }, command);
        bool executable = false;
        auto wake_at = Clock::time_point::max();
        try {
            std::vector<RequestId> eligible;
            std::vector<RequestId> ordered(order_.begin(), order_.end());
            std::stable_sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) {
                return requests_.at(a).priority > requests_.at(b).priority;
            });
            bool admission_blocked = false;
            for (const auto& id : ordered) {
                auto& request = requests_.at(id);
                flush(request);
                if (stopping || (request.deadline && *request.deadline <= Clock::now())) cancel(request);
                if (request.finished) continue;
                if (request.deadline && !request.cancelling) wake_at = std::min(wake_at, *request.deadline);
                if (request.cancelling) { executable |= request.admitted; continue; }
                if (!request.admitted && !reload && !admission_blocked && engine_.status().available > 0) {
                    try {
                        request.admitted = engine_.admit(std::move(request.input)) == Admission::accepted;
                        if (request.admitted) request.input = {};
                        admission_blocked = !request.admitted;
                    }
                    catch (const InferenceInputError& error) {
                        publish(request, {id, Failed{error.code == InferenceInputErrorCode::Unsupported
                            ? "unsupported_input" : "invalid_request", error.what()}}); continue;
                    } catch (const std::invalid_argument& error) {
                        publish(request, {id, Failed{"invalid_request", error.what()}}); continue;
                    }
                }
                if (!request.admitted) continue;
                if (!request.pending.empty()) continue;
                auto& box = *request.outbox;
                std::lock_guard lock(box.mutex_);
                // Reserve one quantum and a separate terminal slot. No wait or
                // consumer acknowledgement occurs on the execution thread.
                if (!box.terminal_ && box.events_.size() + 3 < limits_.events &&
                        box.bytes_ < (limits_.bytes - terminal_reserve) / 2)
                    eligible.push_back(id);
            }
            executable |= !eligible.empty();
            if (executable) {
                auto result = engine_.step(eligible);
                for (auto& event : result.events) {
                    auto found = requests_.find(event.id);
                    if (found != requests_.end()) publish(found->second, std::move(event));
                }
                if (!result.status.healthy) throw std::runtime_error("engine is unhealthy");
                executable = !result.advanced.empty() || !result.events.empty();
                if (result.wake_at) wake_at = std::min(wake_at, *result.wake_at);
            }
        } catch (const std::exception& error) {
            healthy = false;
            for (auto& [id, request] : requests_) {
                try { if (request.admitted) engine_.cancel(id); } catch (...) {}
            }
            try { (void)engine_.step({}); } catch (...) {}
            for (auto& [id, request] : requests_)
                if (!request.finished)
                    try { publish(request, {id, Failed{"backend_failure", error.what()}}); } catch (...) {}
        } catch (...) {
            healthy = false;
            for (auto& [id, request] : requests_)
                try { engine_.cancel(id); } catch (...) {}
            try { (void)engine_.step({}); } catch (...) {}
            for (auto& [id, request] : requests_)
                if (!request.finished)
                    try { publish(request, {id, Failed{"backend_failure", "unknown backend error"}}); } catch (...) {}
        }
        for (auto it = order_.begin(); it != order_.end();) {
            auto box = requests_.at(*it).outbox;
            std::lock_guard lock(box->mutex_);
            if (box->terminal_) {
                release_input_budget(requests_.at(*it));
                requests_.erase(*it); it = order_.erase(it);
            }
            else ++it;
        }
        if (reload && requests_.empty()) {
            try {
                const auto context = engine_.reload(reload->context);
                { std::lock_guard lock(mutex_); info_ = engine_.info(); }
                healthy = true; reload->reply.set_value(context);
            } catch (...) { healthy = false; reload->reply.set_exception(std::current_exception()); }
            reload.reset();
        }
        if (!order_.empty()) { order_.push_back(order_.front()); order_.pop_front(); }
        if (stopping && requests_.empty()) {
            try { engine_.shutdown(); } catch (...) {}
            return;
        }
        if (!executable) {
            std::unique_lock lock(mutex_);
            if (mailbox_.empty() && !stopping_) {
                // A timed recheck also covers a drain racing the wait setup.
                wake_->wait_until(lock, std::min(wake_at, Clock::now() + std::chrono::milliseconds(10)));
            }
        }
    }
}
