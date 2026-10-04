#pragma once

#include "engine.h"

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

class MfqScheduler;
class MfqSchedulerOverloaded : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class MfqScheduledRequest {
public:
    std::vector<mfq::engine::EngineEvent> wait();
    bool done() const;

private:
    friend class MfqScheduler;
class MfqSchedulerOverloaded : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<mfq::engine::EngineEvent> events_;
    std::size_t bytes_ = 0;
    bool terminal_ = false;
    std::shared_ptr<std::condition_variable> wake_;
};

class MfqScheduler {
public:
    struct Limits {
        std::size_t events = 64, bytes = 1024 * 1024;
        std::size_t requests = 256, input_bytes = 512ULL * 1024 * 1024;
    };
    explicit MfqScheduler(mfq::engine::Engine& engine);
    MfqScheduler(mfq::engine::Engine& engine, Limits limits);
    ~MfqScheduler();
    std::shared_ptr<MfqScheduledRequest> submit(mfq::engine::EngineRequest request) const;
    bool cancel_request(const std::string& id) const;
    bool cancel_session(const std::string& id) const;
    void cancel_all() const;
    void shutdown();
    mfq::engine::EngineInfo info() const;
    mfq::engine::EngineStatus status() const;
    std::future<mfq::engine::ControlResult> control_async(mfq::engine::ControlRequest request) const;
    mfq::engine::ControlResult control(mfq::engine::ControlRequest request) const;
    mfq::engine::SessionResult session(mfq::engine::SessionCommand command) const;
    std::int64_t reload(std::int64_t context) const;

    bool supports_multimodal_generation() const { return info().multimodal; }
    bool supports_reload() const { return info().reload; }
    auto chat_template_capabilities() const { return info().chat; }
    std::int32_t vocab_size() const { return info().vocab_size; }
    void prepare_duplex_session(const std::string& prompt, MfqDuplexSessionParams& params) const;
    void prepare_duplex_step(const std::string& text, MfqDuplexStepInput& input) const;
    std::string decode_tokens(const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded = {}) const;

private:
    struct Request {
        mfq::engine::EngineRequest input;
        std::shared_ptr<MfqScheduledRequest> outbox;
        std::string id, session;
        std::optional<mfq::engine::Clock::time_point> deadline;
        int priority = 0;
        std::size_t reserved_bytes = 0;
        Request(mfq::engine::EngineRequest value, std::shared_ptr<MfqScheduledRequest> box)
            : input(std::move(value)), outbox(std::move(box)), id(input.id),
              session(input.input.cache_plan.session_id), deadline(input.deadline), priority(input.priority) {}
        bool admitted = false, cancelling = false;
        bool finished = false;
        std::optional<mfq::engine::Failed> failure;
        // At most one engine quantum waits here; it prevents further execution.
        std::deque<mfq::engine::EngineEvent> pending;
    };
    struct Submit { Request request; std::promise<void> reply; };
    struct Cancel { std::string id; bool session = false, all = false; std::promise<bool> reply; };
    struct Control { mfq::engine::ControlRequest request; std::promise<mfq::engine::ControlResult> reply; };
    struct Session { mfq::engine::SessionCommand request; std::promise<mfq::engine::SessionResult> reply; };
    struct Reload { std::int64_t context; std::promise<std::int64_t> reply; };
    struct Status { std::promise<mfq::engine::EngineStatus> reply; };
    using Command = std::variant<Submit, Cancel, Control, Session, Reload, Status>;
    void enqueue(Command command) const;
    void loop() noexcept;
    void publish(Request& request, mfq::engine::EngineEvent event);
    void flush(Request& request);
    void cancel(Request& request);
    void release_input_budget(const Request& request);

    mfq::engine::Engine& engine_;
    Limits limits_;
    mutable std::mutex mutex_;
    std::shared_ptr<std::condition_variable> wake_ = std::make_shared<std::condition_variable>();
    mutable std::deque<Command> mailbox_;
    mutable std::size_t reserved_requests_ = 0, reserved_bytes_ = 0;
    mfq::engine::EngineInfo info_;
    bool stopping_ = false;
    std::thread worker_;
    std::unordered_map<std::string, Request> requests_;
    std::deque<std::string> order_;
};
