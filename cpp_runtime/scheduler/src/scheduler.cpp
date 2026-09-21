#include "mfq/scheduler.h"

#include <mutex>
#include <unordered_map>
#include <utility>

struct MfqScheduler::State {
    struct Request {
        std::shared_ptr<std::atomic<bool>> cancel_flag;
        std::string session_id;
    };

    std::mutex mutex;
    std::unordered_map<std::string, Request> requests;
};

MfqScheduledRequest::MfqScheduledRequest(
        std::shared_ptr<std::atomic<bool>> cancel_flag,
        std::function<void(std::string)> set_session_id,
        std::function<void()> release)
    : cancel_flag_(std::move(cancel_flag)),
      set_session_id_(std::move(set_session_id)),
      release_(std::move(release)) {}

MfqScheduledRequest::~MfqScheduledRequest() {
    finish();
}

const std::shared_ptr<std::atomic<bool>> &
MfqScheduledRequest::cancel_flag() const noexcept {
    return cancel_flag_;
}

void MfqScheduledRequest::set_session_id(std::string session_id) {
    if (set_session_id_) set_session_id_(std::move(session_id));
}

void MfqScheduledRequest::finish() {
    if (!release_) return;
    auto release = std::move(release_);
    set_session_id_ = {};
    release();
}

MfqScheduler::MfqScheduler(const MfqInferenceEngine & engine)
    : engine_(engine), state_(std::make_shared<State>()) {}

bool MfqScheduler::supports_generation() const noexcept {
    return static_cast<bool>(engine_.generate);
}

int32_t MfqScheduler::generate(
        const std::vector<int64_t> & prompt,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const {
    return engine_.generate(
        prompt, sampling, on_token, on_prefill, cache_plan, token_constraint);
}

bool MfqScheduler::supports_multimodal_generation() const noexcept {
    return static_cast<bool>(engine_.multimodal_generate);
}

int32_t MfqScheduler::generate_multimodal(
        const std::vector<int64_t> & prompt,
        const MfqMultimodalInput & media,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const {
    return engine_.multimodal_generate(
        prompt, media, sampling, on_token, on_prefill, cache_plan,
        token_constraint);
}

bool MfqScheduler::supports_reload() const noexcept {
    return static_cast<bool>(engine_.reload);
}

int64_t MfqScheduler::reload(int64_t context_size) const {
    return engine_.reload(context_size);
}

const MfqDuplexBackend & MfqScheduler::duplex() const noexcept {
    return engine_.duplex;
}

const MfqSessionControl & MfqScheduler::session_control() const noexcept {
    return engine_.session_control;
}

const MfqRuntimeMetricsFn & MfqScheduler::runtime_metrics() const noexcept {
    return engine_.runtime_metrics;
}

std::shared_ptr<MfqScheduledRequest> MfqScheduler::activate_request(
        const std::string & request_id,
        bool replace) const {
    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    if (request_id.empty()) {
        return std::shared_ptr<MfqScheduledRequest>(new MfqScheduledRequest(
            std::move(cancel_flag), nullptr, nullptr));
    }

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->requests.find(request_id);
        if (found != state_->requests.end()) {
            if (!replace) return nullptr;
            found->second.cancel_flag->store(true, std::memory_order_release);
        }
        state_->requests[request_id] = {cancel_flag, {}};
    }

    const std::weak_ptr<State> state = state_;
    auto set_session_id = [state, request_id, cancel_flag](std::string value) {
        const auto shared = state.lock();
        if (!shared) return;
        std::lock_guard<std::mutex> lock(shared->mutex);
        const auto found = shared->requests.find(request_id);
        if (found != shared->requests.end() &&
            found->second.cancel_flag == cancel_flag) {
            found->second.session_id = std::move(value);
        }
    };
    auto release = [state, request_id, cancel_flag] {
        const auto shared = state.lock();
        if (!shared) return;
        std::lock_guard<std::mutex> lock(shared->mutex);
        const auto found = shared->requests.find(request_id);
        if (found != shared->requests.end() &&
            found->second.cancel_flag == cancel_flag) {
            shared->requests.erase(found);
        }
    };
    return std::shared_ptr<MfqScheduledRequest>(new MfqScheduledRequest(
        std::move(cancel_flag), std::move(set_session_id), std::move(release)));
}

bool MfqScheduler::cancel_request(const std::string & request_id) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto found = state_->requests.find(request_id);
    if (found == state_->requests.end()) return false;
    found->second.cancel_flag->store(true, std::memory_order_release);
    return true;
}

bool MfqScheduler::cancel_session(const std::string & session_id) const {
    if (session_id.empty()) return false;
    bool cancelled = false;
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const auto & item : state_->requests) {
        if (item.second.session_id == session_id) {
            item.second.cancel_flag->store(true, std::memory_order_release);
            cancelled = true;
        }
    }
    return cancelled;
}

void MfqScheduler::cancel_all() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const auto & item : state_->requests) {
        item.second.cancel_flag->store(true, std::memory_order_release);
    }
}
