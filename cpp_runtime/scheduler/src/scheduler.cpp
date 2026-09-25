#include "scheduler.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <utility>

struct MfqScheduler::State {
    struct Request {
        std::shared_ptr<std::atomic<bool>> cancel_flag;
        std::string session_id;
    };

    explicit State(std::size_t request_limit)
        : request_limit(std::max<std::size_t>(request_limit, 1)) {}

    std::mutex mutex;
    std::condition_variable changed;
    std::unordered_map<std::string, Request> requests;
    std::deque<std::uint64_t> waiting;
    std::size_t request_limit = 1;
    std::size_t running = 0;
    std::uint64_t next_ticket = 0;
    bool stopping = false;
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

bool MfqScheduledRequest::cancelled() const noexcept {
    return cancel_flag_->load(std::memory_order_acquire);
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
    : engine_(engine),
      state_(std::make_shared<State>(engine.max_concurrent_requests)) {}

MfqScheduler::~MfqScheduler() {
    cancel_all();
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->stopping = true;
    state_->changed.notify_all();
    state_->changed.wait(lock, [&] {
        return state_->running == 0 && state_->waiting.empty();
    });
}

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
    return generate_with_cancel(
        std::make_shared<std::atomic<bool>>(false), prompt, sampling,
        on_token, on_prefill, cache_plan, token_constraint);
}

int32_t MfqScheduler::generate(
        const MfqScheduledRequest & request,
        const std::vector<int64_t> & prompt,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const {
    return generate_with_cancel(
        request.cancel_flag(), prompt, sampling, on_token, on_prefill,
        cache_plan, token_constraint);
}

int32_t MfqScheduler::generate_with_cancel(
        const std::shared_ptr<std::atomic<bool>> & cancel_flag,
        const std::vector<int64_t> & prompt,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const {
    if (!admit(cancel_flag)) return 0;
    try {
        const auto cancelled = [cancel_flag] {
            return cancel_flag->load(std::memory_order_acquire);
        };
        const auto emit = [cancelled, &on_token](int64_t token) {
            return !cancelled() && (!on_token || on_token(token));
        };
        const auto result = engine_.generate(
            prompt, sampling, emit, on_prefill, cache_plan, token_constraint,
            cancelled);
        release_admission();
        return result;
    } catch (...) {
        release_admission();
        throw;
    }
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
    return generate_multimodal_with_cancel(
        std::make_shared<std::atomic<bool>>(false), prompt, media, sampling,
        on_token, on_prefill, cache_plan, token_constraint);
}

int32_t MfqScheduler::generate_multimodal(
        const MfqScheduledRequest & request,
        const std::vector<int64_t> & prompt,
        const MfqMultimodalInput & media,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const {
    return generate_multimodal_with_cancel(
        request.cancel_flag(), prompt, media, sampling, on_token, on_prefill,
        cache_plan, token_constraint);
}

int32_t MfqScheduler::generate_multimodal_with_cancel(
        const std::shared_ptr<std::atomic<bool>> & cancel_flag,
        const std::vector<int64_t> & prompt,
        const MfqMultimodalInput & media,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const {
    if (!admit(cancel_flag)) return 0;
    try {
        const auto cancelled = [cancel_flag] {
            return cancel_flag->load(std::memory_order_acquire);
        };
        const auto emit = [cancelled, &on_token](int64_t token) {
            return !cancelled() && (!on_token || on_token(token));
        };
        const auto result = engine_.multimodal_generate(
            prompt, media, sampling, emit, on_prefill, cache_plan,
            token_constraint, cancelled);
        release_admission();
        return result;
    } catch (...) {
        release_admission();
        throw;
    }
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

bool MfqScheduler::admit(
        const std::shared_ptr<std::atomic<bool>> & cancel_flag) const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    const auto ticket = state_->next_ticket++;
    state_->waiting.push_back(ticket);
    state_->changed.wait(lock, [&] {
        return state_->stopping ||
            cancel_flag->load(std::memory_order_acquire) ||
            (state_->waiting.front() == ticket &&
             state_->running < state_->request_limit);
    });
    if (state_->stopping ||
            cancel_flag->load(std::memory_order_acquire)) {
        const auto found = std::find(
            state_->waiting.begin(), state_->waiting.end(), ticket);
        if (found != state_->waiting.end()) state_->waiting.erase(found);
        state_->changed.notify_all();
        return false;
    }
    state_->waiting.pop_front();
    ++state_->running;
    state_->changed.notify_all();
    return true;
}

void MfqScheduler::release_admission() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->running > 0) --state_->running;
    state_->changed.notify_all();
}

std::shared_ptr<MfqScheduledRequest> MfqScheduler::activate_request(
        const std::string & request_id,
        bool replace) const {
    return activate_request_impl(request_id, {}, replace, false);
}

std::shared_ptr<MfqScheduledRequest> MfqScheduler::activate_request(
        const std::string & request_id,
        const std::string & session_id,
        bool replace_session) const {
    return activate_request_impl(
        request_id, session_id, false, replace_session);
}

std::shared_ptr<MfqScheduledRequest> MfqScheduler::activate_request_impl(
        const std::string & request_id,
        const std::string & session_id,
        bool replace_request,
        bool replace_session) const {
    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    if (request_id.empty()) {
        return std::shared_ptr<MfqScheduledRequest>(new MfqScheduledRequest(
            std::move(cancel_flag), nullptr, nullptr));
    }

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->requests.find(request_id);
        if (found != state_->requests.end()) {
            if (!replace_request) return nullptr;
            found->second.cancel_flag->store(true, std::memory_order_release);
        }
        if (replace_session && !session_id.empty()) {
            for (const auto& item : state_->requests) {
                if (item.second.session_id == session_id) {
                    item.second.cancel_flag->store(
                        true, std::memory_order_release);
                }
            }
        }
        state_->requests[request_id] = {cancel_flag, session_id};
        state_->changed.notify_all();
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
    state_->changed.notify_all();
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
    if (cancelled) state_->changed.notify_all();
    return cancelled;
}

void MfqScheduler::cancel_all() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const auto & item : state_->requests) {
        item.second.cancel_flag->store(true, std::memory_order_release);
    }
    state_->changed.notify_all();
}
