#pragma once

#include "mfq/runtime.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>

class MfqScheduledRequest {
public:
    ~MfqScheduledRequest();

    const std::shared_ptr<std::atomic<bool>> & cancel_flag() const noexcept;
    void set_session_id(std::string session_id);
    void finish();

private:
    friend class MfqScheduler;

    MfqScheduledRequest(
        std::shared_ptr<std::atomic<bool>> cancel_flag,
        std::function<void(std::string)> set_session_id,
        std::function<void()> release);

    std::shared_ptr<std::atomic<bool>> cancel_flag_;
    std::function<void(std::string)> set_session_id_;
    std::function<void()> release_;
};

// Owns dispatch and request lifecycle between transports and the inference
// engine. Backend-specific batch execution remains inside the engine callback.
class MfqScheduler {
public:
    explicit MfqScheduler(const MfqInferenceEngine & engine);

    bool supports_generation() const noexcept;
    int32_t generate(
        const std::vector<int64_t> & prompt,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const;
    bool supports_multimodal_generation() const noexcept;
    int32_t generate_multimodal(
        const std::vector<int64_t> & prompt,
        const MfqMultimodalInput & media,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqTokenConstraintPtr & token_constraint) const;

    bool supports_reload() const noexcept;
    int64_t reload(int64_t context_size) const;
    const MfqDuplexBackend & duplex() const noexcept;
    const MfqSessionControl & session_control() const noexcept;
    const MfqRuntimeMetricsFn & runtime_metrics() const noexcept;

    std::shared_ptr<MfqScheduledRequest> activate_request(
        const std::string & request_id,
        bool replace = false) const;
    bool cancel_request(const std::string & request_id) const;
    bool cancel_session(const std::string & session_id) const;
    void cancel_all() const;

private:
    struct State;

    const MfqInferenceEngine & engine_;
    std::shared_ptr<State> state_;
};
