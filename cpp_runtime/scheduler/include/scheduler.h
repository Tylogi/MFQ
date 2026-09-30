#pragma once

#include "engine.h"
#ifdef MFQ_ENGINE_TEXT
#include "text_processor.h"
#endif

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>

class MfqScheduledRequest {
public:
    ~MfqScheduledRequest();

    const std::shared_ptr<std::atomic<bool>> & cancel_flag() const noexcept;
    bool cancelled() const noexcept;
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
    explicit MfqScheduler(const mfq::engine::Engine& engine);
    ~MfqScheduler();

    bool supports_generation() const noexcept;
    int32_t generate(
        const std::vector<int64_t> & prompt,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const;
    int32_t generate(
        const MfqScheduledRequest & request,
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
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const;
    int32_t generate_multimodal(
        const MfqScheduledRequest & request,
        const std::vector<int64_t> & prompt,
        const MfqMultimodalInput & media,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const;

#ifdef MFQ_ENGINE_TEXT
    mfq::engine::InferenceRequest prepare_inference(
        mfq::engine::InferenceInput input,
        int64_t max_context) const;
    mfq::engine::InferenceResult run_inference(
        const mfq::engine::InferenceRequest & request,
        const MfqScheduledRequest & scheduled,
        const mfq::engine::InferenceEmit & emit,
        mfq::engine::InferenceMetrics * metrics = nullptr,
        bool defer_token_parsing = false,
        const std::function<std::string()> & make_tool_call_id = {}) const;
    mfq::engine::ChatTemplateCapabilities chat_template_capabilities() const;
    int32_t vocab_size() const;
    void prepare_duplex_session(
        const std::string & system_prompt,
        MfqDuplexSessionParams & parameters) const;
    void prepare_duplex_step(
        const std::string & text,
        MfqDuplexStepInput & step) const;
    std::string decode_tokens(
        const std::vector<int64_t> & tokens,
        const std::unordered_set<int64_t> & excluded = {}) const;
#endif

    bool supports_reload() const noexcept;
    int64_t reload(int64_t context_size) const;
    const MfqDuplexBackend & duplex() const noexcept;
    const MfqSessionControl & session_control() const noexcept;
    const MfqRuntimeMetricsFn & runtime_metrics() const noexcept;

    std::shared_ptr<MfqScheduledRequest> activate_request(
        const std::string & request_id,
        bool replace = false) const;
    std::shared_ptr<MfqScheduledRequest> activate_request(
        const std::string & request_id,
        const std::string & session_id,
        bool replace_session) const;
    bool cancel_request(const std::string & request_id) const;
    bool cancel_session(const std::string & session_id) const;
    void cancel_all() const;

private:
    struct State;

    std::shared_ptr<MfqScheduledRequest> activate_request_impl(
        const std::string & request_id,
        const std::string & session_id,
        bool replace_request,
        bool replace_session) const;
    bool admit(const std::shared_ptr<std::atomic<bool>> & cancel_flag) const;
    void release_admission() const;
    int32_t generate_with_cancel(
        const std::shared_ptr<std::atomic<bool>> & cancel_flag,
        const std::vector<int64_t> & prompt,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const;
    int32_t generate_multimodal_with_cancel(
        const std::shared_ptr<std::atomic<bool>> & cancel_flag,
        const std::vector<int64_t> & prompt,
        const MfqMultimodalInput & media,
        const MfqSamplingParams & sampling,
        const MfqTokenCallback & on_token,
        const MfqPrefillCallback & on_prefill,
        const MfqPromptCachePlan & cache_plan,
        const MfqTokenConstraintPtr & token_constraint) const;

    const mfq::engine::Engine& engine_;
    std::shared_ptr<State> state_;
};
