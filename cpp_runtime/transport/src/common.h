#pragma once

#include "transport.h"

#include "chat.h"
#include "mfq_text.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::transport_detail {

class ApiError final : public std::runtime_error {
public:
    ApiError(
            int status,
            std::string type,
            std::string message,
            std::string param = {})
        : std::runtime_error(std::move(message)),
          status(status),
          type(std::move(type)),
          param(std::move(param)) {}

    int status;
    std::string type;
    std::string param;
};

class MfqTokenizer {
public:
    explicit MfqTokenizer(const std::string & path);
    explicit MfqTokenizer(const std::vector<uint8_t> & gguf);
    ~MfqTokenizer();

    MfqTokenizer(const MfqTokenizer &) = delete;
    MfqTokenizer & operator=(const MfqTokenizer &) = delete;

    int32_t vocab_size() const;
    std::string chat_template() const;
    const mfq_text_context * context() const;
    int32_t bos_token() const;
    int32_t eos_token() const;
    int32_t eot_token() const;
    int32_t pad_token() const;
    bool add_bos() const;
    bool add_eos() const;
    std::vector<int64_t> tokenize(
        const std::string & text,
        bool parse_special,
        bool add_special = false) const;
    int64_t special_token_id(const std::string & text) const;
    bool is_eog(int64_t token) const;
    std::string piece(int64_t token, bool special = false) const;

private:
    void load_from_file(const std::string & path);
    void finish_init();

    mfq_text_context * context_ = nullptr;
    const mfq_text_vocab * vocab_ = nullptr;
};

using MfqModelCapabilityProfile = MfqModelCapabilities;

struct RequestWork {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    common_chat_parser_params chat_parser;
    std::unordered_set<int64_t> preserved_tokens;
    std::vector<int64_t> prompt;
    std::vector<std::string> stops;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    MfqTokenConstraintPtr token_constraint;
    std::optional<MfqVisionInput> vision;
};

struct RequestMetrics {
    using Clock = std::chrono::steady_clock;

    Clock::time_point started = Clock::now();
    Clock::time_point first_token;
    size_t prefill_tokens = 0;
    double prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    bool saw_token = false;
    bool saw_prefill = false;

    void mark_prefill(const MfqPrefillTiming & timing) {
        prefill_tokens = timing.prompt_tokens;
        prefill_ms = timing.llm_ms;
        multimodal_ms = timing.multimodal_ms;
        model_prefill_ms = timing.model_ms;
        saw_prefill = timing.llm_ms > 0.0 || timing.model_ms > 0.0;
    }

    void mark_token() {
        if (saw_token) return;
        first_token = Clock::now();
        saw_token = true;
    }
};

struct CompletionResult {
    std::string text;
    std::string reasoning_text;
    std::vector<common_chat_tool_call> tool_calls;
    std::string finish_reason = "length";
    int32_t completion_tokens = 0;
    bool client_connected = true;
    bool cancelled = false;
};

struct RequestMetricValues {
    size_t prefill_tokens = 0;
    double generation_ms = 0.0;
    double ttft_ms = 0.0;
    double prefill_ms = 0.0;
    double prefill_tps = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    double decode_ms = 0.0;
    double generation_tps = 0.0;
    double decode_tps = 0.0;
};

class RuntimeRequestMetrics {
public:
    RuntimeRequestMetrics();
    void begin();
    void complete(
        const std::string & id,
        bool chat,
        bool stream,
        size_t prompt_tokens,
        const CompletionResult & result,
        const RequestMetricValues & values);
    void fail();
    uint64_t active_requests() const;
    json snapshot(
        const MfqRuntimeTransportConfig & config,
        int64_t max_context,
        bool reloading) const;

private:
    mutable std::mutex mutex_;
    std::chrono::steady_clock::time_point started_steady_;
    int64_t started_unix_ = 0;
    uint64_t active_requests_ = 0;
    uint64_t total_requests_ = 0;
    uint64_t failed_requests_ = 0;
    uint64_t total_prompt_tokens_ = 0;
    uint64_t total_completion_tokens_ = 0;
    json last_request_ = nullptr;
};

class ActiveRequest {
public:
    explicit ActiveRequest(RuntimeRequestMetrics & metrics);
    ~ActiveRequest();
    void complete(
        const std::string & id,
        bool chat,
        bool stream,
        size_t prompt_tokens,
        const CompletionResult & result,
        const RequestMetricValues & values);

private:
    RuntimeRequestMetrics & metrics_;
    bool completed_ = false;
};

int64_t unix_time_seconds();
std::string request_id(const char * prefix);
int64_t integer_field(
    const json & body,
    const char * name,
    int64_t fallback);
double number_field(
    const json & body,
    const char * name,
    double fallback);
MfqSamplingParams default_sampling_params(
    const MfqRuntimeTransportConfig & config);
json sampling_params_json(const MfqSamplingParams & sampling);
MfqModelCapabilityProfile architecture_capability_profile(
    const std::string & model_type);
json model_capability_profile_json(
    const MfqModelCapabilityProfile & profile);
json duplex_profile_json(const MfqDuplexSamplingProfile & value);
json tts_profile_json(const MfqTtsSamplingProfile & value);
json chat_template_capabilities_json(const std::string & chat_template);
bool valid_mfq_session_id(const std::string & session_id);
json runtime_generate_body(const json & params);
RequestWork parse_work(
    const json & body,
    bool chat,
    const MfqTokenizer & tokenizer,
    const common_chat_templates * templates,
    int64_t max_context,
    const std::string & model_type,
    const MfqSamplingParams & defaults);
RequestMetricValues request_metric_values(
    const CompletionResult & result,
    const RequestMetrics & metrics);
json request_metric_values_json(
    const RequestMetricValues & values,
    const MfqSamplingParams & sampling);
void log_request_metrics(
    const std::string & id,
    bool chat,
    bool stream,
    size_t prompt_tokens,
    const MfqSamplingParams & sampling,
    const CompletionResult & result,
    const RequestMetricValues & values);
CompletionResult generate_text(
    const RequestWork & work,
    const MfqTokenizer & tokenizer,
    const MfqScheduler & scheduler,
    const std::shared_ptr<std::atomic<bool>> & cancel_requested,
    const std::function<bool(const common_chat_msg_diff &)> & emit,
    RequestMetrics * metrics,
    bool defer_token_parsing);
json usage_json(size_t prompt_tokens, int32_t completion_tokens);
json runtime_generation_event(
    const std::string & event,
    const std::string & request_id,
    int64_t created,
    const std::string & model);
json runtime_generation_result(
    const std::string & request_id,
    int64_t created,
    const std::string & model,
    const CompletionResult & result,
    json usage,
    json metrics);
json chat_diff_json(const common_chat_msg_diff & diff);
MfqMultimodalInput parse_mfq_vision(
    const json & value,
    std::vector<int64_t> & prompt,
    const MfqTokenizer & tokenizer,
    int64_t vocab_size);
std::vector<float> decode_audio_features(
    const std::string & encoded,
    int32_t frames);

} // namespace mfq::transport_detail
