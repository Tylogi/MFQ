#pragma once

#include "transport.h"

#include "inference.h"
#include "text_processor.h"
#include "chat.h"
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

using RequestWork = mfq::engine::InferenceInput;
using RequestInput = mfq::engine::InferenceInput;
using RequestMetrics = mfq::engine::InferenceMetrics;
using CompletionResult = mfq::engine::InferenceResult;

using MfqModelCapabilityProfile = MfqModelCapabilities;

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
json chat_template_capabilities_json(
    const mfq::engine::ChatTemplateCapabilities & capabilities);
bool valid_mfq_session_id(const std::string & session_id);
json runtime_generate_body(const json & params);
RequestInput parse_input(
    const json & body,
    bool chat,
    const MfqSamplingParams & defaults);
RequestMetricValues request_metric_values(
    const CompletionResult & result,
    const RequestMetrics & metrics);
void add_request_runtime_metrics(json& value, const RequestMetrics& metrics);
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
MfqMultimodalInput parse_mfq_vision(const json & value);
std::vector<float> decode_audio_features(
    const std::string & encoded,
    int32_t frames);

} // namespace mfq::transport_detail

namespace mfq::transport_detail {
class CompletionStream {
public:
    CompletionStream(const MfqScheduler& scheduler, const RequestWork& work, std::string id);
    ~CompletionStream();
    std::optional<std::vector<common_chat_msg_diff>> next();
    void cancel();
    CompletionResult result;
    RequestMetrics metrics;
    std::size_t prompt_tokens = 0;
private:
    const MfqScheduler& scheduler_;
    std::string id_;
    std::shared_ptr<MfqScheduledRequest> request_;
    bool terminal_ = false;
};

} // namespace mfq::transport_detail
