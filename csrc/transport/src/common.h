#pragma once

#include "transport.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::transport {

using json = nlohmann::ordered_json;

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

using RequestInput = mfq::engine::InferenceInput;
using RequestMetrics = mfq::engine::InferenceMetrics;

struct CompletionResult {
    std::string text;
    std::string reasoning_text;
    std::vector<mfq::engine::ToolCall> tool_calls;
    std::string finish_reason = "length";
    std::int32_t completion_tokens = 0;
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
json model_capability_profile_json(
    const mfq::engine::ModelCapabilities& capabilities);
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
MfqMultimodalInput parse_mfq_vision(const json & value);
std::vector<float> decode_audio_features(
    const std::string & encoded,
    int32_t frames);

// Shared generation protocol. Adapters only frame these values and signal a
// failed write with cancel(). Destruction cancels unfinished work and accounts
// failures; streaming never retains the generated text or tool arguments.
class CompletionStream {
public:
    CompletionStream(const MfqScheduler& scheduler,
        const MfqRuntimeTransportConfig& config, RuntimeRequestMetrics& metrics,
        const json& params, std::string id = {});
    ~CompletionStream();
    CompletionStream(const CompletionStream&) = delete;
    CompletionStream& operator=(const CompletionStream&) = delete;
    bool streaming() const { return stream_; }
    // Waits for deltas/complete/optional usage, or one non-streaming result.
    // Returns nullopt after terminal delivery or disconnect.
    std::optional<std::vector<json>> next();
    void cancel();
private:
    void fail();
    json event(const char* name) const;
    const MfqScheduler& scheduler_;
    RuntimeRequestMetrics& metrics_store_;
    std::string id_, response_id_, model_;
    std::int64_t created_;
    MfqSamplingParams sampling_;
    CompletionResult result_;
    std::shared_ptr<MfqScheduledRequest> request_;
    bool stream_ = false, include_usage_ = false;
    bool terminal_ = false, accounted_ = false;
};

} // namespace mfq::transport
