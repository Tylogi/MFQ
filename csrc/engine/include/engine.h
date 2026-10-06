#pragma once

#include "mfq/runtime.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace mfq::engine::mtp {

inline constexpr int kMaximumDraftDepth = 5;

struct GenerationStats {
    bool available = false;
    bool used = false;
    std::uint64_t cycles = 0;
    std::uint64_t drafted_tokens = 0;
    std::uint64_t accepted_tokens = 0;
    std::array<std::uint64_t, kMaximumDraftDepth + 1> depth_cycles{};
    std::array<std::uint64_t, kMaximumDraftDepth> position_drafted{};
    std::array<std::uint64_t, kMaximumDraftDepth> position_accepted{};
    std::array<double, kMaximumDraftDepth + 1> measured_depth_ms{};
    int selected_depth = 0;
};

} // namespace mfq::engine::mtp

namespace mfq::engine {

enum class InferenceInputErrorCode {
    Invalid,
    Unsupported,
    ContextLength,
};

class InferenceInputError final : public std::runtime_error {
public:
    InferenceInputError(
        InferenceInputErrorCode code,
        std::string message,
        std::string field = {});

    InferenceInputErrorCode code;
    std::string field;
};

enum class ReasoningFormat { none, automatic, deepseek_legacy, deepseek };
enum class ChatContinuation { none, automatic, reasoning, content };
enum class ChatToolChoice { automatic, required, none };

struct ToolCall {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const ToolCall&) const = default;
};

struct ChatContentPart {
    std::string type;
    std::string text;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::vector<ChatContentPart> content_parts;
    std::vector<ToolCall> tool_calls;
    std::string reasoning_content;
    std::string tool_name;
    std::string tool_call_id;
    std::map<std::string, std::string> extra_fields;
};

struct ChatTool {
    std::string name;
    std::string description;
    std::string parameters;
};

struct ChatInput {
    std::vector<ChatMessage> messages;
    std::vector<ChatTool> tools;
    std::string json_schema;
    std::map<std::string, std::string> template_kwargs;
    ReasoningFormat reasoning_format = ReasoningFormat::automatic;
    ChatContinuation continuation = ChatContinuation::none;
    ChatToolChoice tool_choice = ChatToolChoice::automatic;
    std::optional<std::string> preformatted_prompt;
    std::optional<bool> parallel_tool_calls;
    bool enable_thinking = true;
    bool add_generation_prompt = true;
};

struct InferenceInput {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    std::string prompt;
    std::optional<ChatInput> chat_input;
    std::vector<std::string> stops;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    std::optional<MfqMultimodalInput> media;
};

struct ChatTemplateCapabilities {
    bool thinking = false;
    std::vector<std::string> reasoning_effort_values;
};

struct InferenceMetrics {
    using Clock = std::chrono::steady_clock;

    Clock::time_point started = Clock::now();
    Clock::time_point first_token;
    Clock::time_point last_token;
    std::size_t prefill_tokens = 0;
    double prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    bool saw_token = false;
    bool saw_prefill = false;
    mtp::GenerationStats mtp;

    void mark_prefill(const MfqPrefillTiming& timing);
    void mark_token();
};

struct OutputDiff {
    std::string reasoning_content_delta;
    std::string content_delta;
    std::size_t tool_call_index = std::string::npos;
    ToolCall tool_call_delta;

    bool operator==(const OutputDiff&) const = default;
};

struct OutputDelta {
    std::vector<std::int64_t> token_ids;
    std::vector<OutputDiff> diffs;
};

struct ModelCapabilities {
    std::string family = "unknown";
    bool text = true;
    bool image_input = false;
    bool video_input = false;
    bool audio_input = false;
    bool audio_output = false;
    bool full_duplex = false;
    bool mtp = false;
    std::string source;
};

using RequestId = std::string;
using Clock = std::chrono::steady_clock;
using Metrics = std::vector<std::pair<std::string, double>>;

struct EngineRequest {
    RequestId id;
    InferenceInput input;
    std::optional<Clock::time_point> deadline;
    // Diagnostics only. Production submits owned text/media and is tokenized
    // after admission, using the tokenizer retained by that Engine request.
    std::vector<std::int64_t> token_ids;
};

struct PrefillProgress { MfqPrefillTiming timing; };
struct UsageUpdate { std::size_t prompt_tokens = 0, completion_tokens = 0; };
// Text/tool output travels only in deltas, so terminals have a bounded size.
struct Completed { std::string finish_reason; UsageUpdate usage; InferenceMetrics metrics; };
struct Cancelled { UsageUpdate usage; InferenceMetrics metrics; };
struct Failed { std::string code, message; bool retryable = false; };
using EventData = std::variant<PrefillProgress, OutputDelta, UsageUpdate,
                               Completed, Cancelled, Failed>;
struct EngineEvent { RequestId id; EventData data; };

inline bool terminal(const EventData& event) {
    return std::holds_alternative<Completed>(event) ||
        std::holds_alternative<Cancelled>(event) ||
        std::holds_alternative<Failed>(event);
}

struct EngineInfo {
    std::size_t max_requests = 1;
    std::int64_t max_context = 0;
    std::int32_t vocab_size = 0;
    bool multimodal = false, duplex = false, reload = false;
    ChatTemplateCapabilities chat;
    std::string model_type;
    ModelCapabilities capabilities;
};
struct EngineStatus {
    std::size_t available = 0;
    bool healthy = true;
};
using ControlCompletion = std::variant<std::monostate, MfqDuplexStepResult, MfqDuplexSessionParams,
    MfqDuplexStepInput, std::string, Cancelled, Failed>;
// A pending control publishes exactly one completion from step(). Metrics stay synchronous.
struct ControlPending {};
struct EngineStepResult {
    std::vector<RequestId> advanced;
    std::vector<EngineEvent> events;
    EngineStatus status;
    bool has_work = false;
    std::optional<Clock::time_point> wake_at;
    std::optional<ControlCompletion> control;
    bool control_advanced = false, has_control_work = false;
};
enum class Admission { accepted, deferred };
struct SessionCommand {
    enum class Kind { fork, close, clear, trim, metrics } kind;
    std::string source, target;
    std::uint64_t bytes = 0;
};
struct SessionResult { std::uint64_t count = 0; Metrics metrics; };
struct DecodeTokens {
    std::vector<std::int64_t> tokens;
    std::unordered_set<std::int64_t> excluded;
};
struct PrepareDuplex { std::string prompt; MfqDuplexSessionParams parameters; };
struct PrepareDuplexStep { std::string text; MfqDuplexStepInput input; };
struct StopDuplex {};
struct RuntimeMetrics {};
using ControlRequest = std::variant<DecodeTokens, PrepareDuplex,
    PrepareDuplexStep, MfqDuplexSessionParams, MfqDuplexStepInput, StopDuplex,
    RuntimeMetrics>;
using ControlResult = std::variant<std::monostate, ControlPending,
    std::string, MfqDuplexSessionParams, MfqDuplexStepInput, MfqDuplexStepResult,
    Metrics>;

// Only the scheduler loop calls these methods. Execution returns owned values;
// no caller code or response writer can run on the device execution stack.
class Engine {
public:
    virtual ~Engine() = default;
    virtual EngineInfo info() const = 0;
    // accepted consumes the input; deferred leaves the request intact for retry.
    virtual Admission admit(EngineRequest&& request) = 0;
    virtual void cancel(const RequestId& id) = 0;
    virtual EngineStepResult step(const std::vector<RequestId>& eligible) = 0;
    virtual EngineStatus status() const = 0;
    virtual SessionResult session(const SessionCommand& command) = 0;
    virtual std::int64_t reload(std::int64_t context) = 0;
    virtual void shutdown() = 0;
    virtual ControlResult control(ControlRequest request) = 0;
};

} // namespace mfq::engine
