#pragma once

#include "text_processor.h"

#include <chrono>
#include <optional>
#include <variant>

struct MfqModelCapabilities {
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

namespace mfq::engine {

using RequestId = std::string;
using Clock = std::chrono::steady_clock;
using Metrics = std::vector<std::pair<std::string, double>>;

struct EngineRequest {
    RequestId id;
    InferenceInput input;
    std::optional<Clock::time_point> deadline;
    int priority = 0;
    // Diagnostics only. Production submits owned text/media and is tokenized
    // on admission, never against a tokenizer from before a queued reload.
    std::vector<std::int64_t> token_ids;
};

struct PrefillProgress { MfqPrefillTiming timing; };
using OutputDelta = TokenOutput;
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
    MfqModelCapabilities capabilities;
};
struct EngineStatus {
    std::size_t available = 0;
    bool healthy = true;
};
struct EngineStepResult {
    std::vector<RequestId> advanced;
    std::vector<EngineEvent> events;
    EngineStatus status;
    bool has_work = false;
    std::optional<Clock::time_point> wake_at;
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
using ControlResult = std::variant<std::monostate,
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
