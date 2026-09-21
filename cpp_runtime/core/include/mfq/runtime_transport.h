#pragma once

#include "mfq/runtime.h"
#include "mfq/scheduler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct MfqChatSamplingProfile {
    std::optional<int32_t> max_tokens;
    std::optional<double> temperature;
    std::optional<int32_t> top_k;
    std::optional<double> top_p;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<double> repetition_penalty;
    std::optional<bool> enable_thinking;
    std::optional<bool> enable_vision;
    std::optional<bool> enable_mtp;
    std::optional<int32_t> mtp_max_draft_tokens;
};

struct MfqDuplexSamplingProfile {
    std::optional<std::string> system_prompt;
    std::optional<std::string> decode_mode;
    std::optional<double> temperature;
    std::optional<int32_t> top_k;
    std::optional<double> top_p;
    std::optional<double> text_repetition_penalty;
    std::optional<int32_t> text_repetition_window_size;
    std::optional<double> length_penalty;
    std::optional<double> listen_prob_scale;
    std::optional<int32_t> force_listen_count;
    std::optional<int32_t> max_new_speak_tokens_per_chunk;
};

struct MfqTtsSamplingProfile {
    std::optional<double> temperature;
    std::optional<double> repetition_penalty;
    std::optional<int32_t> token2wav_steps;
};

struct MfqRuntimeProfile {
    MfqChatSamplingProfile chat;
    MfqDuplexSamplingProfile duplex;
    MfqTtsSamplingProfile tts;
    std::string source = "generic-defaults";
};

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

// Model/runtime metadata shared by every private communication transport.
struct MfqRuntimeTransportConfig {
    std::string model_name = "mfq-model";
    std::string model_type;
    std::vector<uint8_t> tokenizer_gguf;
    std::string tokenizer_model;
    int64_t max_context = 0;
    int64_t context_capacity = 0;
    int64_t vocab_size = 0;
    std::optional<MfqModelCapabilities> model_capabilities;
    MfqRuntimeProfile runtime_profile;
};

// Wire-protocol adapter. It communicates only with the scheduler.
class MfqTransport {
public:
    virtual ~MfqTransport() = default;
    virtual int run(const MfqScheduler & scheduler) = 0;
};

// Complete native runtime: transport + scheduler + inference engine.
class MfqRuntime {
public:
    MfqRuntime(
            MfqInferenceEngine engine,
            std::unique_ptr<MfqTransport> transport)
        : engine_(std::move(engine)),
          scheduler_(engine_),
          transport_(std::move(transport)) {
        if (!transport_) {
            throw std::invalid_argument("MFQ runtime requires a transport");
        }
    }

    int run() {
        return transport_->run(scheduler_);
    }

private:
    MfqInferenceEngine engine_;
    MfqScheduler scheduler_;
    std::unique_ptr<MfqTransport> transport_;
};
