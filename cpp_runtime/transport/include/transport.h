#pragma once

#include "engine.h"
#include "scheduler.h"
#include "tokenizer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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

struct MfqHttpRuntimeTransportConfig : MfqRuntimeTransportConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
};

class MfqTransport {
public:
    virtual ~MfqTransport() = default;
    virtual void configure_engine(mfq::engine::Engine &) {}
    virtual int run(const MfqScheduler & scheduler) = 0;
};

class MfqRuntime {
public:
    MfqRuntime(
            std::unique_ptr<mfq::engine::Engine> engine,
            std::unique_ptr<MfqTransport> transport)
        : engine_(require_engine(std::move(engine))),
          transport_(require_transport(std::move(transport))),
          scheduler_(configure_engine(*engine_, *transport_)) {}

    template <typename ConcreteEngine,
              std::enable_if_t<std::is_base_of_v<
                  mfq::engine::Engine, ConcreteEngine>, int> = 0>
    MfqRuntime(
            ConcreteEngine engine,
            std::unique_ptr<MfqTransport> transport)
        : MfqRuntime(
              std::make_unique<ConcreteEngine>(std::move(engine)),
              std::move(transport)) {}

    int run() {
        return transport_->run(scheduler_);
    }

private:
    static std::unique_ptr<mfq::engine::Engine> require_engine(
            std::unique_ptr<mfq::engine::Engine> engine) {
        if (!engine) {
            throw std::invalid_argument("MFQ runtime requires an engine");
        }
        return engine;
    }
    static std::unique_ptr<MfqTransport> require_transport(
            std::unique_ptr<MfqTransport> transport) {
        if (!transport) {
            throw std::invalid_argument("MFQ runtime requires a transport");
        }
        return transport;
    }
    static const mfq::engine::Engine & configure_engine(
            mfq::engine::Engine & engine,
            MfqTransport & transport) {
        transport.configure_engine(engine);
        return engine;
    }

    std::unique_ptr<mfq::engine::Engine> engine_;
    std::unique_ptr<MfqTransport> transport_;
    MfqScheduler scheduler_;
};

std::unique_ptr<MfqTransport> make_mfq_http_transport(
    MfqHttpRuntimeTransportConfig config);

// Reserve the original stdout pipe for NDJSON and redirect ordinary process
// output to stderr. Call before loading a model so startup logs cannot corrupt
// the protocol stream.
void prepare_mfq_stdio_transport();
std::unique_ptr<MfqTransport> make_mfq_stdio_transport(
    MfqRuntimeTransportConfig config);

MfqRuntimeProfile resolve_mfq_runtime_profile(
    const std::string & mfq_path,
    const std::string & model_architecture,
    const std::string & model_type,
    const std::string & model_name,
    const std::string & embedded_profile_json = {},
    const std::string & model_config_json = {},
    const std::string & explicit_profile_path = {});
