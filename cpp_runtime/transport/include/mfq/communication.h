#pragma once

#include "mfq/runtime.h"
#include "mfq/runtime_transport.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// HTTP-specific settings layered over the shared runtime transport metadata.
struct MfqHttpRuntimeTransportConfig : MfqRuntimeTransportConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
};

struct MfqTokenizerProbe {
    int32_t vocab_size = 0;
    int32_t bos_token = -1;
    int32_t eos_token = -1;
    int32_t eot_token = -1;
    int32_t pad_token = -1;
    bool add_bos = false;
    bool add_eos = false;
    std::string chat_template;
    std::vector<int64_t> tokens;
};

std::unique_ptr<MfqTransport> make_mfq_http_transport(
    MfqHttpRuntimeTransportConfig config);

// Reserve the original stdout pipe for NDJSON and redirect ordinary process
// output to stderr. Call before loading a model so startup logs cannot corrupt
// the protocol stream.
void prepare_mfq_stdio_transport();
std::unique_ptr<MfqTransport> make_mfq_stdio_transport(
    MfqRuntimeTransportConfig config);

MfqTokenizerProbe probe_mfq_tokenizer(
    const std::vector<uint8_t> & tokenizer_gguf,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
MfqRuntimeProfile resolve_mfq_runtime_profile(
    const std::string & mfq_path,
    const std::string & model_architecture,
    const std::string & model_type,
    const std::string & model_name,
    const std::string & embedded_profile_json = {},
    const std::string & model_config_json = {},
    const std::string & explicit_profile_path = {});
MfqTokenizerProbe probe_mfq_tokenizer(
    const std::string & tokenizer_model,
    const std::string & text,
    bool add_special = false,
    bool parse_special = true);
