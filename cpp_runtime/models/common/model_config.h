#pragma once

#include "mfq/yarn.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models {

// How the model positions attention queries and keys.
enum class PositionalEncoding { rope, none };

// Backend-neutral transformer geometry normalized from model_config.json.
struct ModelConfig {
    std::string model_type;
    std::int64_t vocab_size = 0;
    std::int64_t hidden_size = 0;
    std::int64_t intermediate_size = 0;
    std::int64_t num_hidden_layers = 0;
    std::int64_t num_attention_heads = 0;
    std::int64_t num_key_value_heads = 0;
    std::int64_t max_position_embeddings = 0;
    std::int64_t head_dim = 0;
    double rope_base = 1'000'000.0;
    double full_rotary_factor = 1.0;
    // none permanently disables rotation and requires rotary_dim == 0.
    PositionalEncoding positional_encoding = PositionalEncoding::rope;
    std::int64_t rotary_dim = 0;
    double rms_norm_eps = 1e-6;
    bool tie_word_embeddings = false;
    std::vector<std::string> layer_types;

    static ModelConfig from_json(std::string_view payload);
};

// Normalize the requested Qwen context and return its positional scaling.
YarnScaling extend_qwen_context(
    std::string& definition, std::string_view backbone, std::int64_t context_size);

} // namespace mfq::models
