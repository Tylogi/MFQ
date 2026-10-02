#pragma once

#include "models/common/model_config.h"

#include <string>
#include <string_view>

namespace mfq::models::minicpmo45 {

struct Config : ModelConfig {
    std::string version;
    std::string hidden_act;
    bool attention_bias = true;
    bool use_sliding_window = true;

    static Config from_json(std::string_view payload);

    template <class Source> static Config from_source(const Source &source) {
        return from_json(source.model_config_json());
    }
};

inline ModelConfig tts_decoder_config() {
    mfq::models::ModelConfig result;
    result.model_type = "minicpmtts";
    result.hidden_size = 768;
    result.intermediate_size = 3072;
    result.num_hidden_layers = 20;
    result.num_attention_heads = 12;
    result.num_key_value_heads = 12;
    result.head_dim = 64;
    result.rotary_dim = 64;
    result.max_position_embeddings = 4096;
    result.rope_base = 10000.0;
    result.rms_norm_eps = 1e-6;
    result.layer_types.assign(20, "full_attention");
    return result;
}

} // namespace mfq::models::minicpmo45
