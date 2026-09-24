#pragma once

#include "model_config.h"

#include <cstdint>
#include <string_view>

namespace mfq::models::gemma4 {

struct Config : ModelConfig {
    std::int64_t global_head_dim = 0;
    std::int64_t num_global_key_value_heads = 0;
    std::int64_t sliding_window = 0;
    std::int64_t num_experts = 0;
    std::int64_t num_experts_per_tok = 0;
    std::int64_t moe_intermediate_size = 0;
    double sliding_rope_base = 10'000.0;
    bool attention_key_equals_value = false;
    double final_logit_softcapping = 0.0;

    static Config from_json(std::string_view payload);

    template <class Source>
    static Config from_source(const Source& source) {
        return from_json(source.model_config_json());
    }
};

} // namespace mfq::models::gemma4
