#include "models/include/gemma4.h"

#include "nlohmann/json.hpp"

#include <cmath>
#include <stdexcept>

namespace mfq::models::gemma4 {

Config Config::from_json(std::string_view payload) {
    const auto root = nlohmann::json::parse(
        payload.begin(), payload.end(), nullptr, false);
    if (!root.is_object()) {
        throw std::runtime_error("Gemma4 model config must be an object");
    }
    const auto& text = root.contains("text_config")
        ? root.at("text_config") : root;
    if (!text.is_object()) {
        throw std::runtime_error("Gemma4 text_config must be an object");
    }
    const auto sliding = text.value(
        "sliding_attention", nlohmann::json::object());
    if (!sliding.is_object()) {
        throw std::runtime_error(
            "Gemma4 sliding_attention must be an object");
    }

    Config config;
    static_cast<ModelConfig&>(config) = ModelConfig::from_json(payload);
    config.global_head_dim = text.value("global_head_dim", config.head_dim);
    config.num_global_key_value_heads = text.value(
        "num_global_key_value_heads", config.num_key_value_heads);
    config.sliding_window = text.value("sliding_window", std::int64_t{0});
    config.num_experts = text.value("num_experts", std::int64_t{0});
    config.num_experts_per_tok = text.value(
        "num_experts_per_tok", std::int64_t{0});
    config.moe_intermediate_size = text.value(
        "moe_intermediate_size", std::int64_t{0});
    config.sliding_rope_base = sliding.value("rope_theta", 10'000.0);
    config.attention_key_equals_value = text.value("attention_k_eq_v", false);
    config.final_logit_softcapping =
        text.value("final_logit_softcapping", 0.0);
    if (config.global_head_dim <= 0 ||
            config.num_global_key_value_heads <= 0 ||
            config.sliding_window < 0 ||
            config.num_experts < 0 ||
            config.num_experts_per_tok < 0 ||
            config.moe_intermediate_size < 0 ||
            ((config.num_experts > 0 ||
              config.num_experts_per_tok > 0 ||
              config.moe_intermediate_size > 0) &&
             (config.num_experts <= 0 ||
              config.num_experts_per_tok <= 0 ||
              config.num_experts_per_tok > config.num_experts ||
              config.moe_intermediate_size <= 0)) ||
            !std::isfinite(config.sliding_rope_base) ||
            config.sliding_rope_base <= 0.0 ||
            !std::isfinite(config.final_logit_softcapping)) {
        throw std::runtime_error("invalid Gemma4 model configuration");
    }
    return config;
}

} // namespace mfq::models::gemma4
