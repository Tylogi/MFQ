#include "models/minicpmo45/config.h"

#include "nlohmann/json.hpp"

#include <stdexcept>

namespace mfq::models::minicpmo45 {

Config Config::from_json(std::string_view payload) {
    const auto root = nlohmann::json::parse(
        payload.begin(), payload.end(), nullptr, false);
    if (!root.is_object()) {
        throw std::runtime_error("MiniCPM-o model config must be an object");
    }
    Config config;
    static_cast<ModelConfig&>(config) = ModelConfig::from_json(payload);
    config.version = root.value("version", std::string{});
    config.hidden_act = root.value("hidden_act", std::string{});
    config.attention_bias = root.value("attention_bias", true);
    config.use_sliding_window = root.value("use_sliding_window", true);
    if (config.version != "4.5") {
        throw std::runtime_error("only MiniCPM-o version 4.5 is supported");
    }
    return config;
}

} // namespace mfq::models::minicpmo45
