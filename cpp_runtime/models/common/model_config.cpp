#include "models/common/model_config.h"

#include "nlohmann/json.hpp"

#include <cmath>
#include <stdexcept>

namespace mfq::models {
namespace {

using json = nlohmann::json;

const json& object_or_self(const json& root, const char* key) {
    const auto found = root.find(key);
    if (found == root.end()) return root;
    if (!found->is_object()) {
        throw std::runtime_error(
            std::string("model config ") + key + " must be an object");
    }
    return *found;
}

const json& object_or_empty(
        const json& parent,
        const char* key,
        const json& empty) {
    const auto found = parent.find(key);
    if (found == parent.end() || found->is_null()) return empty;
    if (!found->is_object()) {
        throw std::runtime_error(
            std::string("model config ") + key + " must be an object");
    }
    return *found;
}

std::int64_t required_positive(const json& object, const char* key) {
    const auto found = object.find(key);
    if (found == object.end() || !found->is_number_integer() ||
            found->get<std::int64_t>() <= 0) {
        throw std::runtime_error(
            std::string("model config requires positive integer ") + key);
    }
    return found->get<std::int64_t>();
}

} // namespace

ModelConfig ModelConfig::from_json(std::string_view payload) {
    json root;
    try {
        root = json::parse(payload.begin(), payload.end());
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid model config JSON: ") + error.what());
    }
    if (!root.is_object()) {
        throw std::runtime_error("model config must be an object");
    }

    const auto& text = object_or_self(root, "text_config");
    static const json empty = json::object();
    const auto& rope = object_or_empty(text, "rope_parameters", empty);
    const auto& full_rope = text.contains("full_attention")
        ? object_or_empty(text, "full_attention", empty)
        : object_or_empty(rope, "full_attention", empty);

    ModelConfig config;
    config.model_type = root.value(
        "model_type", text.value("model_type", std::string{}));
    config.vocab_size = required_positive(text, "vocab_size");
    config.hidden_size = required_positive(text, "hidden_size");
    config.intermediate_size = text.value("intermediate_size", std::int64_t{0});
    if (config.intermediate_size < 0) {
        throw std::runtime_error("model config intermediate_size cannot be negative");
    }
    config.num_hidden_layers = required_positive(text, "num_hidden_layers");
    config.num_attention_heads = required_positive(text, "num_attention_heads");
    config.num_key_value_heads = required_positive(text, "num_key_value_heads");
    config.max_position_embeddings =
        required_positive(text, "max_position_embeddings");
    config.head_dim = text.value(
        "head_dim", config.hidden_size / config.num_attention_heads);
    if (config.head_dim <= 0) {
        throw std::runtime_error("model config head_dim must be positive");
    }
    config.rope_base = full_rope.value(
        "rope_theta", rope.value(
            "rope_theta", text.value("rope_theta", 1'000'000.0)));
    config.full_rotary_factor = full_rope.value(
        "partial_rotary_factor", rope.value(
            "partial_rotary_factor",
            text.value("partial_rotary_factor", 1.0)));
    config.rotary_dim = static_cast<std::int64_t>(std::llround(
        config.full_rotary_factor * static_cast<double>(config.head_dim)));
    config.rms_norm_eps = text.value("rms_norm_eps", 1e-6);
    config.tie_word_embeddings = text.value(
        "tie_word_embeddings", root.value("tie_word_embeddings", false));
    config.layer_types = text.value(
        "layer_types", std::vector<std::string>{});
    if (config.layer_types.empty()) {
        config.layer_types.assign(
            static_cast<std::size_t>(config.num_hidden_layers),
            "full_attention");
    }
    if (config.layer_types.size() !=
            static_cast<std::size_t>(config.num_hidden_layers)) {
        throw std::runtime_error(
            "model config layer_types length does not match num_hidden_layers");
    }
    if (!std::isfinite(config.rope_base) || config.rope_base <= 0.0 ||
            !std::isfinite(config.full_rotary_factor) ||
            config.full_rotary_factor <= 0.0 || config.rotary_dim <= 0 ||
            !std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0) {
        throw std::runtime_error("invalid model RoPE or RMSNorm configuration");
    }
    return config;
}

} // namespace mfq::models
