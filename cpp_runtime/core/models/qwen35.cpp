#include "models/qwen35.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mfq::models::qwen35 {

Config Config::from_json(
        std::string_view payload,
        const MfqModelGraph& graph) {
    const auto document = nlohmann::json::parse(
        payload.begin(), payload.end(), nullptr, false);
    if (!document.is_object()) {
        throw std::runtime_error("Qwen model config must be an object");
    }
    const auto& text = document.contains("text_config")
        ? document.at("text_config") : document;
    if (!text.is_object()) {
        throw std::runtime_error("Qwen text_config must be an object");
    }

    Config config;
    static_cast<ModelConfig&>(config) = ModelConfig::from_json(payload);
    config.num_experts = text.value(
        "num_experts",
        text.value("n_routed_experts", std::int64_t{0}));
    config.num_experts_per_tok = text.value(
        "num_experts_per_tok",
        text.value("top_k_experts", std::int64_t{0}));
    config.moe_intermediate_size =
        text.value("moe_intermediate_size", std::int64_t{0});
    config.shared_expert_intermediate_size =
        text.value("shared_expert_intermediate_size", std::int64_t{0});
    const auto shared_experts =
        text.value("n_shared_experts", std::int64_t{0});
    if (config.shared_expert_intermediate_size == 0 &&
            shared_experts > 0 && config.moe_intermediate_size > 0) {
        config.shared_expert_intermediate_size =
            shared_experts * config.moe_intermediate_size;
    }
    config.routed_scaling_factor =
        text.value("routed_scaling_factor", 1.0);
    config.norm_topk_prob = text.value("norm_topk_prob", false);
    config.expert_gating_func =
        text.value("scoring_func", std::string("softmax"));
    if (config.num_experts > 0) {
        if (config.num_experts_per_tok <= 0 ||
                config.num_experts_per_tok > config.num_experts ||
                config.moe_intermediate_size <= 0 ||
                config.shared_expert_intermediate_size <= 0) {
            throw std::runtime_error(
                "invalid Qwen MoE configuration");
        }
    } else if (config.intermediate_size <= 0) {
        throw std::runtime_error(
            "dense Qwen model config intermediate_size must be positive");
    }
    config.attention_output_gate = text.value("attn_output_gate", false);
    config.mtp_num_hidden_layers =
        text.value("mtp_num_hidden_layers", std::int64_t{0});
    config.mtp_use_dedicated_embeddings =
        text.value("mtp_use_dedicated_embeddings", false);
    config.linear_conv_kernel_dim =
        text.value("linear_conv_kernel_dim", std::int64_t{4});
    config.linear_key_head_dim =
        text.value("linear_key_head_dim", std::int64_t{128});
    config.linear_value_head_dim =
        text.value("linear_value_head_dim", std::int64_t{128});
    config.linear_num_key_heads = text.value(
        "linear_num_key_heads", config.num_key_value_heads);
    config.linear_num_value_heads = text.value(
        "linear_num_value_heads", config.num_attention_heads);
    if (config.mtp_num_hidden_layers < 0 ||
            config.linear_conv_kernel_dim <= 0 ||
            config.linear_key_head_dim <= 0 ||
            config.linear_value_head_dim <= 0 ||
            config.linear_num_key_heads <= 0 ||
            config.linear_num_value_heads <= 0) {
        throw std::runtime_error("invalid Qwen model configuration");
    }

    const nlohmann::json empty = nlohmann::json::object();
    const auto& rope_parameters = text.contains("rope_parameters")
        ? text.at("rope_parameters") : empty;
    if (!rope_parameters.is_object()) {
        throw std::runtime_error("Qwen rope_parameters must be an object");
    }
    const auto& full_rope = rope_parameters.contains("full_attention")
        ? rope_parameters.at("full_attention") : empty;
    if (!full_rope.is_object()) {
        throw std::runtime_error(
            "Qwen rope_parameters.full_attention must be an object");
    }

    if (graph.component("vision") == nullptr) return config;
    if (!document.contains("vision_config") ||
            !document.at("vision_config").is_object()) {
        throw std::runtime_error(
            "Qwen grid-ViT requires an object vision_config");
    }
    const auto& vision = document.at("vision_config");
    if (vision.value(
            "hidden_act", std::string("gelu_pytorch_tanh")) !=
            "gelu_pytorch_tanh") {
        throw std::runtime_error(
            "Qwen grid-ViT requires gelu_pytorch_tanh hidden_act");
    }
    GridVisionConfig parsed;
    parsed.hidden_size = vision.at("hidden_size").get<std::int64_t>();
    parsed.intermediate_size =
        vision.at("intermediate_size").get<std::int64_t>();
    parsed.depth = vision.at("depth").get<std::int64_t>();
    parsed.num_heads = vision.at("num_heads").get<std::int64_t>();
    parsed.in_channels = vision.value("in_channels", std::int64_t{3});
    parsed.patch_size = vision.at("patch_size").get<std::int64_t>();
    parsed.temporal_patch_size =
        vision.at("temporal_patch_size").get<std::int64_t>();
    parsed.spatial_merge_size =
        vision.at("spatial_merge_size").get<std::int64_t>();
    parsed.out_hidden_size =
        vision.at("out_hidden_size").get<std::int64_t>();
    parsed.num_position_embeddings =
        vision.at("num_position_embeddings").get<std::int64_t>();
    parsed.rope_theta = vision.value("rope_theta", 10'000.0);
    if (vision.contains("rope_parameters")) {
        if (!vision.at("rope_parameters").is_object()) {
            throw std::runtime_error(
                "vision_config.rope_parameters must be an object");
        }
        parsed.rope_theta = vision.at("rope_parameters").value(
            "rope_theta", parsed.rope_theta);
    }
    parsed.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
    parsed.validate();
    if (parsed.out_hidden_size != config.hidden_size) {
        throw std::runtime_error(
            "Qwen grid-ViT output width must equal text hidden_size");
    }
    const auto token_id = [&](const char* name) {
        if (!document.contains(name) ||
                !document.at(name).is_number_integer()) {
            throw std::runtime_error(
                std::string("Qwen grid-ViT requires ") + name);
        }
        const auto value = document.at(name).get<std::int64_t>();
        if (value < 0 || value >= config.vocab_size) {
            throw std::runtime_error(
                std::string("invalid Qwen grid-ViT ") + name);
        }
        return value;
    };
    config.image_token_id = token_id("image_token_id");
    config.video_token_id = token_id("video_token_id");
    if (*config.image_token_id == *config.video_token_id) {
        throw std::runtime_error(
            "Qwen grid-ViT placeholder token IDs must differ");
    }
    const auto* rope = full_rope.contains("mrope_section")
        ? &full_rope : &rope_parameters;
    if (!rope->contains("mrope_section") ||
            !rope->at("mrope_section").is_array()) {
        throw std::runtime_error(
            "Qwen grid-MRoPE requires rope mrope_section");
    }
    config.mrope_sections =
        rope->at("mrope_section").get<std::vector<std::int64_t>>();
    config.mrope_interleaved =
        rope_parameters.value("mrope_interleaved", false);
    if (config.mrope_sections.size() != 3 ||
            std::any_of(
                config.mrope_sections.begin(), config.mrope_sections.end(),
                [](std::int64_t value) { return value < 0; }) ||
            std::accumulate(
                config.mrope_sections.begin(), config.mrope_sections.end(),
                std::int64_t{0}) != config.rotary_dim / 2) {
        throw std::runtime_error(
            "Qwen grid-MRoPE sections must be three nonnegative values summing to rotary_dim / 2");
    }
    config.grid_vision = parsed;
    return config;
}

} // namespace mfq::models::qwen35
