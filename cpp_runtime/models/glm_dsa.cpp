#include "models/include/glm_dsa.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <stdexcept>

namespace mfq::models::glm_dsa {

Config Config::from_json(std::string_view payload) {
    const auto root = nlohmann::json::parse(
        payload.begin(), payload.end(), nullptr, false);
    if (!root.is_object()) {
        throw std::runtime_error("GLM DSA model config must be an object");
    }
    const auto& text = root.contains("text_config")
        ? root.at("text_config") : root;
    if (!text.is_object()) {
        throw std::runtime_error("GLM DSA text_config must be an object");
    }

    Config config;
    static_cast<ModelConfig&>(config) = ModelConfig::from_json(payload);
    config.indexer_types = text.value(
        "indexer_types", std::vector<std::string>{});
    config.mlp_layer_types = text.value(
        "mlp_layer_types", std::vector<std::string>{});
    const auto index_topk_frequency =
        text.value("index_topk_freq", std::int64_t{1});
    const auto index_skip_topk_offset =
        text.value("index_skip_topk_offset", std::int64_t{0});
    const auto first_dense_layers =
        text.value("first_k_dense_replace", std::int64_t{0});
    const auto moe_layer_frequency =
        text.value("moe_layer_freq", std::int64_t{1});
    if (config.indexer_types.empty()) {
        config.indexer_types.reserve(
            static_cast<std::size_t>(config.num_hidden_layers));
        const auto frequency = std::max<std::int64_t>(
            1, index_topk_frequency);
        for (std::int64_t layer = 0;
             layer < config.num_hidden_layers; ++layer) {
            const auto phase = std::max<std::int64_t>(
                layer - index_skip_topk_offset + 1, 0);
            config.indexer_types.push_back(
                phase % frequency == 0 ? "full" : "shared");
        }
    }
    if (config.mlp_layer_types.empty()) {
        config.mlp_layer_types.reserve(
            static_cast<std::size_t>(config.num_hidden_layers));
        for (std::int64_t layer = 0;
             layer < config.num_hidden_layers; ++layer) {
            const bool sparse = layer >= first_dense_layers &&
                layer % std::max<std::int64_t>(1, moe_layer_frequency) == 0;
            config.mlp_layer_types.push_back(sparse ? "sparse" : "dense");
        }
    }
    if (config.indexer_types.size() !=
            static_cast<std::size_t>(config.num_hidden_layers) ||
            config.mlp_layer_types.size() !=
            static_cast<std::size_t>(config.num_hidden_layers)) {
        throw std::runtime_error(
            "GLM DSA layer schedules do not match num_hidden_layers");
    }
    bool have_full_indexer = false;
    for (std::size_t layer = 0; layer < config.indexer_types.size(); ++layer) {
        const auto& indexer = config.indexer_types[layer];
        if (indexer == "full") {
            have_full_indexer = true;
        } else if (indexer != "shared" || !have_full_indexer) {
            throw std::runtime_error(
                "invalid GLM DSA indexer schedule at layer " +
                std::to_string(layer));
        }
        const auto& mlp = config.mlp_layer_types[layer];
        if (mlp != "dense" && mlp != "sparse") {
            throw std::runtime_error(
                "invalid GLM DSA MLP schedule at layer " +
                std::to_string(layer));
        }
    }

    config.num_experts = text.value(
        "num_experts", text.value("n_routed_experts", std::int64_t{0}));
    config.num_experts_per_tok = text.value(
        "num_experts_per_tok", text.value("top_k_experts", std::int64_t{0}));
    config.moe_intermediate_size =
        text.value("moe_intermediate_size", std::int64_t{0});
    config.shared_expert_count =
        text.value("n_shared_experts", std::int64_t{0});
    config.shared_expert_intermediate_size =
        text.value("shared_expert_intermediate_size", std::int64_t{0});
    config.q_lora_rank = text.value("q_lora_rank", std::int64_t{0});
    config.kv_lora_rank = text.value("kv_lora_rank", std::int64_t{0});
    config.qk_head_dim = text.value("qk_head_dim", std::int64_t{0});
    config.qk_nope_head_dim =
        text.value("qk_nope_head_dim", std::int64_t{0});
    config.qk_rope_head_dim =
        text.value("qk_rope_head_dim", std::int64_t{0});
    config.v_head_dim = text.value("v_head_dim", std::int64_t{0});
    config.index_head_dim = text.value("index_head_dim", std::int64_t{0});
    config.index_n_heads = text.value("index_n_heads", std::int64_t{0});
    config.index_topk = text.value("index_topk", std::int64_t{0});
    config.expert_group_count = text.value("n_group", std::int64_t{1});
    config.selected_group_count = text.value("topk_group", std::int64_t{1});
    config.routed_scaling_factor =
        text.value("routed_scaling_factor", 1.0);
    config.norm_topk_prob = text.value("norm_topk_prob", false);
    config.swiglu_limit = text.value("swiglu_limit", 0.0);
    config.scoring_func = text.value("scoring_func", std::string("softmax"));
    config.hidden_act = text.value("hidden_act", std::string{});
    config.topk_method = text.value("topk_method", std::string{});
    config.attention_bias = text.value("attention_bias", false);
    config.rope_interleave = text.value("rope_interleave", true);
    config.indexer_rope_interleave =
        text.value("indexer_rope_interleave", true);
    return config;
}

} // namespace mfq::models::glm_dsa
