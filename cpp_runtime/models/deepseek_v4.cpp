#include "models/include/deepseek_v4.h"

#include "nlohmann/json.hpp"

#include <cmath>
#include <stdexcept>

namespace mfq::models::deepseek_v4 {

Config Config::from_json(std::string_view payload) {
    const auto root = nlohmann::json::parse(
        payload.begin(), payload.end(), nullptr, false);
    if (!root.is_object()) {
        throw std::runtime_error("DeepSeek V4 model config must be an object");
    }
    const auto& text = root.contains("text_config")
        ? root.at("text_config") : root;
    if (!text.is_object()) {
        throw std::runtime_error("DeepSeek V4 text_config must be an object");
    }

    Config config;
    static_cast<ModelConfig&>(config) = ModelConfig::from_json(payload);
    config.compress_ratios = text.value(
        "compress_ratios", std::vector<std::int64_t>{});
    if (config.compress_ratios.size() <
            static_cast<std::size_t>(config.num_hidden_layers)) {
        throw std::runtime_error(
            "DeepSeek V4 compress_ratios is shorter than num_hidden_layers");
    }
    config.compress_ratios.resize(
        static_cast<std::size_t>(config.num_hidden_layers));
    config.hash_layer_count = text.value(
        "num_hash_layers", text.value("n_hash_layers", std::int64_t{0}));
    config.compress_rope_base = text.value("compress_rope_theta", 0.0);
    const auto rope = text.value(
        "rope_scaling", nlohmann::json::object());
    if (!rope.is_object()) {
        throw std::runtime_error("DeepSeek V4 rope_scaling must be an object");
    }
    config.rope_original_positions = rope.value(
        "original_max_position_embeddings", std::int64_t{0});
    config.rope_factor = rope.value("factor", 1.0);
    config.rope_beta_fast = rope.value("beta_fast", 32.0);
    config.rope_beta_slow = rope.value("beta_slow", 1.0);
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
    config.qk_nope_head_dim =
        text.value("qk_nope_head_dim", std::int64_t{0});
    config.qk_rope_head_dim =
        text.value("qk_rope_head_dim", std::int64_t{0});
    config.v_head_dim = text.value("v_head_dim", std::int64_t{0});
    config.index_head_dim = text.value("index_head_dim", std::int64_t{0});
    config.index_n_heads = text.value("index_n_heads", std::int64_t{0});
    config.index_topk = text.value("index_topk", std::int64_t{0});
    config.hc_mult = text.value("hc_mult", std::int64_t{1});
    config.hc_sinkhorn_iters =
        text.value("hc_sinkhorn_iters", std::int64_t{0});
    config.hc_eps = text.value("hc_eps", 1e-6);
    config.o_groups = text.value("o_groups", std::int64_t{1});
    config.o_lora_rank = text.value("o_lora_rank", std::int64_t{0});
    config.swiglu_limit = text.value("swiglu_limit", 0.0);
    config.routed_scaling_factor =
        text.value("routed_scaling_factor", 1.0);
    config.norm_topk_prob = text.value("norm_topk_prob", false);
    config.scoring_func = text.value("scoring_func", std::string("softmax"));
    if (config.hash_layer_count < 0 ||
            !std::isfinite(config.compress_rope_base) ||
            !std::isfinite(config.rope_factor) || config.rope_factor <= 0.0 ||
            !std::isfinite(config.rope_beta_fast) ||
            !std::isfinite(config.rope_beta_slow)) {
        throw std::runtime_error("invalid DeepSeek V4 model configuration");
    }
    return config;
}

} // namespace mfq::models::deepseek_v4
