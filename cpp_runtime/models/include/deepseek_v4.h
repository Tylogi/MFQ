#pragma once

#include "model_config.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models::deepseek_v4 {

struct Config : ModelConfig {
    std::vector<std::int64_t> compress_ratios;
    std::int64_t hash_layer_count = 0;
    double compress_rope_base = 0.0;
    std::int64_t rope_original_positions = 0;
    double rope_factor = 1.0;
    double rope_beta_fast = 32.0;
    double rope_beta_slow = 1.0;
    std::int64_t num_experts = 0;
    std::int64_t num_experts_per_tok = 0;
    std::int64_t moe_intermediate_size = 0;
    std::int64_t shared_expert_count = 0;
    std::int64_t shared_expert_intermediate_size = 0;
    std::int64_t q_lora_rank = 0;
    std::int64_t kv_lora_rank = 0;
    std::int64_t qk_nope_head_dim = 0;
    std::int64_t qk_rope_head_dim = 0;
    std::int64_t v_head_dim = 0;
    std::int64_t index_head_dim = 0;
    std::int64_t index_n_heads = 0;
    std::int64_t index_topk = 0;
    std::int64_t hc_mult = 1;
    std::int64_t hc_sinkhorn_iters = 0;
    double hc_eps = 1e-6;
    std::int64_t o_groups = 1;
    std::int64_t o_lora_rank = 0;
    double swiglu_limit = 0.0;
    double routed_scaling_factor = 1.0;
    bool norm_topk_prob = false;
    std::string scoring_func = "softmax";

    static Config from_json(std::string_view payload);

    template <class Source>
    static Config from_source(const Source& source) {
        return from_json(source.model_config_json());
    }
};

} // namespace mfq::models::deepseek_v4
