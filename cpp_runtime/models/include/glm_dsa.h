#pragma once

#include "model_config.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models::glm_dsa {

struct Config : ModelConfig {
    std::vector<std::string> indexer_types;
    std::vector<std::string> mlp_layer_types;
    std::int64_t num_experts = 0;
    std::int64_t num_experts_per_tok = 0;
    std::int64_t moe_intermediate_size = 0;
    std::int64_t shared_expert_count = 0;
    std::int64_t shared_expert_intermediate_size = 0;
    std::int64_t q_lora_rank = 0;
    std::int64_t kv_lora_rank = 0;
    std::int64_t qk_head_dim = 0;
    std::int64_t qk_nope_head_dim = 0;
    std::int64_t qk_rope_head_dim = 0;
    std::int64_t v_head_dim = 0;
    std::int64_t index_head_dim = 0;
    std::int64_t index_n_heads = 0;
    std::int64_t index_topk = 0;
    std::int64_t expert_group_count = 1;
    std::int64_t selected_group_count = 1;
    double routed_scaling_factor = 1.0;
    bool norm_topk_prob = false;
    double swiglu_limit = 0.0;
    std::string scoring_func = "softmax";
    std::string hidden_act;
    std::string topk_method;
    bool attention_bias = false;
    bool rope_interleave = true;
    bool indexer_rope_interleave = true;

    static Config from_json(std::string_view payload);

    template <class Source>
    static Config from_source(const Source& source) {
        return from_json(source.model_config_json());
    }
};

} // namespace mfq::models::glm_dsa
