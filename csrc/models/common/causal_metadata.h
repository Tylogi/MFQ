#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mfq::models {

struct CausalLmMetadata {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t rotary_dim = 0;
    int64_t num_experts = 0;
    int64_t hc_mult = 1;
    double rope_base = 0.0;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 0.0;
    double hc_eps = 1e-6;
    double final_logit_softcapping = 0.0;
    double embedding_scale = 1.0;
    bool tie_word_embeddings = false;
    bool rope_interleaved = false;
    bool multi_axis_positions = false;
    bool flash_next = false;
    bool gemma4 = false;
    bool decode_graph_double_warmup = false;
    std::string model_type;
    std::vector<std::string> layer_types;
};

struct CausalState {
    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    int64_t speculative_start = -1;
    int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;
};

} // namespace mfq::models
