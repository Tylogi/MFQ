#pragma once

#include "grid_vision.h"
#include "mfq_legacy_tensor_names.h"
#include "mfq_model_graph.h"
#include "models/model_config.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models::qwen35 {

struct Config : ModelConfig {
    MfqLegacyTensorLayout legacy_tensor_layout;
    bool attention_output_gate = false;
    std::int64_t num_experts = 0;
    std::int64_t num_experts_per_tok = 0;
    std::int64_t moe_intermediate_size = 0;
    std::int64_t shared_expert_intermediate_size = 0;
    double routed_scaling_factor = 1.0;
    bool norm_topk_prob = false;
    std::string expert_gating_func = "softmax";
    std::int64_t mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings = false;
    std::int64_t linear_conv_kernel_dim = 4;
    std::int64_t linear_key_head_dim = 128;
    std::int64_t linear_value_head_dim = 128;
    std::int64_t linear_num_key_heads = 0;
    std::int64_t linear_num_value_heads = 0;
    std::vector<std::int64_t> mrope_sections;
    bool mrope_interleaved = false;
    std::optional<GridVisionConfig> grid_vision;
    std::optional<std::int64_t> image_token_id;
    std::optional<std::int64_t> video_token_id;

    std::int64_t linear_k_size() const noexcept {
        return linear_num_key_heads * linear_key_head_dim;
    }
    std::int64_t linear_v_size() const noexcept {
        return linear_num_value_heads * linear_value_head_dim;
    }

    static Config from_json(
        std::string_view payload,
        const MfqModelGraph& graph);

    template <class Source>
    static Config from_source(const Source& source) {
        const auto graph = source.resolved_model_graph();
        if (graph.backbone != "qwen3_5" &&
                graph.backbone != "generic_qwen") {
            throw std::runtime_error(
                "Qwen3.5 loading requires a Qwen-compatible model graph");
        }
        return from_json(source.model_config_json(), graph);
    }
};

} // namespace mfq::models::qwen35
