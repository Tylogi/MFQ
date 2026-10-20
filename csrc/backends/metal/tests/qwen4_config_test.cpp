#include "models/flash_next/mlx_qwen4_causal_lm.h"
#include "nlohmann/json.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

constexpr const char* kConfig = R"JSON(
{
  "model_type": "qwen4_exp",
  "text_config": {
    "model_type": "qwen4_exp_text",
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 4,
    "max_position_embeddings": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "full_attention_interval": 4,
    "hc_count": 2,
    "hc_lowrank": 4,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention"
    ],
    "partial_rotary_factor": 0.5,
    "rope_parameters": {"rope_theta": 10000000.0},
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 4,
    "linear_value_head_dim": 4,
    "linear_conv_kernel_dim": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 8,
    "shared_expert_intermediate_size": 8,
    "indexer_n_heads": 2,
    "indexer_head_dim": 4,
    "indexer_compress_ratio": 2,
    "indexer_budget": 8,
    "indexer_kv_heads": 1,
    "ple_conv_kernel_size": 4,
    "ple_embed_dim": 16,
    "ple_layer_ids": [2],
    "ngram_size": 3,
    "ngram_vocab_size_base": 100,
    "heads_per_ngram": 4,
    "split_ngram_parts": 2,
    "hidden_act": "silu",
    "output_gate_type": "sigmoid",
    "attention_bias": false,
    "rms_norm_eps": 0.000001,
    "norm_topk_prob": true,
    "tie_word_embeddings": false,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "eos_token_id": 7
  }
}
)JSON";

void test_config_contract() {
    const auto config = mfq::metal::Qwen4Config::from_json(kConfig);
    require(config.model_type == "qwen4_exp", "outer model type mismatch");
    require(
        config.text_model_type == "qwen4_exp_text",
        "text model type mismatch");
    require(config.hidden_size == 16, "hidden size mismatch");
    require(config.num_hidden_layers == 4, "layer count mismatch");
    require(config.rotary_dim == 2, "rotary dimension mismatch");
    require(config.layer_types.size() == 4, "layer schedule mismatch");
    require(config.ple_layer_ids == std::vector<std::int64_t>{2},
        "PLE schedule mismatch");
    require(config.eos_token_id == 7, "EOS token mismatch");
    require(config.mtp_num_hidden_layers == 1, "MTP layer count mismatch");
    require(!config.output_gate_silu, "sigmoid output gate mismatch");
}

void test_invalid_schedule_is_rejected() {
    std::string invalid(kConfig);
    const auto marker = invalid.find("\"full_attention\"");
    require(marker != std::string::npos, "test schedule marker is absent");
    invalid.replace(marker, std::string("\"full_attention\"").size(),
        "\"linear_attention\"");
    try {
        (void)mfq::metal::Qwen4Config::from_json(invalid);
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("invalid Qwen4 layer schedule was accepted");
}

void test_vision_config() {
    auto root = nlohmann::json::parse(kConfig);
    root["image_token_id"] = 5;
    root["video_token_id"] = 6;
    root["vision_config"] = {
        {"hidden_size", 8}, {"intermediate_size", 16}, {"depth", 1},
        {"num_heads", 2}, {"patch_size", 2}, {"temporal_patch_size", 1},
        {"spatial_merge_size", 2}, {"out_hidden_size", 16},
        {"num_position_embeddings", 16}};
    const auto config = mfq::metal::Qwen4Config::from_json(root.dump());
    require(config.vision && config.vision->depth == 1, "vision config missing");
    require(config.image_token_id == 5 && config.video_token_id == 6, "media IDs mismatch");
    for (const auto invalid : {"width", "token"}) {
        auto other = root;
        if (std::string_view(invalid) == "width") other["vision_config"]["out_hidden_size"] = 32;
        else other["image_token_id"] = 32;
        bool rejected = false;
        try { (void)mfq::metal::Qwen4Config::from_json(other.dump()); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "invalid vision geometry or media ID accepted");
    }
}

} // namespace

int main() {
    try {
        test_config_contract();
        test_invalid_schedule_is_rejected();
        test_vision_config();
        std::cout << "Qwen4-Exp config contract test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
