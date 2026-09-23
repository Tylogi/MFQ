#include "models/deepseek_v4.h"
#include "models/flash_next.h"
#include "models/gemma4.h"
#include "models/glm_dsa.h"
#include "models/minicpmo45.h"
#include "models/qwen35.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        const auto deepseek = mfq::models::deepseek_v4::Config::from_json(R"({
            "model_type":"deepseek_v4","vocab_size":128,"hidden_size":64,
            "num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":1,"max_position_embeddings":4096,
            "head_dim":16,"compress_ratios":[2,4,8],"num_hash_layers":1,
            "compress_rope_theta":160000,
            "rope_scaling":{"original_max_position_embeddings":4096,
            "factor":2,"beta_fast":16,"beta_slow":2}})");
        require(
            deepseek.compress_ratios == std::vector<std::int64_t>({2, 4}) &&
                deepseek.hash_layer_count == 1 &&
                deepseek.compress_rope_base == 160000.0 &&
                deepseek.rope_original_positions == 4096 &&
                deepseek.rope_factor == 2.0 &&
                deepseek.rope_beta_fast == 16.0 &&
                deepseek.rope_beta_slow == 2.0,
            "DeepSeek V4 config was not normalized");

        const auto gemma = mfq::models::gemma4::Config::from_json(R"({
            "model_type":"gemma4","vocab_size":128,"hidden_size":64,
            "intermediate_size":192,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,
            "max_position_embeddings":4096,"head_dim":16,
            "global_head_dim":32,"num_global_key_value_heads":4,
            "sliding_window":1024,"attention_k_eq_v":true,
            "num_experts":8,"num_experts_per_tok":2,
            "moe_intermediate_size":48,
            "full_attention":{"rope_theta":20000,
            "partial_rotary_factor":0.5},
            "sliding_attention":{"rope_theta":5000}})");
        require(
            gemma.global_head_dim == 32 &&
                gemma.num_global_key_value_heads == 4 &&
                gemma.sliding_window == 1024 &&
                gemma.rope_base == 20000.0 && gemma.rotary_dim == 8 &&
                gemma.sliding_rope_base == 5000.0 &&
                gemma.attention_key_equals_value &&
                gemma.num_experts == 8 &&
                gemma.num_experts_per_tok == 2 &&
                gemma.moe_intermediate_size == 48,
            "Gemma4 config was not normalized");

        const auto glm = mfq::models::glm_dsa::Config::from_json(R"({
            "model_type":"glm_dsa","vocab_size":128,"hidden_size":64,
            "intermediate_size":192,"num_hidden_layers":3,
            "num_attention_heads":4,"num_key_value_heads":4,
            "max_position_embeddings":4096,"head_dim":16,
            "indexer_types":["full","shared","full"],
            "mlp_layer_types":["dense","sparse","dense"]})");
        require(
            glm.indexer_types.size() == 3 &&
                glm.indexer_types[1] == "shared" &&
                glm.mlp_layer_types[1] == "sparse",
            "GLM DSA config was not normalized");

        bool rejected = false;
        try {
            static_cast<void>(mfq::models::glm_dsa::Config::from_json(R"({
                "model_type":"glm_dsa","vocab_size":128,"hidden_size":64,
                "intermediate_size":192,"num_hidden_layers":3,
                "num_attention_heads":4,"num_key_value_heads":4,
                "max_position_embeddings":4096,"head_dim":16,
                "indexer_types":["shared","full","full"],
                "mlp_layer_types":["dense","dense","dense"]})"));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "invalid GLM DSA schedule was accepted");

        mfq::MfqModelGraph graph;
        graph.backbone = "qwen3_5";
        const auto qwen = mfq::models::qwen35::Config::from_json(R"({
            "model_type":"qwen3_5","text_config":{
            "model_type":"qwen3_5_text","vocab_size":128,
            "hidden_size":64,"intermediate_size":192,
            "num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":2,"max_position_embeddings":4096,
            "head_dim":16,"attn_output_gate":true,
            "layer_types":["full_attention","linear_attention"],
            "rope_parameters":{"full_attention":{"rope_theta":10000.0,
            "partial_rotary_factor":0.5}}}})", graph);
        require(
            qwen.model_type == "qwen3_5" && qwen.hidden_size == 64 &&
                qwen.head_dim == 16 && qwen.layer_types.size() == 2 &&
                qwen.layer_types[1] == "linear_attention" &&
                qwen.attention_output_gate && qwen.rope_base == 10000.0 &&
                qwen.rotary_dim == 8 && qwen.linear_k_size() == 256 &&
                qwen.linear_v_size() == 512 && !qwen.grid_vision,
            "Qwen3.5 config was not normalized");

        const auto qwen_moe = mfq::models::qwen35::Config::from_json(R"({
            "model_type":"qwen3_5_moe","text_config":{
            "model_type":"qwen3_5_moe_text","vocab_size":128,
            "hidden_size":64,"num_hidden_layers":2,
            "num_attention_heads":4,"num_key_value_heads":2,
            "max_position_embeddings":4096,"head_dim":16,
            "num_experts":8,"num_experts_per_tok":2,
            "moe_intermediate_size":48,
            "shared_expert_intermediate_size":96,
            "layer_types":["full_attention","linear_attention"]}})", graph);
        require(
            qwen_moe.intermediate_size == 0 &&
                qwen_moe.num_experts == 8 &&
                qwen_moe.num_experts_per_tok == 2 &&
                qwen_moe.moe_intermediate_size == 48 &&
                qwen_moe.shared_expert_intermediate_size == 96,
            "Qwen3.5 MoE config was not normalized without intermediate_size");

        const auto minicpm = mfq::models::minicpmo45::Config::from_json(R"({
            "version":"4.5","model_type":"minicpmo45","vocab_size":128,
            "hidden_size":4096,"intermediate_size":12288,
            "num_hidden_layers":36,"num_attention_heads":32,
            "num_key_value_heads":8,"max_position_embeddings":4096,
            "head_dim":128,"hidden_act":"silu","attention_bias":false,
            "use_sliding_window":false})");
        require(
            minicpm.version == "4.5" && !minicpm.attention_bias &&
                !minicpm.use_sliding_window,
            "MiniCPM-o config was not normalized");

        std::cout << "model config tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model config tests failed: " << error.what() << '\n';
        return 1;
    }
}
