#include "models/deepseek_v4/config.h"
#include "models/glm5_next/config.h"
#include "models/qwen4_exp/config.h"
#include "models/gemma4/config.h"
#include "models/glm_dsa/config.h"
#include "models/nemotron_h/config.h"
#include "models/minicpmo45/config.h"
#include "models/qwen35/config.h"
#include "models/common/model_config.h"

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

        const auto nope = mfq::models::ModelConfig::from_json(R"({
            "model_type":"nemotron_h","vocab_size":128,"hidden_size":64,
            "num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":2,"max_position_embeddings":4096,
            "head_dim":16,"positional_encoding":"none"})");
        require(
            nope.positional_encoding == mfq::models::PositionalEncoding::none &&
                nope.rotary_dim == 0,
            "NoPE model config retained a rotary dimension");
        rejected = false;
        try {
            static_cast<void>(mfq::models::ModelConfig::from_json(R"({
                "model_type":"nemotron_h","vocab_size":128,"hidden_size":64,
                "num_hidden_layers":2,"num_attention_heads":4,
                "num_key_value_heads":2,"max_position_embeddings":4096,
                "head_dim":16,"positional_encoding":"absolute"})"));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "unknown positional_encoding was accepted");

        auto released = nlohmann::json::parse(R"({
            "model_type":"nemotron_h_omni","tie_word_embeddings":false,
            "llm_config":{"model_type":"nemotron_h","vocab_size":128,
            "hidden_size":64,"intermediate_size":128,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":16,"max_position_embeddings":4096,
            "layer_norm_epsilon":1e-5,"layers_block_type":["mamba","moe","attention"],
            "mamba_num_heads":4,"mamba_head_dim":8,"ssm_state_size":32,
            "n_groups":2,"conv_kernel":4,"use_conv_bias":true,
            "mamba_hidden_act":"silu","mamba_ssm_cache_dtype":"float32",
            "time_step_limit":[0.001,0.1],"n_routed_experts":4,
            "num_experts_per_tok":2,"moe_intermediate_size":8,"moe_latent_size":16,
            "moe_shared_expert_intermediate_size":12,"mlp_hidden_act":"relu2",
            "norm_topk_prob":true,"routed_scaling_factor":5.0,"n_shared_experts":1,
            "n_group":1,"topk_group":1,"num_nextn_predict_layers":1,
            "mtp_layers_block_type":["attention","moe"]}})");
        const auto predictor =
            mfq::models::nemotron_h::Config::from_json(released.dump(), mfq::MfqModelGraph{});
        require(predictor.num_nextn_predict_layers == 1 &&
                    predictor.mtp_mixers ==
                        std::vector<mfq::models::nemotron_h::MixerKind>{
                            mfq::models::nemotron_h::MixerKind::full_attention,
                            mfq::models::nemotron_h::MixerKind::moe},
                "Nemotron-H MTP schedule was not normalized");
        // The predictor stage owns attention history only; a mamba sub-block
        // and a second stage have no runtime representation.
        for (const auto& [field, value] :
             std::vector<std::pair<std::string, nlohmann::json>>{
                 {"mtp_layers_block_type", nlohmann::json::array({"mamba", "moe"})},
                 {"num_nextn_predict_layers", 2}}) {
            auto broken = released;
            broken["llm_config"][field] = value;
            rejected = false;
            try {
                static_cast<void>(
                    mfq::models::nemotron_h::Config::from_json(broken.dump(), mfq::MfqModelGraph{}));
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            require(rejected, "invalid Nemotron-H MTP schedule was accepted");
        }
        auto orphaned = released;
        orphaned["llm_config"].erase("num_nextn_predict_layers");
        rejected = false;
        try {
            static_cast<void>(
                mfq::models::nemotron_h::Config::from_json(orphaned.dump(), mfq::MfqModelGraph{}));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "orphaned Nemotron-H MTP schedule was accepted");

        auto flash = nlohmann::json::parse(R"({
            "model_type":"qwen4_exp_text","vocab_size":128,"hidden_size":32,
            "num_hidden_layers":2,"max_position_embeddings":512,
            "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
            "full_attention_interval":2,"hc_count":4,"hc_lowrank":8,
            "layer_types":["linear_attention","full_attention"],
            "linear_num_key_heads":2,"linear_num_value_heads":4,
            "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_conv_kernel_dim":4,
            "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,
            "shared_expert_intermediate_size":16,"indexer_n_heads":2,
            "indexer_head_dim":8,"indexer_kv_heads":1,"indexer_compress_ratio":2,
            "indexer_budget":8,"ple_conv_kernel_size":4,"ngram_size":2,
            "heads_per_ngram":4,"split_ngram_parts":2,"ngram_vocab_size_base":16,
            "ple_embed_dim":32,"mtp_num_hidden_layers":0,"mtp":{"num_hidden_layers":1}})");
        require(mfq::models::qwen4_exp::Config::parse(flash).predictor_layers == 0,
                "disabled predictor retained in source config must not activate MTP");
        flash["mtp_num_hidden_layers"] = 1;
        require(mfq::models::qwen4_exp::Config::parse(flash).predictor_layers == 1,
                "matching active predictor configuration was rejected");
        for (auto count : {-1, 2}) {
            flash["mtp_num_hidden_layers"] = count;
            rejected = false;
            try {
                static_cast<void>(mfq::models::qwen4_exp::Config::parse(flash));
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            require(rejected, "invalid active predictor configuration was accepted");
        }

        for (const auto* wrapper : {"text_config", "language_config"}) {
            auto definition = nlohmann::json{{wrapper, {
                {"max_position_embeddings", 128},
                {"rope_scaling", {{"type", "yarn"}, {"factor", 4.0},
                    {"original_max_position_embeddings", 64}, {"beta_fast", 16.0}, {"beta_slow", 2.0}}}
            }}}.dump();
            const auto scaling = mfq::models::extend_qwen_context(definition, "qwen4_exp", 256);
            require(scaling.original_context == 64 && scaling.factor == 4.0 &&
                    scaling.beta_fast == 16.0 && scaling.beta_slow == 2.0,
                "shared YaRN parser lost scaling metadata");
            require(nlohmann::json::parse(definition).at(wrapper).at("max_position_embeddings") == 256,
                "shared YaRN parser did not extend the text context");
            rejected = false;
            try {
                (void)mfq::models::extend_qwen_context(definition, "qwen4_exp", 257);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "shared YaRN parser accepted a context beyond its configured factor");
        }
        std::string native_definition = R"({"max_position_embeddings":64})";
        require(!mfq::models::extend_qwen_context(native_definition, "qwen3_5", 32).enabled(),
            "short context unexpectedly enabled YaRN");
        require(nlohmann::json::parse(native_definition).at("max_position_embeddings") == 64,
            "short context changed the model's native capacity");
        std::string nested_rope = R"({"max_position_embeddings":64,
            "rope_parameters":{"full_attention":{"rope_type":"default","factor":2}}})";
        require(mfq::models::extend_qwen_context(nested_rope, "qwen3_5", 128).factor == 2.0,
            "shared YaRN parser lost nested full-attention scaling");
        for (const auto* invalid : {
                R"({"max_position_embeddings":0})",
                R"({"max_position_embeddings":64,"rope_parameters":{"rope_type":"linear"}})",
                R"({"max_position_embeddings":64,"rope_parameters":{"factor":0.5}})"}) {
            std::string definition = invalid;
            rejected = false;
            try {
                (void)mfq::models::extend_qwen_context(definition, "qwen3_5", 128);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "shared YaRN parser accepted unsupported positional metadata");
        }
        auto vision = nlohmann::json::parse(R"({
            "image_token_id":5,"video_token_id":6,"vision_config":{
                "hidden_size":8,"intermediate_size":16,"depth":1,"num_heads":2,
                "patch_size":2,"temporal_patch_size":1,"out_hidden_size":32,
                "num_position_embeddings":16,"rope_parameters":{"rope_theta":20000}}})");
        auto text = mfq::models::qwen4_exp::Config{};
        text.hidden = 32;
        text.vocab = 128;
        text.sections = {1, 1, 2};
        const auto grid = mfq::models::qwen4_exp::VisionConfig::from_json(vision.dump(), text);
        require(grid.grid.in_channels == 3 && grid.grid.spatial_merge_size == 2 &&
                grid.grid.rope_theta == 20000 && grid.grid.layer_norm_eps == 1e-6 &&
                grid.image_token_id == 5 && grid.video_token_id == 6,
            "shared Qwen4 vision parser lost defaults or placeholder IDs");
        for (const auto* invalid : {"hidden_act", "deepstack_visual_indexes", "out_hidden_size",
                                   "num_position_embeddings", "video_token_id"}) {
            auto invalid_vision = vision;
            if (std::string(invalid) == "hidden_act") invalid_vision["vision_config"][invalid] = "gelu";
            else if (std::string(invalid) == "deepstack_visual_indexes")
                invalid_vision["vision_config"][invalid] = std::vector<int>{0};
            else if (std::string(invalid) == "video_token_id") invalid_vision[invalid] = 5;
            else invalid_vision["vision_config"][invalid] = 17;
            rejected = false;
            try {
                (void)mfq::models::qwen4_exp::VisionConfig::from_json(invalid_vision.dump(), text);
            } catch (const std::exception&) {
                rejected = true;
            }
            require(rejected, "shared Qwen4 vision parser accepted unsupported component metadata");
        }

        std::cout << "model config tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model config tests failed: " << error.what() << '\n';
        return 1;
    }
}
