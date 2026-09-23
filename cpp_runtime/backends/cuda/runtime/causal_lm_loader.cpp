#include "causal_lm.h"

#include "cuda_transformer_loader.h"
#include "moe_expert_cache.h"
#include "../models/registry.h"
#include "../models/deepseek_v4/deepseek_v4_causal_lm.h"
#include "../models/deepseek_v41/deepseek_v41_causal_lm.h"
#include "../models/glm5_next/causal_lm.h"
#include "../models/qwen4_exp/causal_lm.h"
#include "../models/gemma4/gemma4_causal_lm.h"
#include "../models/glm_dsa/glm_dsa_causal_lm.h"
#include "../models/qwen35/qwen35_causal_lm.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <utility>

namespace {

float round_to_bfloat16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

template <typename Model, typename Loader>
void load_model_blocks(Model& model, Loader&& load) {
    const auto layer_count = model.num_hidden_layers();
    model.blocks.reserve(static_cast<std::size_t>(layer_count));
    for (int layer = 0; layer < layer_count; ++layer) {
        const int device = g_layer_placement.device_for_layer(layer);
        const bool cpu_offloaded = layer < g_dense_cpu_layer_count;
        g_loading_cpu_layer = cpu_offloaded;
        g_layer_placement.load_device = device;
        MfqCudaGuard layer_guard(device);
        const std::string type(model.layer_type(layer));
        std::cerr << "loading layer " << layer << ' ' << type << ' '
                  << (cpu_offloaded ? "CPU" : "CUDA") << std::endl;
        auto block = load(layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = cpu_offloaded;
        model.blocks.push_back(std::move(block));
    }
}

} // namespace

template <mfq::cuda::CudaBackbone Kind>
mfq::cuda::CausalLmFor<Kind> mfq::cuda::load_causal_lm(
        const std::string& model_path,
        const std::string& config_path,
        int64_t context_size_override,
        bool load_blocks,
        bool defer_moe_cache_finalize,
        std::shared_ptr<const mfq::ModelSource> model_source) {
    CausalLmFor<Kind> model;
    model.source = model_source
        ? std::move(model_source)
        : mfq::open_model_source(model_path);
    const auto& source = *model.source;
    validate_model_source(source);
    model.graph = source.resolved_model_graph();
    model.plan = cuda_model_plan(model.graph);
    MFQ_RUNTIME_CHECK(
        model.plan.backbone == Kind,
        "loaded CUDA backbone does not match the requested causal LM type");

    const auto payload = load_model_config_json(source, config_path);
    if constexpr (Kind == CudaBackbone::generic_qwen) {
        model.config = qwen35::Config::from_json(payload, model.graph);
        model.config.legacy_tensor_layout =
            source.legacy_tensor_compatibility().layout;
        if (model.graph.component("vision") != nullptr &&
                model.plan.vision != CudaVisionAdapter::grid_vit) {
            throw std::runtime_error(
                "Qwen CUDA vision requires grid_vit/grid_vision.v1/grid_mrope");
        }
    } else if constexpr (Kind == CudaBackbone::minicpmo45) {
        model.config =
            mfq::models::minicpmo45::Config::from_json(payload);
        if (model.config.model_type.empty()) {
            model.config.model_type = model.graph.architecture;
        }
        model.config.rotary_dim = model.config.head_dim;
        model.config.layer_types.assign(
            static_cast<std::size_t>(model.config.num_hidden_layers),
            "full_attention");
        if (model.config.hidden_size != 4096 ||
                model.config.intermediate_size != 12288 ||
                model.config.num_hidden_layers != 36 ||
                model.config.num_attention_heads != 32 ||
                model.config.num_key_value_heads != 8 ||
                model.config.head_dim != 128 ||
                model.config.hidden_act != "silu" ||
                model.config.attention_bias ||
                model.config.use_sliding_window) {
            throw std::runtime_error(
                "unsupported MiniCPM-o 4.5 Qwen3 CUDA configuration");
        }
    } else if constexpr (Kind == CudaBackbone::minicpmo_tts) {
        model.config = mfq::models::ModelConfig::from_json(payload);
    } else if constexpr (Kind == CudaBackbone::gemma4) {
        model.config = gemma4::Config::from_json(payload);
        model.embed_scale = round_to_bfloat16(static_cast<float>(
            std::sqrt(static_cast<double>(model.config.hidden_size))));
    } else if constexpr (Kind == CudaBackbone::glm_dsa) {
        model.config = glm_dsa::Config::from_json(payload);
        model.config.layer_types.assign(
            static_cast<std::size_t>(model.config.num_hidden_layers),
            "glm_dsa");
        model.config.rotary_dim = model.config.qk_rope_head_dim;
        const auto& config = model.config;
        if (config.q_lora_rank <= 0 || config.kv_lora_rank <= 0 ||
                config.qk_nope_head_dim <= 0 ||
                config.qk_rope_head_dim <= 0 || config.v_head_dim <= 0 ||
                config.index_head_dim <= 0 || config.index_n_heads <= 0 ||
                config.index_topk <= 0 || config.num_experts <= 0 ||
                config.num_experts_per_tok <= 0 ||
                config.num_attention_heads != 64 ||
                config.num_key_value_heads != 64 ||
                config.kv_lora_rank != 512 ||
                config.qk_nope_head_dim != 192 ||
                config.qk_rope_head_dim != 64 || config.v_head_dim != 256 ||
                config.index_head_dim != 128 || config.index_n_heads != 32 ||
                config.index_topk != 2048 ||
                config.qk_head_dim !=
                    config.qk_nope_head_dim + config.qk_rope_head_dim ||
                config.attention_bias || !config.rope_interleave ||
                !config.indexer_rope_interleave ||
                config.hidden_act != "silu" ||
                config.expert_group_count != 1 ||
                config.selected_group_count != 1 ||
                config.shared_expert_count != 1 ||
                config.scoring_func != "sigmoid" ||
                config.topk_method != "noaux_tc") {
            throw std::runtime_error(
                "unsupported GLM DSA CUDA configuration");
        }
    } else if constexpr (Kind == CudaBackbone::glm5_next) {
        model.config =
            mfq::models::flash_next::GlmConfig::from_json(payload);
    } else if constexpr (Kind == CudaBackbone::qwen4_exp) {
        model.config =
            mfq::models::flash_next::QwenConfig::from_json(payload);
    } else if constexpr (Kind == CudaBackbone::deepseek_v4) {
        model.config = deepseek_v4::Config::from_json(payload);
        model.config.layer_types.assign(
            static_cast<std::size_t>(model.config.num_hidden_layers),
            "deepseek_v4");
    } else if constexpr (Kind == CudaBackbone::deepseek_v41) {
        model.config =
            mfq::models::deepseek_v41::Config::from_json(payload);
    }

    if (model.num_hidden_layers() != model.graph.topology.text_layers) {
        throw std::runtime_error(
            "model graph/config text-layer topology mismatch");
    }
    if constexpr (Kind == CudaBackbone::qwen4_exp ||
                  Kind == CudaBackbone::glm5_next) {
        flash_next::validate_load_options();
    }
    if constexpr (Kind == CudaBackbone::deepseek_v41) {
        deepseek_v41_runtime::validate_load_options();
    }
    if constexpr (Kind == CudaBackbone::deepseek_v4) {
        deepseek_v4::validate_load_options(model.config);
    }
    if (g_expert_parallel.enabled() && model.num_experts() <= 0) {
        throw std::runtime_error(
            "--expert-parallel requires a model with routed experts");
    }

    g_layer_placement.prepare(model.num_hidden_layers());
    g_dense_cpu_layer_count = 0;
    if (g_n_gpu_layers >= 0) {
        g_dense_cpu_layer_count = static_cast<int>(std::max<int64_t>(
            model.num_hidden_layers() - g_n_gpu_layers, 0));
        if (g_dense_cpu_layer_count > 0) {
            if (model_parallel_enabled() || g_layer_placement.enabled()) {
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            }
            constexpr bool dense_qwen_style =
                Kind == CudaBackbone::generic_qwen ||
                Kind == CudaBackbone::minicpmo45 ||
                Kind == CudaBackbone::minicpmo_tts;
            bool supported = dense_qwen_style && model.num_experts() <= 0;
            for (int layer = 0; supported &&
                    layer < model.num_hidden_layers(); ++layer) {
                const auto type = model.layer_type(layer);
                supported = type == "full_attention" ||
                    type == "linear_attention";
            }
            if (!supported) {
                throw std::runtime_error(
                    "--n-gpu-layers currently supports dense Qwen-style blocks only");
            }
            std::cerr << "dense_layer_placement cpu=0-"
                      << (g_dense_cpu_layer_count - 1)
                      << " gpu=" << g_dense_cpu_layer_count << '-'
                      << (model.num_hidden_layers() - 1)
                      << " cpu_threads=" << mfq_get_num_threads()
                      << std::endl;
        }
    }
    g_layer_placement.load_device = g_layer_placement.primary_device();
    MfqCudaGuard model_guard(g_layer_placement.primary_device());
    if (model.plan.vision == CudaVisionAdapter::grid_vit &&
            g_dense_cpu_layer_count > 0) {
        throw std::runtime_error(
            "CUDA grid-Vision currently requires GPU-resident text layers");
    }

    if (context_size_override > 0) {
        if (context_size_override > model.max_position_embeddings()) {
            throw std::runtime_error(
                "--ctx-size exceeds max_position_embeddings");
        }
        model.set_max_position_embeddings(context_size_override);
    }

    if constexpr (Kind == CudaBackbone::generic_qwen ||
                  Kind == CudaBackbone::minicpmo45 ||
                  Kind == CudaBackbone::minicpmo_tts ||
                  Kind == CudaBackbone::glm_dsa) {
        auto make_rope = [&](mfq_tensor_backend::Device device) {
            RopeCache rope(
                model.max_position_embeddings(), model.rotary_dim(),
                model.rope_base(), 0, -1, device,
                Kind == CudaBackbone::minicpmo45);
            if constexpr (Kind == CudaBackbone::generic_qwen) {
                rope.configure_mrope(
                    model.config.mrope_sections,
                    model.config.mrope_interleaved,
                    model.config.rotary_dim,
                    device);
            }
            return rope;
        };
        const auto primary = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            g_layer_placement.primary_device());
        model.rope = make_rope(primary);
        if (g_dense_cpu_layer_count > 0) {
            model.cpu_rope = make_rope(
                mfq_tensor_backend::Device(mfq_tensor_backend::kCPU));
        }
        if (g_layer_placement.enabled()) {
            for (int device : g_layer_placement.devices) {
                MfqCudaGuard rope_guard(device);
                model.device_ropes.emplace(
                    device,
                    make_rope(mfq_tensor_backend::Device(
                        mfq_tensor_backend::kCUDA, device)));
            }
        }
    }

    const std::string embed_name = "model.token_embedding.weight";
    const std::string norm_name = "model.output_norm.weight";
    const std::string output_name = "model.output.weight";
    model.embed = load_quant_linear(source, embed_name);
    if constexpr (Kind == CudaBackbone::qwen4_exp) {
        model.final_mixer =
            flash_next::load_final_mixer(source, model.config);
    } else {
        model.output_norm = load_dense_gpu(source, norm_name);
    }
    if constexpr (Kind == CudaBackbone::deepseek_v4) {
        auto head = deepseek_v4::load_output_head(source);
        model.hc_head_fn = std::move(head.function);
        model.hc_head_scale = std::move(head.scale);
        model.hc_head_base = std::move(head.base);
    }
    if (model.tie_word_embeddings() || !has_tensor(source, output_name)) {
        model.lm_head = g_tensor_parallel.enabled()
            ? load_quant_linear(
                source, embed_name, TensorParallelAxis::Output)
            : model.embed;
    } else {
        model.lm_head = load_quant_linear(
            source, output_name, TensorParallelAxis::Output);
    }

    if (load_blocks) {
        if constexpr (Kind == CudaBackbone::qwen4_exp ||
                      Kind == CudaBackbone::glm5_next) {
            load_model_blocks(model, [&](int layer, int, const std::string&) {
                return flash_next::load_block(
                    source, model.config, layer);
            });
        } else if constexpr (Kind == CudaBackbone::deepseek_v41) {
            model.shared = deepseek_v41_runtime::load_shared_state(
                source, model.config);
            load_model_blocks(model, [&](int layer, int,
                                         const std::string& type) {
                MFQ_RUNTIME_CHECK(
                    type == "deepseek_v41" && model.shared,
                    "invalid DeepSeek-V4.1 block loader state");
                return deepseek_v41_runtime::load_block(
                    source, layer, model.shared);
            });
        } else if constexpr (Kind == CudaBackbone::deepseek_v4) {
            std::unordered_map<int, std::shared_ptr<Dsv4SharedState>> states;
            load_model_blocks(model, [&](int layer, int device,
                                         const std::string& type) {
                auto& state = states[device];
                if (!state) state = std::make_shared<Dsv4SharedState>();
                return deepseek_v4::load_block(
                    source, model.config, layer, type, state);
            });
        } else if constexpr (Kind == CudaBackbone::glm_dsa) {
            std::unordered_map<int, std::shared_ptr<GlmDsaSharedState>> states;
            load_model_blocks(model, [&](int layer, int device,
                                         const std::string& type) {
                auto& state = states[device];
                if (!state) state = std::make_shared<GlmDsaSharedState>();
                return glm_dsa::load_block(
                    source, model.config, layer, type, state);
            });
        } else if constexpr (Kind == CudaBackbone::gemma4) {
            load_model_blocks(model, [&](int layer, int,
                                         const std::string& type) {
                return gemma4::load_block(
                    source, model.config, layer, type);
            });
        } else if constexpr (Kind == CudaBackbone::generic_qwen) {
            load_model_blocks(model, [&](int layer, int,
                                         const std::string& type) {
                return qwen35::load_block(
                    source, model.config, layer, type);
            });
        } else if constexpr (Kind == CudaBackbone::minicpmo45 ||
                             Kind == CudaBackbone::minicpmo_tts) {
            load_model_blocks(model, [&](int layer, int,
                                         const std::string& type) {
                return load_transformer_block(
                    source, model.config, layer, type,
                    Kind == CudaBackbone::minicpmo45);
            });
        }
    }

    g_loading_cpu_layer = false;
    g_layer_placement.load_device = g_layer_placement.primary_device();
    if (moe_expert_cache_has_sources() &&
            !moe_expert_cache_finalized() &&
            !defer_moe_cache_finalize) {
        finalize_moe_expert_cache();
    }
    return model;
}

#define MFQ_INSTANTIATE_CAUSAL_LM(BACKBONE, TYPE)                         \
    template TYPE mfq::cuda::load_causal_lm<BACKBONE>(                   \
        const std::string&, const std::string&, int64_t, bool, bool,     \
        std::shared_ptr<const mfq::ModelSource>)

MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::generic_qwen, mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::minicpmo45, mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::minicpmo_tts, mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::gemma4, mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::glm_dsa, mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::glm5_next, mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::qwen4_exp, mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::deepseek_v4, mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(
    mfq::cuda::CudaBackbone::deepseek_v41, mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_CAUSAL_LM
