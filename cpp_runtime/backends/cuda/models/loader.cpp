#include "causal_lm.h"

#include "storage/moe_expert_cache.h"
#include "../models/registry.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace {

template <typename Model, typename Loader>
void load_model_blocks(Model& model, Loader&& load) {
    auto& execution = cuda_execution_context();
    const auto layer_count = model.num_hidden_layers();
    model.blocks.reserve(static_cast<std::size_t>(layer_count));
    for (int layer = 0; layer < layer_count; ++layer) {
        const int device = execution.layer_placement.device_for_layer(layer);
        const bool cpu_offloaded = layer < execution.dense_cpu_layer_count;
        execution.loading_cpu_layer = cpu_offloaded;
        execution.layer_placement.load_device = device;
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

mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::Qwen35CausalLm&) {
    return mfq::cuda::CudaBackbone::generic_qwen;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::MiniCPMO45CausalLm&) {
    return mfq::cuda::CudaBackbone::minicpmo45;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::MiniCPMOTtsCausalLm&) {
    return mfq::cuda::CudaBackbone::minicpmo_tts;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::Gemma4CausalLm&) {
    return mfq::cuda::CudaBackbone::gemma4;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::GlmDsaCausalLm&) {
    return mfq::cuda::CudaBackbone::glm_dsa;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::Glm5CausalLm&) {
    return mfq::cuda::CudaBackbone::glm5_next;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::Qwen4CausalLm&) {
    return mfq::cuda::CudaBackbone::qwen4_exp;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::DeepseekV4CausalLm&) {
    return mfq::cuda::CudaBackbone::deepseek_v4;
}
mfq::cuda::CudaBackbone expected_backbone(
        const mfq::cuda::DeepseekV41CausalLm&) {
    return mfq::cuda::CudaBackbone::deepseek_v41;
}

} // namespace

template <typename Model>
Model mfq::cuda::load_causal_lm(
        CudaExecutionContext& execution,
        const std::string& model_path,
        const std::string& config_path,
        int64_t context_size_override,
        bool load_blocks,
        bool defer_moe_cache_finalize,
        std::shared_ptr<const mfq::ModelSource> model_source) {
    Model model;
    model.execution = &execution;
    model.source = model_source
        ? std::move(model_source)
        : mfq::open_model_source(model_path);
    const auto& source = *model.source;
    validate_model_source(source);
    model.graph = source.resolved_model_graph();
    model.plan = cuda_model_plan(model.graph);
    MFQ_RUNTIME_CHECK(
        cuda_backbone(model.graph.backbone) == expected_backbone(model),
        "loaded CUDA backbone does not match the requested causal LM type");

    model.adapter_load_config(
        load_model_config_json(source, config_path), model.graph, source);

    if (model.num_hidden_layers() != model.graph.topology.text_layers) {
        throw std::runtime_error(
            "model graph/config text-layer topology mismatch");
    }
    model.adapter_validate_load_options();
    if (execution.expert_parallel.enabled() && model.num_experts() <= 0) {
        throw std::runtime_error(
            "--expert-parallel requires a model with routed experts");
    }

    execution.layer_placement.prepare(model.num_hidden_layers());
    execution.dense_cpu_layer_count = 0;
    if (execution.n_gpu_layers >= 0) {
        execution.dense_cpu_layer_count = static_cast<int>(
            std::max<int64_t>(
                model.num_hidden_layers() - execution.n_gpu_layers, 0));
        if (execution.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled() ||
                    execution.layer_placement.enabled()) {
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            }
            bool supported =
                model.adapter_supports_dense_cpu_offload() &&
                model.num_experts() <= 0;
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
                      << (execution.dense_cpu_layer_count - 1)
                      << " gpu=" << execution.dense_cpu_layer_count << '-'
                      << (model.num_hidden_layers() - 1)
                      << " cpu_threads=" << mfq_get_num_threads()
                      << std::endl;
        }
    }
    execution.layer_placement.load_device =
        execution.layer_placement.primary_device();
    MfqCudaGuard model_guard(execution.layer_placement.primary_device());
    if (model.plan.vision == CudaVisionAdapter::grid_vit &&
            execution.dense_cpu_layer_count > 0) {
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

    if (model.adapter_uses_common_rope()) {
        auto make_rope = [&](mfq_tensor_backend::Device device) {
            RopeCache rope(
                model.max_position_embeddings(), model.rotary_dim(),
                model.rope_base(), 0, -1, device,
                model.metadata.rope_interleaved);
            model.adapter_configure_rope(rope, device);
            return rope;
        };
        const auto primary = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            execution.layer_placement.primary_device());
        model.rope = make_rope(primary);
        if (execution.dense_cpu_layer_count > 0) {
            model.cpu_rope = make_rope(
                mfq_tensor_backend::Device(mfq_tensor_backend::kCPU));
        }
        if (execution.layer_placement.enabled()) {
            for (int device : execution.layer_placement.devices) {
                MfqCudaGuard rope_guard(device);
                model.device_ropes.emplace(
                    device,
                    make_rope(mfq_tensor_backend::Device(
                        mfq_tensor_backend::kCUDA, device)));
            }
        }
    }

    const std::string embed_name = "model.token_embedding.weight";
    const std::string output_name = "model.output.weight";
    model.embed = load_quant_linear(source, embed_name);
    model.adapter_load_final_state(source, model.output_norm);
    if (model.tie_word_embeddings() || !has_tensor(source, output_name)) {
        model.lm_head = execution.tensor_parallel.enabled()
            ? load_quant_linear(
                source, embed_name, TensorParallelAxis::Output)
            : model.embed;
    } else {
        model.lm_head = load_quant_linear(
            source, output_name, TensorParallelAxis::Output);
    }

    if (load_blocks) {
        model.adapter_prepare_blocks(source);
        load_model_blocks(model, [&](int layer, int device,
                                     const std::string& type) {
            return model.adapter_load_block(
                source, layer, device, type);
        });
    }

    execution.loading_cpu_layer = false;
    execution.layer_placement.load_device =
        execution.layer_placement.primary_device();
    if (moe_expert_cache_has_sources() &&
            !moe_expert_cache_finalized() &&
            !defer_moe_cache_finalize) {
        finalize_moe_expert_cache();
    }
    return model;
}

#define MFQ_INSTANTIATE_CAUSAL_LM(TYPE)                                   \
    template TYPE mfq::cuda::load_causal_lm<TYPE>(                        \
        CudaExecutionContext&, const std::string&, const std::string&,     \
        int64_t, bool, bool, std::shared_ptr<const mfq::ModelSource>)

MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::DeepseekV41CausalLm);

#undef MFQ_INSTANTIATE_CAUSAL_LM
