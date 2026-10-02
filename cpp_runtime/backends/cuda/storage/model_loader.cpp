#include "storage/weight_loader.h"
#include "models/causal_models.h"

#include "cuda_execution.h"
#include "models/registry.h"
#include "storage/moe_expert_cache.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace {

// Allocation, packing, placement and device lifetime are native. The shared
// causal model supplies weight relationships and layer traversal.
template <class Model> struct CudaWeightLoader {
    Model &model;
    const mfq::ModelSource &source;
    auto embedding(const std::string &name) {
        return load_quant_linear(*model.execution, source, name);
    }
    auto output(const std::string &name) {
        return load_quant_linear(*model.execution, source, name, TensorParallelAxis::Output);
    }
    auto tied_output(const QuantLinear &embedding, const std::string &name) {
        return model.execution->tensor_parallel.enabled() ? output(name) : embedding;
    }
    bool has_weight(const std::string &name) { return has_tensor(source, name); }
    void final_state(mfq_tensor_backend::Tensor &norm) {
        model.adapter_load_final_state(source, norm);
    }
    void prepare_blocks() { model.adapter_prepare_blocks(source); }
    auto block(int layer, const std::string &type) {
        auto &execution = *model.execution;
        const int device = execution.layer_placement.device_for_layer(layer);
        const bool cpu_offloaded = layer < execution.dense_cpu_layer_count;
        execution.loading_cpu_layer = cpu_offloaded;
        execution.layer_placement.load_device = device;
        MfqCudaGuard layer_guard(device);
        std::cerr << "loading layer " << layer << ' ' << type << ' '
                  << (cpu_offloaded ? "CPU" : "CUDA") << std::endl;
        auto block = model.adapter_load_block(source, layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = cpu_offloaded;
        return block;
    }
};

} // namespace

template <typename Model>
Model mfq::cuda::load_causal_lm(CudaExecutionContext &execution, const std::string &model_path,
                                const std::string &config_path, int64_t context_size_override,
                                bool load_blocks, bool defer_moe_cache_finalize,
                                std::shared_ptr<const mfq::ModelSource> model_source) {
    Model model;
    model.execution = &execution;
    model.source = model_source ? std::move(model_source) : mfq::open_model_source(model_path);
    const auto &source = *model.source;
    validate_model_source(source);
    model.graph = source.resolved_model_graph();
    model.plan = cuda_model_plan(model.graph);
    model.load_definition(load_model_config_json(source, config_path), model.graph, source,
                          context_size_override);

    model.adapter_validate_load_options();
    if (execution.expert_parallel.enabled() && model.num_experts() <= 0) {
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    }

    execution.layer_placement.prepare(model.num_hidden_layers());
    execution.dense_cpu_layer_count = 0;
    if (execution.n_gpu_layers >= 0) {
        execution.dense_cpu_layer_count = static_cast<int>(
            std::max<int64_t>(model.num_hidden_layers() - execution.n_gpu_layers, 0));
        if (execution.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution) || execution.layer_placement.enabled()) {
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            }
            bool supported = model.adapter_supports_dense_cpu_offload() && model.num_experts() <= 0;
            for (int layer = 0; supported && layer < model.num_hidden_layers(); ++layer) {
                const auto type = model.layer_type(layer);
                supported = type == "full_attention" || type == "linear_attention";
            }
            if (!supported) {
                throw std::runtime_error(
                    "--n-gpu-layers currently supports dense Qwen-style blocks only");
            }
            std::cerr << "dense_layer_placement cpu=0-" << (execution.dense_cpu_layer_count - 1)
                      << " gpu=" << execution.dense_cpu_layer_count << '-'
                      << (model.num_hidden_layers() - 1) << " cpu_threads=" << mfq_get_num_threads()
                      << std::endl;
        }
    }
    execution.layer_placement.load_device = execution.layer_placement.primary_device();
    MfqCudaGuard model_guard(execution.layer_placement.primary_device());
    if (model.plan.vision == CudaVisionAdapter::grid_vit && execution.dense_cpu_layer_count > 0) {
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    }

    if (model.adapter_uses_common_rope()) {
        auto make_rope = [&](mfq_tensor_backend::Device device) {
            RopeCache rope(model.max_position_embeddings(), model.rotary_dim(), model.rope_base(),
                           0, -1, device, model.metadata.rope_interleaved);
            model.adapter_configure_rope(rope, device);
            return rope;
        };
        const auto primary = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
                                                        execution.layer_placement.primary_device());
        model.rope = make_rope(primary);
        if (execution.dense_cpu_layer_count > 0) {
            model.cpu_rope = make_rope(mfq_tensor_backend::Device(mfq_tensor_backend::kCPU));
        }
        if (execution.layer_placement.enabled()) {
            for (int device : execution.layer_placement.devices) {
                MfqCudaGuard rope_guard(device);
                model.device_ropes.emplace(device, make_rope(mfq_tensor_backend::Device(
                                                       mfq_tensor_backend::kCUDA, device)));
            }
        }
    }

    CudaWeightLoader<Model> weights{model, source};
    model.load_weights(weights, load_blocks);

    execution.loading_cpu_layer = false;
    execution.layer_placement.load_device = execution.layer_placement.primary_device();
    if (moe_expert_cache_has_sources(execution.moe_expert_cache) &&
        !moe_expert_cache_finalized(execution.moe_expert_cache) && !defer_moe_cache_finalize) {
        finalize_moe_expert_cache(execution.moe_expert_cache);
    }
    return model;
}

#define MFQ_INSTANTIATE_CAUSAL_LM(TYPE)                                                            \
    template TYPE mfq::cuda::load_causal_lm<TYPE>(CudaExecutionContext &, const std::string &,     \
                                                  const std::string &, int64_t, bool, bool,        \
                                                  std::shared_ptr<const mfq::ModelSource>)

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
