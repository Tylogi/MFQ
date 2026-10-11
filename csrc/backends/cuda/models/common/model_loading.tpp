#pragma once

#include "causal_model_ops.h"
#include "runtime/execution_options.h"
#include "storage/mapped_embedding.h"
#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"

#include <algorithm>
#include <iostream>

namespace mfq::cuda {
namespace {
// Bind the shared family's weight traversal to native materialization/placement.
template <class Model> struct CudaWeightLoader {
    Model& model;
    const mfq::ModelSource& source;
    auto embedding(const std::string& name) {
        const auto mapped = runtime_options::embedding_mapped();
        const bool mapped_control = mapped && *mapped == 2;
        const bool use_mapped = mapped ? *mapped == 1 || mapped_control :
            model.execution->config.moe_pipeline && model.execution->config.moe_preload_all &&
            model.execution->config.moe_ram_pcie;
        const auto& dtype = require_tensor(source, name).dtype;
        if (use_mapped && model.has_independent_output_weights(*this) &&
                !model.execution->tensor_parallel.enabled() && !model.execution->loading_cpu_layer &&
                (dtype == "BF16" || dtype == "F16" || dtype == "F32")) {
            auto cpu = load_dense_cpu(*model.execution, source, name);
            try {
                auto result = map_dense_embedding(
                    cpu, active_weight_load_device(*model.execution), mapped_control);
                std::cerr << "mapped_token_embedding bytes=" << cpu.numel() * cpu.element_size()
                          << " rows=" << cpu.size(0) << " width=" << cpu.size(1)
                          << " dtype=" << dtype
                          << " read=" << (mapped_control ? "device_control" : "mapped_host")
                          << std::endl;
                return result;
            } catch (const Error& error) {
                if (mapped) throw;
                (void)cudaGetLastError();
                std::cerr << "mapped_token_embedding unavailable; using device table: "
                          << error.what() << std::endl;
            }
        }
        return load_quant_linear(*model.execution, source, name);
    }
    auto output(const std::string& name) {
        return load_quant_linear(*model.execution, source, name, TensorParallelAxis::Output);
    }
    auto tied_output(const QuantLinear& embedding, const std::string& name) {
        return model.execution->tensor_parallel.enabled() ? output(name) : embedding;
    }
    bool has_weight(const std::string& name) { return has_tensor(source, name); }
    void final_state(mfq_tensor_backend::Tensor& norm) { model.adapter_load_final_state(source, norm); }
    void prepare_blocks() { model.adapter_prepare_blocks(source); }
    auto block(int layer, const std::string& type) {
        auto& execution = *model.execution;
        const int device = execution.layer_placement.device_for_layer(layer);
        const bool offloaded = layer < execution.dense_cpu_layer_count;
        execution.loading_cpu_layer = offloaded;
        execution.layer_placement.load_device = device;
        MfqCudaGuard guard(device);
        std::cerr << "loading layer " << layer << ' ' << type << ' ' << (offloaded ? "CPU" : "CUDA") << '\n';
        auto block = model.adapter_load_block(source, layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = offloaded;
        return block;
    }
};
} // namespace

template <typename Model>
Model load_causal_lm(CudaExecutionContext& execution, const std::string& model_path,
        const std::string& config_path, int64_t context_size_override, bool load_blocks,
        bool defer_moe_cache_finalize, std::shared_ptr<const mfq::ModelSource> model_source) {
    Model model;
    model.execution = &execution;
    model.source = model_source ? std::move(model_source) : mfq::open_model_source(model_path);
    const auto& source = *model.source;
    validate_model_source(source);
    model.graph = source.resolved_model_graph();
    model.plan = cuda_model_plan(model.graph);
    model.load_definition(load_model_config_json(source, config_path), model.graph, source, context_size_override);
    model.adapter_validate_load_options();
    if (execution.expert_parallel.enabled() && model.num_experts() <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    execution.layer_placement.prepare(model.num_hidden_layers());
    execution.dense_cpu_layer_count = 0;
    if (execution.n_gpu_layers >= 0) {
        execution.dense_cpu_layer_count = static_cast<int>(std::max<int64_t>(model.num_hidden_layers() - execution.n_gpu_layers, 0));
        if (execution.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution) || execution.layer_placement.enabled())
                throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            bool supported = model.adapter_supports_dense_cpu_offload() && model.num_experts() <= 0;
            for (int layer = 0; supported && layer < model.num_hidden_layers(); ++layer) {
                const auto type = model.layer_type(layer);
                supported = type == "full_attention" || type == "linear_attention";
            }
            if (!supported) throw std::runtime_error("--n-gpu-layers currently supports dense Qwen-style blocks only");
            std::cerr << "dense_layer_placement cpu=0-" << execution.dense_cpu_layer_count - 1
                << " gpu=" << execution.dense_cpu_layer_count << '-' << model.num_hidden_layers() - 1
                << " cpu_threads=" << mfq_get_num_threads() << '\n';
        }
    }
    execution.layer_placement.load_device = execution.layer_placement.primary_device();
    CudaExecutionScope guard(execution);
    if (model.plan.vision == CudaVisionAdapter::grid_vit && execution.dense_cpu_layer_count > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    if (model.adapter_uses_common_rope()) {
        const auto make_rope = [&](mfq_tensor_backend::Device device) {
            RopeCache rope(model.max_position_embeddings(), model.rotary_dim(), model.rope_base(), 0, -1, device, model.metadata.rope_interleaved);
            model.adapter_configure_rope(rope, device);
            return rope;
        };
        model.rope = make_rope(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, execution.layer_placement.primary_device()));
        if (execution.dense_cpu_layer_count > 0) model.cpu_rope = make_rope(mfq_tensor_backend::Device(mfq_tensor_backend::kCPU));
        if (execution.layer_placement.enabled())
            for (int device : execution.layer_placement.devices) {
                MfqCudaGuard guard(device);
                model.device_ropes.emplace(device, make_rope(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device)));
            }
    }
    CudaWeightLoader<Model> weights{model, source};
    model.load_weights(weights, load_blocks);
    execution.loading_cpu_layer = false;
    execution.layer_placement.load_device = execution.layer_placement.primary_device();
    if (moe_expert_cache_has_sources(execution.moe_expert_cache) &&
            !moe_expert_cache_finalized(execution.moe_expert_cache) && !defer_moe_cache_finalize)
        finalize_moe_expert_cache(execution.moe_expert_cache);
    return model;
}
} // namespace mfq::cuda
