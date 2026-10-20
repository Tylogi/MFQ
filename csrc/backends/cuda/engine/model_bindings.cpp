#include "models/common/full_block.h"
#include "native/weight_memory.h"
#include "cuda_model_list.h"
#include "native/weight_memory.h"
#include "kernels/sampling_ops.h"
#include "kernels/gr_prefill_blas.h"
#include "nlohmann/json.hpp"
#include <cmath>

#include "models/common/causal_model_ops.h"
#include "ops/execution/execution_options.h"
#include "storage/weights/mapped_embedding.h"
#include "storage/weights/loaders/weight.h"
#include "storage/weights/moe/expert_cache.h"
#include <algorithm>
#include <iostream>

namespace mfq::cuda::internal {
void prepare_sampling_runtime(int64_t vocabulary) {
    namespace tb=mfq_tensor_backend;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto before=context->local_memory_usage();
    {
        const auto gpu=tb::TensorOptions().device(tb::kCUDA);
        auto random=tb::full({1},.37,gpu.dtype(tb::kFloat32));
        auto counts=tb::zeros({vocabulary},gpu.dtype(tb::kInt32));
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto logits=tb::zeros({1,vocabulary},gpu.dtype(dtype));
            // Lazy CUDA modules and sampler workspaces must be charged before
            // expert arenas consume the remaining device budget. Exercise both
            // compact and fallback sampling without touching request RNG/state.
            for(int k:{20,64,65})if(k<=vocabulary)
                (void)sample_top_k_top_p_cuda(logits,random,.7,k,.8);
            (void)sample_softmax_cuda(logits,random,.7);
            (void)sample_apply_penalties_cuda(logits,counts,.2,.05,1.0);
            if(vocabulary>=64)(void)sample_top_k_compact_cuda(logits,64);
        }
        prepare_gr_prefill_blas_runtime();
    }
    context->reclaim_idle_memory();
    std::cerr<<"runtime_sampling_prepare vocab="<<vocabulary<<" local_before="<<before
        <<" local_after="<<context->local_memory_usage()<<"\n";
}


}

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
        const auto& dtype=require_tensor(source,name).dtype;
        const bool nint=dtype=="NINT";
        if(use_mapped && model.has_independent_output_weights(*this) &&
                !model.execution->tensor_parallel.enabled() && !model.execution->loading_cpu_layer &&
                (dtype=="BF16" || dtype=="F16" || dtype=="F32" || nint)) {
            try {
                const mfq::cuda::WeightAllocationScope weight_memory;
                if(nint) {
                    QuantLinear cpu;
                    {
                        struct CpuLoadScope {
                            bool& flag;
                            explicit CpuLoadScope(bool& value):flag(value){flag=true;}
                            ~CpuLoadScope(){flag=false;}
                        } scope(model.execution->loading_cpu_layer);
                        cpu=load_quant_linear(*model.execution,source,name);
                    }
                    auto result=map_nint_embedding(std::move(cpu),active_weight_load_device(*model.execution),mapped_control);
                    std::cerr<<"mapped_token_embedding rows="<<result.out()<<" width="<<result.neuron_len()
                             <<" dtype="<<dtype<<" read="<<(mapped_control?"device_control":"mapped_host")<<std::endl;
                    return result;
                }
                auto cpu=load_dense_cpu(*model.execution,source,name);
                auto result=map_dense_embedding(cpu,active_weight_load_device(*model.execution),mapped_control);
                std::cerr<<"mapped_token_embedding bytes="<<cpu.numel()*cpu.element_size()
                         <<" rows="<<cpu.size(0)<<" width="<<cpu.size(1)<<" dtype="<<dtype
                         <<" read="<<(mapped_control?"device_control":"mapped_host")<<std::endl;
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
    void final_state(mfq_tensor_backend::Tensor& norm) { model.load_final_state(source, norm); }
    void prepare_blocks() { model.prepare_blocks(source); }
    auto block(int layer, const std::string& type) {
        auto& execution = *model.execution;
        const int device = execution.layer_placement.device_for_layer(layer);
        const bool offloaded = layer < execution.dense_cpu_layer_count;
        execution.loading_cpu_layer = offloaded;
        execution.layer_placement.load_device = device;
        MfqCudaGuard guard(device);
        std::cerr << "loading layer " << layer << ' ' << type << ' ' << (offloaded ? "CPU" : "CUDA") << '\n';
        auto block = model.load_block(source, layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = offloaded;
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) full->preallocate_kv();
        return block;
    }
};
} // namespace

template <typename Model>
Model load_causal_lm(CudaExecutionContext& execution, const std::string& model_path,
        const std::string& config_path, int64_t context_size_override, bool load_blocks,
        bool defer_moe_cache_finalize, std::shared_ptr<const mfq::ModelSource> model_source, bool load_input_output) {
    Model model;
    model.execution = &execution;
    model.source = model_source ? std::move(model_source) : mfq::open_model_source(model_path);
    const auto& source = *model.source;
    validate_model_source(source);
    model.graph = source.resolved_model_graph();
    model.plan = Model::plan_components(model.graph);
    auto definition=load_model_config_json(source, config_path);
    execution.yarn={};
    if(execution.yarn_context_extension) {
        if(model.graph.backbone!="qwen3_5" && model.graph.backbone!="qwen4_exp")
            throw std::invalid_argument("YaRN extension requires a Qwen model");
        auto config=nlohmann::json::parse(definition);
        auto& text=config.contains("text_config") ? config["text_config"] : config.contains("language_config") ? config["language_config"] : config;
        auto rope=text.value("rope_parameters",text.value("rope_scaling",nlohmann::json::object()));
        if(rope.contains("full_attention"))rope=rope["full_attention"];
        auto native=text.value("max_position_embeddings",int64_t(0));
        const auto kind=rope.value("rope_type",rope.value("type",std::string("default")));
        if(kind=="yarn")native=rope.value("original_max_position_embeddings",native);
        const auto maximum=rope.value("factor",4.0);
        if((kind!="default" && kind!="yarn") || native<=0 || !std::isfinite(maximum) || maximum<1 ||
            context_size_override>std::floor(native*maximum))throw std::invalid_argument("context exceeds Qwen YaRN limit");
        execution.yarn.original_context=native;
        execution.yarn.factor=std::max(1.0,double(context_size_override)/native);
        execution.yarn.beta_fast=rope.value("beta_fast",32.0);
        execution.yarn.beta_slow=rope.value("beta_slow",1.0);
        if(execution.yarn.enabled())text["max_position_embeddings"]=context_size_override;
        definition=config.dump();
    }
    model.load_definition(definition, model.graph, source,
                          context_size_override);

    model.validate_load_options();
    if (execution.expert_parallel.enabled() && model.num_experts() <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    execution.layer_placement.prepare(model.num_hidden_layers());
    execution.dense_cpu_layer_count = 0;
    if (execution.n_gpu_layers >= 0) {
        execution.dense_cpu_layer_count = static_cast<int>(std::max<int64_t>(model.num_hidden_layers() - execution.n_gpu_layers, 0));
        if (execution.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution) || execution.layer_placement.enabled())
                throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            bool supported = Model::policy.dense_cpu_offload && model.num_experts() <= 0;
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
    if (Model::policy.common_rope) {
        const auto make_rope = [&](mfq_tensor_backend::Device device) {
            RopeCache rope(model.max_position_embeddings(), model.rotary_dim(), model.rope_base(), 0, -1, device, model.metadata.rope_interleaved,execution.yarn);
            model.configure_rope(rope, device);
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
    if (load_input_output) {
        model.load_weights(weights, load_blocks);
    } else {
        if (load_blocks) throw std::invalid_argument("streamed model setup requires deferred blocks");
        model.load_final_state(source, model.output_norm);
        model.prepare_blocks(source);
    }

    execution.loading_cpu_layer = false;
    execution.layer_placement.load_device = execution.layer_placement.primary_device();
    if (moe_expert_cache_has_sources(execution.stores.moe_expert_cache) &&
        !moe_expert_cache_finalized(execution.stores.moe_expert_cache) && !defer_moe_cache_finalize) {
        if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
            mfq::cuda::internal::prepare_sampling_runtime(model.vocab_size());
        finalize_moe_expert_cache(execution.stores.moe_expert_cache);
    }
    return model;
}
} // namespace mfq::cuda
#include "models/common/mtp.h"
#include "storage/session/text_cache_impl.h"

#define MFQ_BIND_CUDA_MODEL(TYPE) \
    template TYPE mfq::cuda::load_causal_lm<TYPE>(CudaExecutionContext&, const std::string&, const std::string&, int64_t, bool, bool, std::shared_ptr<const mfq::ModelSource>, bool); \
    template mfq::StepSequence<mfq::cuda::internal::TextSessionRestore> mfq::cuda::internal::TextSessionCache::restore_steps(TYPE&, MtpModule*, const std::string&, const std::vector<int64_t>&, size_t, const std::string&); \
    template mfq::cuda::internal::TextSessionRestore mfq::cuda::internal::TextSessionCache::restore_best(TYPE&, MtpModule*, const std::string&, const std::vector<int64_t>&, size_t, const std::string&);

MFQ_CUDA_EACH_MODEL(MFQ_BIND_CUDA_MODEL)
#undef MFQ_BIND_CUDA_MODEL
