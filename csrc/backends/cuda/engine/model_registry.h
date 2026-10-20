#pragma once

#include "cuda_model_list.h"
#include "model_components.h"
#include "load_options.h"
#include "storage/weights/moe/kv_memory.h"
#include "storage/weights/moe/expert_cache.h"

#include <chrono>
#include <memory>
#include <variant>

namespace mfq::cuda::internal {
void prepare_sampling_runtime(int64_t vocabulary);

template <typename F>
auto with_loaded_cuda_model(CudaExecutionContext& execution, const CudaLoadOptions& options,
        bool optional, F&& run) {
    CudaExecutionScope scope(execution);
    auto source = mfq::open_model_source(options.model_path);
    auto load = [&]<typename Model>() {
        auto started = std::chrono::steady_clock::now();
        Model model = load_causal_lm<Model>(
            execution,
            options.model_path,
            options.config_path,
            options.context_size,
            true,
            optional,
            source);
        auto components =
            load_runtime_components(model, optional);
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            prepare_sampling_runtime(model.vocab_size());
        if(execution.stores.kv_memory) {
            execution.stores.kv_memory->finalize(moe_expert_cache_minimum_bytes(execution.stores.moe_expert_cache));
            set_moe_kv_expert_loan(execution.stores.moe_expert_cache,execution.stores.kv_memory->expert_loan());
            if(auto offload=execution.stores.kv_memory->offload())execution.stores.qsa_kv_store=std::move(offload);
        }
        if (moe_expert_cache_has_sources(execution.stores.moe_expert_cache) &&
                !moe_expert_cache_finalized(execution.stores.moe_expert_cache)) {
            std::size_t graph_reserve=0;
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                graph_reserve=qwen4_mtp_graph_reserve(model,components.mtp.get());
                if(graph_reserve)std::cerr<<"runtime_mtp_flexible_budget maximum_depth="
                    <<components.mtp->maximum_draft_depth()<<" bytes="<<graph_reserve
                    <<" initial_depth=3 lend_unused_to_experts=1\n";
            }
            finalize_moe_expert_cache(execution.stores.moe_expert_cache,
                (execution.stores.kv_memory?execution.stores.kv_memory->growth_reserve():0)+graph_reserve,
                execution.stores.kv_memory?execution.stores.kv_memory->expert_budget_charge():0);
        }
        mfq_cuda_synchronize();
        const auto loaded = std::chrono::steady_clock::now();
        report_cuda_memory(execution.config, "loaded");
        return run(model, components, started, loaded);
    };
    const auto& backbone = source->resolved_model_graph().backbone;
#define MFQ_CASE(TYPE) if (TYPE::accepts_backbone(backbone)) return load.template operator()<TYPE>();
    MFQ_CUDA_EACH_MODEL(MFQ_CASE)
#undef MFQ_CASE
    throw std::runtime_error("unsupported CUDA model backbone: " + backbone);
}
} // namespace mfq::cuda::internal
