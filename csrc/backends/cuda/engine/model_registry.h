#pragma once

#include "cuda_model_list.h"
#include "model_components.h"
#include "load_options.h"
#include "storage/moe/expert_cache.h"

#include <chrono>
#include <memory>
#include <variant>

namespace mfq::cuda::internal {

template <typename F>
auto with_loaded_cuda_model(CudaExecutionContext& execution, const CudaLoadOptions& options,
        bool optional, F&& run) {
    CudaExecutionScope scope(execution);
    auto source = mfq::open_model_source(options.model_path);
    const auto load = [&]<typename Model>() {
        const auto started = std::chrono::steady_clock::now();
        auto model = load_causal_lm<Model>(execution, options.model_path, options.config_path,
            options.context_size, true, optional, source);
        auto components = load_runtime_components(model, optional);
        if (moe_expert_cache_has_sources(execution.moe_expert_cache) && !moe_expert_cache_finalized(execution.moe_expert_cache))
            finalize_moe_expert_cache(execution.moe_expert_cache);
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
