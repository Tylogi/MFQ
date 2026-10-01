#pragma once

#include "causal_models.h"
#include "cuda_execution.h"
#include "storage/moe_expert_cache.h"
#include "options.h"
#include "models/components.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::internal {

template <typename F>
auto with_loaded_cuda_model(
        CudaExecutionContext& execution,
        const CudaLoadOptions& options,
        bool load_optional_components,
        F&& run) {
    auto source = mfq::open_model_source(options.model_path);
    auto load = [&]<typename Model>() {
        auto started = std::chrono::steady_clock::now();
        Model model = load_causal_lm<Model>(
            execution,
            options.model_path,
            options.config_path,
            options.context_size,
            true,
            load_optional_components,
            source);
        auto runtime_components =
            load_runtime_components(model, load_optional_components);
        if (moe_expert_cache_has_sources(execution.moe_expert_cache) &&
                !moe_expert_cache_finalized(execution.moe_expert_cache)) {
            finalize_moe_expert_cache(execution.moe_expert_cache);
        }
        mfq_cuda_synchronize();
        auto loaded = std::chrono::steady_clock::now();
        report_cuda_memory(execution.config, "loaded");
        return run(model, runtime_components, started, loaded);
    };

    switch (cuda_backbone(source->resolved_model_graph().backbone)) {
        case CudaBackbone::generic_qwen:
            return load.template operator()<Qwen35CausalLm>();
        case CudaBackbone::minicpmo45:
            return load.template operator()<MiniCPMO45CausalLm>();
        case CudaBackbone::minicpmo_tts:
            return load.template operator()<MiniCPMOTtsCausalLm>();
        case CudaBackbone::gemma4:
            return load.template operator()<Gemma4CausalLm>();
        case CudaBackbone::glm_dsa:
            return load.template operator()<GlmDsaCausalLm>();
        case CudaBackbone::glm5_next:
            return load.template operator()<Glm5CausalLm>();
        case CudaBackbone::qwen4_exp:
            return load.template operator()<Qwen4CausalLm>();
        case CudaBackbone::deepseek_v4:
            return load.template operator()<DeepseekV4CausalLm>();
        case CudaBackbone::deepseek_v41:
            return load.template operator()<DeepseekV41CausalLm>();
        case CudaBackbone::unsupported:
            throw std::runtime_error("unsupported CUDA model backbone");
    }
    throw std::runtime_error("invalid CUDA model backbone");
}

} // namespace mfq::cuda::internal
