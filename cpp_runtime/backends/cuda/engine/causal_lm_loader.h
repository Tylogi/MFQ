#pragma once

#include "causal_lm.h"
#include "cuda_execution.h"
#include "moe_expert_cache.h"
#include "options.h"
#include "models/components.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::internal {

template <typename F>
auto with_loaded_cuda_model(const CudaLoadOptions& options,
        bool load_optional_components, F&& run) {
    auto& execution = cuda_execution_context();
    auto source = mfq::open_model_source(options.model_path);
    auto load = [&]<CudaBackbone Backbone>() {
        using Model = CausalLmFor<Backbone>;
        auto started = std::chrono::steady_clock::now();
        Model model = load_causal_lm<Backbone>(
            execution,
            options.model_path,
            options.config_path,
            options.context_size,
            true,
            load_optional_components,
            source);
        auto runtime_components =
            load_runtime_components(model, load_optional_components);
        if (moe_expert_cache_has_sources() &&
                !moe_expert_cache_finalized()) {
            finalize_moe_expert_cache();
        }
        mfq_cuda_synchronize();
        auto loaded = std::chrono::steady_clock::now();
        report_cuda_memory("loaded");
        return run.template operator()<Backbone>(
            model, runtime_components, started, loaded);
    };

    switch (cuda_model_plan(source->resolved_model_graph()).backbone) {
        case CudaBackbone::generic_qwen:
            return load.template operator()<CudaBackbone::generic_qwen>();
        case CudaBackbone::minicpmo45:
            return load.template operator()<CudaBackbone::minicpmo45>();
        case CudaBackbone::minicpmo_tts:
            return load.template operator()<CudaBackbone::minicpmo_tts>();
        case CudaBackbone::gemma4:
            return load.template operator()<CudaBackbone::gemma4>();
        case CudaBackbone::glm_dsa:
            return load.template operator()<CudaBackbone::glm_dsa>();
        case CudaBackbone::glm5_next:
            return load.template operator()<CudaBackbone::glm5_next>();
        case CudaBackbone::qwen4_exp:
            return load.template operator()<CudaBackbone::qwen4_exp>();
        case CudaBackbone::deepseek_v4:
            return load.template operator()<CudaBackbone::deepseek_v4>();
        case CudaBackbone::deepseek_v41:
            return load.template operator()<CudaBackbone::deepseek_v41>();
        case CudaBackbone::unsupported:
            throw std::runtime_error("unsupported CUDA model backbone");
    }
    throw std::runtime_error("invalid CUDA model backbone");
}

} // namespace mfq::cuda::internal
