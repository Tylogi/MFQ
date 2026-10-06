#pragma once

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "models/common/runtime_components.h"
#include "load_options.h"
#include "storage/moe_expert_cache.h"

#include <chrono>
#include <memory>
#include <variant>

namespace mfq::cuda::internal {

template <template<class> class State>
using CudaModelStates = std::variant<
    std::unique_ptr<State<Qwen35CausalLm>>, std::unique_ptr<State<MiniCPMO45CausalLm>>,
    std::unique_ptr<State<MiniCPMOTtsCausalLm>>, std::unique_ptr<State<Gemma4CausalLm>>,
    std::unique_ptr<State<GlmDsaCausalLm>>, std::unique_ptr<State<Glm5CausalLm>>,
    std::unique_ptr<State<Qwen4CausalLm>>, std::unique_ptr<State<DeepseekV4CausalLm>>,
    std::unique_ptr<State<DeepseekV41CausalLm>>>;

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
    switch (cuda_backbone(source->resolved_model_graph().backbone)) {
        case CudaBackbone::generic_qwen: return load.template operator()<Qwen35CausalLm>();
        case CudaBackbone::minicpmo45: return load.template operator()<MiniCPMO45CausalLm>();
        case CudaBackbone::minicpmo_tts: return load.template operator()<MiniCPMOTtsCausalLm>();
        case CudaBackbone::gemma4: return load.template operator()<Gemma4CausalLm>();
        case CudaBackbone::glm_dsa: return load.template operator()<GlmDsaCausalLm>();
        case CudaBackbone::glm5_next: return load.template operator()<Glm5CausalLm>();
        case CudaBackbone::qwen4_exp: return load.template operator()<Qwen4CausalLm>();
        case CudaBackbone::deepseek_v4: return load.template operator()<DeepseekV4CausalLm>();
        case CudaBackbone::deepseek_v41: return load.template operator()<DeepseekV41CausalLm>();
        case CudaBackbone::unsupported: throw std::runtime_error("unsupported CUDA model backbone");
    }
    throw std::runtime_error("invalid CUDA model backbone");
}
} // namespace mfq::cuda::internal
