#pragma once

#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_deepseek_v41_causal_lm.h"
#include "mlx_minicpmo45.h"
#include "mlx_qwen4_causal_lm.h"
#include "mlx_qwen35_causal_lm.h"

#include "mfq_model_graph.h"
#include "mfq/runtime.h"
#include "mlx_generation_job.h"

#include <memory>
#include <mutex>
#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

// Native payload translation is private to the Metal engine. The scheduler
// receives owned values from the generation job, never these device functions.
struct MlxDuplexComponent {
    std::function<void(const MfqDuplexSessionParams&)> start;
    std::function<MfqDuplexStepResult(const MfqDuplexStepInput&)> step;
    std::function<void()> stop;
    explicit operator bool() const { return bool(start) && bool(step) && bool(stop); }
};
struct MlxEngineComponents {
    std::function<std::int32_t(const std::vector<std::int64_t>&,
        const MfqMultimodalInput&, const MfqSamplingParams&,
        MlxGenerationJob&, const MfqTokenConstraintPtr&)> multimodal_generate;
    MlxDuplexComponent duplex;
    bool mtp_available = false;
};

MlxEngineComponents make_mlx_engine_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxQwen35CausalLm>> runtime,
    mlx::core::Stream runtime_stream);

MlxEngineComponents make_mlx_engine_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxMiniCPMO45Runtime>> runtime,
    mlx::core::Stream runtime_stream);

MlxEngineComponents make_mlx_engine_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxDeepseekV4CausalLm>> runtime,
    mlx::core::Stream runtime_stream);

MlxEngineComponents make_mlx_engine_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxDeepseekV41CausalLm>> runtime,
    mlx::core::Stream runtime_stream);

MlxEngineComponents make_mlx_engine_components(
    const MfqModelGraph* graph,
    std::shared_ptr<std::mutex> runtime_mutex,
    std::shared_ptr<std::optional<MlxQwen4CausalLm>> runtime,
    mlx::core::Stream runtime_stream);

} // namespace mfq::metal
