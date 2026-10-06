#pragma once

#include "step_sequence.h"

#include "models/common/causal_model_ops.h"
#include "storage/session_state.h"
#include "batch_state.h"
#include "qwen35/causal_lm.h"
#include "qwen35/config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mfq {
class ModelSource;
}
struct Block;

namespace mfq::cuda::qwen35 {

using Config = mfq::models::qwen35::Config;

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const Config &config, int layer, const std::string &type,
                                    std::string_view tensor_root = "model");

bool supports_text_session_state(const std::vector<std::unique_ptr<::Block>> &blocks);

mfq::StepSequence<TextSessionState> capture_text_session_state(const std::vector<std::unique_ptr<::Block>> &blocks,
                                            const std::vector<std::int64_t> &tokens,
                                            std::int64_t cache_position);

mfq::StepSequence<std::monostate> restore_text_session_state(std::vector<std::unique_ptr<::Block>> &blocks,
                                const TextSessionState &state);

} // namespace mfq::cuda::qwen35

namespace mfq::engine {
template<class Ops> class ContinuousBatch;
struct InferenceRequest;
}
namespace mfq::cuda {
class QwenBatchOperations;
namespace internal { class TextSessionCache; }
struct CudaContinuousBatchConfig;

struct Qwen35Model : CausalResources {
    template <class Backend> using CausalModel = mfq::models::qwen35::CausalLm<Backend>;
    mfq::models::qwen35::Config config;
    static constexpr bool adapter_supports_batching = true;
    using Batch = mfq::engine::ContinuousBatch<QwenBatchOperations>;
    static std::unique_ptr<Batch> adapter_create_batch(
        Qwen35CausalLm&, CudaExecutionContext&, std::size_t chunk, const CudaContinuousBatchConfig&);
    static bool adapter_owns_generation(const mfq::engine::InferenceRequest&,
        const internal::TextSessionCache&, bool predictor);
    static qwen35::Qwen35GenerationState adapter_request_state(Qwen35CausalLm&, MtpModule*);

    void adapter_validate_components(const mfq::ModelGraph &graph) const;
    void adapter_configure_rope(RopeCache &rope, mfq_tensor_backend::Device device) const;
    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    mfq_tensor_backend::Tensor adapter_embed(mfq_tensor_backend::Tensor output) const;

    bool adapter_supports_prepared_prompt() const noexcept;
};

template <> struct CudaSessionCodec<Qwen35Model> {
    using Model = CausalLm<Qwen35Model>;
    static TextSessionStateKind kind(const Model &model);
    static bool supports_paged(const Model &model);
    static mfq::StepSequence<TextSessionState> capture_steps(const Model &model, const std::vector<int64_t> &tokens);
    static TextSessionState capture(const Model &model, const std::vector<int64_t> &tokens) { return mfq::finish_steps(capture_steps(model, tokens)); }
    static mfq::StepSequence<std::monostate> restore_steps(Model &model, const TextSessionState &state);
    static void restore(Model &model, const TextSessionState &state) { (void)mfq::finish_steps(restore_steps(model, state)); }
};

extern template struct FullAttentionSessionCodec<Qwen35Model>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct qwen35::CausalLm<cuda::CudaCausalOps<cuda::Qwen35Model>>;
} // namespace mfq::models
