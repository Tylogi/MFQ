#pragma once

#include "../causal_ops.h"
#include "../full_attention_session_codec.h"
#include "inference.h"
#include "models/block.h"
#include "models/minicpmo45/causal_lm.h"
#include "models/minicpmo45/config.h"
#include "quant_linear.h"

#include <memory>
#include <string>
#include <vector>

namespace mfq::cuda {

struct MiniCPMO45Model : CausalResources {
    template <class Backend> using CausalModel = mfq::models::minicpmo45::CausalLm<Backend>;
    mfq::models::minicpmo45::Config config;

    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    mfq_tensor_backend::Tensor adapter_embed(mfq_tensor_backend::Tensor output) const;
    bool mask_all_ones(const mfq_tensor_backend::Tensor &mask) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(mfq_tensor_backend::Tensor hidden,
                                                      int64_t batch, int64_t tokens) const;
    mfq_tensor_backend::Tensor
    adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_logits(const QuantLinear &lm_head,
                                              mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_last_logits(const QuantLinear &lm_head,
                                                   mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(const QuantLinear &lm_head,
                                                  mfq_tensor_backend::Tensor hidden) const;
};

template <>
struct CudaSessionCodec<MiniCPMO45Model> : FullAttentionSessionCodec<MiniCPMO45Model> {};

extern template struct FullAttentionSessionCodec<MiniCPMO45Model>;

struct MiniCPMOTtsModel : CausalResources {
    template <class Backend> using CausalModel = mfq::models::minicpmo45::TtsCausalLm<Backend>;
    mfq::models::ModelConfig config;

    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);
};

template <>
struct CudaSessionCodec<MiniCPMOTtsModel> : FullAttentionSessionCodec<MiniCPMOTtsModel> {};

extern template struct FullAttentionSessionCodec<MiniCPMOTtsModel>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct minicpmo45::CausalLm<cuda::CudaCausalOps<cuda::MiniCPMO45Model>>;
extern template struct minicpmo45::TtsCausalLm<cuda::CudaCausalOps<cuda::MiniCPMOTtsModel>>;
} // namespace mfq::models

namespace mfq::cuda::minicpmo45 {

class Components {
  public:
    explicit Components(mfq::cuda::MiniCPMO45CausalLm language);
    ~Components();

    mfq::cuda::MiniCPMO45CausalLm &language() noexcept;
    CudaPreparedPrompt prepare(const std::vector<int64_t> &prompt, const MfqMultimodalInput &media);
    void start(const MfqDuplexSessionParams &parameters);
    MfqDuplexStepResult step(const MfqDuplexStepInput &input);
    void stop();

  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace mfq::cuda::minicpmo45
