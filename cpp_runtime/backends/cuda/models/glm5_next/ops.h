#pragma once

#include "../causal_ops.h"
#include "models/block.h"
#include "models/glm5_next/causal_lm.h"
#include "models/glm5_next/config.h"
#include "quant_linear.h"

#include <memory>

namespace mfq::cuda {

struct Glm5Model : CausalResources {
    template <class Backend> using CausalModel = mfq::models::glm5_next::CausalLm<Backend>;
    mfq::models::glm5_next::Config config;

    void adapter_validate_load_options() const;
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    mfq_tensor_backend::Tensor collapse_hidden(mfq_tensor_backend::Tensor hidden, int64_t batch,
                                               int64_t tokens) const;
    mfq_tensor_backend::Tensor normalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                const mfq_tensor_backend::Tensor &output_norm,
                                                int64_t batch, int64_t tokens) const;
    mfq_tensor_backend::Tensor
    adapter_raw_hidden(const mfq_tensor_backend::Tensor &hidden,
                       const mfq_tensor_backend::Tensor &finalized) const;
    mfq_tensor_backend::Tensor adapter_last_logits(const QuantLinear &lm_head,
                                                   mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(const QuantLinear &lm_head,
                                                  mfq_tensor_backend::Tensor hidden) const;
};

extern template struct CudaSessionCodec<Glm5Model>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct glm5_next::CausalLm<cuda::CudaCausalOps<cuda::Glm5Model>>;
} // namespace mfq::models
