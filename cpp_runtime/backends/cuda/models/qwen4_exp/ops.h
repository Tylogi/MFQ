#pragma once

#include "../causal_ops.h"
#include "layers.h"
#include "models/block.h"
#include "models/qwen4_exp/causal_lm.h"
#include "models/qwen4_exp/config.h"
#include "quant_linear.h"

#include <memory>

namespace mfq::cuda::qwen4_exp {

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const mfq::models::qwen4_exp::Config &config, int layer);
std::unique_ptr<Gr> load_final_mixer(CudaExecutionContext &execution,
                                     const mfq::ModelSource &source,
                                     const mfq::models::qwen4_exp::Config &config);
Tensor finalize_hidden(const Gr &mixer, const Tensor &hidden);

} // namespace mfq::cuda::qwen4_exp

namespace mfq::cuda {

struct Qwen4Model : CausalResources {
    template <class Backend> using CausalModel = mfq::models::qwen4_exp::CausalLm<Backend>;
    mfq::models::qwen4_exp::Config config;
    std::unique_ptr<qwen4_exp::Gr> final_mixer;
    mfq_tensor_backend::Tensor positions;
    int64_t batch = 0;

    void adapter_validate_load_options() const;
    void adapter_load_final_state(const mfq::ModelSource &source,
                                  mfq_tensor_backend::Tensor &output_norm);
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    void adapter_reset(int64_t batch);

    mfq_tensor_backend::Tensor
    adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                            const mfq_tensor_backend::Tensor &local_positions, int device) const;
    void adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions, int64_t batch,
                                int64_t tokens);
    mfq_tensor_backend::Tensor
    adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_last_logits(const QuantLinear &lm_head,
                                                   mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(const QuantLinear &lm_head,
                                                  mfq_tensor_backend::Tensor hidden) const;
    void adapter_rollback_speculative(int64_t keep);
};

extern template struct CudaSessionCodec<Qwen4Model>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
