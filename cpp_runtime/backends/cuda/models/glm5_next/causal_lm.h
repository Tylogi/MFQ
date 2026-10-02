#pragma once

#include "../causal_lm.h"
#include "models/block.h"
#include "quant_linear.h"
#include "models/include/glm5_next.h"
#include "layers.h"

#include <memory>

namespace mfq::cuda::glm5_next {

std::unique_ptr<::Block> load_block(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const mfq::models::glm5_next::Config& config,
    int layer);
Tensor finalize_hidden(
    const Tensor& hidden,
    const Tensor& output_norm,
    double epsilon);

} // namespace mfq::cuda::glm5_next

namespace mfq::cuda {

struct Glm5Model : CausalLmArchitecture {
    mfq::models::glm5_next::Config config;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    void adapter_validate_load_options() const;
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.maximum = value;
    }
    void adapter_validate_forward(
        int64_t batch,
        int64_t tokens,
        int64_t cache_position,
        bool has_position_override,
        bool has_cache_position_override,
        bool has_attention_mask) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_raw_hidden(
        const mfq_tensor_backend::Tensor& hidden,
        const mfq_tensor_backend::Tensor& finalized) const;
    mfq_tensor_backend::Tensor adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    bool adapter_supports_speculation() const noexcept;
};

extern template struct CudaSessionCodec<Glm5Model>;

extern template struct CausalLm<Glm5Model>;

} // namespace mfq::cuda
