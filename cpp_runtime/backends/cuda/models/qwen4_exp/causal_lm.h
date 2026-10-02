#pragma once

#include "../causal_ops.h"
#include "models/block.h"
#include "quant_linear.h"
#include "models/qwen4_exp/config.h"
#include "layers.h"

#include <memory>

namespace mfq::cuda::qwen4_exp {

std::unique_ptr<::Block> load_block(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const mfq::models::qwen4_exp::Config& config,
    int layer);
std::unique_ptr<Gr> load_final_mixer(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const mfq::models::qwen4_exp::Config& config);
Tensor finalize_hidden(const Gr& mixer, const Tensor& hidden);

} // namespace mfq::cuda::qwen4_exp

namespace mfq::cuda {

struct Qwen4Model : CausalLmArchitecture {
    mfq::models::qwen4_exp::Config config;
    std::unique_ptr<qwen4_exp::Gr> final_mixer;
    mfq_tensor_backend::Tensor positions;
    int64_t batch = 0;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    void adapter_validate_load_options() const;
    void adapter_load_final_state(
        const mfq::ModelSource& source,
        mfq_tensor_backend::Tensor& output_norm);
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.maximum = value;
    }
    void adapter_reset(int64_t batch);
    bool adapter_requires_batch_reset(int64_t batch) const noexcept;
    void adapter_validate_forward(
        int64_t batch,
        int64_t tokens,
        int64_t cache_position,
        bool has_position_override,
        bool has_cache_position_override,
        bool has_attention_mask) const;
    bool adapter_allows_speculative_position_override() const noexcept;
    CudaPreparedPositions adapter_prepare_positions(
        mfq_tensor_backend::Tensor positions,
        int64_t batch,
        int64_t tokens);
    void adapter_validate_positions(
        const mfq_tensor_backend::Tensor& positions,
        int64_t batch,
        int64_t tokens,
        bool has_mrope) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_block_positions(
        const mfq_tensor_backend::Tensor& full_positions,
        const mfq_tensor_backend::Tensor& local_positions,
        int device) const;
    void adapter_finish_forward(
        const mfq_tensor_backend::Tensor& full_positions,
        int64_t batch,
        int64_t tokens);
    bool adapter_force_cache_advance() const noexcept;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    bool adapter_supports_speculation() const noexcept;
    void adapter_rollback_speculative(int64_t keep);
};

extern template struct CudaSessionCodec<Qwen4Model>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
