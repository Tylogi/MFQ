#include "causal_lm.h"
#include "../session_codec_impl.h"

namespace mfq::cuda::glm5_next {

std::unique_ptr<::Block> load_block(
        CudaExecutionContext& execution,
        const mfq::ModelSource& source,
        const mfq::models::glm5_next::Config& config,
        int layer) {
    return std::make_unique<Glm5NextBlock>(
        execution, source, config, layer);
}

Tensor finalize_hidden(
        const Tensor& hidden,
        const Tensor& output_norm,
        double epsilon) {
    return rms_norm(hidden.mean(2), output_norm, epsilon);
}

} // namespace mfq::cuda::glm5_next

namespace mfq::cuda {

void Glm5Model::adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph&,
        const mfq::ModelSource&) {
    config = mfq::models::glm5_next::Config::from_json(payload);
    metadata.vocab_size = config.vocab;
    metadata.hidden_size = config.hidden;
    metadata.num_hidden_layers = config.layers;
    metadata.num_attention_heads = config.heads;
    metadata.num_key_value_heads = 1;
    metadata.head_dim = config.nope;
    metadata.max_position_embeddings = config.maximum;
    metadata.num_experts = config.experts;
    metadata.hc_mult = config.streams;
    metadata.rope_base = 1.0;
    metadata.rms_norm_eps = config.eps;
    metadata.hc_eps = config.hc_eps;
    metadata.tie_word_embeddings = config.tied_embeddings;
    metadata.flash_next = true;
    metadata.model_type = "glm5_next";
    metadata.layer_types = config.layer_types;
}

void Glm5Model::adapter_validate_load_options() const {
    glm5_next::validate_load_options(*execution);
}

std::unique_ptr<Block>
Glm5Model::adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int,
        const std::string&) {
    return glm5_next::load_block(
        *execution, source, config, layer);
}

void Glm5Model::adapter_validate_forward(
        int64_t,
        int64_t tokens,
        int64_t cache_position,
        bool has_position_override,
        bool has_cache_position_override,
        bool has_attention_mask) const {
    MFQ_RUNTIME_CHECK(
        tokens > 0 && cache_position + tokens <= metadata.max_position_embeddings &&
            !has_position_override && !has_cache_position_override &&
            !has_attention_mask,
        "GLM Flash-Next currently requires contiguous causal cache positions "
        "without an external mask");
}

mfq_tensor_backend::Tensor
Glm5Model::adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const {
    return hidden.to(mfq_tensor_backend::kFloat16)
        .unsqueeze(2)
        .expand({batch, tokens, metadata.hc_mult, metadata.hidden_size})
        .contiguous();
}

mfq_tensor_backend::Tensor
Glm5Model::adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t,
        int64_t) const {
    return glm5_next::finalize_hidden(
        hidden, output_norm, metadata.rms_norm_eps);
}

mfq_tensor_backend::Tensor
Glm5Model::adapter_raw_hidden(
        const mfq_tensor_backend::Tensor&,
        const mfq_tensor_backend::Tensor& finalized) const {
    return finalized;
}

mfq_tensor_backend::Tensor
Glm5Model::adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor
Glm5Model::adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const {
    return mfq_tensor_backend::argmax(
        adapter_logits(lm_head, std::move(hidden)), -1)
        .to(mfq_tensor_backend::kInt64);
}

bool Glm5Model::adapter_supports_speculation() const noexcept {
    return true;
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Glm5Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct CausalLm<cuda::CudaCausalOps<cuda::Glm5Model>>;
} // namespace mfq::models
