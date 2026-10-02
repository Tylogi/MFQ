#include "ops.h"
#include "../session_codec_impl.h"

namespace mfq::cuda::glm5_next {

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const mfq::models::glm5_next::Config &config, int layer) {
    return std::make_unique<Glm5NextBlock>(execution, source, config, layer);
}

} // namespace mfq::cuda::glm5_next

namespace mfq::cuda {

void Glm5Model::adapter_validate_load_options() const {
    glm5_next::validate_load_options(*execution);
}

std::unique_ptr<Block> Glm5Model::adapter_load_block(const mfq::ModelSource &source, int layer, int,
                                                     const std::string &) {
    return glm5_next::load_block(*execution, source, config, layer);
}

mfq_tensor_backend::Tensor Glm5Model::collapse_hidden(mfq_tensor_backend::Tensor hidden, int64_t,
                                                      int64_t) const {
    return hidden.mean(2);
}

mfq_tensor_backend::Tensor
Glm5Model::normalize_hidden(mfq_tensor_backend::Tensor hidden,
                            const mfq_tensor_backend::Tensor &output_norm, int64_t, int64_t) const {
    return glm5_next::rms_norm(hidden, output_norm, metadata.rms_norm_eps);
}

mfq_tensor_backend::Tensor
Glm5Model::adapter_raw_hidden(const mfq_tensor_backend::Tensor &,
                              const mfq_tensor_backend::Tensor &finalized) const {
    return finalized;
}

mfq_tensor_backend::Tensor Glm5Model::adapter_last_logits(const QuantLinear &lm_head,
                                                          mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor Glm5Model::adapter_next_token(const QuantLinear &lm_head,
                                                         mfq_tensor_backend::Tensor hidden) const {
    return mfq_tensor_backend::argmax(adapter_logits(lm_head, std::move(hidden)), -1)
        .to(mfq_tensor_backend::kInt64);
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Glm5Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct glm5_next::CausalLm<cuda::CudaCausalOps<cuda::Glm5Model>>;
} // namespace mfq::models
