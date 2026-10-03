#include "ops.h"
#include "storage/transformer_loader.h"
#include "storage/session_codec.h"
#include "core/full_block.h"

namespace mfq::cuda {

std::unique_ptr<Block> minicpmo45::load_language_block(CudaExecutionContext& execution,
    const mfq::ModelSource& source, const mfq::models::ModelConfig& config, int layer,
    const std::string& type, mfq::models::minicpmo45::LanguageComponent component,
    std::string_view tensor_root) {
    auto block = std::make_unique<FullBlock>();
    block->official_bf16 = component == mfq::models::minicpmo45::LanguageComponent::text;
    TransformerWeightLoader loader{execution, source};
    loader.preserve_projection_boundaries = block->official_bf16;
    mfq::models::minicpmo45::load_language_block(*block, loader, config, layer, type, component,
                                                tensor_root);
    return block;
}


bool MiniCPMO45Model::adapter_uses_common_rope() const noexcept { return true; }

bool MiniCPMO45Model::adapter_supports_dense_cpu_offload() const noexcept { return true; }

std::unique_ptr<Block> MiniCPMO45Model::adapter_load_block(const mfq::ModelSource &source,
                                                           int layer, int,
                                                           const std::string &type) {
    return minicpmo45::load_language_block(*execution, source, config, layer, type,
        mfq::models::minicpmo45::LanguageComponent::text);
}

bool MiniCPMOTtsModel::adapter_uses_common_rope() const noexcept { return true; }

bool MiniCPMOTtsModel::adapter_supports_dense_cpu_offload() const noexcept { return true; }

std::unique_ptr<Block> MiniCPMOTtsModel::adapter_load_block(const mfq::ModelSource &source,
                                                            int layer, int,
                                                            const std::string &type) {
    return minicpmo45::load_language_block(*execution, source, config, layer, type,
        mfq::models::minicpmo45::LanguageComponent::standalone_tts);
}

mfq_tensor_backend::Tensor MiniCPMO45Model::adapter_embed(mfq_tensor_backend::Tensor output) const {
    return output.to(mfq_tensor_backend::kBFloat16).contiguous();
}

bool MiniCPMO45Model::mask_all_ones(const mfq_tensor_backend::Tensor &mask) const {
    return mask.eq(1).all().item<bool>();
}

mfq_tensor_backend::Tensor
MiniCPMO45Model::adapter_prepare_hidden(mfq_tensor_backend::Tensor hidden, int64_t, int64_t) const {
    return hidden.to(mfq_tensor_backend::kBFloat16).contiguous();
}

mfq_tensor_backend::Tensor
MiniCPMO45Model::adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                         const mfq_tensor_backend::Tensor &output_norm,
                                         int64_t batch, int64_t tokens) const {
    return execution->profiler.measure("model.output_norm", [&]() {
        return qwen_rms_norm_bf16(execution->config,
                                  hidden.reshape({batch * tokens, metadata.hidden_size}),
                                  output_norm, metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

mfq_tensor_backend::Tensor
MiniCPMO45Model::adapter_logits(const QuantLinear &lm_head,
                                mfq_tensor_backend::Tensor hidden) const {
    return execution->profiler.measure("model.lm_head", [&]() {
        return lm_head.forward(*execution, hidden).to(mfq_tensor_backend::kBFloat16).contiguous();
    });
}

mfq_tensor_backend::Tensor
MiniCPMO45Model::adapter_last_logits(const QuantLinear &lm_head,
                                     mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, hidden.to(mfq_tensor_backend::kBFloat16).contiguous());
}

mfq_tensor_backend::Tensor
MiniCPMO45Model::adapter_next_token(const QuantLinear &lm_head,
                                    mfq_tensor_backend::Tensor hidden) const {
    return mfq_tensor_backend::argmax(adapter_last_logits(lm_head, std::move(hidden)), -1)
        .to(mfq_tensor_backend::kInt64);
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct FullAttentionSessionCodec<MiniCPMO45Model>;
template struct FullAttentionSessionCodec<MiniCPMOTtsModel>;

} // namespace mfq::cuda

namespace mfq::models {
template struct minicpmo45::CausalLm<cuda::CudaCausalOps<cuda::MiniCPMO45Model>>;
template struct minicpmo45::TtsCausalLm<cuda::CudaCausalOps<cuda::MiniCPMOTtsModel>>;
} // namespace mfq::models
