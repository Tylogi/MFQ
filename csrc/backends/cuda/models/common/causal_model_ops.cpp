#include "kernels/sampling_ops.h"
#include "storage/weights/loaders/weight.h"
#include "models/common/causal_model_ops.h"
#include "models/common/full_block.h"

namespace mfq::cuda {

namespace {

mfq_tensor_backend::Tensor normalize_hidden(mfq_tensor_backend::Tensor hidden,
                                            const mfq_tensor_backend::Tensor &output_norm,
                                            const mfq::models::CausalLmMetadata &metadata,
                                            int64_t batch, int64_t tokens, CudaProfiler &profiler) {
    return profiler.measure("model.output_norm", [&]() {
        auto flat = hidden.reshape({batch * tokens, metadata.hidden_size});
        return qwen_rms_norm(flat.to(mfq_tensor_backend::kFloat32), output_norm,
                             metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

} // namespace

void CausalResources::validate_load_options() const {}

void CausalResources::configure_rope(RopeCache &, mfq_tensor_backend::Device) const {}

void CausalResources::load_final_state(const mfq::ModelSource &source,
                                               mfq_tensor_backend::Tensor &output_norm) {
    output_norm = load_dense_gpu(*execution, source, "model.output_norm.weight");
}

void CausalResources::prepare_blocks(const mfq::ModelSource &) {}

mfq_tensor_backend::Tensor CausalResources::post_embed(mfq_tensor_backend::Tensor output) const {
    return output;
}

void CausalResources::reset_native(int64_t) {}

bool CausalResources::requires_batch_reset(int64_t) const noexcept { return false; }

mfq_tensor_backend::Tensor
CausalResources::prepare_hidden(mfq_tensor_backend::Tensor hidden, int64_t, int64_t) const {
    return hidden;
}

void CausalResources::begin_forward(bool) {}

mfq_tensor_backend::Tensor
CausalResources::block_positions(const mfq_tensor_backend::Tensor &,
                                         const mfq_tensor_backend::Tensor &local_positions,
                                         int) const {
    return local_positions;
}

void CausalResources::finish_forward(const mfq_tensor_backend::Tensor &, int64_t, int64_t) {
}

mfq_tensor_backend::Tensor
CausalResources::finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                         const mfq_tensor_backend::Tensor &output_norm,
                                         int64_t batch, int64_t tokens) const {
    return normalize_hidden(std::move(hidden), output_norm, metadata, batch, tokens,
                            execution->profiler);
}

mfq_tensor_backend::Tensor
CausalResources::extract_raw_hidden(const mfq_tensor_backend::Tensor &hidden,
                                    const mfq_tensor_backend::Tensor &) const {
    return hidden;
}

mfq_tensor_backend::Tensor
CausalResources::logits_from_head(const QuantLinear &lm_head,
                                mfq_tensor_backend::Tensor hidden) const {
    auto logits = execution->profiler.measure(
        "model.lm_head", [&]() { return lm_head.forward(*execution, hidden); });
    return logits;
}

mfq_tensor_backend::Tensor
CausalResources::last_logits_from_head(const QuantLinear &lm_head,
                                     mfq_tensor_backend::Tensor hidden) const {
    return lm_head.forward(*execution, hidden.to(mfq_tensor_backend::kFloat16).contiguous());
}

mfq_tensor_backend::Tensor
CausalResources::next_token_from_head(const QuantLinear &lm_head,
                                    mfq_tensor_backend::Tensor hidden) const {
    auto input = hidden.to(mfq_tensor_backend::kFloat16).contiguous();
    auto logits = execution->profiler.measure("model.lm_head",
                                              [&]() { return lm_head.forward(*execution, input); });
    return sample_greedy_cuda(logits.contiguous().view({input.size(0), -1}));
}

void CausalResources::native_begin_speculation() {}
void CausalResources::native_commit_speculation() {}
void CausalResources::native_rollback_speculation(int64_t) {}

} // namespace mfq::cuda
