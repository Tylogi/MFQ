#include "ops.h"
#include "../session_codec_impl.h"

namespace mfq::cuda::qwen4_exp {

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const mfq::models::qwen4_exp::Config &config, int layer) {
    return std::make_unique<Qwen4Block>(execution, source, config, layer);
}

std::unique_ptr<Gr> load_final_mixer(CudaExecutionContext &execution,
                                     const mfq::ModelSource &source,
                                     const mfq::models::qwen4_exp::Config &config) {
    return std::make_unique<Gr>(execution, source, config, "model.mhc.pre", false);
}

Tensor finalize_hidden(const Gr &mixer, const Tensor &hidden) { return mixer.pre(hidden)[0]; }

} // namespace mfq::cuda::qwen4_exp

namespace mfq::cuda {

void Qwen4Model::adapter_validate_load_options() const {
    qwen4_exp::validate_load_options(*execution);
}

void Qwen4Model::adapter_load_final_state(const mfq::ModelSource &source,
                                          mfq_tensor_backend::Tensor &output_norm) {
    final_mixer = qwen4_exp::load_final_mixer(*execution, source, config);
    output_norm = mfq_tensor_backend::Tensor();
}

std::unique_ptr<Block> Qwen4Model::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                      int, const std::string &) {
    return qwen4_exp::load_block(*execution, source, config, layer);
}

void Qwen4Model::adapter_reset(int64_t new_batch) {
    positions = mfq_tensor_backend::Tensor();
    batch = new_batch;
}

CudaPreparedPositions Qwen4Model::adapter_prepare_positions(mfq_tensor_backend::Tensor current,
                                                            int64_t, int64_t) {
    if ((current.dim() == 2 || current.dim() == 3) && current.size(0) == 4) {
        current = current.narrow(0, 1, 3);
    }
    auto full = current;
    if (positions.defined()) {
        full = mfq_tensor_backend::cat({positions, current}, -1);
    }
    return {std::move(current), std::move(full)};
}

void Qwen4Model::adapter_validate_positions(const mfq_tensor_backend::Tensor &current,
                                            int64_t batch_size, int64_t tokens, bool) const {
    MFQ_RUNTIME_CHECK(
        (current.dim() == 1 || (current.dim() == 2 && current.size(0) == 3) ||
         (current.dim() == 3 && current.size(0) == 3 && current.size(1) == batch_size)) &&
            current.size(-1) == tokens,
        "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_prepare_hidden(mfq_tensor_backend::Tensor hidden,
                                                              int64_t, int64_t) const {
    return hidden.to(mfq_tensor_backend::kFloat16).repeat({1, 1, metadata.hc_mult});
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_block_positions(const mfq_tensor_backend::Tensor &full_positions,
                                    const mfq_tensor_backend::Tensor &, int device) const {
    return tensor_to_cuda_device(execution->model_parallel_collectives, full_positions, device);
}

void Qwen4Model::adapter_finish_forward(const mfq_tensor_backend::Tensor &full_positions,
                                        int64_t new_batch, int64_t) {
    positions = full_positions;
    batch = new_batch;
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                               const mfq_tensor_backend::Tensor &,
                                                               int64_t, int64_t) const {
    return qwen4_exp::finalize_hidden(*final_mixer, hidden);
}

mfq_tensor_backend::Tensor
Qwen4Model::adapter_last_logits(const QuantLinear &lm_head,
                                mfq_tensor_backend::Tensor hidden) const {
    return adapter_logits(lm_head, std::move(hidden));
}

mfq_tensor_backend::Tensor Qwen4Model::adapter_next_token(const QuantLinear &lm_head,
                                                          mfq_tensor_backend::Tensor hidden) const {
    return mfq_tensor_backend::argmax(adapter_logits(lm_head, std::move(hidden)), -1)
        .to(mfq_tensor_backend::kInt64);
}

void Qwen4Model::adapter_rollback_speculative(int64_t keep) {
    if (positions.defined())
        positions = positions.narrow(-1, 0, keep);
}

} // namespace mfq::cuda

namespace mfq::cuda {

template struct CudaSessionCodec<Qwen4Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace mfq::models
