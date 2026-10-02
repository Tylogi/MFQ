#pragma once

#include "../mtp.h"
#include "models/full_block.h"
#include "models/qwen35/causal_lm.h"
#include "ops.h"

// Qwen3.5's predictor shares the main embedding/output head and owns only
// its fusion/norm/attention/FFN weights and an independent attention history.
struct Qwen35Mtp final : MtpModule {
    mfq::cuda::qwen35::Config config;
    QuantLinear fusion;
    mfq_tensor_backend::Tensor hidden_norm, embedding_norm, output_norm;
    std::vector<std::unique_ptr<Block>> blocks;
    CudaExecutionContext *execution = nullptr;
    int64_t cache_pos = 0;

    static std::optional<Qwen35Mtp> load_if_present(const mfq::ModelSource &file,
                                                    const mfq::cuda::qwen35::Config &main,
                                                    CudaExecutionContext &execution) {
        const bool fusion_present = has_tensor(file, "predictor.fusion.weight");
        const bool any = fusion_present || has_tensor(file, "predictor.hidden_norm.weight") ||
                         has_tensor(file, "predictor.embedding_norm.weight") ||
                         has_tensor(file, "predictor.output_norm.weight") ||
                         has_tensor(file, "predictor.block.0.attention.query.weight");
        if (!fusion_present) {
            MFQ_RUNTIME_CHECK(!any, "Qwen model source contains an incomplete MTP head");
            return std::nullopt;
        }
        MFQ_RUNTIME_CHECK(main.mtp_num_hidden_layers > 0 && !main.mtp_use_dedicated_embeddings,
                          "Qwen MTP requires declared predictor layers and shared embeddings");
        Qwen35Mtp result;
        result.config = main;
        result.execution = &execution;
        result.hidden_norm = load_dense_gpu(execution, file, "predictor.hidden_norm.weight");
        result.embedding_norm = load_dense_gpu(execution, file, "predictor.embedding_norm.weight");
        result.output_norm = load_dense_gpu(execution, file, "predictor.output_norm.weight");
        result.fusion = load_quant_linear(execution, file, "predictor.fusion.weight");
        MFQ_RUNTIME_CHECK(result.fusion.neuron_len() == 2 * main.hidden_size &&
                              result.fusion.out() == main.hidden_size &&
                              result.hidden_norm.numel() == main.hidden_size &&
                              result.embedding_norm.numel() == main.hidden_size &&
                              result.output_norm.numel() == main.hidden_size,
                          "Qwen MTP component dimensions disagree with the backbone");
        for (int layer = 0; layer < main.mtp_num_hidden_layers; ++layer) {
            auto block = mfq::cuda::qwen35::load_block(execution, file, result.config, layer,
                                                       "full_attention", "predictor");
            block->cuda_device = execution.layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch = 1) override {
        cache_pos = 0;
        for (auto &block : blocks)
            block->reset(batch);
    }

    mfq_tensor_backend::Tensor forward(const MtpTarget &target, mfq_tensor_backend::Tensor hidden,
                                       mfq_tensor_backend::Tensor next_ids) override {
        return evaluate(target, std::move(hidden), std::move(next_ids), {});
    }

    MtpStep step_positioned(const MtpTarget &target, mfq_tensor_backend::Tensor hidden,
                            mfq_tensor_backend::Tensor next_ids,
                            mfq_tensor_backend::Tensor positions) override {
        auto output =
            evaluate(target, std::move(hidden), std::move(next_ids), std::move(positions));
        return {output, output};
    }

    mfq_tensor_backend::Tensor evaluate(const MtpTarget &target, mfq_tensor_backend::Tensor hidden,
                                        mfq_tensor_backend::Tensor next_ids,
                                        mfq_tensor_backend::Tensor positions) {
        struct Ops {
            using Tensor = mfq_tensor_backend::Tensor;
            Qwen35Mtp &model;
            const MtpTarget &target;
            static int rank(const Tensor &x) { return x.dim(); }
            static int64_t size(const Tensor &x, int axis) { return x.size(axis); }
            static int64_t elements(const Tensor &x) { return x.numel(); }
            static bool defined(const Tensor &x) { return x.defined(); }
            Tensor embed(Tensor ids, const Tensor &like) {
                return target.embed(ids).to(like.scalar_type());
            }
            Tensor normalize(Tensor x, mfq::models::qwen35::PredictorNorm role) {
                using Norm = mfq::models::qwen35::PredictorNorm;
                const auto &weight = role == Norm::embedding ? model.embedding_norm
                                     : role == Norm::hidden  ? model.hidden_norm
                                                             : model.output_norm;
                return qwen_rms_norm(x.reshape({-1, model.config.hidden_size})
                                         .to(mfq_tensor_backend::kFloat32),
                                     weight, model.config.rms_norm_eps,
                                     model.config.legacy_tensor_layout.norm_weight_offset)
                    .reshape(x.sizes());
            }
            void trace(const char *stage, const Tensor &x) {
                trace_gemma_stage(*model.execution, 0, stage, x);
            }
            Tensor fuse(Tensor e, Tensor h) {
                return model.fusion.forward(*model.execution, mfq_tensor_backend::cat({e, h}, -1));
            }
            Tensor positions(Tensor explicit_pos, int64_t start, int64_t tokens) {
                return explicit_pos.defined()
                           ? tensor_to_cuda_device(
                                 model.execution->model_parallel_collectives, explicit_pos,
                                 model.execution->layer_placement.primary_device())
                                 .to(mfq_tensor_backend::kInt64)
                                 .contiguous()
                           : mfq_tensor_backend::arange(start, start + tokens,
                                                        mfq_tensor_backend::TensorOptions()
                                                            .device(mfq_tensor_backend::kCUDA)
                                                            .dtype(mfq_tensor_backend::kInt64));
            }
            static Tensor lengths(int64_t batch, int64_t end, const Tensor &positions) {
                return mfq_tensor_backend::full({batch}, end, positions.options());
            }
            static Tensor cache_positions(const Tensor &positions, int64_t start, int64_t tokens) {
                return mfq_tensor_backend::arange(start, start + tokens, positions.options());
            }
            Tensor layer(std::unique_ptr<Block> &block, Tensor hidden, const Tensor &pos,
                         const std::optional<Tensor> &lengths,
                         const std::optional<Tensor> &cache_positions) {
                return block->forward(*model.execution, hidden, pos, model.cache_pos, lengths,
                                      *target.rope, cache_positions);
            }
        } ops{*this, target};
        return mfq::models::qwen35::mtp_predictor(ops, std::move(hidden), std::move(next_ids),
                                                  std::move(positions));
    }

    int64_t cache_position() const noexcept override { return cache_pos; }

    void trim_cache_to(int64_t position) override {
        MFQ_RUNTIME_CHECK(position >= 0 && position <= cache_pos,
                          "Qwen MTP cache trim position is invalid");
        // Full-attention cache storage is position addressed. Lowering the
        // logical boundary makes the next predictor pass overwrite the
        // discarded draft suffix without copying history-sized tensors.
        cache_pos = position;
    }

    bool supports_session_state() const noexcept override { return true; }

    MtpSessionState
    capture_session_state(int64_t target_cache_position,
                          const mfq_tensor_backend::Tensor &last_target_hidden) const override {
        const int64_t predictor_position = target_cache_position - 1;
        if (predictor_position <= 0 || predictor_position > cache_pos ||
            !last_target_hidden.defined() || last_target_hidden.dim() != 3 ||
            last_target_hidden.size(0) != 1 || last_target_hidden.size(1) != 1 ||
            last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Qwen MTP session boundary is unavailable");
        }
        MtpSessionState state;
        state.cache_pos = predictor_position;
        state.blocks.reserve(blocks.size());
        for (const auto &block : blocks) {
            MfqCudaGuard guard(block->cuda_device);
            const auto *full = dynamic_cast<const FullBlock *>(block.get());
            if (full == nullptr) {
                throw std::runtime_error("Qwen MTP session layer type changed");
            }
            state.blocks.push_back(
                capture_full_attention_session_state(*full, predictor_position, state.bytes));
        }
        state.last_target_hidden = last_target_hidden.clone();
        state.bytes += session_tensor_bytes(state.last_target_hidden);
        return state;
    }

    void restore_session_state(const MtpSessionState &state) override {
        if (state.cache_pos <= 0 || state.blocks.size() != blocks.size() ||
            !state.last_target_hidden.defined() || state.last_target_hidden.dim() != 3 ||
            state.last_target_hidden.size(0) != 1 || state.last_target_hidden.size(1) != 1 ||
            state.last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Qwen MTP session state is incompatible");
        }
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            auto &block = blocks[index];
            MfqCudaGuard guard(block->cuda_device);
            auto *full = dynamic_cast<FullBlock *>(block.get());
            if (full == nullptr) {
                throw std::runtime_error("Qwen MTP session layer type changed");
            }
            restore_full_attention_session_state(*full, state.blocks[index]);
        }
        cache_pos = state.cache_pos;
    }

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }
};
