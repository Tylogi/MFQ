#pragma once

#include "qwen35_causal_lm.h"

// Qwen3.5's predictor shares the main embedding/output head and owns only
// its fusion/norm/attention/FFN weights and an independent attention history.
struct Qwen35Mtp final : MtpModule {
    mfq::cuda::qwen35::Config config;
    QuantLinear fusion;
    mfq_tensor_backend::Tensor hidden_norm, embedding_norm, output_norm;
    std::vector<std::unique_ptr<Block>> blocks;
    int64_t cache_pos = 0;

    static std::optional<Qwen35Mtp> load_if_present(
            const mfq::ModelSource& file,
            const mfq::cuda::qwen35::Config& main) {
        const bool fusion_present = has_tensor(file, "predictor.fusion.weight");
        const bool any = fusion_present ||
            has_tensor(file, "predictor.hidden_norm.weight") ||
            has_tensor(file, "predictor.embedding_norm.weight") ||
            has_tensor(file, "predictor.output_norm.weight") ||
            has_tensor(file, "predictor.block.0.attention.query.weight");
        if (!fusion_present) {
            MFQ_RUNTIME_CHECK(!any, "Qwen model source contains an incomplete MTP head");
            return std::nullopt;
        }
        MFQ_RUNTIME_CHECK(
            main.mtp_num_hidden_layers > 0 &&
                !main.mtp_use_dedicated_embeddings,
            "Qwen MTP requires declared predictor layers and shared embeddings");
        Qwen35Mtp result;
        result.config = main;
        result.hidden_norm = load_dense_gpu(
            file, "predictor.hidden_norm.weight");
        result.embedding_norm = load_dense_gpu(
            file, "predictor.embedding_norm.weight");
        result.output_norm = load_dense_gpu(
            file, "predictor.output_norm.weight");
        result.fusion = load_quant_linear(file, "predictor.fusion.weight");
        MFQ_RUNTIME_CHECK(
            result.fusion.neuron_len() == 2 * main.hidden_size &&
                result.fusion.out() == main.hidden_size &&
                result.hidden_norm.numel() == main.hidden_size &&
                result.embedding_norm.numel() == main.hidden_size &&
                result.output_norm.numel() == main.hidden_size,
            "Qwen MTP component dimensions disagree with the backbone");
        for (int layer = 0; layer < main.mtp_num_hidden_layers; ++layer) {
            auto block = mfq::cuda::qwen35::load_block(
                file, result.config, layer, "full_attention", "predictor");
            block->cuda_device = g_layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        return result;
    }

    void reset(int64_t batch = 1) override {
        cache_pos = 0;
        for (auto& block : blocks) block->reset(batch);
    }

    mfq_tensor_backend::Tensor forward(
            const MtpTarget& target,
            mfq_tensor_backend::Tensor hidden,
            mfq_tensor_backend::Tensor next_ids) override {
        return evaluate(
            target, std::move(hidden), std::move(next_ids), {});
    }

    MtpStep step_positioned(
            const MtpTarget& target,
            mfq_tensor_backend::Tensor hidden,
            mfq_tensor_backend::Tensor next_ids,
            mfq_tensor_backend::Tensor positions) override {
        auto output = evaluate(
            target, std::move(hidden), std::move(next_ids),
            std::move(positions));
        return {output, output};
    }

    mfq_tensor_backend::Tensor evaluate(
            const MtpTarget& target,
            mfq_tensor_backend::Tensor hidden,
            mfq_tensor_backend::Tensor next_ids,
            mfq_tensor_backend::Tensor positions) {
        MFQ_RUNTIME_CHECK(
            hidden.dim() == 3 && next_ids.dim() == 2 &&
                hidden.size(0) == next_ids.size(0) &&
                hidden.size(1) == next_ids.size(1) &&
                hidden.size(2) == config.hidden_size &&
                hidden.size(1) > 0,
            "Qwen MTP inputs must be matching [B,T,H] hidden states and "
            "[B,T] next-token IDs");
        const auto batch = hidden.size(0);
        const auto tokens = hidden.size(1);
        MFQ_RUNTIME_CHECK(
            cache_pos + tokens <= config.max_position_embeddings,
            "Qwen MTP history exceeds context capacity");
        auto embedded = target.embed(next_ids).to(hidden.scalar_type());
        auto e = qwen_rms_norm(
            embedded.reshape({batch * tokens, config.hidden_size})
                .to(mfq_tensor_backend::kFloat32),
            embedding_norm, config.rms_norm_eps, 1.0)
            .reshape_as(hidden);
        auto h = qwen_rms_norm(
            hidden.reshape({batch * tokens, config.hidden_size})
                .to(mfq_tensor_backend::kFloat32),
            hidden_norm, config.rms_norm_eps, 1.0)
            .reshape_as(hidden);
        trace_gemma_stage(0, "mtp.embedding_norm", e);
        trace_gemma_stage(0, "mtp.hidden_norm", h);
        auto x = fusion.forward(mfq_tensor_backend::cat({e, h}, -1));
        trace_gemma_stage(0, "mtp.fusion", x);
        auto pos = positions.defined()
            ? tensor_to_cuda_device(
                  positions, g_layer_placement.primary_device())
                  .to(mfq_tensor_backend::kInt64).contiguous()
            : mfq_tensor_backend::arange(
                  cache_pos, cache_pos + tokens,
                  mfq_tensor_backend::TensorOptions()
                      .device(mfq_tensor_backend::kCUDA)
                      .dtype(mfq_tensor_backend::kInt64));
        MFQ_RUNTIME_CHECK(
            (pos.dim() == 1 && pos.numel() == tokens) ||
                (pos.dim() == 2 && pos.size(1) == tokens &&
                 (pos.size(0) == batch ||
                  (pos.size(0) == 3 && !config.mrope_sections.empty()))),
            "Qwen MTP positions must have shape [T], [B,T], or "
            "configured grid-MRoPE [3,T]");
        MfqOptional<mfq_tensor_backend::Tensor> length = mfq_nullopt;
        if (tokens == 1 && cache_pos > 0) {
            length = mfq_tensor_backend::full(
                {batch}, cache_pos + 1, pos.options());
        }
        MfqOptional<mfq_tensor_backend::Tensor> cache_positions = mfq_nullopt;
        if (positions.defined()) {
            cache_positions = mfq_tensor_backend::arange(
                cache_pos, cache_pos + tokens, pos.options());
        }
        for (auto& block : blocks) {
            x = block->forward(
                x, pos, cache_pos, length, *target.rope,
                cache_positions);
        }
        cache_pos += tokens;
        auto output = qwen_rms_norm(
            x.reshape({batch * tokens, config.hidden_size})
                .to(mfq_tensor_backend::kFloat32),
            output_norm, config.rms_norm_eps, 1.0)
            .reshape({batch, tokens, config.hidden_size});
        trace_gemma_stage(0, "mtp.output_norm", output);
        return output;
    }

    int64_t cache_position() const noexcept override { return cache_pos; }

    void trim_cache_to(int64_t position) override {
        MFQ_RUNTIME_CHECK(
            position >= 0 && position <= cache_pos,
            "Qwen MTP cache trim position is invalid");
        // Full-attention cache storage is position addressed. Lowering the
        // logical boundary makes the next predictor pass overwrite the
        // discarded draft suffix without copying history-sized tensors.
        cache_pos = position;
    }

    bool supports_session_state() const noexcept override { return true; }

    MtpSessionState capture_session_state(
            int64_t target_cache_position,
            const mfq_tensor_backend::Tensor& last_target_hidden) const override {
        const int64_t predictor_position = target_cache_position - 1;
        if (predictor_position <= 0 || predictor_position > cache_pos ||
                !last_target_hidden.defined() ||
                last_target_hidden.dim() != 3 ||
                last_target_hidden.size(0) != 1 ||
                last_target_hidden.size(1) != 1 ||
                last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error(
                "Qwen MTP session boundary is unavailable");
        }
        MtpSessionState state;
        state.cache_pos = predictor_position;
        state.blocks.reserve(blocks.size());
        for (const auto& block : blocks) {
            MfqCudaGuard guard(block->cuda_device);
            const auto* full = dynamic_cast<const FullBlock*>(block.get());
            if (full == nullptr) {
                throw std::runtime_error(
                    "Qwen MTP session layer type changed");
            }
            state.blocks.push_back(capture_full_attention_session_state(
                *full, predictor_position, state.bytes));
        }
        state.last_target_hidden = last_target_hidden.clone();
        state.bytes += session_tensor_bytes(state.last_target_hidden);
        return state;
    }

    void restore_session_state(const MtpSessionState& state) override {
        if (state.cache_pos <= 0 || state.blocks.size() != blocks.size() ||
                !state.last_target_hidden.defined() ||
                state.last_target_hidden.dim() != 3 ||
                state.last_target_hidden.size(0) != 1 ||
                state.last_target_hidden.size(1) != 1 ||
                state.last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Qwen MTP session state is incompatible");
        }
        for (std::size_t index = 0; index < blocks.size(); ++index) {
            auto& block = blocks[index];
            MfqCudaGuard guard(block->cuda_device);
            auto* full = dynamic_cast<FullBlock*>(block.get());
            if (full == nullptr) {
                throw std::runtime_error(
                    "Qwen MTP session layer type changed");
            }
            restore_full_attention_session_state(
                *full, state.blocks[index]);
        }
        cache_pos = state.cache_pos;
    }

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }
};
