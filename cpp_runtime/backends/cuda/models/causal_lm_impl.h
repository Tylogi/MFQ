#pragma once

#include "causal_lm.h"
#include "full_attention_session_codec.h"
#include "session_state.h"

namespace mfq::cuda {

template <typename Model>
int64_t CausalLm<Model>::vocab_size() const noexcept {
        return this->metadata.vocab_size;
    }

template <typename Model>
int64_t CausalLm<Model>::hidden_size() const noexcept {
        return this->metadata.hidden_size;
    }

template <typename Model>
int64_t CausalLm<Model>::num_hidden_layers() const noexcept {
        return this->metadata.num_hidden_layers;
    }

template <typename Model>
int64_t CausalLm<Model>::num_attention_heads() const noexcept {
        return this->metadata.num_attention_heads;
    }

template <typename Model>
int64_t CausalLm<Model>::num_key_value_heads() const noexcept {
        return this->metadata.num_key_value_heads;
    }

template <typename Model>
int64_t CausalLm<Model>::head_dim() const noexcept {
        return this->metadata.head_dim;
    }

template <typename Model>
int64_t CausalLm<Model>::max_position_embeddings() const noexcept {
        return this->metadata.max_position_embeddings;
    }

template <typename Model>
int64_t CausalLm<Model>::rotary_dim() const noexcept {
    return this->metadata.rotary_dim;
}

template <typename Model>
double CausalLm<Model>::rope_base() const noexcept {
    return this->metadata.rope_base;
}

template <typename Model>
void CausalLm<Model>::set_max_position_embeddings(int64_t value) noexcept {
        this->adapter_set_max_position_embeddings(value);
        this->metadata.max_position_embeddings = value;
    }

template <typename Model>
double CausalLm<Model>::rms_norm_eps() const noexcept {
        return this->metadata.rms_norm_eps;
    }

template <typename Model>
double CausalLm<Model>::norm_weight_offset() const noexcept {
        return this->metadata.norm_weight_offset;
    }

template <typename Model>
bool CausalLm<Model>::tie_word_embeddings() const noexcept {
        return this->metadata.tie_word_embeddings;
    }

template <typename Model>
int64_t CausalLm<Model>::num_experts() const noexcept {
        return this->metadata.num_experts;
    }

template <typename Model>
int64_t CausalLm<Model>::hc_mult() const noexcept {
        return this->metadata.hc_mult;
    }

template <typename Model>
double CausalLm<Model>::hc_eps() const noexcept {
        return this->metadata.hc_eps;
    }

template <typename Model>
double CausalLm<Model>::final_logit_softcapping() const noexcept {
        return this->metadata.final_logit_softcapping;
    }

template <typename Model>
double CausalLm<Model>::embedding_scale() const noexcept {
        return this->metadata.embedding_scale;
    }

template <typename Model>
std::string_view CausalLm<Model>::model_type() const noexcept {
        return this->metadata.model_type;
    }

template <typename Model>
std::string_view CausalLm<Model>::layer_type(int64_t layer) const {
        return this->metadata.layer_types.at(
            static_cast<std::size_t>(layer));
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::embed_forward(mfq_tensor_backend::Tensor ids) const {
        auto token_ids = ids.contiguous().to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64);
        return this->adapter_embed(
            quant_embedding_lookup(embed, token_ids));
    }

template <typename Model>
void CausalLm<Model>::reset(int64_t B) {
        cache_pos = 0;
        decode_position_delta = 0;
        this->adapter_reset(B);
        speculative_start = -1;
        speculative_confirmed = 0;
        speculative_suffix_forward = false;
        for (auto & b : blocks) {
            MfqCudaGuard guard(b->cuda_device);
            b->reset(B);
        }
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::hidden_forward(mfq_tensor_backend::Tensor ids,
                             MfqOptional<mfq_tensor_backend::Tensor> pos_override,
                             MfqOptional<mfq_tensor_backend::Tensor> seq_len,
                             std::vector<mfq_tensor_backend::Tensor> * block_trace,
                             MfqOptional<mfq_tensor_backend::Tensor> cache_positions_override,
                             mfq_tensor_backend::Tensor* raw_hidden,
                             int64_t confirmed_prefix,
                             int64_t planned_kv_length,
                             int64_t decode_attention_parts) {
        const int primary = this->execution->layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        ids = tensor_to_cuda_device(*this->execution,
            ids.to(mfq_tensor_backend::kInt64), primary);
        if (ids.dim() == 1) ids = ids.unsqueeze(0);
        auto x = this->execution->profiler.measure(
            "model.embed", [&]() { return embed_forward(ids); });
        return hidden_forward_inputs(
            ids, x, pos_override, seq_len, block_trace,
            mfq_nullopt, false, cache_positions_override, raw_hidden,
            confirmed_prefix, planned_kv_length, decode_attention_parts);
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::hidden_forward_speculative_suffix(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor* raw_hidden) {
        MFQ_RUNTIME_CHECK(
            speculative_start >= 0 && speculative_confirmed == 0 &&
                !speculative_suffix_forward,
            "DeepSeek-V4.1 speculative suffix is not active");
        speculative_suffix_forward = true;
        try {
            auto result = hidden_forward(
                std::move(ids), mfq_nullopt, mfq_nullopt,
                nullptr, mfq_nullopt, raw_hidden);
            speculative_suffix_forward = false;
            return result;
        } catch (...) {
            speculative_suffix_forward = false;
            throw;
        }
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::hidden_forward_inputs(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor input_embeddings,
        MfqOptional<mfq_tensor_backend::Tensor> pos_override,
        MfqOptional<mfq_tensor_backend::Tensor> seq_len,
        std::vector<mfq_tensor_backend::Tensor> * block_trace,
        MfqOptional<mfq_tensor_backend::Tensor> attention_mask,
        bool advance_cache_with_position_ids,
        MfqOptional<mfq_tensor_backend::Tensor> cache_positions_override,
        mfq_tensor_backend::Tensor* raw_hidden,
        int64_t confirmed_prefix,
        int64_t planned_kv_length,
        int64_t decode_attention_parts) {
        const int primary = this->execution->layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        ids = tensor_to_cuda_device(*this->execution,
            ids.to(mfq_tensor_backend::kInt64), primary);
        if (ids.dim() == 1) ids = ids.unsqueeze(0);
        if (input_embeddings.dim() != 3 ||
                input_embeddings.size(0) != ids.size(0) ||
                input_embeddings.size(1) != ids.size(1) ||
                input_embeddings.size(2) != hidden_size()) {
            throw std::runtime_error(
                "inputs_embeds shape must match [batch,tokens,hidden_size]");
        }
        const int64_t B = ids.size(0);
        const int64_t T = ids.size(1);
        if (this->adapter_requires_batch_reset(B)) reset(B);
        this->adapter_validate_forward(
            B, T, cache_pos, pos_override.has_value(),
            cache_positions_override.has_value(), attention_mask.has_value());
        MFQ_RUNTIME_CHECK(speculative_start < 0 || speculative_suffix_forward,
            "commit or roll back the pending speculative pass before forwarding");
        MFQ_RUNTIME_CHECK(confirmed_prefix >= 0 &&
            (confirmed_prefix == 0 || (confirmed_prefix < T && B == 1 && cache_pos > 0 &&
                (!pos_override.has_value() ||
                 this->adapter_allows_speculative_position_override()) &&
                !cache_positions_override.has_value() &&
                !attention_mask.has_value() && supports_speculation())),
            "unsupported speculative backbone geometry");
        if (confirmed_prefix > 0) {
            MFQ_RUNTIME_CHECK(
                cache_pos + T <= max_position_embeddings(),
                "speculative pass exceeds context capacity");
            speculative_start = cache_pos;
            speculative_confirmed = confirmed_prefix;
        }
        if (cache_pos == 0) reset(B);
        auto cache_positions = cache_positions_override.has_value()
            ? tensor_to_cuda_device(*this->execution,
                cache_positions_override.value(), primary)
                .to(mfq_tensor_backend::kInt64).contiguous()
            : mfq_tensor_backend::arange(
                cache_pos, cache_pos + T,
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                    .dtype(mfq_tensor_backend::kInt64));
        if (!((cache_positions.dim() == 1 && cache_positions.numel() == T) ||
              (cache_positions.dim() == 2 && cache_positions.size(0) == B &&
               cache_positions.size(1) == T))) {
            throw std::runtime_error(
                "cache_positions must have shape [tokens] or [batch,tokens]");
        }
        auto pos = pos_override.has_value()
            ? tensor_to_cuda_device(*this->execution,
                pos_override.value(), primary).to(mfq_tensor_backend::kInt64).contiguous()
            : (decode_position_delta == 0
                ? cache_positions
                : cache_positions + decode_position_delta);
        auto prepared_positions = this->adapter_prepare_positions(
            std::move(pos), B, T);
        pos = std::move(prepared_positions.positions);
        auto full_positions = std::move(prepared_positions.full_positions);
        this->adapter_validate_positions(
            pos, B, T, rope.sections.numel() > 0);
        if (attention_mask.has_value()) {
            auto mask = attention_mask.value();
            if (mask.dim() != 2 || mask.size(0) != B ||
                    mask.size(1) < cache_pos + T) {
                throw std::runtime_error(
                    "attention_mask must cover [batch,cached_plus_current_tokens]");
            }
        }
        auto effective_attention_mask = this->adapter_attention_mask(
            attention_mask, T, cache_pos);
        mfq_tensor_backend::Tensor cpu_ids, cpu_pos, cpu_cache_positions;
        MfqOptional<mfq_tensor_backend::Tensor> cpu_attention_mask = mfq_nullopt;
        MfqOptional<mfq_tensor_backend::Tensor> cpu_seq_len = mfq_nullopt;
        if (this->execution->dense_cpu_layer_count > 0) {
            cpu_ids = ids.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
            cpu_pos = pos.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
            cpu_cache_positions = cache_positions
                .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
            if (seq_len.has_value()) {
                cpu_seq_len = seq_len.value()
                    .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64).contiguous();
            }
            if (effective_attention_mask.has_value()) {
                cpu_attention_mask = effective_attention_mask.value()
                    .to(mfq_tensor_backend::kCPU).contiguous();
            }
        }
        auto x = this->adapter_prepare_hidden(
            tensor_to_cuda_device(*this->execution, input_embeddings, primary).contiguous(),
            B, T);
        this->adapter_begin_forward(raw_hidden != nullptr);
        if (block_trace != nullptr) block_trace->push_back(x.to(mfq_tensor_backend::kFloat32).clone());
        for (auto & b : blocks) {
            MfqCudaGuard block_guard(b->cuda_device);
            auto local_ids = b->cpu_offloaded
                ? cpu_ids
                : tensor_to_cuda_device(*this->execution, ids, b->cuda_device);
            auto local_pos = b->cpu_offloaded
                ? cpu_pos
                : tensor_to_cuda_device(*this->execution, pos, b->cuda_device);
            MfqOptional<mfq_tensor_backend::Tensor> local_cache_positions = mfq_nullopt;
            // Grid-MRoPE semantic coordinates never double as physical KV
            // slots, during either prepared prefill or delta-adjusted decode.
            if (this->adapter_pass_cache_positions(
                    rope.sections.numel() > 0,
                    cache_positions_override.has_value())) {
                local_cache_positions = b->cpu_offloaded
                    ? cpu_cache_positions
                    : tensor_to_cuda_device(*this->execution,
                        cache_positions, b->cuda_device);
            }
            MfqOptional<mfq_tensor_backend::Tensor> local_seq_len = mfq_nullopt;
            if (seq_len.has_value()) {
                local_seq_len = b->cpu_offloaded
                    ? cpu_seq_len.value()
                    : tensor_to_cuda_device(*this->execution,
                        seq_len.value(), b->cuda_device);
            }
            MfqOptional<mfq_tensor_backend::Tensor> local_attention_mask = mfq_nullopt;
            if (effective_attention_mask.has_value()) {
                local_attention_mask = b->cpu_offloaded
                    ? cpu_attention_mask.value()
                    : tensor_to_cuda_device(*this->execution,
                        effective_attention_mask.value(), b->cuda_device);
            }
            x = b->cpu_offloaded
                ? x.to(mfq_tensor_backend::kCPU).contiguous()
                : tensor_to_cuda_device(*this->execution, x, b->cuda_device);
            b->set_token_ids(local_ids);
            const RopeCache & active_rope = b->cpu_offloaded
                ? cpu_rope
                : (device_ropes.empty()
                    ? rope : device_ropes.at(b->cuda_device));
            Block::Context context;
            context.token_ids=local_ids;
            context.positions=local_pos;
            context.full_positions = this->adapter_block_positions(
                full_positions, local_pos, b->cuda_device);
            context.cache_position=cache_pos;
            context.confirmed_prefix=confirmed_prefix;
            context.planned_kv_length=planned_kv_length;
            context.decode_attention_parts=decode_attention_parts;
            context.sequence_lengths=local_seq_len;
            context.cache_positions=local_cache_positions;
            if (this->adapter_pass_attention_mask()) {
                context.attention_mask = local_attention_mask;
            }
            x = b->forward_context(
                *this->execution, std::move(x), context, active_rope);
            if (block_trace != nullptr) {
                block_trace->push_back(
                    tensor_to_cuda_device(*this->execution, x, primary)
                        .to(mfq_tensor_backend::kFloat32).clone());
            }
        }
        this->adapter_finish_forward(full_positions, B, T);
        if (this->adapter_force_cache_advance() ||
                !pos_override.has_value() ||
                advance_cache_with_position_ids) {
            cache_pos += T;
        }
        x = tensor_to_cuda_device(*this->execution, x, primary);
        auto finalized=finalize_hidden(x, B, T);
        if (raw_hidden != nullptr) {
            *raw_hidden = this->adapter_raw_hidden(x, finalized);
        }
        return finalized;
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::forward_inputs(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor input_embeddings,
        MfqOptional<mfq_tensor_backend::Tensor> pos_override,
        MfqOptional<mfq_tensor_backend::Tensor> seq_len) {
        return logits_from_hidden(hidden_forward_inputs(
            ids, input_embeddings, pos_override, seq_len));
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::apply_final_logit_softcap(
        mfq_tensor_backend::Tensor logits) const {
        const double cap = final_logit_softcapping();
        return cap > 0.0
            ? mfq_tensor_backend::tanh(logits / cap) * cap
            : logits;
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::logits_from_hidden(mfq_tensor_backend::Tensor y) {
        return this->adapter_logits(lm_head, std::move(y));
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::forward(mfq_tensor_backend::Tensor ids) {
        return logits_from_hidden(hidden_forward(ids));
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::last_logits_prepared(
        const CudaPreparedPrompt& prepared) {
        MFQ_RUNTIME_CHECK(
            prepared.transformed() && !prepared.token_ids.empty() &&
                prepared.embeddings.defined() && prepared.positions.defined() &&
                this->adapter_supports_prepared_prompt(),
            "invalid prepared prompt for CUDA text runtime");
        auto options = mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64)
            .device(mfq_tensor_backend::kCUDA);
        auto ids = mfq_tensor_backend::tensor(prepared.token_ids, options)
            .reshape({1, -1}).contiguous();
        reset(1);
        try {
            auto hidden = hidden_forward_inputs(
                ids, prepared.embeddings, prepared.positions,
                mfq_nullopt, nullptr, mfq_nullopt, true);
            decode_position_delta = prepared.decode_position_delta;
            auto last = hidden.index({Slice(), -1, Slice()})
                .to(mfq_tensor_backend::kFloat16).contiguous();
            return lm_head.forward(*this->execution, last);
        } catch (...) {
            reset(1);
            throw;
        }
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::last_logits(mfq_tensor_backend::Tensor ids) {
        MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt;
        const int64_t token_count = ids.dim() == 1 ? ids.size(0) : ids.size(1);
        const int64_t batch_size = ids.dim() == 1 ? 1 : ids.size(0);
        if (this->adapter_uses_decode_sequence_length() &&
                cache_pos > 0 && token_count == 1) {
            seq_len = mfq_tensor_backend::full(
                {batch_size}, cache_pos + 1,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
        }
        auto y = hidden_forward(ids, mfq_nullopt, seq_len);
        return this->adapter_last_logits(
            lm_head, y.index({Slice(), -1, Slice()}));
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::hidden_forward_static(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor pos,
        mfq_tensor_backend::Tensor seq_len,
        int64_t planned_kv_length,
        int64_t decode_attention_parts) {
        return hidden_forward(
            std::move(ids), pos, seq_len, nullptr, pos, nullptr, 0,
            planned_kv_length, decode_attention_parts);
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::last_logits_static(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor pos,
        mfq_tensor_backend::Tensor seq_len,
        int64_t planned_kv_length,
        int64_t decode_attention_parts) {
        auto y = hidden_forward_static(
            std::move(ids), std::move(pos), std::move(seq_len),
            planned_kv_length, decode_attention_parts);
        return this->adapter_last_logits(
            lm_head, y.index({Slice(), -1, Slice()}));
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::next_token(mfq_tensor_backend::Tensor ids) {
        MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt;
        const int64_t token_count = ids.dim() == 1 ? ids.size(0) : ids.size(1);
        const int64_t batch_size = ids.dim() == 1 ? 1 : ids.size(0);
        if (this->adapter_uses_decode_sequence_length() &&
                cache_pos > 0 && token_count == 1) {
            seq_len = mfq_tensor_backend::full(
                {batch_size}, cache_pos + 1,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
        }
        auto y = hidden_forward(ids, mfq_nullopt, seq_len);
        return next_token_from_hidden(y);
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::next_token_static(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor pos,
        mfq_tensor_backend::Tensor seq_len,
        int64_t planned_kv_length,
        int64_t decode_attention_parts) {
        auto y = hidden_forward_static(
            std::move(ids), std::move(pos), std::move(seq_len),
            planned_kv_length, decode_attention_parts);
        return next_token_from_hidden(y);
    }

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::next_token_from_hidden(mfq_tensor_backend::Tensor y) {
        return this->adapter_next_token(
            lm_head, y.index({Slice(), -1, Slice()}));
    }

template <typename Model>
bool CausalLm<Model>::supports_speculation() const {
    return this->adapter_supports_speculation() &&
        !blocks.empty() && std::all_of(
        blocks.begin(), blocks.end(),
        [](const auto& block) { return block->supports_speculation(); });
}

template <typename Model>
bool CausalLm<Model>::supports_suffix_speculation() const {
    return this->adapter_supports_suffix_speculation() &&
        !blocks.empty() && std::all_of(
        blocks.begin(), blocks.end(),
        [](const auto& block) { return block->supports_speculation(); });
}

template <typename Model>
void CausalLm<Model>::begin_speculative_suffix(int64_t draft_tokens) {
    MFQ_RUNTIME_CHECK(
        supports_suffix_speculation() &&
            speculative_start < 0 && draft_tokens > 0 &&
            cache_pos + draft_tokens <= max_position_embeddings(),
        "invalid speculative suffix");
    speculative_start = cache_pos;
    speculative_confirmed = 0;
    this->adapter_begin_speculative();
    std::size_t begun = 0;
    try {
        for (auto& block : blocks) {
            MfqCudaGuard guard(block->cuda_device);
            block->begin_speculative(draft_tokens);
            ++begun;
        }
    } catch (...) {
        for (std::size_t index = 0; index < begun; ++index) {
            try {
                MfqCudaGuard guard(blocks[index]->cuda_device);
                blocks[index]->commit_speculative();
            } catch (...) {}
        }
        try { this->adapter_rollback_speculative(cache_pos); } catch (...) {}
        speculative_start = -1;
        speculative_confirmed = 0;
        throw;
    }
}

template <typename Model>
void CausalLm<Model>::commit_speculative() {
    MFQ_RUNTIME_CHECK(
        speculative_start >= 0,
        "no speculative transaction to commit");
    for (auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        block->commit_speculative();
    }
    this->adapter_commit_speculative();
    speculative_start = -1;
    speculative_confirmed = 0;
}

template <typename Model>
void CausalLm<Model>::rollback_speculative(int64_t accepted_suffix) {
    MFQ_RUNTIME_CHECK(
        speculative_start >= 0 && accepted_suffix >= 0,
        "no speculative transaction to roll back");
    const int64_t keep =
        speculative_start + speculative_confirmed + accepted_suffix;
    for (auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        block->rollback_speculative(keep);
    }
    this->adapter_rollback_speculative(keep);
    cache_pos = keep;
    speculative_start = -1;
    speculative_confirmed = 0;
}

template <typename Model>
mfq_tensor_backend::Tensor CausalLm<Model>::finalize_hidden(
        mfq_tensor_backend::Tensor x, int64_t batch, int64_t tokens) {
    return this->adapter_finalize_hidden(
        std::move(x), output_norm, batch, tokens);
}

template <typename Model>
TextSessionStateKind CausalLm<Model>::text_session_state_kind() const {
    return CudaSessionCodec<Model>::kind(*this);
}

template <typename Model>
bool CausalLm<Model>::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

template <typename Model>
bool CausalLm<Model>::supports_paged_text_session_state() const {
    return CudaSessionCodec<Model>::supports_paged(*this);
}

template <typename Model>
TextSessionState CausalLm<Model>::capture_text_session_state(
        const std::vector<int64_t>& tokens) const {
    auto state = CudaSessionCodec<Model>::capture(*this, tokens);
    state.decode_position_delta = decode_position_delta;
    return state;
}

template <typename Model>
void CausalLm<Model>::restore_text_session_state(
        const TextSessionState& state) {
    CudaSessionCodec<Model>::restore(*this, state);
    decode_position_delta = state.decode_position_delta;
}

template <typename Model>
TextSessionStateKind CudaSessionCodec<Model>::kind(
        const CausalLm<Model>&) {
    return TextSessionStateKind::Unsupported;
}

template <typename Model>
bool CudaSessionCodec<Model>::supports_paged(
        const CausalLm<Model>&) {
    return false;
}

template <typename Model>
TextSessionState CudaSessionCodec<Model>::capture(
        const CausalLm<Model>&,
        const std::vector<int64_t>&) {
    throw std::runtime_error(
        "text session state is unsupported by this model");
}

template <typename Model>
void CudaSessionCodec<Model>::restore(
        CausalLm<Model>&,
        const TextSessionState&) {
    throw CudaSessionStateError(
        "text session state is unsupported by this model");
}

template <typename Model>
TextSessionStateKind FullAttentionSessionCodec<Model>::kind(
        const CausalLm<Model>& model) {
    return !model.blocks.empty() && std::all_of(
        model.blocks.begin(), model.blocks.end(),
        [](const std::unique_ptr<Block>& block) {
            return dynamic_cast<const FullBlock*>(block.get()) != nullptr;
        })
        ? TextSessionStateKind::FullAttention
        : TextSessionStateKind::Unsupported;
}

template <typename Model>
bool FullAttentionSessionCodec<Model>::supports_paged(
        const CausalLm<Model>& model) {
    return kind(model) == TextSessionStateKind::FullAttention &&
        std::all_of(
            model.blocks.begin(), model.blocks.end(),
            [](const std::unique_ptr<Block>& block) {
                const auto* full =
                    dynamic_cast<const FullBlock*>(block.get());
                return full != nullptr && !full->sliding;
            });
}

template <typename Model>
TextSessionState FullAttentionSessionCodec<Model>::capture(
        const CausalLm<Model>& model,
        const std::vector<int64_t>& tokens) {
    if (kind(model) != TextSessionStateKind::FullAttention) {
        throw std::runtime_error(
            "text session state is unsupported by this block layout");
    }
    if (model.cache_pos <= 0 ||
            static_cast<size_t>(model.cache_pos) != tokens.size()) {
        throw std::runtime_error(
            "text session token count does not match the model cache");
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = model.cache_pos;
    state.payload = std::vector<FullBlockSessionState>{};
    auto& layers = std::get<std::vector<FullBlockSessionState>>(state.payload);
    layers.reserve(model.blocks.size());
    for (const auto& block : model.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        const auto* full = dynamic_cast<const FullBlock*>(block.get());
        if (full == nullptr) {
            throw std::runtime_error(
                "full-attention session layer changed");
        }
        layers.push_back(capture_full_attention_session_state(
            *full, model.cache_pos, state.bytes));
    }
    return state;
}

template <typename Model>
void FullAttentionSessionCodec<Model>::restore(
        CausalLm<Model>& model,
        const TextSessionState& state) {
    const auto* layers = std::get_if<
        std::vector<FullBlockSessionState>>(&state.payload);
    if (kind(model) != TextSessionStateKind::FullAttention ||
            state.kind() != TextSessionStateKind::FullAttention ||
            state.cache_pos <= 0 ||
            static_cast<size_t>(state.cache_pos) != state.tokens.size() ||
            layers == nullptr || layers->size() != model.blocks.size()) {
        throw CudaSessionStateError(
            "full-attention text session state is incompatible");
    }
    for (size_t index = 0; index < model.blocks.size(); ++index) {
        auto& block = model.blocks[index];
        MfqCudaGuard guard(block->cuda_device);
        auto* full = dynamic_cast<FullBlock*>(block.get());
        if (full == nullptr) {
            throw CudaSessionStateError(
                "full-attention session layer changed");
        }
        restore_full_attention_session_state(
            *full, (*layers)[index]);
    }
    model.cache_pos = state.cache_pos;
}

} // namespace mfq::cuda
