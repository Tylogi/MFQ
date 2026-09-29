#include "causal_lm.h"

void CudaPagedWriter::raw(const void * data, size_t size) {
        const auto * begin = static_cast<const uint8_t *>(data);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

void CudaPagedWriter::tensor(const mfq_tensor_backend::Tensor & source) {
        if (!source.defined() || !source.is_cuda() ||
                source.dim() <= 0 || source.dim() > 8) {
            throw std::runtime_error(
                "invalid CUDA paged cache tensor");
        }
        const auto dtype = source.scalar_type();
        if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
                dtype != mfq_tensor_backend::kFloat32) {
            throw std::runtime_error(
                "unsupported CUDA paged cache tensor dtype");
        }
        const auto cpu = source.to(mfq_tensor_backend::kCPU).contiguous();
        scalar<int32_t>(static_cast<int32_t>(dtype));
        scalar<int32_t>(source.get_device());
        scalar<uint32_t>(static_cast<uint32_t>(cpu.dim()));
        scalar<uint32_t>(0);
        for (const auto dimension : cpu.sizes()) scalar<int64_t>(dimension);
        const auto bytes = static_cast<uint64_t>(
            cpu.numel() * cpu.element_size());
        scalar<uint64_t>(bytes);
        raw(cpu.data_ptr(), static_cast<size_t>(bytes));
    }

CudaPagedPayload CudaPagedWriter::finish() && {
        return std::make_shared<const std::vector<uint8_t>>(
            std::move(bytes_));
    }

void CudaPagedReader::raw(void * destination, size_t size, const char * name) {
        if (remaining() < size) {
            throw CudaSessionStateError(
                std::string("truncated CUDA paged cache ") + name);
        }
        std::memcpy(destination, cursor_, size);
        cursor_ += size;
    }

std::pair<mfq_tensor_backend::Tensor, int> CudaPagedReader::tensor() {
        const auto dtype_value = scalar<int32_t>("dtype");
        const auto device = scalar<int32_t>("device");
        const auto rank = scalar<uint32_t>("rank");
        (void)scalar<uint32_t>("tensor flags");
        if (rank == 0 || rank > 8 || device < 0) {
            throw CudaSessionStateError(
                "invalid CUDA paged cache tensor header");
        }
        const auto dtype = static_cast<mfq_tensor_backend::ScalarType>(dtype_value);
        if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
                dtype != mfq_tensor_backend::kFloat32) {
            throw CudaSessionStateError(
                "invalid CUDA paged cache tensor dtype");
        }
        std::vector<int64_t> shape;
        shape.reserve(rank);
        uint64_t elements = 1;
        for (uint32_t index = 0; index < rank; ++index) {
            const auto dimension = scalar<int64_t>("shape");
            if (dimension <= 0 || elements >
                    std::numeric_limits<uint64_t>::max() /
                        static_cast<uint64_t>(dimension)) {
                throw CudaSessionStateError(
                    "invalid CUDA paged cache tensor shape");
            }
            shape.push_back(dimension);
            elements *= static_cast<uint64_t>(dimension);
        }
        const auto bytes = scalar<uint64_t>("tensor size");
        const uint64_t element_size =
            dtype == mfq_tensor_backend::kFloat32 ? 4 : 2;
        if (elements > std::numeric_limits<uint64_t>::max() / element_size ||
                bytes != elements * element_size || bytes > remaining()) {
            throw CudaSessionStateError(
                "invalid CUDA paged cache tensor size");
        }
        auto cpu = mfq_tensor_backend::empty(
            shape,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(dtype));
        raw(cpu.data_ptr(), static_cast<size_t>(bytes), "tensor");
        return {std::move(cpu), device};
    }

void CudaPagedReader::expect_end() const {
        if (cursor_ != end_) {
            throw CudaSessionStateError(
                "trailing CUDA paged cache payload bytes");
        }
    }

size_t CudaPagedReader::remaining() const {
        return static_cast<size_t>(end_ - cursor_);
    }

namespace mfq::cuda {

namespace {

mfq_tensor_backend::Tensor softcap_logits(
        mfq_tensor_backend::Tensor logits,
        double cap) {
    return cap > 0.0
        ? mfq_tensor_backend::tanh(logits / cap) * cap
        : logits;
}

mfq_tensor_backend::Tensor normalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        const CausalLmMetadata& metadata,
        int64_t batch,
        int64_t tokens,
        bool bfloat16) {
    return g_profiler.measure("model.output_norm", [&]() {
        auto flat = hidden.reshape(
            {batch * tokens, metadata.hidden_size});
        if (bfloat16) {
            return qwen_rms_norm_bf16(
                flat, output_norm, metadata.rms_norm_eps,
                metadata.norm_weight_offset)
                .reshape({batch, tokens, metadata.hidden_size});
        }
        return qwen_rms_norm(
            flat.to(mfq_tensor_backend::kFloat32), output_norm,
            metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

} // namespace

void CausalLmArchitecture::adapter_validate_load_options() const {}

bool CausalLmArchitecture::adapter_uses_common_rope() const noexcept {
    return false;
}

void CausalLmArchitecture::adapter_configure_rope(
        RopeCache&,
        mfq_tensor_backend::Device) const {}

void CausalLmArchitecture::adapter_load_final_state(
        const mfq::ModelSource& source,
        mfq_tensor_backend::Tensor& output_norm) {
    output_norm = load_dense_gpu(source, "model.output_norm.weight");
}

void CausalLmArchitecture::adapter_prepare_blocks(
        const mfq::ModelSource&) {}

bool CausalLmArchitecture::adapter_supports_dense_cpu_offload() const noexcept {
    return false;
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_embed(
        mfq_tensor_backend::Tensor output) const {
    return output;
}

void CausalLmArchitecture::adapter_reset(int64_t) {}

bool CausalLmArchitecture::adapter_requires_batch_reset(int64_t) const noexcept {
    return false;
}

void CausalLmArchitecture::adapter_validate_forward(
        int64_t, int64_t, int64_t, bool, bool, bool) const {}

bool CausalLmArchitecture::adapter_allows_speculative_position_override()
        const noexcept {
    return false;
}

CudaPreparedPositions CausalLmArchitecture::adapter_prepare_positions(
        mfq_tensor_backend::Tensor positions,
        int64_t,
        int64_t) {
    return {positions, positions};
}

void CausalLmArchitecture::adapter_validate_positions(
        const mfq_tensor_backend::Tensor& positions,
        int64_t batch,
        int64_t tokens,
        bool) const {
    if (!((positions.dim() == 1 && positions.numel() == tokens) ||
          (positions.dim() == 2 && positions.size(0) == batch &&
           positions.size(1) == tokens))) {
        throw std::runtime_error(
            "position_ids must have shape [tokens] or [batch,tokens]");
    }
}

MfqOptional<mfq_tensor_backend::Tensor>
CausalLmArchitecture::adapter_attention_mask(
        MfqOptional<mfq_tensor_backend::Tensor> mask,
        int64_t,
        int64_t) const {
    return mask;
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t,
        int64_t) const {
    return hidden;
}

void CausalLmArchitecture::adapter_begin_forward(bool) {}

bool CausalLmArchitecture::adapter_pass_cache_positions(
        bool has_mrope,
        bool has_override) const noexcept {
    return has_mrope || has_override;
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_block_positions(
        const mfq_tensor_backend::Tensor&,
        const mfq_tensor_backend::Tensor& local_positions,
        int) const {
    return local_positions;
}

bool CausalLmArchitecture::adapter_pass_attention_mask() const noexcept {
    return false;
}

void CausalLmArchitecture::adapter_finish_forward(
        const mfq_tensor_backend::Tensor&,
        int64_t,
        int64_t) {}

bool CausalLmArchitecture::adapter_force_cache_advance() const noexcept {
    return false;
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const {
    return normalize_hidden(
        std::move(hidden), output_norm, metadata, batch, tokens, false);
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_raw_hidden(
        const mfq_tensor_backend::Tensor& hidden,
        const mfq_tensor_backend::Tensor&) const {
    return hidden;
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const {
    auto logits = g_profiler.measure(
        "model.lm_head", [&]() { return lm_head.forward(hidden); });
    return softcap_logits(
        std::move(logits), metadata.final_logit_softcapping);
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const {
    return softcap_logits(
        lm_head.forward(
            hidden.to(mfq_tensor_backend::kFloat16).contiguous()),
        metadata.final_logit_softcapping);
}

mfq_tensor_backend::Tensor CausalLmArchitecture::adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const {
    auto input = hidden.to(mfq_tensor_backend::kFloat16).contiguous();
    auto logits = g_profiler.measure(
        "model.lm_head", [&]() { return lm_head.forward(input); });
    return sample_greedy_cuda(
        logits.contiguous().view({input.size(0), -1}));
}

bool CausalLmArchitecture::adapter_uses_decode_sequence_length()
        const noexcept {
    return true;
}

bool CausalLmArchitecture::adapter_supports_prepared_prompt()
        const noexcept {
    return false;
}

bool CausalLmArchitecture::adapter_supports_speculation()
        const noexcept {
    return false;
}

bool CausalLmArchitecture::adapter_supports_suffix_speculation()
        const noexcept {
    return false;
}

void CausalLmArchitecture::adapter_begin_speculative() {}
void CausalLmArchitecture::adapter_commit_speculative() {}
void CausalLmArchitecture::adapter_rollback_speculative(int64_t) {}

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
        const int primary = execution->layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        ids = tensor_to_cuda_device(
            ids.to(mfq_tensor_backend::kInt64), primary);
        if (ids.dim() == 1) ids = ids.unsqueeze(0);
        auto x = g_profiler.measure("model.embed", [&]() { return embed_forward(ids); });
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
        const int primary = execution->layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        ids = tensor_to_cuda_device(
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
            ? tensor_to_cuda_device(
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
            ? tensor_to_cuda_device(
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
        if (execution->dense_cpu_layer_count > 0) {
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
            tensor_to_cuda_device(input_embeddings, primary).contiguous(),
            B, T);
        this->adapter_begin_forward(raw_hidden != nullptr);
        if (block_trace != nullptr) block_trace->push_back(x.to(mfq_tensor_backend::kFloat32).clone());
        for (auto & b : blocks) {
            MfqCudaGuard block_guard(b->cuda_device);
            auto local_ids = b->cpu_offloaded
                ? cpu_ids
                : tensor_to_cuda_device(ids, b->cuda_device);
            auto local_pos = b->cpu_offloaded
                ? cpu_pos
                : tensor_to_cuda_device(pos, b->cuda_device);
            MfqOptional<mfq_tensor_backend::Tensor> local_cache_positions = mfq_nullopt;
            // Grid-MRoPE semantic coordinates never double as physical KV
            // slots, during either prepared prefill or delta-adjusted decode.
            if (this->adapter_pass_cache_positions(
                    rope.sections.numel() > 0,
                    cache_positions_override.has_value())) {
                local_cache_positions = b->cpu_offloaded
                    ? cpu_cache_positions
                    : tensor_to_cuda_device(
                        cache_positions, b->cuda_device);
            }
            MfqOptional<mfq_tensor_backend::Tensor> local_seq_len = mfq_nullopt;
            if (seq_len.has_value()) {
                local_seq_len = b->cpu_offloaded
                    ? cpu_seq_len.value()
                    : tensor_to_cuda_device(
                        seq_len.value(), b->cuda_device);
            }
            MfqOptional<mfq_tensor_backend::Tensor> local_attention_mask = mfq_nullopt;
            if (effective_attention_mask.has_value()) {
                local_attention_mask = b->cpu_offloaded
                    ? cpu_attention_mask.value()
                    : tensor_to_cuda_device(
                        effective_attention_mask.value(), b->cuda_device);
            }
            x = b->cpu_offloaded
                ? x.to(mfq_tensor_backend::kCPU).contiguous()
                : tensor_to_cuda_device(x, b->cuda_device);
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
                std::move(x), context, active_rope);
            if (block_trace != nullptr) {
                block_trace->push_back(
                    tensor_to_cuda_device(x, primary)
                        .to(mfq_tensor_backend::kFloat32).clone());
            }
        }
        this->adapter_finish_forward(full_positions, B, T);
        if (this->adapter_force_cache_advance() ||
                !pos_override.has_value() ||
                advance_cache_with_position_ids) {
            cache_pos += T;
        }
        x = tensor_to_cuda_device(x, primary);
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
        return softcap_logits(
            std::move(logits), final_logit_softcapping());
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
            return lm_head.forward(last);
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

} // namespace mfq::cuda

std::vector<CudaPagedPayload> encode_cuda_paged_session(
        const TextSessionState & state,
        size_t block_size,
        size_t first_block,
        size_t maximum_blocks) {
    static constexpr std::array<uint8_t, 8> magic{
        'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    const auto* layers = std::get_if<
        std::vector<FullBlockSessionState>>(&state.payload);
    if (state.kind() != TextSessionStateKind::FullAttention ||
            state.cache_pos <= 0 || layers == nullptr || layers->empty() ||
            state.tokens.size() != static_cast<size_t>(state.cache_pos)) {
        throw std::runtime_error(
            "CUDA session state is not block-sliceable");
    }
    const size_t full_blocks = state.tokens.size() / block_size;
    if (first_block > full_blocks) {
        throw std::runtime_error("invalid CUDA cache first block");
    }
    const size_t end_block = maximum_blocks > full_blocks - first_block
        ? full_blocks
        : first_block + maximum_blocks;
    std::vector<CudaPagedPayload> result;
    result.reserve(end_block - first_block);
    for (size_t block_index = first_block;
            block_index < end_block; ++block_index) {
        const int64_t start = static_cast<int64_t>(block_index * block_size);
        CudaPagedWriter writer;
        writer.raw(magic.data(), magic.size());
        writer.scalar<uint32_t>(1);
        writer.scalar<uint32_t>(static_cast<uint32_t>(start));
        writer.scalar<uint32_t>(static_cast<uint32_t>(block_size));
        writer.scalar<uint32_t>(static_cast<uint32_t>(layers->size()));
        for (const auto & layer : *layers) {
            if (layer.ring || layer.capacity <= 0 ||
                    !layer.k.defined() || !layer.v.defined() ||
                    layer.k.dim() != 4 || layer.v.sizes() != layer.k.sizes() ||
                    layer.k.size(2) < start + static_cast<int64_t>(block_size)) {
                throw std::runtime_error(
                    "CUDA session layer cannot be block-sliced");
            }
            writer.scalar<int64_t>(layer.capacity);
            writer.tensor(layer.k.narrow(
                2, start, static_cast<int64_t>(block_size)));
            writer.tensor(layer.v.narrow(
                2, start, static_cast<int64_t>(block_size)));
        }
        result.push_back(std::move(writer).finish());
    }
    return result;
}

CudaPagedPayload encode_cuda_paged_block(
        const TextSessionState & state,
        size_t block_size,
        size_t block_index) {
    auto result = encode_cuda_paged_session(
        state, block_size, block_index, 1);
    if (result.size() != 1) {
        throw std::runtime_error("invalid CUDA cache block index");
    }
    return std::move(result.front());
}

CudaDecodedPagedBlock decode_cuda_paged_block(
        const std::vector<uint8_t> & payload) {
    static constexpr std::array<uint8_t, 8> magic{
        'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    CudaPagedReader reader(payload);
    std::array<uint8_t, 8> found_magic{};
    reader.raw(found_magic.data(), found_magic.size(), "magic");
    if (found_magic != magic || reader.scalar<uint32_t>("version") != 1) {
        throw CudaSessionStateError(
            "unsupported CUDA paged cache payload");
    }
    CudaDecodedPagedBlock block;
    block.start = reader.scalar<uint32_t>("start");
    block.count = reader.scalar<uint32_t>("count");
    const auto layer_count = reader.scalar<uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw CudaSessionStateError(
            "invalid CUDA paged cache geometry");
    }
    block.layers.reserve(layer_count);
    for (uint32_t index = 0; index < layer_count; ++index) {
        CudaDecodedPagedLayer layer;
        layer.capacity = reader.scalar<int64_t>("capacity");
        auto key = reader.tensor();
        auto value = reader.tensor();
        if (key.second != value.second || key.first.dim() != 4 ||
                value.first.sizes() != key.first.sizes() ||
                key.first.size(2) != static_cast<int64_t>(block.count)) {
            throw CudaSessionStateError(
                "invalid CUDA paged cache layer");
        }
        layer.device = key.second;
        layer.k = std::move(key.first);
        layer.v = std::move(value.first);
        block.layers.push_back(std::move(layer));
    }
    reader.expect_end();
    return block;
}

TextSessionState decode_cuda_paged_session(
        const std::vector<CudaPagedPayload> & payloads,
        const std::vector<int64_t> & tokens,
        size_t block_size) {
    if (payloads.empty() || tokens.size() != payloads.size() * block_size) {
        throw CudaSessionStateError(
            "CUDA paged cache chain length mismatch");
    }
    std::vector<CudaDecodedPagedBlock> blocks;
    blocks.reserve(payloads.size());
    size_t expected_start = 0;
    size_t layer_count = 0;
    for (const auto & payload : payloads) {
        if (!payload) {
            throw CudaSessionStateError(
                "CUDA paged cache block payload is null");
        }
        auto block = decode_cuda_paged_block(*payload);
        if (block.start != expected_start || block.count != block_size ||
                (layer_count != 0 && block.layers.size() != layer_count)) {
            throw CudaSessionStateError(
                "incompatible CUDA paged cache chain");
        }
        expected_start += block.count;
        layer_count = block.layers.size();
        blocks.push_back(std::move(block));
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = static_cast<int64_t>(tokens.size());
    state.payload = std::vector<FullBlockSessionState>{};
    auto& layers = std::get<std::vector<FullBlockSessionState>>(state.payload);
    layers.reserve(layer_count);
    for (size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        const auto & final = blocks.back().layers[layer_index];
        std::vector<mfq_tensor_backend::Tensor> keys;
        std::vector<mfq_tensor_backend::Tensor> values;
        keys.reserve(blocks.size());
        values.reserve(blocks.size());
        for (const auto & block : blocks) {
            const auto & layer = block.layers[layer_index];
            if (layer.capacity != final.capacity ||
                    layer.device != final.device ||
                    layer.k.scalar_type() != final.k.scalar_type() ||
                    layer.k.size(0) != final.k.size(0) ||
                    layer.k.size(1) != final.k.size(1) ||
                    layer.k.size(3) != final.k.size(3)) {
                throw CudaSessionStateError(
                    "inconsistent CUDA paged cache topology");
            }
            keys.push_back(layer.k);
            values.push_back(layer.v);
        }
        auto key = mfq_tensor_backend::cat(keys, 2).contiguous();
        auto value = mfq_tensor_backend::cat(values, 2).contiguous();
        MfqCudaGuard guard(final.device);
        const auto options = key.options().device(
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, final.device));
        key = key.to(options, true, false).contiguous();
        value = value.to(options, true, false).contiguous();
        state.bytes += static_cast<size_t>(
            key.numel() * key.element_size() +
            value.numel() * value.element_size());
        layers.push_back(FullBlockSessionState{
            std::move(key),
            std::move(value),
            final.capacity,
            false,
        });
    }
    return state;
}

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor & tensor) {
    if (!tensor.defined()) return 0;
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

bool session_tensor_layout_matches(
        const mfq_tensor_backend::Tensor & target,
        const mfq_tensor_backend::Tensor & source) {
    return target.defined() && source.defined() &&
        target.sizes() == source.sizes() &&
        target.scalar_type() == source.scalar_type() &&
        target.device() == source.device();
}

void restore_session_tensor(
        mfq_tensor_backend::Tensor & target,
        const mfq_tensor_backend::Tensor & source) {
    if (!source.defined()) {
        target = mfq_tensor_backend::Tensor();
        return;
    }
    if (!session_tensor_layout_matches(target, source)) {
        target = mfq_tensor_backend::empty(source.sizes(), source.options());
    }
    target.copy_(source);
}

void restore_session_prefix_tensor(
        mfq_tensor_backend::Tensor & target,
        const mfq_tensor_backend::Tensor & source,
        int64_t dimension,
        int64_t capacity) {
    if (!source.defined() || dimension < 0 ||
            dimension >= source.dim() || capacity <= 0 ||
            source.size(dimension) > capacity) {
        throw CudaSessionStateError(
            "session prefix tensor layout is invalid");
    }
    auto target_shape = source.sizes().vec();
    target_shape.at(static_cast<size_t>(dimension)) = capacity;
    const bool compatible = target.defined() &&
        target.sizes() == mfq_tensor_backend::IntArrayRef(target_shape) &&
        target.scalar_type() == source.scalar_type() &&
        target.device() == source.device();
    if (!compatible) {
        target = mfq_tensor_backend::empty(target_shape, source.options());
    }
    if (source.size(dimension) > 0) {
        target.narrow(dimension, 0, source.size(dimension)).copy_(source);
    }
}

FullBlockSessionState capture_full_attention_session_state(
        const FullBlock & block,
        int64_t cache_pos,
        size_t & bytes) {
    if (!block.cache.k.defined() || !block.cache.v.defined() ||
            block.cache.k.dim() != 4 ||
            block.cache.v.sizes() != block.cache.k.sizes() ||
            block.cache.k.size(0) != 1) {
        throw std::runtime_error(
            "full-attention KV cache is unavailable");
    }
    FullBlockSessionState state;
    state.capacity = block.cache.k.size(2);
    state.ring = block.cache.ring;
    const int64_t saved_tokens = state.ring
        ? state.capacity
        : std::min<int64_t>(cache_pos, state.capacity);
    state.k = block.cache.k.narrow(2, 0, saved_tokens).clone();
    state.v = block.cache.v.narrow(2, 0, saved_tokens).clone();
    bytes += session_tensor_bytes(state.k);
    bytes += session_tensor_bytes(state.v);
    return state;
}

void restore_full_attention_session_state(
        FullBlock & block,
        const FullBlockSessionState & state) {
    if (!state.k.defined() || !state.v.defined() ||
            state.capacity <= 0 || state.k.dim() != 4 ||
            state.v.sizes() != state.k.sizes() ||
            state.k.size(0) != 1 ||
            state.k.size(2) > state.capacity) {
        throw CudaSessionStateError(
            "full-attention session KV layout is invalid");
    }
    restore_session_prefix_tensor(
        block.cache.k, state.k, 2, state.capacity);
    restore_session_prefix_tensor(
        block.cache.v, state.v, 2, state.capacity);
    block.cache.ring = state.ring;
}

Dsv4PoolSessionState capture_dsv4_pool_session_state(
        const Dsv4PoolState & source,
        int64_t cache_pos,
        size_t & bytes) {
    Dsv4PoolSessionState state;
    state.ratio = source.ratio;
    state.head_dim = source.head_dim;
    state.cache_quant_mode = source.cache_quant_mode;
    state.capacity = source.capacity;
    state.overlap = source.overlap;
    if (source.ratio <= 0) return state;
    if (source.capacity <= 0 || !source.state_kv.defined() ||
            !source.state_gate.defined() || !source.pool.defined() ||
            source.pool.dim() != 3 || source.pool.size(0) != 1 ||
            source.pool.size(1) != source.capacity ||
            (source.overlap &&
             (!source.previous_kv.defined() ||
              !source.previous_gate.defined()))) {
        throw std::runtime_error(
            "DeepSeek V4 session compressor state is unavailable");
    }
    const int64_t visible = std::min<int64_t>(
        cache_pos / source.ratio, source.capacity);
    state.state_kv = source.state_kv.clone();
    state.state_gate = source.state_gate.clone();
    if (source.overlap) {
        state.previous_kv = source.previous_kv.clone();
        state.previous_gate = source.previous_gate.clone();
    }
    state.pool = source.pool.narrow(1, 0, visible).clone();
    bytes += session_tensor_bytes(state.state_kv);
    bytes += session_tensor_bytes(state.state_gate);
    bytes += session_tensor_bytes(state.previous_kv);
    bytes += session_tensor_bytes(state.previous_gate);
    bytes += session_tensor_bytes(state.pool);
    return state;
}

void restore_dsv4_pool_session_state(
        Dsv4PoolState & target,
        const Dsv4PoolSessionState & state) {
    if (target.ratio != state.ratio ||
            target.head_dim != state.head_dim ||
            target.cache_quant_mode != state.cache_quant_mode ||
            target.overlap != state.overlap) {
        throw CudaSessionStateError(
            "DeepSeek V4 session compressor configuration changed");
    }
    if (state.ratio <= 0) return;
    if (state.capacity <= 0 || !state.state_kv.defined() ||
            !state.state_gate.defined() || !state.pool.defined() ||
            state.pool.dim() != 3 || state.pool.size(0) != 1 ||
            state.pool.size(1) > state.capacity ||
            (state.overlap &&
             (!state.previous_kv.defined() ||
              !state.previous_gate.defined()))) {
        throw CudaSessionStateError(
            "DeepSeek V4 saved compressor state is invalid");
    }
    target.capacity = state.capacity;
    restore_session_tensor(target.state_kv, state.state_kv);
    restore_session_tensor(target.state_gate, state.state_gate);
    if (state.overlap) {
        restore_session_tensor(target.previous_kv, state.previous_kv);
        restore_session_tensor(target.previous_gate, state.previous_gate);
    } else {
        target.previous_kv = mfq_tensor_backend::Tensor();
        target.previous_gate = mfq_tensor_backend::Tensor();
    }
    restore_session_prefix_tensor(
        target.pool, state.pool, 1, state.capacity);
}

namespace mfq::cuda {

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

template struct FullAttentionSessionCodec<Qwen35Model>;
template struct FullAttentionSessionCodec<MiniCPMO45Model>;
template struct FullAttentionSessionCodec<MiniCPMOTtsModel>;
template struct CudaSessionCodec<Gemma4Model>;
template struct CudaSessionCodec<Glm5Model>;
template struct CudaSessionCodec<Qwen4Model>;
template struct CudaSessionCodec<DeepseekV41Model>;

#define MFQ_INSTANTIATE_CAUSAL_MODEL(MODEL) \
    template struct CausalLm<MODEL>;

MFQ_INSTANTIATE_CAUSAL_MODEL(Qwen35Model)
MFQ_INSTANTIATE_CAUSAL_MODEL(MiniCPMO45Model)
MFQ_INSTANTIATE_CAUSAL_MODEL(MiniCPMOTtsModel)
MFQ_INSTANTIATE_CAUSAL_MODEL(Gemma4Model)
MFQ_INSTANTIATE_CAUSAL_MODEL(GlmDsaModel)
MFQ_INSTANTIATE_CAUSAL_MODEL(Glm5Model)
MFQ_INSTANTIATE_CAUSAL_MODEL(Qwen4Model)
MFQ_INSTANTIATE_CAUSAL_MODEL(DeepseekV4Model)
MFQ_INSTANTIATE_CAUSAL_MODEL(DeepseekV41Model)

#undef MFQ_INSTANTIATE_CAUSAL_MODEL

} // namespace mfq::cuda
