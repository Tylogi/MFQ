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
            throw std::runtime_error(
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
            throw std::runtime_error(
                "invalid CUDA paged cache tensor header");
        }
        const auto dtype = static_cast<mfq_tensor_backend::ScalarType>(dtype_value);
        if (dtype != mfq_tensor_backend::kFloat16 && dtype != mfq_tensor_backend::kBFloat16 &&
                dtype != mfq_tensor_backend::kFloat32) {
            throw std::runtime_error(
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
                throw std::runtime_error(
                    "invalid CUDA paged cache tensor shape");
            }
            shape.push_back(dimension);
            elements *= static_cast<uint64_t>(dimension);
        }
        const auto bytes = scalar<uint64_t>("tensor size");
        auto cpu = mfq_tensor_backend::empty(
            shape,
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(dtype));
        const auto expected = static_cast<uint64_t>(
            cpu.numel() * cpu.element_size());
        if (bytes != expected || bytes > remaining()) {
            throw std::runtime_error(
                "invalid CUDA paged cache tensor size");
        }
        raw(cpu.data_ptr(), static_cast<size_t>(bytes), "tensor");
        return {std::move(cpu), device};
    }

void CudaPagedReader::expect_end() const {
        if (cursor_ != end_) {
            throw std::runtime_error(
                "trailing CUDA paged cache payload bytes");
        }
    }

size_t CudaPagedReader::remaining() const {
        return static_cast<size_t>(end_ - cursor_);
    }

namespace mfq::cuda {

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::vocab_size() const noexcept {
        if constexpr (is_flash_next || is_deepseek_v41) {
            return this->config.vocab;
        } else {
            return this->config.vocab_size;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::hidden_size() const noexcept {
        if constexpr (is_flash_next || is_deepseek_v41) {
            return this->config.hidden;
        } else {
            return this->config.hidden_size;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::num_hidden_layers() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.layers;
        } else if constexpr (is_deepseek_v41) {
            return this->config.n_layers;
        } else {
            return this->config.num_hidden_layers;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::num_attention_heads() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.heads;
        } else if constexpr (is_deepseek_v41) {
            return this->config.n_heads;
        } else {
            return this->config.num_attention_heads;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::num_key_value_heads() const noexcept {
        if constexpr (is_qwen4) {
            return this->config.kv_heads;
        } else if constexpr (is_glm5) {
            return 1;
        } else if constexpr (is_deepseek_v41) {
            return this->config.n_kv_heads;
        } else {
            return this->config.num_key_value_heads;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::head_dim() const noexcept {
        if constexpr (is_qwen4) {
            return this->config.width;
        } else if constexpr (is_glm5) {
            return this->config.nope;
        } else {
            return this->config.head_dim;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::max_position_embeddings() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.maximum;
        } else {
            return this->config.max_position_embeddings;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::rotary_dim() const noexcept {
    if constexpr (is_qwen4) return this->config.rotary;
    else if constexpr (is_glm5) return 0;
    else if constexpr (is_deepseek_v41) return this->config.rope_head_dim;
    else return this->config.rotary_dim;
}

template <CudaBackbone Backbone>
double CausalLm<Backbone>::rope_base() const noexcept {
    if constexpr (is_glm5) return 1.0;
    else if constexpr (is_deepseek_v41) return this->config.rope_theta;
    else return this->config.rope_base;
}

template <CudaBackbone Backbone>
void CausalLm<Backbone>::set_max_position_embeddings(int64_t value) noexcept {
        if constexpr (is_flash_next) this->config.maximum = value;
        else this->config.max_position_embeddings = value;
    }

template <CudaBackbone Backbone>
double CausalLm<Backbone>::rms_norm_eps() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.eps;
        } else if constexpr (is_deepseek_v41) {
            return this->config.rms_eps;
        } else {
            return this->config.rms_norm_eps;
        }
    }

template <CudaBackbone Backbone>
double CausalLm<Backbone>::norm_weight_offset() const noexcept {
        if constexpr (Backbone == CudaBackbone::generic_qwen) {
            return this->config.legacy_tensor_layout.norm_weight_offset;
        } else if constexpr (
                is_flash_next || is_dsv4 || is_gemma4 || is_minicpmo45) {
            return 0.0;
        } else {
            return 1.0;
        }
    }

template <CudaBackbone Backbone>
bool CausalLm<Backbone>::tie_word_embeddings() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.tied_embeddings;
        } else if constexpr (is_deepseek_v41) {
            return false;
        } else {
            return this->config.tie_word_embeddings;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::num_experts() const noexcept {
        if constexpr (is_qwen4 || is_glm5) {
            return this->config.experts;
        } else if constexpr (Backbone == CudaBackbone::generic_qwen) {
            return this->config.num_experts;
        } else if constexpr (is_deepseek_v41) {
            return this->config.n_experts;
        } else if constexpr (Backbone == CudaBackbone::deepseek_v4 ||
                             Backbone == CudaBackbone::glm_dsa ||
                             Backbone == CudaBackbone::gemma4) {
            return this->config.num_experts;
        } else {
            return 0;
        }
    }

template <CudaBackbone Backbone>
int64_t CausalLm<Backbone>::hc_mult() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.streams;
        } else if constexpr (is_deepseek_v41 ||
                             Backbone == CudaBackbone::deepseek_v4) {
            return this->config.hc_mult;
        } else {
            return 1;
        }
    }

template <CudaBackbone Backbone>
double CausalLm<Backbone>::hc_eps() const noexcept {
        if constexpr (is_glm5 || is_deepseek_v41 ||
                      Backbone == CudaBackbone::deepseek_v4) {
            return this->config.hc_eps;
        } else {
            return 1e-6;
        }
    }

template <CudaBackbone Backbone>
double CausalLm<Backbone>::final_logit_softcapping() const noexcept {
        if constexpr (is_gemma4) {
            return this->config.final_logit_softcapping;
        } else {
            return 0.0;
        }
    }

template <CudaBackbone Backbone>
double CausalLm<Backbone>::embedding_scale() const noexcept {
        if constexpr (is_gemma4) return this->embed_scale;
        else return 1.0;
    }

template <CudaBackbone Backbone>
std::string_view CausalLm<Backbone>::model_type() const noexcept {
        if constexpr (is_qwen4) {
            return "qwen4_exp";
        } else if constexpr (is_glm5) {
            return "glm5_next";
        } else if constexpr (is_deepseek_v41) {
            return this->config.text_model_type;
        } else {
            return this->config.model_type;
        }
    }

template <CudaBackbone Backbone>
std::string_view CausalLm<Backbone>::layer_type(int64_t layer) const {
        if constexpr (is_deepseek_v41) {
            return "deepseek_v41";
        } else {
            return this->config.layer_types.at(
                static_cast<std::size_t>(layer));
        }
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::embed_forward(mfq_tensor_backend::Tensor ids) const {
        auto token_ids = ids.contiguous().to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64);
        auto output = quant_embedding_lookup(embed, token_ids);
        if constexpr (is_minicpmo45) {
            return output.to(mfq_tensor_backend::kBFloat16).contiguous();
        }
        if constexpr (Backbone == CudaBackbone::generic_qwen) {
            return output.to(mfq_tensor_backend::kFloat16).contiguous();
        }
        return output;
    }

template <CudaBackbone Backbone>
void CausalLm<Backbone>::reset(int64_t B) {
        cache_pos = 0;
        decode_position_delta = 0;
        if constexpr (is_qwen4) {
            this->positions = {};
            this->batch = B;
        }
        speculative_start = -1;
        speculative_confirmed = 0;
        speculative_suffix_forward = false;
        for (auto & b : blocks) {
            MfqCudaGuard guard(b->cuda_device);
            b->reset(B);
        }
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::hidden_forward(mfq_tensor_backend::Tensor ids,
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

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::hidden_forward_speculative_suffix(
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

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::hidden_forward_inputs(
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
        if constexpr (is_glm5 || is_deepseek_v41) {
            MFQ_RUNTIME_CHECK(T > 0 &&
                cache_pos + T <= max_position_embeddings() &&
                !pos_override.has_value() && !cache_positions_override.has_value() && !attention_mask.has_value(),
                is_deepseek_v41
                    ? "DeepSeek-V4.1 currently requires contiguous causal cache positions without an external mask"
                    : "GLM Flash-Next currently requires contiguous causal cache positions without an external mask");
        }
        if constexpr (is_qwen4) {
            if (this->batch!=0 && this->batch!=B) reset(B);
            MFQ_RUNTIME_CHECK(T > 0 &&
                cache_pos + T <= max_position_embeddings() &&
                !cache_positions_override.has_value() && !attention_mask.has_value(),
                "Qwen4 requires unpadded causal cache positions");
        }
        MFQ_RUNTIME_CHECK(speculative_start < 0 || speculative_suffix_forward,
            "commit or roll back the pending speculative pass before forwarding");
        MFQ_RUNTIME_CHECK(confirmed_prefix >= 0 &&
            (confirmed_prefix == 0 || (confirmed_prefix < T && B == 1 && cache_pos > 0 &&
                (!pos_override.has_value() || is_qwen4) && !cache_positions_override.has_value() &&
                !attention_mask.has_value() && supports_qwen_speculation())),
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
        if constexpr (is_qwen4) {
            if ((pos.dim()==2 || pos.dim()==3) && pos.size(0)==4) pos=pos.narrow(0,1,3);
            MFQ_RUNTIME_CHECK((pos.dim()==1 || (pos.dim()==2 && pos.size(0)==3) ||
                (pos.dim()==3 && pos.size(0)==3 && pos.size(1)==B)) && pos.size(-1)==T,
                "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
        } else if (!((pos.dim() == 1 && pos.numel() == T) ||
              (pos.dim() == 2 && pos.size(0) == B && pos.size(1) == T) ||
              (Backbone == mfq::cuda::CudaBackbone::generic_qwen &&
               rope.sections.numel() > 0 && pos.dim() == 2 &&
               pos.size(0) == 3 && pos.size(1) == T))) {
            throw std::runtime_error(
                "position_ids must have shape [tokens], [batch,tokens], or configured grid-MRoPE [3,tokens]");
        }
        auto full_positions = pos;
        if constexpr (is_qwen4) {
            if (this->positions.defined()) {
                full_positions = mfq_tensor_backend::cat(
                    {this->positions, pos}, -1);
            }
        }
        if (attention_mask.has_value()) {
            auto mask = attention_mask.value();
            if (mask.dim() != 2 || mask.size(0) != B ||
                    mask.size(1) < cache_pos + T) {
                throw std::runtime_error(
                    "attention_mask must cover [batch,cached_plus_current_tokens]");
            }
        }
        auto effective_attention_mask = attention_mask;
        if constexpr (is_minicpmo45) {
            if (attention_mask.has_value() &&
                    (T == 1 || cache_pos == 0) &&
                    attention_mask.value().eq(1).all().item<bool>()) {
                effective_attention_mask = mfq_nullopt;
            }
        }
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
        auto x = tensor_to_cuda_device(
            input_embeddings, primary).contiguous();
        if constexpr (is_minicpmo45) {
            x = x.to(mfq_tensor_backend::kBFloat16).contiguous();
        }
        if constexpr (is_gemma4) {
            x = g_profiler.measure("model.embed_scale", [&]() {
                return x * embedding_scale();
            });
        }
        if constexpr (is_dsv4 || is_glm5 || is_deepseek_v41) {
            x = x.to(mfq_tensor_backend::kFloat16)
                .unsqueeze(2)
                .expand({B, T, hc_mult(), hidden_size()})
                .contiguous();
        }
        if constexpr (is_deepseek_v41) {
            MFQ_RUNTIME_CHECK(
                this->shared,
                "DeepSeek-V4.1 target capture state is unavailable");
            this->shared->begin_forward(
                raw_hidden != nullptr,
                this->shared->config.dspark_target_layer_ids.size());
        }
        if constexpr (is_qwen4) {
            x = x.to(mfq_tensor_backend::kFloat16)
                .repeat({1, 1, hc_mult()});
        }
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
            if constexpr (is_minicpmo45) {
                local_cache_positions = b->cpu_offloaded
                    ? cpu_cache_positions
                    : tensor_to_cuda_device(
                        cache_positions, b->cuda_device);
            } else if (rope.sections.numel() > 0 ||
                    cache_positions_override.has_value()) {
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
            if constexpr (is_qwen4) {
                context.full_positions = tensor_to_cuda_device(
                    full_positions, b->cuda_device);
            } else {
                context.full_positions = local_pos;
            }
            context.cache_position=cache_pos;
            context.confirmed_prefix=confirmed_prefix;
            context.planned_kv_length=planned_kv_length;
            context.decode_attention_parts=decode_attention_parts;
            context.sequence_lengths=local_seq_len;
            context.cache_positions=local_cache_positions;
            if constexpr (is_minicpmo45) {
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
        if constexpr (is_qwen4) {
            this->positions = full_positions;
            this->batch = B;
        }
        if constexpr (is_qwen4) {
            cache_pos += T;
        } else if (!pos_override.has_value() ||
                advance_cache_with_position_ids) {
            cache_pos += T;
        }
        x = tensor_to_cuda_device(x, primary);
        auto finalized=finalize_hidden(x, B, T);
        if (raw_hidden != nullptr) {
            if constexpr (is_deepseek_v41) {
                *raw_hidden = this->shared->dspark_target_hidden();
            } else if constexpr (is_glm5) {
                *raw_hidden = finalized;
            } else {
                *raw_hidden = x;
            }
        }
        return finalized;
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::forward_inputs(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor input_embeddings,
        MfqOptional<mfq_tensor_backend::Tensor> pos_override,
        MfqOptional<mfq_tensor_backend::Tensor> seq_len) {
        return logits_from_hidden(hidden_forward_inputs(
            ids, input_embeddings, pos_override, seq_len));
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::apply_final_logit_softcap(
        mfq_tensor_backend::Tensor logits) const {
        const double cap = final_logit_softcapping();
        return cap > 0.0
            ? mfq_tensor_backend::tanh(logits / cap) * cap
            : logits;
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::logits_from_hidden(mfq_tensor_backend::Tensor y) {
        auto logits = g_profiler.measure(
            "model.lm_head", [&]() { return lm_head.forward(y); });
        if constexpr (is_minicpmo45) {
            logits = logits.to(mfq_tensor_backend::kBFloat16).contiguous();
        }
        return apply_final_logit_softcap(std::move(logits));
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::forward(mfq_tensor_backend::Tensor ids) {
        return logits_from_hidden(hidden_forward(ids));
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::last_logits_prepared(
        const CudaPreparedPrompt& prepared) {
        MFQ_RUNTIME_CHECK(
            prepared.transformed() && !prepared.token_ids.empty() &&
                prepared.embeddings.defined() && prepared.positions.defined() &&
                Backbone == mfq::cuda::CudaBackbone::generic_qwen,
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

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::last_logits(mfq_tensor_backend::Tensor ids) {
        MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt;
        const int64_t token_count = ids.dim() == 1 ? ids.size(0) : ids.size(1);
        const int64_t batch_size = ids.dim() == 1 ? 1 : ids.size(0);
        if (!is_minicpmo45 && cache_pos > 0 && token_count == 1) {
            seq_len = mfq_tensor_backend::full(
                {batch_size}, cache_pos + 1,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
        }
        auto y = hidden_forward(ids, mfq_nullopt, seq_len);
        auto last = y.index({Slice(), -1, Slice()});
        if constexpr (is_flash_next) return logits_from_hidden(last);
        if constexpr (is_minicpmo45) {
            return logits_from_hidden(
                last.to(mfq_tensor_backend::kBFloat16).contiguous());
        }
        last = last.to(mfq_tensor_backend::kFloat16).contiguous();
        return apply_final_logit_softcap(lm_head.forward(last));
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::hidden_forward_static(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor pos,
        mfq_tensor_backend::Tensor seq_len,
        int64_t planned_kv_length,
        int64_t decode_attention_parts) {
        return hidden_forward(
            std::move(ids), pos, seq_len, nullptr, pos, nullptr, 0,
            planned_kv_length, decode_attention_parts);
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::last_logits_static(
        mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor pos,
        mfq_tensor_backend::Tensor seq_len,
        int64_t planned_kv_length,
        int64_t decode_attention_parts) {
        auto y = hidden_forward_static(
            std::move(ids), std::move(pos), std::move(seq_len),
            planned_kv_length, decode_attention_parts);
        auto last = y.index({Slice(), -1, Slice()});
        if constexpr (is_minicpmo45) {
            return logits_from_hidden(
                last.to(mfq_tensor_backend::kBFloat16).contiguous());
        }
        last = last.to(mfq_tensor_backend::kFloat16).contiguous();
        return apply_final_logit_softcap(lm_head.forward(last));
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::next_token(mfq_tensor_backend::Tensor ids) {
        MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt;
        const int64_t token_count = ids.dim() == 1 ? ids.size(0) : ids.size(1);
        const int64_t batch_size = ids.dim() == 1 ? 1 : ids.size(0);
        if (!is_minicpmo45 && cache_pos > 0 && token_count == 1) {
            seq_len = mfq_tensor_backend::full(
                {batch_size}, cache_pos + 1,
                mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCUDA));
        }
        auto y = hidden_forward(ids, mfq_nullopt, seq_len);
        return next_token_from_hidden(y);
    }

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::next_token_static(
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

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::next_token_from_hidden(mfq_tensor_backend::Tensor y) {
        auto last = y.index({Slice(), -1, Slice()});
        if constexpr (is_flash_next) return mfq_tensor_backend::argmax(logits_from_hidden(last),-1).to(mfq_tensor_backend::kInt64);
        if constexpr (is_minicpmo45) {
            auto logits = logits_from_hidden(
                last.to(mfq_tensor_backend::kBFloat16).contiguous());
            return mfq_tensor_backend::argmax(logits, -1).to(mfq_tensor_backend::kInt64);
        }
        last = last.to(mfq_tensor_backend::kFloat16).contiguous();
        auto logits = g_profiler.measure("model.lm_head", [&]() { return lm_head.forward(last); });
        return sample_greedy_cuda(logits.contiguous().view({last.size(0), -1}));
    }

} // namespace mfq::cuda

std::vector<CudaPagedPayload> encode_cuda_paged_session(
        const TextSessionState & state,
        size_t block_size,
        size_t first_block,
        size_t maximum_blocks) {
    static constexpr std::array<uint8_t, 8> magic{
        'M', 'F', 'Q', 'C', 'U', 'D', '1', 0};
    if (state.kind != TextSessionStateKind::FullAttention ||
            state.cache_pos <= 0 || state.blocks.empty() ||
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
        writer.scalar<uint32_t>(static_cast<uint32_t>(state.blocks.size()));
        for (const auto & layer : state.blocks) {
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
        throw std::runtime_error(
            "unsupported CUDA paged cache payload");
    }
    CudaDecodedPagedBlock block;
    block.start = reader.scalar<uint32_t>("start");
    block.count = reader.scalar<uint32_t>("count");
    const auto layer_count = reader.scalar<uint32_t>("layer count");
    if (block.count == 0 || layer_count == 0 || layer_count > 4096) {
        throw std::runtime_error(
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
            throw std::runtime_error(
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
        throw std::runtime_error(
            "CUDA paged cache chain length mismatch");
    }
    std::vector<CudaDecodedPagedBlock> blocks;
    blocks.reserve(payloads.size());
    size_t expected_start = 0;
    size_t layer_count = 0;
    for (const auto & payload : payloads) {
        if (!payload) {
            throw std::runtime_error(
                "CUDA paged cache block payload is null");
        }
        auto block = decode_cuda_paged_block(*payload);
        if (block.start != expected_start || block.count != block_size ||
                (layer_count != 0 && block.layers.size() != layer_count)) {
            throw std::runtime_error(
                "incompatible CUDA paged cache chain");
        }
        expected_start += block.count;
        layer_count = block.layers.size();
        blocks.push_back(std::move(block));
    }
    TextSessionState state;
    state.tokens = tokens;
    state.kind = TextSessionStateKind::FullAttention;
    state.cache_pos = static_cast<int64_t>(tokens.size());
    state.blocks.reserve(layer_count);
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
                throw std::runtime_error(
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
        state.blocks.push_back(FullBlockSessionState{
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
        throw std::runtime_error("session prefix tensor layout is invalid");
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
        throw std::runtime_error(
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
        throw std::runtime_error(
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
        throw std::runtime_error(
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

template <CudaBackbone Backbone>
bool CausalLm<Backbone>::supports_qwen_speculation() const {
    if constexpr (
            Backbone != CudaBackbone::generic_qwen && !is_flash_next) {
        return false;
    }
    return !blocks.empty() && std::all_of(
        blocks.begin(), blocks.end(),
        [](const auto& block) { return block->supports_speculation(); });
}

template <CudaBackbone Backbone>
bool CausalLm<Backbone>::supports_deepseek_v41_speculation() const {
    if constexpr (!is_deepseek_v41) return false;
    return !blocks.empty() && std::all_of(
        blocks.begin(), blocks.end(),
        [](const auto& block) { return block->supports_speculation(); });
}

template <CudaBackbone Backbone>
void CausalLm<Backbone>::begin_speculative_suffix(int64_t draft_tokens) {
    if constexpr (!is_deepseek_v41) {
        throw std::runtime_error(
            "speculative suffix requires DeepseekV41CausalLm");
    } else {
        MFQ_RUNTIME_CHECK(
            supports_deepseek_v41_speculation() &&
                speculative_start < 0 && draft_tokens > 0 &&
                cache_pos + draft_tokens <= max_position_embeddings() &&
                this->shared,
            "invalid DeepSeek-V4.1 speculative suffix");
        speculative_start = cache_pos;
        speculative_confirmed = 0;
        this->shared->begin_speculative();
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
            try { this->shared->rollback_speculative(); } catch (...) {}
            speculative_start = -1;
            speculative_confirmed = 0;
            throw;
        }
    }
}

template <CudaBackbone Backbone>
void CausalLm<Backbone>::commit_speculative() {
    MFQ_RUNTIME_CHECK(
        speculative_start >= 0,
        "no speculative transaction to commit");
    for (auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        block->commit_speculative();
    }
    if constexpr (is_deepseek_v41) {
        MFQ_RUNTIME_CHECK(
            this->shared,
            "DeepSeek-V4.1 speculative state is unavailable");
        this->shared->commit_speculative();
    }
    speculative_start = -1;
    speculative_confirmed = 0;
}

template <CudaBackbone Backbone>
void CausalLm<Backbone>::rollback_speculative(int64_t accepted_suffix) {
    MFQ_RUNTIME_CHECK(
        speculative_start >= 0 && accepted_suffix >= 0,
        "no speculative transaction to roll back");
    const int64_t keep =
        speculative_start + speculative_confirmed + accepted_suffix;
    for (auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        block->rollback_speculative(keep);
    }
    if constexpr (is_deepseek_v41) {
        MFQ_RUNTIME_CHECK(
            this->shared,
            "DeepSeek-V4.1 speculative state is unavailable");
        this->shared->rollback_speculative();
    }
    cache_pos = keep;
    if constexpr (is_qwen4) {
        if (this->positions.defined()) {
            this->positions = this->positions.narrow(-1, 0, cache_pos);
        }
    }
    speculative_start = -1;
    speculative_confirmed = 0;
}

template <CudaBackbone Backbone>
mfq_tensor_backend::Tensor CausalLm<Backbone>::finalize_hidden(
        mfq_tensor_backend::Tensor x, int64_t batch, int64_t tokens) {
    if constexpr (is_qwen4) return this->final_mixer->pre(x)[0];
    if constexpr (is_glm5) {
        return glm5_next::rms_norm(
            x.mean(2), output_norm, rms_norm_eps());
    }
    if constexpr (Backbone == CudaBackbone::deepseek_v4) {
        x = g_profiler.measure("model.dsv4_hc_head", [&]() {
            auto flat = x.flatten(2).to(mfq_tensor_backend::kFloat32);
            auto inverse_rms = mfq_tensor_backend::rsqrt(
                flat.square().mean(-1, true) + rms_norm_eps());
            auto mixes = mfq_tensor_backend::matmul(
                flat, this->hc_head_fn.transpose(0, 1)) * inverse_rms;
            auto pre = mfq_tensor_backend::sigmoid(
                mixes * this->hc_head_scale + this->hc_head_base) + hc_eps();
            return (
                pre.unsqueeze(-1) *
                flat.reshape({batch, tokens, hc_mult(), hidden_size()}))
                .sum(2).to(mfq_tensor_backend::kFloat16).contiguous();
        });
    }
    if constexpr (is_deepseek_v41) {
        MFQ_RUNTIME_CHECK(
            this->shared,
            "DeepSeek-V4.1 final state is unavailable");
        x = g_profiler.measure(
            "model.deepseek_v41.final_collapse", [&]() {
                return this->shared->final_collapse(
                    x, this->shared->config.n_layers);
            });
    }
    return g_profiler.measure("model.output_norm", [&]() {
        if constexpr (is_minicpmo45) {
            return qwen_rms_norm_bf16(
                x.reshape({batch * tokens, hidden_size()}), output_norm,
                rms_norm_eps(), norm_weight_offset())
                .reshape({batch, tokens, hidden_size()});
        }
        return qwen_rms_norm(
            x.reshape({batch * tokens, hidden_size()})
                .to(mfq_tensor_backend::kFloat32),
            output_norm, rms_norm_eps(), norm_weight_offset())
            .reshape({batch, tokens, hidden_size()});
    });
}

template <CudaBackbone Backbone>
TextSessionStateKind CudaSessionCodec<Backbone>::kind(
        const CausalLm<Backbone>& model) {
    const auto& blocks = model.blocks;
    if (blocks.empty()) return TextSessionStateKind::Unsupported;
    if constexpr (
            Backbone == CudaBackbone::generic_qwen ||
            Backbone == CudaBackbone::minicpmo45 ||
            Backbone == CudaBackbone::minicpmo_tts) {
        const bool full_attention = std::all_of(
            blocks.begin(), blocks.end(),
            [](const std::unique_ptr<Block>& block) {
                return dynamic_cast<const FullBlock*>(block.get()) != nullptr;
            });
        if (full_attention) return TextSessionStateKind::FullAttention;
        if constexpr (Backbone == CudaBackbone::generic_qwen) {
            return mfq::cuda::qwen35::supports_text_session_state(blocks)
                ? TextSessionStateKind::HybridAttention
                : TextSessionStateKind::Unsupported;
        }
    } else if constexpr (Backbone == CudaBackbone::deepseek_v4) {
        return std::all_of(
            blocks.begin(), blocks.end(),
            [](const std::unique_ptr<Block>& block) {
                return dynamic_cast<const Dsv4Block*>(block.get()) != nullptr;
            })
            ? TextSessionStateKind::DeepseekV4
            : TextSessionStateKind::Unsupported;
    } else if constexpr (Backbone == CudaBackbone::glm_dsa) {
        return std::all_of(
            blocks.begin(), blocks.end(),
            [](const std::unique_ptr<Block>& block) {
                return dynamic_cast<const GlmDsaBlock*>(block.get()) != nullptr;
            })
            ? TextSessionStateKind::GlmDsa
            : TextSessionStateKind::Unsupported;
    }
    return TextSessionStateKind::Unsupported;
}

template <CudaBackbone Backbone>
bool CudaSessionCodec<Backbone>::supports_paged(
        const CausalLm<Backbone>& model) {
    if (kind(model) != TextSessionStateKind::FullAttention) return false;
    return std::all_of(
        model.blocks.begin(), model.blocks.end(),
        [](const std::unique_ptr<Block>& block) {
            const auto* full = dynamic_cast<const FullBlock*>(block.get());
            return full != nullptr && !full->sliding;
        });
}

template <CudaBackbone Backbone>
TextSessionState CudaSessionCodec<Backbone>::capture(
        const CausalLm<Backbone>& model,
        const std::vector<int64_t>& tokens) {
    const auto& blocks = model.blocks;
    const auto cache_pos = model.cache_pos;
    const auto state_kind = kind(model);
    if (state_kind == TextSessionStateKind::Unsupported) {
        throw std::runtime_error(
            "text session state is unsupported by this block layout");
    }
    if (cache_pos <= 0 || static_cast<size_t>(cache_pos) != tokens.size()) {
        throw std::runtime_error(
            "text session token count does not match the model cache");
    }
    TextSessionState state;
    state.tokens = tokens;
    state.kind = state_kind;
    state.cache_pos = cache_pos;
    if constexpr (
            Backbone == CudaBackbone::generic_qwen ||
            Backbone == CudaBackbone::minicpmo45 ||
            Backbone == CudaBackbone::minicpmo_tts) {
        if constexpr (Backbone == CudaBackbone::generic_qwen) {
            if (state.kind == TextSessionStateKind::HybridAttention) {
                return mfq::cuda::qwen35::capture_text_session_state(
                    blocks, tokens, cache_pos);
            }
        }
        state.blocks.reserve(blocks.size());
        for (const auto& block : blocks) {
            MfqCudaGuard guard(block->cuda_device);
            const auto* full = dynamic_cast<const FullBlock*>(block.get());
            if (full == nullptr) {
                throw std::runtime_error(
                    "full-attention session layer changed");
            }
            state.blocks.push_back(capture_full_attention_session_state(
                *full, cache_pos, state.bytes));
        }
    } else if constexpr (Backbone == CudaBackbone::deepseek_v4) {
        state.dsv4_blocks.reserve(blocks.size());
        for (const auto& block : blocks) {
            MfqCudaGuard guard(block->cuda_device);
            const auto* dsv4 = dynamic_cast<const Dsv4Block*>(block.get());
            if (dsv4 == nullptr || !dsv4->local_cache.defined()) {
                throw std::runtime_error(
                    "DeepSeek V4 local session cache is unavailable");
            }
            Dsv4BlockSessionState saved;
            saved.local_cache = dsv4->local_cache.clone();
            state.bytes += session_tensor_bytes(saved.local_cache);
            saved.compressor = capture_dsv4_pool_session_state(
                dsv4->compressor, cache_pos, state.bytes);
            saved.indexer_compressor = capture_dsv4_pool_session_state(
                dsv4->indexer_compressor, cache_pos, state.bytes);
            state.dsv4_blocks.push_back(std::move(saved));
        }
    } else if constexpr (Backbone == CudaBackbone::glm_dsa) {
        state.glm_dsa_blocks.reserve(blocks.size());
        for (const auto& block : blocks) {
            MfqCudaGuard guard(block->cuda_device);
            const auto* glm = dynamic_cast<const GlmDsaBlock*>(block.get());
            if (glm == nullptr || !glm->kv_cache.defined() ||
                    glm->kv_cache.dim() != 4 ||
                    glm->kv_cache.size(0) != 1 ||
                    cache_pos > glm->kv_cache.size(2)) {
                throw std::runtime_error(
                    "GLM DSA session MLA cache is unavailable");
            }
            GlmDsaBlockSessionState saved;
            saved.full_indexer = glm->full_indexer;
            saved.kv_capacity = glm->kv_cache.size(2);
            saved.kv_cache = glm->kv_cache.narrow(2, 0, cache_pos).clone();
            state.bytes += session_tensor_bytes(saved.kv_cache);
            if (glm->full_indexer) {
                if (!glm->index_cache.defined() ||
                        glm->index_cache.dim() != 3 ||
                        glm->index_cache.size(0) != 1 ||
                        cache_pos > glm->index_cache.size(1)) {
                    throw std::runtime_error(
                        "GLM DSA session index cache is unavailable");
                }
                saved.index_capacity = glm->index_cache.size(1);
                saved.index_cache = glm->index_cache.narrow(
                    1, 0, cache_pos).clone();
                state.bytes += session_tensor_bytes(saved.index_cache);
            }
            state.glm_dsa_blocks.push_back(std::move(saved));
        }
    }
    return state;
}

template <CudaBackbone Backbone>
void CudaSessionCodec<Backbone>::restore(
        CausalLm<Backbone>& model,
        const TextSessionState& state) {
    auto& blocks = model.blocks;
    const auto model_kind = kind(model);
    if (model_kind == TextSessionStateKind::Unsupported ||
            state.kind != model_kind || state.cache_pos <= 0 ||
            static_cast<size_t>(state.cache_pos) != state.tokens.size()) {
        throw std::runtime_error("text session state is incompatible");
    }
    if constexpr (
            Backbone == CudaBackbone::generic_qwen ||
            Backbone == CudaBackbone::minicpmo45 ||
            Backbone == CudaBackbone::minicpmo_tts) {
        if constexpr (Backbone == CudaBackbone::generic_qwen) {
            if (state.kind == TextSessionStateKind::HybridAttention) {
                mfq::cuda::qwen35::restore_text_session_state(blocks, state);
                model.cache_pos = state.cache_pos;
                return;
            }
        }
        if (state.blocks.size() != blocks.size()) {
            throw std::runtime_error(
                "full-attention session layer count changed");
        }
        for (size_t index = 0; index < blocks.size(); ++index) {
            auto& block = blocks[index];
            MfqCudaGuard guard(block->cuda_device);
            auto* full = dynamic_cast<FullBlock*>(block.get());
            if (full == nullptr) {
                throw std::runtime_error(
                    "full-attention session layer changed");
            }
            restore_full_attention_session_state(
                *full, state.blocks[index]);
        }
    } else if constexpr (Backbone == CudaBackbone::deepseek_v4) {
        if (state.dsv4_blocks.size() != blocks.size()) {
            throw std::runtime_error(
                "DeepSeek V4 session layer count changed");
        }
        for (size_t index = 0; index < blocks.size(); ++index) {
            auto& block = blocks[index];
            MfqCudaGuard guard(block->cuda_device);
            auto* dsv4 = dynamic_cast<Dsv4Block*>(block.get());
            const auto& saved = state.dsv4_blocks[index];
            if (dsv4 == nullptr || !saved.local_cache.defined() ||
                    saved.local_cache.dim() != 3 ||
                    saved.local_cache.size(0) != 1 ||
                    saved.local_cache.size(1) != 128) {
                throw std::runtime_error(
                    "DeepSeek V4 saved local cache is invalid");
            }
            restore_session_tensor(dsv4->local_cache, saved.local_cache);
            restore_dsv4_pool_session_state(
                dsv4->compressor, saved.compressor);
            restore_dsv4_pool_session_state(
                dsv4->indexer_compressor, saved.indexer_compressor);
            dsv4->shared_state->ensure();
        }
    } else if constexpr (Backbone == CudaBackbone::glm_dsa) {
        if (state.glm_dsa_blocks.size() != blocks.size()) {
            throw std::runtime_error(
                "GLM DSA session layer count changed");
        }
        for (size_t index = 0; index < blocks.size(); ++index) {
            auto& block = blocks[index];
            MfqCudaGuard guard(block->cuda_device);
            auto* glm = dynamic_cast<GlmDsaBlock*>(block.get());
            const auto& saved = state.glm_dsa_blocks[index];
            if (glm == nullptr || glm->full_indexer != saved.full_indexer ||
                    !saved.kv_cache.defined() ||
                    saved.kv_cache.dim() != 4 ||
                    saved.kv_cache.size(0) != 1 ||
                    saved.kv_cache.size(2) != state.cache_pos) {
                throw std::runtime_error(
                    "GLM DSA saved MLA cache is invalid");
            }
            restore_session_prefix_tensor(
                glm->kv_cache, saved.kv_cache, 2, saved.kv_capacity);
            if (saved.full_indexer) {
                if (!saved.index_cache.defined() ||
                        saved.index_cache.dim() != 3 ||
                        saved.index_cache.size(0) != 1 ||
                        saved.index_cache.size(1) != state.cache_pos) {
                    throw std::runtime_error(
                        "GLM DSA saved index cache is invalid");
                }
                restore_session_prefix_tensor(
                    glm->index_cache, saved.index_cache,
                    1, saved.index_capacity);
            } else {
                glm->index_cache = mfq_tensor_backend::Tensor();
            }
            glm->shared_state->reset();
        }
    }
    model.cache_pos = state.cache_pos;
}

#define MFQ_INSTANTIATE_SESSION_CODEC(BACKBONE) \
    template struct CudaSessionCodec<BACKBONE>; \
    template struct CausalLm<BACKBONE>;

MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::generic_qwen)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::minicpmo45)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::minicpmo_tts)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::gemma4)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::glm_dsa)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::glm5_next)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::qwen4_exp)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::deepseek_v4)
MFQ_INSTANTIATE_SESSION_CODEC(CudaBackbone::deepseek_v41)

#undef MFQ_INSTANTIATE_SESSION_CODEC

} // namespace mfq::cuda
