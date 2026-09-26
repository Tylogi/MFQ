#include "causal_lm.h"

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
    template struct CudaSessionCodec<BACKBONE>;

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
