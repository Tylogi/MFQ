#pragma once

#include "cuda_transformer.h"
#include "cuda_model_plan.h"
#include "../models/deepseek_v4/deepseek_v4_causal_lm.h"
#include "../models/deepseek_v41/deepseek_v41_causal_lm.h"
#include "../models/glm5_next/causal_lm.h"
#include "../models/qwen4_exp/causal_lm.h"
#include "../models/glm_dsa/glm_dsa_causal_lm.h"
#include "models/gemma4.h"
#include "models/minicpmo45.h"
#include "models/qwen35.h"
#include "../models/qwen35/qwen35_causal_lm.h"
#include "mfq_cuda_paged_kv.h"
#include "prepared_prompt.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct FullBlockSessionState {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    int64_t capacity = 0;
    bool ring = false;
};

enum class TextSessionStateKind {
    Unsupported,
    FullAttention,
    HybridAttention,
    DeepseekV4,
    GlmDsa,
};

struct Dsv4PoolSessionState {
    int64_t ratio = 0;
    int64_t head_dim = 0;
    int64_t cache_quant_mode = 0;
    int64_t capacity = 0;
    bool overlap = false;
    mfq_tensor_backend::Tensor state_kv;
    mfq_tensor_backend::Tensor state_gate;
    mfq_tensor_backend::Tensor previous_kv;
    mfq_tensor_backend::Tensor previous_gate;
    mfq_tensor_backend::Tensor pool;
};

struct Dsv4BlockSessionState {
    mfq_tensor_backend::Tensor local_cache;
    Dsv4PoolSessionState compressor;
    Dsv4PoolSessionState indexer_compressor;
};

struct GlmDsaBlockSessionState {
    mfq_tensor_backend::Tensor kv_cache;
    mfq_tensor_backend::Tensor index_cache;
    int64_t kv_capacity = 0;
    int64_t index_capacity = 0;
    bool full_indexer = false;
};

enum class HybridBlockSessionStateKind {
    FullAttention,
    Recurrent,
};

struct HybridBlockSessionState {
    HybridBlockSessionStateKind kind =
        HybridBlockSessionStateKind::FullAttention;
    FullBlockSessionState full_attention;
    mfq_tensor_backend::Tensor convolution_state;
    mfq_tensor_backend::Tensor recurrent_state;
};

struct MtpSessionState {
    std::vector<FullBlockSessionState> blocks;
    mfq_tensor_backend::Tensor last_target_hidden;
    int64_t cache_pos = 0;
    size_t bytes = 0;
};

struct TextSessionState {
    std::vector<int64_t> tokens;
    std::string input_key;
    TextSessionStateKind kind = TextSessionStateKind::Unsupported;
    std::vector<FullBlockSessionState> blocks;
    std::vector<HybridBlockSessionState> hybrid_blocks;
    std::vector<Dsv4BlockSessionState> dsv4_blocks;
    std::vector<GlmDsaBlockSessionState> glm_dsa_blocks;
    std::optional<MtpSessionState> mtp;
    int64_t cache_pos = 0;
    size_t bytes = 0;
    uint64_t last_used = 0;
};

using CudaPagedPayload =
    std::shared_ptr<const std::vector<uint8_t>>;

class CudaPagedWriter {
public:
    template <typename T>
    void scalar(T value) {
        using Unsigned = std::make_unsigned_t<T>;
        const auto converted = static_cast<Unsigned>(value);
        for (size_t index = 0; index < sizeof(T); ++index) {
            bytes_.push_back(static_cast<uint8_t>(
                converted >> (index * 8)));
        }
    }

    void raw(const void * data, size_t size) {
        const auto * begin = static_cast<const uint8_t *>(data);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

    void tensor(const mfq_tensor_backend::Tensor & source) {
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

    CudaPagedPayload finish() && {
        return std::make_shared<const std::vector<uint8_t>>(
            std::move(bytes_));
    }

private:
    std::vector<uint8_t> bytes_;
};

class CudaPagedReader {
public:
    explicit CudaPagedReader(const std::vector<uint8_t> & bytes)
        : cursor_(bytes.data()), end_(bytes.data() + bytes.size()) {}

    template <typename T>
    T scalar(const char * name) {
        if (remaining() < sizeof(T)) {
            throw std::runtime_error(
                std::string("truncated CUDA paged cache ") + name);
        }
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned value = 0;
        for (size_t index = 0; index < sizeof(T); ++index) {
            value |= static_cast<Unsigned>(cursor_[index]) << (index * 8);
        }
        cursor_ += sizeof(T);
        return static_cast<T>(value);
    }

    void raw(void * destination, size_t size, const char * name) {
        if (remaining() < size) {
            throw std::runtime_error(
                std::string("truncated CUDA paged cache ") + name);
        }
        std::memcpy(destination, cursor_, size);
        cursor_ += size;
    }

    std::pair<mfq_tensor_backend::Tensor, int> tensor() {
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

    void expect_end() const {
        if (cursor_ != end_) {
            throw std::runtime_error(
                "trailing CUDA paged cache payload bytes");
        }
    }

private:
    size_t remaining() const {
        return static_cast<size_t>(end_ - cursor_);
    }

    const uint8_t * cursor_;
    const uint8_t * end_;
};

struct CudaDecodedPagedLayer {
    int64_t capacity = 0;
    int device = -1;
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
};

struct CudaDecodedPagedBlock {
    uint32_t start = 0;
    uint32_t count = 0;
    std::vector<CudaDecodedPagedLayer> layers;
};

std::vector<CudaPagedPayload> encode_cuda_paged_session(
        const TextSessionState & state,
        size_t block_size,
        size_t first_block,
        size_t maximum_blocks = std::numeric_limits<size_t>::max());

CudaPagedPayload encode_cuda_paged_block(
        const TextSessionState & state,
        size_t block_size,
        size_t block_index);

CudaDecodedPagedBlock decode_cuda_paged_block(
        const std::vector<uint8_t> & payload);

TextSessionState decode_cuda_paged_session(
        const std::vector<CudaPagedPayload> & payloads,
        const std::vector<int64_t> & tokens,
        size_t block_size);

size_t session_tensor_bytes(const mfq_tensor_backend::Tensor & tensor);

bool session_tensor_layout_matches(
        const mfq_tensor_backend::Tensor & target,
        const mfq_tensor_backend::Tensor & source);

void restore_session_tensor(
        mfq_tensor_backend::Tensor & target,
        const mfq_tensor_backend::Tensor & source);

void restore_session_prefix_tensor(
        mfq_tensor_backend::Tensor & target,
        const mfq_tensor_backend::Tensor & source,
        int64_t dimension,
        int64_t capacity);

FullBlockSessionState capture_full_attention_session_state(
        const FullBlock & block,
        int64_t cache_pos,
        size_t & bytes);

void restore_full_attention_session_state(
        FullBlock & block,
        const FullBlockSessionState & state);

Dsv4PoolSessionState capture_dsv4_pool_session_state(
        const Dsv4PoolState & source,
        int64_t cache_pos,
        size_t & bytes);

void restore_dsv4_pool_session_state(
        Dsv4PoolState & target,
        const Dsv4PoolSessionState & state);

namespace mfq::cuda {

template <CudaBackbone Backbone>
struct CausalLmArchitectureState;

template <>
struct CausalLmArchitectureState<CudaBackbone::generic_qwen> {
    mfq::models::qwen35::Config config;
};

template <>
struct CausalLmArchitectureState<CudaBackbone::minicpmo45> {
    mfq::models::minicpmo45::Config config;
};

template <>
struct CausalLmArchitectureState<CudaBackbone::minicpmo_tts> {
    mfq::models::ModelConfig config;
};

template <>
struct CausalLmArchitectureState<CudaBackbone::gemma4> {
    mfq::models::gemma4::Config config;
    double embed_scale = 1.0;
};

template <>
struct CausalLmArchitectureState<CudaBackbone::glm_dsa> {
    mfq::models::glm_dsa::Config config;
};

template <>
struct CausalLmArchitectureState<CudaBackbone::glm5_next> {
    mfq::models::flash_next::GlmConfig config{};
};

template <>
struct CausalLmArchitectureState<CudaBackbone::qwen4_exp> {
    mfq::models::flash_next::QwenConfig config{};
    std::unique_ptr<flash_runtime::Gr> final_mixer;
    mfq_tensor_backend::Tensor positions;
    int64_t batch = 0;
};

template <>
struct CausalLmArchitectureState<CudaBackbone::deepseek_v4> {
    mfq::models::deepseek_v4::Config config;
    mfq_tensor_backend::Tensor hc_head_fn;
    mfq_tensor_backend::Tensor hc_head_scale;
    mfq_tensor_backend::Tensor hc_head_base;
};

template <>
struct CausalLmArchitectureState<CudaBackbone::deepseek_v41> {
    mfq::models::deepseek_v41::Config config;
    std::shared_ptr<deepseek_v41_runtime::SharedState> shared;
};

template <CudaBackbone Backbone>
struct CausalLm : CausalLmArchitectureState<Backbone> {
    static constexpr CudaBackbone backbone = Backbone;
    static constexpr bool is_qwen4 = Backbone == CudaBackbone::qwen4_exp;
    static constexpr bool is_glm5 = Backbone == CudaBackbone::glm5_next;
    static constexpr bool is_flash_next = is_qwen4 || is_glm5;
    static constexpr bool is_dsv4 =
        Backbone == CudaBackbone::deepseek_v4 ||
        Backbone == CudaBackbone::deepseek_v41;
    static constexpr bool is_deepseek_v41 =
        Backbone == CudaBackbone::deepseek_v41;
    static constexpr bool is_minicpmo45 =
        Backbone == CudaBackbone::minicpmo45;
    static constexpr bool is_gemma4 = Backbone == CudaBackbone::gemma4;
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    QuantLinear embed;
    std::vector<std::unique_ptr<Block>> blocks;
    mfq_tensor_backend::Tensor output_norm;
    QuantLinear lm_head;
    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    int64_t speculative_start = -1;
    int64_t speculative_confirmed = 0;
    bool speculative_suffix_forward = false;

    int64_t vocab_size() const noexcept {
        if constexpr (is_flash_next || is_deepseek_v41) {
            return this->config.vocab;
        } else {
            return this->config.vocab_size;
        }
    }

    int64_t hidden_size() const noexcept {
        if constexpr (is_flash_next || is_deepseek_v41) {
            return this->config.hidden;
        } else {
            return this->config.hidden_size;
        }
    }

    int64_t num_hidden_layers() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.layers;
        } else if constexpr (is_deepseek_v41) {
            return this->config.n_layers;
        } else {
            return this->config.num_hidden_layers;
        }
    }

    int64_t num_attention_heads() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.heads;
        } else if constexpr (is_deepseek_v41) {
            return this->config.n_heads;
        } else {
            return this->config.num_attention_heads;
        }
    }

    int64_t num_key_value_heads() const noexcept {
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

    int64_t head_dim() const noexcept {
        if constexpr (is_qwen4) {
            return this->config.width;
        } else if constexpr (is_glm5) {
            return this->config.nope;
        } else {
            return this->config.head_dim;
        }
    }

    int64_t max_position_embeddings() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.maximum;
        } else {
            return this->config.max_position_embeddings;
        }
    }

    int64_t rotary_dim() const noexcept {
        return this->config.rotary_dim;
    }

    double rope_base() const noexcept {
        return this->config.rope_base;
    }

    void set_max_position_embeddings(int64_t value) noexcept {
        if constexpr (is_flash_next) this->config.maximum = value;
        else this->config.max_position_embeddings = value;
    }

    double rms_norm_eps() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.eps;
        } else if constexpr (is_deepseek_v41) {
            return this->config.rms_eps;
        } else {
            return this->config.rms_norm_eps;
        }
    }

    double norm_weight_offset() const noexcept {
        if constexpr (Backbone == CudaBackbone::generic_qwen) {
            return this->config.legacy_tensor_layout.norm_weight_offset;
        } else if constexpr (
                is_flash_next || is_dsv4 || is_gemma4 || is_minicpmo45) {
            return 0.0;
        } else {
            return 1.0;
        }
    }

    bool tie_word_embeddings() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.tied_embeddings;
        } else if constexpr (is_deepseek_v41) {
            return false;
        } else {
            return this->config.tie_word_embeddings;
        }
    }

    int64_t num_experts() const noexcept {
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

    int64_t hc_mult() const noexcept {
        if constexpr (is_flash_next) {
            return this->config.streams;
        } else if constexpr (is_deepseek_v41 ||
                             Backbone == CudaBackbone::deepseek_v4) {
            return this->config.hc_mult;
        } else {
            return 1;
        }
    }

    double hc_eps() const noexcept {
        if constexpr (is_glm5 || is_deepseek_v41 ||
                      Backbone == CudaBackbone::deepseek_v4) {
            return this->config.hc_eps;
        } else {
            return 1e-6;
        }
    }

    double final_logit_softcapping() const noexcept {
        if constexpr (is_gemma4) {
            return this->config.final_logit_softcapping;
        } else {
            return 0.0;
        }
    }

    double embedding_scale() const noexcept {
        if constexpr (is_gemma4) return this->embed_scale;
        else return 1.0;
    }

    std::string_view model_type() const noexcept {
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

    std::string_view layer_type(int64_t layer) const {
        if constexpr (is_deepseek_v41) {
            return "deepseek_v41";
        } else {
            return this->config.layer_types.at(
                static_cast<std::size_t>(layer));
        }
    }

    bool supports_qwen_speculation() const {
        if constexpr (
                Backbone != CudaBackbone::generic_qwen && !is_flash_next) {
            return false;
        }
        return !blocks.empty() && std::all_of(
            blocks.begin(), blocks.end(),
            [](const auto& block) { return block->supports_speculation(); });
    }

    bool supports_deepseek_v41_speculation() const {
        if constexpr (!is_deepseek_v41) return false;
        return !blocks.empty() && std::all_of(
            blocks.begin(), blocks.end(),
            [](const auto& block) { return block->supports_speculation(); });
    }

    void begin_speculative_suffix(int64_t draft_tokens) {
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

    void commit_speculative() {
        MFQ_RUNTIME_CHECK(speculative_start >= 0, "no speculative transaction to commit");
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

    void rollback_speculative(int64_t accepted_suffix = 0) {
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
        // Full-attention KV slots beyond this logical length are overwritten
        // by the next pass; no history-sized cache copy is needed.
        cache_pos = keep;
        if constexpr (is_qwen4) {
            if (this->positions.defined()) {
                this->positions = this->positions.narrow(-1, 0, cache_pos);
            }
        }
        speculative_start = -1;
        speculative_confirmed = 0;
    }

    mfq_tensor_backend::Tensor embed_forward(mfq_tensor_backend::Tensor ids) const {
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

    void reset(int64_t B) {
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

    TextSessionStateKind text_session_state_kind() const {
        if (blocks.empty()) return TextSessionStateKind::Unsupported;
        if constexpr (
                Backbone == CudaBackbone::generic_qwen ||
                Backbone == CudaBackbone::minicpmo45 ||
                Backbone == CudaBackbone::minicpmo_tts) {
            const bool full_attention = std::all_of(
                blocks.begin(), blocks.end(),
                [](const std::unique_ptr<Block>& block) {
                    return dynamic_cast<const FullBlock*>(block.get()) !=
                        nullptr;
                });
            if (full_attention) return TextSessionStateKind::FullAttention;
            if constexpr (Backbone == CudaBackbone::generic_qwen) {
                return mfq::cuda::qwen35::supports_text_session_state(blocks)
                    ? TextSessionStateKind::HybridAttention
                    : TextSessionStateKind::Unsupported;
            }
            return TextSessionStateKind::Unsupported;
        } else if constexpr (Backbone == CudaBackbone::deepseek_v4) {
            return std::all_of(
                blocks.begin(), blocks.end(),
                [](const std::unique_ptr<Block>& block) {
                    return dynamic_cast<const Dsv4Block*>(block.get()) !=
                        nullptr;
                })
                ? TextSessionStateKind::DeepseekV4
                : TextSessionStateKind::Unsupported;
        } else if constexpr (Backbone == CudaBackbone::glm_dsa) {
            return std::all_of(
                blocks.begin(), blocks.end(),
                [](const std::unique_ptr<Block>& block) {
                    return dynamic_cast<const GlmDsaBlock*>(block.get()) !=
                        nullptr;
                })
                ? TextSessionStateKind::GlmDsa
                : TextSessionStateKind::Unsupported;
        }
        return TextSessionStateKind::Unsupported;
    }

    bool supports_text_session_state() const {
        return text_session_state_kind() != TextSessionStateKind::Unsupported;
    }

    bool supports_paged_text_session_state() const {
        if (text_session_state_kind() !=
                TextSessionStateKind::FullAttention) {
            return false;
        }
        return std::all_of(
            blocks.begin(), blocks.end(),
            [](const std::unique_ptr<Block> & block) {
                const auto * full =
                    dynamic_cast<const FullBlock *>(block.get());
                return full != nullptr && !full->sliding;
            });
    }

    TextSessionState capture_text_session_state(
            const std::vector<int64_t> & tokens) const {
        if (!supports_text_session_state()) {
            throw std::runtime_error(
                "text session state is unsupported by this block layout");
        }
        if (cache_pos <= 0 ||
                static_cast<size_t>(cache_pos) != tokens.size()) {
            throw std::runtime_error(
                "text session token count does not match the model cache");
        }
        TextSessionState state;
        state.tokens = tokens;
        state.kind = text_session_state_kind();
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
            for (const auto & block : blocks) {
                MfqCudaGuard guard(block->cuda_device);
                const auto * full =
                    dynamic_cast<const FullBlock *>(block.get());
                if (full == nullptr) {
                    throw std::runtime_error(
                        "full-attention session layer changed");
                }
                state.blocks.push_back(capture_full_attention_session_state(
                    *full, cache_pos, state.bytes));
            }
        } else if constexpr (Backbone == CudaBackbone::deepseek_v4) {
            state.dsv4_blocks.reserve(blocks.size());
            for (const auto & block : blocks) {
                MfqCudaGuard guard(block->cuda_device);
                const auto * dsv4 =
                    dynamic_cast<const Dsv4Block *>(block.get());
                if (dsv4 == nullptr || !dsv4->local_cache.defined()) {
                    throw std::runtime_error(
                        "DeepSeek V4 local session cache is unavailable");
                }
                Dsv4BlockSessionState saved;
                saved.local_cache = dsv4->local_cache.clone();
                state.bytes += session_tensor_bytes(saved.local_cache);
                saved.compressor = capture_dsv4_pool_session_state(
                    dsv4->compressor, cache_pos, state.bytes);
                saved.indexer_compressor =
                    capture_dsv4_pool_session_state(
                        dsv4->indexer_compressor,
                        cache_pos, state.bytes);
                state.dsv4_blocks.push_back(std::move(saved));
            }
        } else if constexpr (Backbone == CudaBackbone::glm_dsa) {
            state.glm_dsa_blocks.reserve(blocks.size());
            for (const auto & block : blocks) {
                MfqCudaGuard guard(block->cuda_device);
                const auto * glm =
                    dynamic_cast<const GlmDsaBlock *>(block.get());
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
                saved.kv_cache = glm->kv_cache.narrow(
                    2, 0, cache_pos).clone();
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

    void restore_text_session_state(const TextSessionState & state) {
        if (!supports_text_session_state() ||
                state.kind != text_session_state_kind() ||
                state.cache_pos <= 0 ||
                static_cast<size_t>(state.cache_pos) != state.tokens.size()) {
            throw std::runtime_error("text session state is incompatible");
        }
        if constexpr (
                Backbone == CudaBackbone::generic_qwen ||
                Backbone == CudaBackbone::minicpmo45 ||
                Backbone == CudaBackbone::minicpmo_tts) {
            if constexpr (Backbone == CudaBackbone::generic_qwen) {
                if (state.kind == TextSessionStateKind::HybridAttention) {
                    mfq::cuda::qwen35::restore_text_session_state(
                        blocks, state);
                    cache_pos = state.cache_pos;
                    return;
                }
            }
            if (state.blocks.size() != blocks.size()) {
                throw std::runtime_error(
                    "full-attention session layer count changed");
            }
            for (size_t index = 0; index < blocks.size(); ++index) {
                auto & block = blocks[index];
                MfqCudaGuard guard(block->cuda_device);
                auto * full = dynamic_cast<FullBlock *>(block.get());
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
                auto & block = blocks[index];
                MfqCudaGuard guard(block->cuda_device);
                auto * dsv4 = dynamic_cast<Dsv4Block *>(block.get());
                const auto & saved = state.dsv4_blocks[index];
                if (dsv4 == nullptr || !saved.local_cache.defined() ||
                        saved.local_cache.dim() != 3 ||
                        saved.local_cache.size(0) != 1 ||
                        saved.local_cache.size(1) != 128) {
                    throw std::runtime_error(
                        "DeepSeek V4 saved local cache is invalid");
                }
                restore_session_tensor(
                    dsv4->local_cache, saved.local_cache);
                restore_dsv4_pool_session_state(
                    dsv4->compressor, saved.compressor);
                restore_dsv4_pool_session_state(
                    dsv4->indexer_compressor,
                    saved.indexer_compressor);
                dsv4->shared_state->ensure();
            }
        } else if constexpr (Backbone == CudaBackbone::glm_dsa) {
            if (state.glm_dsa_blocks.size() != blocks.size()) {
                throw std::runtime_error(
                    "GLM DSA session layer count changed");
            }
            for (size_t index = 0; index < blocks.size(); ++index) {
                auto & block = blocks[index];
                MfqCudaGuard guard(block->cuda_device);
                auto * glm = dynamic_cast<GlmDsaBlock *>(block.get());
                const auto & saved = state.glm_dsa_blocks[index];
                if (glm == nullptr ||
                        glm->full_indexer != saved.full_indexer ||
                        !saved.kv_cache.defined() ||
                        saved.kv_cache.dim() != 4 ||
                        saved.kv_cache.size(0) != 1 ||
                        saved.kv_cache.size(2) != state.cache_pos) {
                    throw std::runtime_error(
                        "GLM DSA saved MLA cache is invalid");
                }
                restore_session_prefix_tensor(
                    glm->kv_cache, saved.kv_cache,
                    2, saved.kv_capacity);
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
        cache_pos = state.cache_pos;
    }

    mfq_tensor_backend::Tensor finalize_hidden(mfq_tensor_backend::Tensor x, int64_t B, int64_t T) {
        if constexpr (is_qwen4) return this->final_mixer->pre(x)[0];
        if constexpr (is_glm5) {
            return mfq::flash_next::rms_norm(
                x.mean(2), output_norm, rms_norm_eps());
        }
        if constexpr (Backbone == CudaBackbone::deepseek_v4) {
            x = g_profiler.measure("model.dsv4_hc_head", [&]() {
                auto flat = x.flatten(2).to(mfq_tensor_backend::kFloat32);
                auto inverse_rms = mfq_tensor_backend::rsqrt(
                    flat.square().mean(-1, true) + rms_norm_eps());
                auto mixes = mfq_tensor_backend::matmul(
                    flat, this->hc_head_fn.transpose(0, 1)) *
                    inverse_rms;
                auto pre = mfq_tensor_backend::sigmoid(
                    mixes * this->hc_head_scale +
                    this->hc_head_base) + hc_eps();
                return (
                    pre.unsqueeze(-1) *
                    flat.reshape({B, T, hc_mult(), hidden_size()}))
                    .sum(2).to(mfq_tensor_backend::kFloat16).contiguous();
            });
        }
        if constexpr (is_deepseek_v41) {
            MFQ_RUNTIME_CHECK(
                this->shared,
                "DeepSeek-V4.1 final state is unavailable");
            x = g_profiler.measure("model.deepseek_v41.final_collapse", [&]() {
                return this->shared->final_collapse(
                    x, this->shared->config.n_layers);
            });
        }
        return g_profiler.measure("model.output_norm", [&]() {
            if constexpr (is_minicpmo45) {
                return qwen_rms_norm_bf16(
                    x.reshape({B * T, hidden_size()}), output_norm,
                    rms_norm_eps(), norm_weight_offset())
                    .reshape({B, T, hidden_size()});
            }
            return qwen_rms_norm(
                x.reshape({B * T, hidden_size()})
                    .to(mfq_tensor_backend::kFloat32),
                output_norm, rms_norm_eps(), norm_weight_offset())
                .reshape({B, T, hidden_size()});
        });
    }

    mfq_tensor_backend::Tensor hidden_forward(mfq_tensor_backend::Tensor ids,
                                 MfqOptional<mfq_tensor_backend::Tensor> pos_override = mfq_nullopt,
                                 MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt,
                                 std::vector<mfq_tensor_backend::Tensor> * block_trace = nullptr,
                                 MfqOptional<mfq_tensor_backend::Tensor> cache_positions_override = mfq_nullopt,
                                 mfq_tensor_backend::Tensor* raw_hidden = nullptr,
                                 int64_t confirmed_prefix = 0) {
        const int primary = g_layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        ids = tensor_to_cuda_device(
            ids.to(mfq_tensor_backend::kInt64), primary);
        if (ids.dim() == 1) ids = ids.unsqueeze(0);
        auto x = g_profiler.measure("model.embed", [&]() { return embed_forward(ids); });
        return hidden_forward_inputs(
            ids, x, pos_override, seq_len, block_trace,
            mfq_nullopt, false, cache_positions_override, raw_hidden, confirmed_prefix);
    }

    mfq_tensor_backend::Tensor hidden_forward_speculative_suffix(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor* raw_hidden = nullptr) {
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

    mfq_tensor_backend::Tensor hidden_forward_inputs(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor input_embeddings,
            MfqOptional<mfq_tensor_backend::Tensor> pos_override = mfq_nullopt,
            MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt,
            std::vector<mfq_tensor_backend::Tensor> * block_trace = nullptr,
            MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt,
            bool advance_cache_with_position_ids = false,
            MfqOptional<mfq_tensor_backend::Tensor> cache_positions_override = mfq_nullopt,
            mfq_tensor_backend::Tensor* raw_hidden = nullptr,
            int64_t confirmed_prefix = 0) {
        const int primary = g_layer_placement.primary_device();
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
        if (g_dense_cpu_layer_count > 0) {
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

    mfq_tensor_backend::Tensor forward_inputs(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor input_embeddings,
            MfqOptional<mfq_tensor_backend::Tensor> pos_override = mfq_nullopt,
            MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt) {
        return logits_from_hidden(hidden_forward_inputs(
            ids, input_embeddings, pos_override, seq_len));
    }

    mfq_tensor_backend::Tensor apply_final_logit_softcap(
            mfq_tensor_backend::Tensor logits) const {
        const double cap = final_logit_softcapping();
        return cap > 0.0
            ? mfq_tensor_backend::tanh(logits / cap) * cap
            : logits;
    }

    mfq_tensor_backend::Tensor logits_from_hidden(mfq_tensor_backend::Tensor y) {
        auto logits = g_profiler.measure(
            "model.lm_head", [&]() { return lm_head.forward(y); });
        if constexpr (is_minicpmo45) {
            logits = logits.to(mfq_tensor_backend::kBFloat16).contiguous();
        }
        return apply_final_logit_softcap(std::move(logits));
    }

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor ids) {
        return logits_from_hidden(hidden_forward(ids));
    }

    mfq_tensor_backend::Tensor last_logits_prepared(
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

    mfq_tensor_backend::Tensor last_logits(mfq_tensor_backend::Tensor ids) {
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

    mfq_tensor_backend::Tensor last_logits_static(mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor seq_len) {
        auto y = hidden_forward(ids, pos, seq_len, nullptr, pos);
        auto last = y.index({Slice(), -1, Slice()});
        if constexpr (is_minicpmo45) {
            return logits_from_hidden(
                last.to(mfq_tensor_backend::kBFloat16).contiguous());
        }
        last = last.to(mfq_tensor_backend::kFloat16).contiguous();
        return apply_final_logit_softcap(lm_head.forward(last));
    }

    mfq_tensor_backend::Tensor next_token(mfq_tensor_backend::Tensor ids) {
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

    mfq_tensor_backend::Tensor next_token_static(mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor pos, mfq_tensor_backend::Tensor seq_len) {
        auto y = hidden_forward(ids, pos, seq_len, nullptr, pos);
        return next_token_from_hidden(y);
    }

    mfq_tensor_backend::Tensor next_token_from_hidden(mfq_tensor_backend::Tensor y) {
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
};

struct Qwen35CausalLm final : CausalLm<CudaBackbone::generic_qwen> {};
struct MiniCPMO45CausalLm final : CausalLm<CudaBackbone::minicpmo45> {};
struct MiniCPMOTtsCausalLm final : CausalLm<CudaBackbone::minicpmo_tts> {};
struct Gemma4CausalLm final : CausalLm<CudaBackbone::gemma4> {};
struct GlmDsaCausalLm final : CausalLm<CudaBackbone::glm_dsa> {};
struct Glm5CausalLm final : CausalLm<CudaBackbone::glm5_next> {};
struct Qwen4CausalLm final : CausalLm<CudaBackbone::qwen4_exp> {};
struct DeepseekV4CausalLm final : CausalLm<CudaBackbone::deepseek_v4> {};
struct DeepseekV41CausalLm final : CausalLm<CudaBackbone::deepseek_v41> {};

template <CudaBackbone Backbone>
struct CausalLmType;

template <>
struct CausalLmType<CudaBackbone::generic_qwen> { using type = Qwen35CausalLm; };
template <>
struct CausalLmType<CudaBackbone::minicpmo45> { using type = MiniCPMO45CausalLm; };
template <>
struct CausalLmType<CudaBackbone::minicpmo_tts> { using type = MiniCPMOTtsCausalLm; };
template <>
struct CausalLmType<CudaBackbone::gemma4> { using type = Gemma4CausalLm; };
template <>
struct CausalLmType<CudaBackbone::glm_dsa> { using type = GlmDsaCausalLm; };
template <>
struct CausalLmType<CudaBackbone::glm5_next> { using type = Glm5CausalLm; };
template <>
struct CausalLmType<CudaBackbone::qwen4_exp> { using type = Qwen4CausalLm; };
template <>
struct CausalLmType<CudaBackbone::deepseek_v4> { using type = DeepseekV4CausalLm; };
template <>
struct CausalLmType<CudaBackbone::deepseek_v41> { using type = DeepseekV41CausalLm; };

template <CudaBackbone Backbone>
using CausalLmFor = typename CausalLmType<Backbone>::type;

template <CudaBackbone Backbone>
CausalLmFor<Backbone> load_causal_lm(
    const std::string& model_path,
    const std::string& config_path,
    std::int64_t context_size_override = 0,
    bool load_blocks = true,
    bool defer_moe_cache_finalize = false,
    std::shared_ptr<const mfq::ModelSource> source = {});

} // namespace mfq::cuda
