#pragma once

#include "models/transformer.h"
#include "cuda_model_plan.h"
#include "../models/deepseek_v4/causal_lm.h"
#include "../models/deepseek_v41/causal_lm.h"
#include "../models/glm5_next/causal_lm.h"
#include "../models/qwen4_exp/causal_lm.h"
#include "../models/glm_dsa/causal_lm.h"
#include "models/include/gemma4.h"
#include "models/include/minicpmo45.h"
#include "models/include/qwen35.h"
#include "../models/qwen35/causal_lm.h"
#include "mfq_cuda_paged_kv.h"

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
#include <variant>
#include <vector>

struct FullBlockSessionState {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    int64_t capacity = 0;
    bool ring = false;
};

class CudaSessionStateError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
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

using TextSessionPayload = std::variant<
    std::monostate,
    std::vector<FullBlockSessionState>,
    std::vector<HybridBlockSessionState>,
    std::vector<Dsv4BlockSessionState>,
    std::vector<GlmDsaBlockSessionState>>;

struct TextSessionState {
    std::vector<int64_t> tokens;
    std::string input_key;
    TextSessionPayload payload;
    std::optional<MtpSessionState> mtp;
    int64_t cache_pos = 0;
    size_t bytes = 0;

    TextSessionStateKind kind() const noexcept {
        return static_cast<TextSessionStateKind>(payload.index());
    }
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

    void raw(const void * data, size_t size);

    void tensor(const mfq_tensor_backend::Tensor & source);

    CudaPagedPayload finish() &&;

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
            throw CudaSessionStateError(
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

    void raw(void * destination, size_t size, const char * name);

    std::pair<mfq_tensor_backend::Tensor, int> tensor();

    void expect_end() const;

private:
    size_t remaining() const;

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

struct CausalLmMetadata {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t rotary_dim = 0;
    int64_t num_experts = 0;
    int64_t hc_mult = 1;
    double rope_base = 0.0;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 0.0;
    double hc_eps = 1e-6;
    double final_logit_softcapping = 0.0;
    double embedding_scale = 1.0;
    bool tie_word_embeddings = false;
    bool rope_interleaved = false;
    bool multi_axis_positions = false;
    bool flash_next = false;
    bool gemma4 = false;
    bool decode_graph_double_warmup = false;
    std::string model_type;
    std::vector<std::string> layer_types;
};

struct CudaPreparedPositions {
    mfq_tensor_backend::Tensor positions;
    mfq_tensor_backend::Tensor full_positions;
};

struct CausalLmArchitecture {
    CausalLmMetadata metadata;

    void adapter_validate_load_options() const;
    bool adapter_uses_common_rope() const noexcept;
    void adapter_configure_rope(
        RopeCache& rope,
        mfq_tensor_backend::Device device) const;
    void adapter_load_final_state(
        const mfq::ModelSource& source,
        mfq_tensor_backend::Tensor& output_norm);
    void adapter_prepare_blocks(const mfq::ModelSource& source);
    bool adapter_supports_dense_cpu_offload() const noexcept;

    mfq_tensor_backend::Tensor adapter_embed(
        mfq_tensor_backend::Tensor output) const;
    void adapter_reset(int64_t batch);
    bool adapter_requires_batch_reset(int64_t batch) const noexcept;
    void adapter_validate_forward(
        int64_t batch,
        int64_t tokens,
        int64_t cache_position,
        bool has_position_override,
        bool has_cache_position_override,
        bool has_attention_mask) const;
    bool adapter_allows_speculative_position_override() const noexcept;
    CudaPreparedPositions adapter_prepare_positions(
        mfq_tensor_backend::Tensor positions,
        int64_t batch,
        int64_t tokens);
    void adapter_validate_positions(
        const mfq_tensor_backend::Tensor& positions,
        int64_t batch,
        int64_t tokens,
        bool has_mrope) const;
    MfqOptional<mfq_tensor_backend::Tensor> adapter_attention_mask(
        MfqOptional<mfq_tensor_backend::Tensor> mask,
        int64_t tokens,
        int64_t cache_position) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    void adapter_begin_forward(bool capture_raw_hidden);
    bool adapter_pass_cache_positions(
        bool has_mrope,
        bool has_override) const noexcept;
    mfq_tensor_backend::Tensor adapter_block_positions(
        const mfq_tensor_backend::Tensor& full_positions,
        const mfq_tensor_backend::Tensor& local_positions,
        int device) const;
    bool adapter_pass_attention_mask() const noexcept;
    void adapter_finish_forward(
        const mfq_tensor_backend::Tensor& full_positions,
        int64_t batch,
        int64_t tokens);
    bool adapter_force_cache_advance() const noexcept;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_raw_hidden(
        const mfq_tensor_backend::Tensor& hidden,
        const mfq_tensor_backend::Tensor& finalized) const;
    mfq_tensor_backend::Tensor adapter_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    bool adapter_uses_decode_sequence_length() const noexcept;
    bool adapter_supports_prepared_prompt() const noexcept;
    bool adapter_supports_speculation() const noexcept;
    bool adapter_supports_suffix_speculation() const noexcept;
    void adapter_begin_speculative();
    void adapter_commit_speculative();
    void adapter_rollback_speculative(int64_t keep);
};

struct Qwen35Model : CausalLmArchitecture {
    mfq::models::qwen35::Config config;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    void adapter_configure_rope(
        RopeCache& rope,
        mfq_tensor_backend::Device device) const;
    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
    mfq_tensor_backend::Tensor adapter_embed(
        mfq_tensor_backend::Tensor output) const;
    void adapter_validate_positions(
        const mfq_tensor_backend::Tensor& positions,
        int64_t batch,
        int64_t tokens,
        bool has_mrope) const;
    bool adapter_supports_prepared_prompt() const noexcept;
    bool adapter_supports_speculation() const noexcept;
};

struct MiniCPMO45Model : CausalLmArchitecture {
    mfq::models::minicpmo45::Config config;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
    mfq_tensor_backend::Tensor adapter_embed(
        mfq_tensor_backend::Tensor output) const;
    MfqOptional<mfq_tensor_backend::Tensor> adapter_attention_mask(
        MfqOptional<mfq_tensor_backend::Tensor> mask,
        int64_t tokens,
        int64_t cache_position) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    bool adapter_pass_cache_positions(bool, bool) const noexcept;
    bool adapter_pass_attention_mask() const noexcept;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    bool adapter_uses_decode_sequence_length() const noexcept;
};

struct MiniCPMOTtsModel : CausalLmArchitecture {
    mfq::models::ModelConfig config;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    bool adapter_uses_common_rope() const noexcept;
    bool adapter_supports_dense_cpu_offload() const noexcept;
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
};

struct Gemma4Model : CausalLmArchitecture {
    mfq::models::gemma4::Config config;
    double embed_scale = 1.0;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
};

struct GlmDsaModel : CausalLmArchitecture {
    mfq::models::glm_dsa::Config config;
    std::unordered_map<int, std::shared_ptr<GlmDsaSharedState>> block_states;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    bool adapter_uses_common_rope() const noexcept;
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
};

struct Glm5Model : CausalLmArchitecture {
    mfq::models::glm5_next::Config config;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    void adapter_validate_load_options() const;
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.maximum = value;
    }
    void adapter_validate_forward(
        int64_t batch,
        int64_t tokens,
        int64_t cache_position,
        bool has_position_override,
        bool has_cache_position_override,
        bool has_attention_mask) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_raw_hidden(
        const mfq_tensor_backend::Tensor& hidden,
        const mfq_tensor_backend::Tensor& finalized) const;
    mfq_tensor_backend::Tensor adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    bool adapter_supports_speculation() const noexcept;
};

struct Qwen4Model : CausalLmArchitecture {
    mfq::models::qwen4_exp::Config config;
    std::unique_ptr<qwen4_exp::Gr> final_mixer;
    mfq_tensor_backend::Tensor positions;
    int64_t batch = 0;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    void adapter_validate_load_options() const;
    void adapter_load_final_state(
        const mfq::ModelSource& source,
        mfq_tensor_backend::Tensor& output_norm);
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.maximum = value;
    }
    void adapter_reset(int64_t batch);
    bool adapter_requires_batch_reset(int64_t batch) const noexcept;
    void adapter_validate_forward(
        int64_t batch,
        int64_t tokens,
        int64_t cache_position,
        bool has_position_override,
        bool has_cache_position_override,
        bool has_attention_mask) const;
    bool adapter_allows_speculative_position_override() const noexcept;
    CudaPreparedPositions adapter_prepare_positions(
        mfq_tensor_backend::Tensor positions,
        int64_t batch,
        int64_t tokens);
    void adapter_validate_positions(
        const mfq_tensor_backend::Tensor& positions,
        int64_t batch,
        int64_t tokens,
        bool has_mrope) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_block_positions(
        const mfq_tensor_backend::Tensor& full_positions,
        const mfq_tensor_backend::Tensor& local_positions,
        int device) const;
    void adapter_finish_forward(
        const mfq_tensor_backend::Tensor& full_positions,
        int64_t batch,
        int64_t tokens);
    bool adapter_force_cache_advance() const noexcept;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_last_logits(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    mfq_tensor_backend::Tensor adapter_next_token(
        const QuantLinear& lm_head,
        mfq_tensor_backend::Tensor hidden) const;
    bool adapter_supports_speculation() const noexcept;
    void adapter_rollback_speculative(int64_t keep);
};

struct DeepseekV4Model : CausalLmArchitecture {
    mfq::models::deepseek_v4::Config config;
    deepseek_v4::OutputHeadWeights output_head;
    std::unordered_map<int, std::shared_ptr<Dsv4SharedState>> block_states;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    void adapter_validate_load_options() const;
    void adapter_load_final_state(
        const mfq::ModelSource& source,
        mfq_tensor_backend::Tensor& output_norm);
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
};

struct DeepseekV41Model : CausalLmArchitecture {
    mfq::models::deepseek_v41::Config config;
    std::shared_ptr<deepseek_v41_runtime::SharedState> shared;

    void adapter_load_config(
        std::string_view payload,
        const mfq::ModelGraph& graph,
        const mfq::ModelSource& source);
    void adapter_validate_load_options() const;
    void adapter_load_final_state(
        const mfq::ModelSource& source,
        mfq_tensor_backend::Tensor& output_norm);
    void adapter_prepare_blocks(const mfq::ModelSource& source);
    std::unique_ptr<Block> adapter_load_block(
        const mfq::ModelSource& source,
        int layer,
        int device,
        const std::string& type);
    void adapter_set_max_position_embeddings(int64_t value) {
        config.max_position_embeddings = value;
    }
    void adapter_validate_forward(
        int64_t batch,
        int64_t tokens,
        int64_t cache_position,
        bool has_position_override,
        bool has_cache_position_override,
        bool has_attention_mask) const;
    mfq_tensor_backend::Tensor adapter_prepare_hidden(
        mfq_tensor_backend::Tensor hidden,
        int64_t batch,
        int64_t tokens) const;
    void adapter_begin_forward(bool capture_raw_hidden);
    mfq_tensor_backend::Tensor adapter_finalize_hidden(
        mfq_tensor_backend::Tensor hidden,
        const mfq_tensor_backend::Tensor& output_norm,
        int64_t batch,
        int64_t tokens) const;
    mfq_tensor_backend::Tensor adapter_raw_hidden(
        const mfq_tensor_backend::Tensor& hidden,
        const mfq_tensor_backend::Tensor& finalized) const;
    bool adapter_supports_suffix_speculation() const noexcept;
    void adapter_begin_speculative();
    void adapter_commit_speculative();
    void adapter_rollback_speculative(int64_t keep);
};

template <typename Model>
struct CausalLm;

template <typename Model>
struct CudaSessionCodec {
    using CausalModel = CausalLm<Model>;
    static TextSessionStateKind kind(const CausalModel& model);
    static bool supports_paged(const CausalModel& model);
    static TextSessionState capture(
        const CausalModel& model,
        const std::vector<int64_t>& tokens);
    static void restore(
        CausalModel& model,
        const TextSessionState& state);
};

template <typename Model>
struct FullAttentionSessionCodec {
    using CausalModel = CausalLm<Model>;
    static TextSessionStateKind kind(const CausalModel& model);
    static bool supports_paged(const CausalModel& model);
    static TextSessionState capture(
        const CausalModel& model,
        const std::vector<int64_t>& tokens);
    static void restore(
        CausalModel& model,
        const TextSessionState& state);
};

template <>
struct CudaSessionCodec<Qwen35Model> {
    using Model = CausalLm<Qwen35Model>;
    static TextSessionStateKind kind(const Model& model);
    static bool supports_paged(const Model& model);
    static TextSessionState capture(
        const Model& model,
        const std::vector<int64_t>& tokens);
    static void restore(
        Model& model,
        const TextSessionState& state);
};

template <>
struct CudaSessionCodec<MiniCPMO45Model>
    : FullAttentionSessionCodec<MiniCPMO45Model> {};

template <>
struct CudaSessionCodec<MiniCPMOTtsModel>
    : FullAttentionSessionCodec<MiniCPMOTtsModel> {};

template <>
struct CudaSessionCodec<DeepseekV4Model> {
    using Model = CausalLm<DeepseekV4Model>;
    static TextSessionStateKind kind(const Model& model);
    static bool supports_paged(const Model& model);
    static TextSessionState capture(
        const Model& model,
        const std::vector<int64_t>& tokens);
    static void restore(
        Model& model,
        const TextSessionState& state);
};

template <>
struct CudaSessionCodec<GlmDsaModel> {
    using Model = CausalLm<GlmDsaModel>;
    static TextSessionStateKind kind(const Model& model);
    static bool supports_paged(const Model& model);
    static TextSessionState capture(
        const Model& model,
        const std::vector<int64_t>& tokens);
    static void restore(
        Model& model,
        const TextSessionState& state);
};

template <typename Model>
struct CausalLm : Model {
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    CudaExecutionContext* execution = &cuda_execution_context();
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

    int64_t vocab_size() const noexcept;

    int64_t hidden_size() const noexcept;

    int64_t num_hidden_layers() const noexcept;

    int64_t num_attention_heads() const noexcept;

    int64_t num_key_value_heads() const noexcept;

    int64_t head_dim() const noexcept;

    int64_t max_position_embeddings() const noexcept;

    int64_t rotary_dim() const noexcept;

    double rope_base() const noexcept;

    void set_max_position_embeddings(int64_t value) noexcept;

    double rms_norm_eps() const noexcept;

    double norm_weight_offset() const noexcept;

    bool tie_word_embeddings() const noexcept;

    int64_t num_experts() const noexcept;

    int64_t hc_mult() const noexcept;

    double hc_eps() const noexcept;

    double final_logit_softcapping() const noexcept;

    double embedding_scale() const noexcept;

    std::string_view model_type() const noexcept;

    std::string_view layer_type(int64_t layer) const;

    bool supports_speculation() const;
    bool supports_suffix_speculation() const;
    void begin_speculative_suffix(int64_t draft_tokens);
    void commit_speculative();
    void rollback_speculative(int64_t accepted_suffix = 0);

    mfq_tensor_backend::Tensor embed_forward(mfq_tensor_backend::Tensor ids) const;

    void reset(int64_t B);

    TextSessionStateKind text_session_state_kind() const {
        return CudaSessionCodec<Model>::kind(*this);
    }

    bool supports_text_session_state() const {
        return text_session_state_kind() != TextSessionStateKind::Unsupported;
    }

    bool supports_paged_text_session_state() const {
        return CudaSessionCodec<Model>::supports_paged(*this);
    }

    TextSessionState capture_text_session_state(
            const std::vector<int64_t>& tokens) const {
        return CudaSessionCodec<Model>::capture(*this, tokens);
    }

    void restore_text_session_state(const TextSessionState& state) {
        CudaSessionCodec<Model>::restore(*this, state);
    }

    mfq_tensor_backend::Tensor finalize_hidden(
        mfq_tensor_backend::Tensor x, int64_t batch, int64_t tokens);

    mfq_tensor_backend::Tensor hidden_forward(mfq_tensor_backend::Tensor ids,
                                 MfqOptional<mfq_tensor_backend::Tensor> pos_override = mfq_nullopt,
                                 MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt,
                                 std::vector<mfq_tensor_backend::Tensor> * block_trace = nullptr,
                                 MfqOptional<mfq_tensor_backend::Tensor> cache_positions_override = mfq_nullopt,
                                 mfq_tensor_backend::Tensor* raw_hidden = nullptr,
                                 int64_t confirmed_prefix = 0,
                                 int64_t planned_kv_length = 0,
                                 int64_t decode_attention_parts = 0);

    mfq_tensor_backend::Tensor hidden_forward_speculative_suffix(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor* raw_hidden = nullptr);

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
            int64_t confirmed_prefix = 0,
            int64_t planned_kv_length = 0,
            int64_t decode_attention_parts = 0);

    mfq_tensor_backend::Tensor forward_inputs(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor input_embeddings,
            MfqOptional<mfq_tensor_backend::Tensor> pos_override = mfq_nullopt,
            MfqOptional<mfq_tensor_backend::Tensor> seq_len = mfq_nullopt);

    mfq_tensor_backend::Tensor apply_final_logit_softcap(
            mfq_tensor_backend::Tensor logits) const;

    mfq_tensor_backend::Tensor logits_from_hidden(mfq_tensor_backend::Tensor y);

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor ids);

    mfq_tensor_backend::Tensor last_logits_prepared(
            const CudaPreparedPrompt& prepared);

    mfq_tensor_backend::Tensor last_logits(mfq_tensor_backend::Tensor ids);

    mfq_tensor_backend::Tensor hidden_forward_static(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor pos,
            mfq_tensor_backend::Tensor seq_len,
            int64_t planned_kv_length,
            int64_t decode_attention_parts);

    mfq_tensor_backend::Tensor last_logits_static(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor pos,
            mfq_tensor_backend::Tensor seq_len,
            int64_t planned_kv_length = 0,
            int64_t decode_attention_parts = 0);

    mfq_tensor_backend::Tensor next_token(mfq_tensor_backend::Tensor ids);

    mfq_tensor_backend::Tensor next_token_static(
            mfq_tensor_backend::Tensor ids,
            mfq_tensor_backend::Tensor pos,
            mfq_tensor_backend::Tensor seq_len,
            int64_t planned_kv_length = 0,
            int64_t decode_attention_parts = 0);

    mfq_tensor_backend::Tensor next_token_from_hidden(mfq_tensor_backend::Tensor y);
};

using Qwen35CausalLm = CausalLm<Qwen35Model>;
using MiniCPMO45CausalLm = CausalLm<MiniCPMO45Model>;
using MiniCPMOTtsCausalLm = CausalLm<MiniCPMOTtsModel>;
using Gemma4CausalLm = CausalLm<Gemma4Model>;
using GlmDsaCausalLm = CausalLm<GlmDsaModel>;
using Glm5CausalLm = CausalLm<Glm5Model>;
using Qwen4CausalLm = CausalLm<Qwen4Model>;
using DeepseekV4CausalLm = CausalLm<DeepseekV4Model>;
using DeepseekV41CausalLm = CausalLm<DeepseekV41Model>;

template <typename Model>
Model load_causal_lm(
    CudaExecutionContext& execution,
    const std::string& model_path,
    const std::string& config_path,
    std::int64_t context_size_override = 0,
    bool load_blocks = true,
    bool defer_moe_cache_finalize = false,
    std::shared_ptr<const mfq::ModelSource> source = {});

template <typename Model>
Model load_causal_lm(
        const std::string& model_path,
        const std::string& config_path,
        std::int64_t context_size_override = 0,
        bool load_blocks = true,
        bool defer_moe_cache_finalize = false,
        std::shared_ptr<const mfq::ModelSource> source = {}) {
    return load_causal_lm<Model>(
        cuda_execution_context(), model_path, config_path,
        context_size_override, load_blocks,
        defer_moe_cache_finalize, std::move(source));
}

} // namespace mfq::cuda
