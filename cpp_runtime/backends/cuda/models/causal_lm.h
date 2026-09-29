#pragma once

#include "models/transformer.h"
#include "cuda_model_plan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

enum class TextSessionStateKind : int;
struct TextSessionState;


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
    CudaExecutionContext* execution = nullptr;

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

struct Qwen35Model;
struct MiniCPMO45Model;
struct MiniCPMOTtsModel;
struct Gemma4Model;
struct GlmDsaModel;
struct Glm5Model;
struct Qwen4Model;
struct DeepseekV4Model;
struct DeepseekV41Model;

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
struct CausalLm : Model {
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

    TextSessionStateKind text_session_state_kind() const;

    bool supports_text_session_state() const;

    bool supports_paged_text_session_state() const;

    TextSessionState capture_text_session_state(
        const std::vector<int64_t>& tokens) const;

    void restore_text_session_state(const TextSessionState& state);

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

} // namespace mfq::cuda
