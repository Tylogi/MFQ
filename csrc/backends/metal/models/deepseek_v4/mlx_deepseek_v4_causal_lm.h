#pragma once

#include "mfq/token_constraint.h"
#include "deepseek_v4_model.h"
#include "mlx_deepseek_v4_attention.h"
#include "mlx_deepseek_v4_dspark.h"
#include "mlx_deepseek_v4_hc.h"
#include "mlx_deepseek_v4_moe.h"
#include "mlx_deepseek_v4_vision.h"
#include "mlx_mtp.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

using MlxDeepseekV4PrefillCallback = std::function<void(
    std::size_t prompt_tokens,
    double llm_ms,
    double multimodal_ms,
    double model_ms)>;

struct MlxDeepseekV4LayerComponents {
    MlxDeepseekV4Attention attention;
    MlxDeepseekV4Moe moe;
    mlx::core::array attention_norm;
    mlx::core::array ffn_norm;
    MlxLinear hc_attention_fn;
    mlx::core::array hc_attention_base;
    mlx::core::array hc_attention_scale;
    MlxLinear hc_ffn_fn;
    mlx::core::array hc_ffn_base;
    mlx::core::array hc_ffn_scale;
};

// One complete DeepSeek-V4 decoder layer. The input and output both retain
// the four hyper-connection streams:
//   [batch,tokens,4,hidden].
class MlxDeepseekV4Layer {
public:
    static MlxDeepseekV4Layer load(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        std::size_t index,
        int max_context,
        const mlx::core::array& available,
        std::pair<mlx::core::array, mlx::core::array>
            rope_base,
        std::pair<mlx::core::array, mlx::core::array>
            rope_compressed,
        std::shared_ptr<MlxMfeOffloadCache> offload =
            nullptr,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache =
            nullptr);

    MlxDeepseekV4Layer(
        DeepseekV4Config config,
        std::size_t index,
        MlxDeepseekV4LayerComponents components);

    mlx::core::array forward(
        const mlx::core::array& hidden,
        const mlx::core::array& token_ids,
        MlxDeepseekV4LayerState& state,
        int pos0) const;

    // The model-level prefill scheduler owns this handle when it pipelines
    // layer L+1's SSD read with layer L's Metal work. Passing the handle into
    // forward avoids starting a second, same-layer read.
    mlx::core::array forward(
        const mlx::core::array& hidden,
        const mlx::core::array& token_ids,
        MlxDeepseekV4LayerState& state,
        int pos0,
        MlxSsdPrefetchedExpertLayer* prefetched) const;

    mlx::core::array forward(
        const mlx::core::array& hidden,
        const mlx::core::array& token_ids,
        MlxDeepseekV4LayerState& state,
        int pos0,
        MlxSsdPrefetchedExpertLayer* prefetched,
        const MlxDeepseekV4ImageVisibility* visibility) const;

    std::optional<MlxSsdPrefetchedExpertLayer>
    prefetch_routed(std::size_t rows) const {
        return components_.moe.prefetch_routed(rows);
    }

    mlx::core::array operator()(
        const mlx::core::array& hidden,
        const mlx::core::array& token_ids,
        MlxDeepseekV4LayerState& state,
        int pos0) const {
        return forward(hidden, token_ids, state, pos0);
    }

    void commit_speculative(
        MlxDeepseekV4LayerState& state) const noexcept;
    void rollback_speculative(
        MlxDeepseekV4LayerState& state,
        int accepted_tokens) const;

    const DeepseekV4Config& config() const noexcept {
        return config_;
    }
    std::size_t index() const noexcept {
        return index_;
    }
    int ratio() const noexcept {
        return ratio_;
    }
    int max_context() const noexcept {
        return components_.attention.max_context();
    }
    bool uses_streamed_experts() const noexcept {
        return components_.moe.uses_streamed_experts();
    }
    int recommended_prefill_chunk_size() const noexcept {
        return components_.moe.recommended_prefill_chunk_size();
    }

private:
    MlxDeepseekV4HcPreResult hc_pre_norm(
        const mlx::core::array& residual,
        const MlxLinear& function,
        const mlx::core::array& scale,
        const mlx::core::array& base,
        const mlx::core::array& norm,
        const MlxRmsNorm& normalizer) const;

    mlx::core::array hc_post(
        const mlx::core::array& branch,
        const mlx::core::array& residual,
        const mlx::core::array& post,
        const mlx::core::array& combination) const;

    void validate_components() const;

    DeepseekV4Config config_;
    std::size_t index_;
    int ratio_;
    MlxDeepseekV4LayerComponents components_;
    MlxRmsNorm attention_norm_;
    MlxRmsNorm ffn_norm_;
};

using MlxDeepseekV4TokenCallback = MlxGenerationTokenCallback;

struct MlxDeepseekV4TextSessionState {
    std::vector<std::int64_t> tokens;
    std::vector<MlxDeepseekV4LayerState> layers;
    std::optional<MlxDeepseekV4DSparkState> dspark;
    int cache_position = 0;
    int cache_batch = 0;
    std::size_t bytes = 0;
};

// Complete native C++/MLX DeepSeek-V4 text runtime. It owns all layer caches
// and never invokes Python or a subprocess.
class MlxDeepseekV4CausalLm {
public:
    // MFE is fully resident by default, matching the CUDA runtime.  Passing
    // a cache budget explicitly opts into the bounded expert-residency mode.
    static MlxDeepseekV4CausalLm load(
        const MfqContainer& model,
        int max_context = 4096,
        std::optional<std::size_t> expert_cache_bytes =
            std::nullopt);

    static MlxDeepseekV4CausalLm load(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        const DeepseekV4TensorNames& names,
        int max_context,
        std::optional<std::size_t> expert_cache_bytes =
            std::nullopt);

    MlxDeepseekV4CausalLm(
        DeepseekV4Config config,
        MlxEmbedding embedding,
        std::vector<MlxDeepseekV4Layer> layers,
        mlx::core::array output_norm,
        MlxLinear output,
        MlxLinear hc_head_fn,
        mlx::core::array hc_head_base,
        mlx::core::array hc_head_scale,
        int max_context,
        mlx::core::Dtype activation_dtype =
            mlx::core::float16,
        std::shared_ptr<MlxMfeOffloadCache>
            expert_offload = nullptr,
        std::shared_ptr<MlxMoeSsdExpertCache>
            ssd_expert_cache = nullptr,
        std::optional<MlxDeepseekV4Vision> vision =
            std::nullopt,
        std::optional<MlxDeepseekV4DSpark> dspark =
            std::nullopt);

    // Accepts [tokens] or [batch,tokens]. Like the Python reference,
    // use_cache=false starts a fresh cache and still leaves that cache ready
    // for a subsequent decode.
    mlx::core::array forward(
        const mlx::core::array& token_ids,
        bool use_cache = false);

    mlx::core::array operator()(
        const mlx::core::array& token_ids,
        bool use_cache = false) {
        return forward(token_ids, use_cache);
    }

    // Chunked cache-building prefill. full_logits=false returns
    // [batch,vocab] for the final prompt token.
    mlx::core::array prefill(
        const mlx::core::array& token_ids,
        int chunk_size = 512,
        bool full_logits = true);

    // Appends to an existing prefill/forward cache.
    mlx::core::array decode(
        const mlx::core::array& token_ids);

    void reset_cache(int batch = 1);
    void clear_cache() noexcept;

    // Returns the number of emitted tokens. The callback observes an EOS
    // token before generation stops. When eos_token_ids is omitted, the
    // normalized model-config EOS set is used.
    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxDeepseekV4TokenCallback& callback = {},
        const std::optional<std::vector<std::int64_t>>&
            eos_token_ids = std::nullopt,
        int chunk_size = 512,
        const std::function<void(std::size_t, double)>&
            prefill_callback = {},
        std::optional<std::size_t> stable_prefix_tokens =
            std::nullopt,
        const MfqTokenConstraintPtr& token_constraint = {});

    std::int32_t generate_multimodal(
        const std::vector<std::int64_t>& prompt,
        const std::vector<MlxDeepseekV4ImageInput>& images,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxDeepseekV4TokenCallback& callback = {},
        const std::optional<std::vector<std::int64_t>>&
            eos_token_ids = std::nullopt,
        const MlxDeepseekV4PrefillCallback&
            prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {});

    const DeepseekV4Config& config() const noexcept {
        return config_;
    }
    std::size_t layer_count() const noexcept {
        return layers_.size();
    }
    int max_context() const noexcept {
        return max_context_;
    }
    int cache_position() const noexcept {
        return cache_position_;
    }
    int cache_batch() const noexcept {
        return cache_batch_;
    }
    bool cache_ready() const noexcept {
        return cache_batch_ != 0;
    }
    const MlxDeepseekV4LayerState& layer_state(
        std::size_t index) const;
    bool uses_streamed_experts() const noexcept;
    std::size_t expert_cache_limit_bytes() const noexcept;
    std::size_t expert_resident_packed_bytes() const;
    std::size_t cached_expert_count() const;
    std::optional<MlxSsdExpertCacheStats>
    ssd_expert_cache_stats() const;
    void prewarm_ssd_expert_arena();
    void clear_expert_cache();
    MlxDeepseekV4TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const;
    void restore_text_session_state(
        const MlxDeepseekV4TextSessionState& state);
    bool supports_text_session_state() const noexcept {
        return true;
    }
    bool supports_multimodal() const noexcept {
        return vision_.has_value();
    }
    bool supports_mtp() const noexcept {
        return dspark_.has_value();
    }
    int preferred_prefill_chunk_size(
        int portable_default) const noexcept;
    const MlxMtpGenerationStats& last_mtp_stats() const noexcept {
        return last_mtp_stats_;
    }

private:
    enum class ExpertBackend {
        resident,
        streaming,
    };

    void validate_components() const;
    mlx::core::array normalize_ids(
        const mlx::core::array& token_ids,
        bool allow_empty) const;
    mlx::core::array forward_chunk(
        const mlx::core::array& token_ids,
        int pos0,
        bool full_logits,
        const std::optional<mlx::core::array>& input_embeddings =
            std::nullopt,
        const MlxDeepseekV4ImageVisibility* visibility = nullptr,
        mlx::core::array* dspark_hidden = nullptr,
        bool skip_lm_head = false);
    mlx::core::array prefill_impl(
        const mlx::core::array& token_ids,
        int chunk_size,
        bool full_logits,
        bool reset);
    mlx::core::array head(
        const mlx::core::array& hidden) const;
    void materialize_states(
        const std::vector<MlxDeepseekV4LayerState>& states) const;
    void append_state_arrays(
        const MlxDeepseekV4LayerState& state,
        std::vector<mlx::core::array>& arrays) const;
    void materialize_layer_range(
        const mlx::core::array& hidden,
        std::size_t begin,
        std::size_t end) const;
    void capture_dspark_target(
        std::size_t layer,
        const mlx::core::array& hidden,
        std::vector<std::optional<mlx::core::array>>* targets) const;
    mlx::core::array forward_resident_layers(
        mlx::core::array hidden,
        const mlx::core::array& token_ids,
        int pos0,
        const MlxDeepseekV4ImageVisibility* visibility,
        std::vector<std::optional<mlx::core::array>>* targets,
        bool eager_layer_materialization);
    mlx::core::array forward_streaming_layers(
        mlx::core::array hidden,
        const mlx::core::array& token_ids,
        int pos0,
        const MlxDeepseekV4ImageVisibility* visibility,
        std::vector<std::optional<mlx::core::array>>* targets,
        bool eager_layer_materialization);
    void begin_speculative_target(
        int confirmed_tokens,
        int total_tokens);
    void commit_speculative_target() noexcept;
    void rollback_speculative_target(
        int accepted_tokens,
        int draft_tokens);
    std::int32_t generate_impl(
        const std::vector<std::int64_t>& prompt,
        const std::vector<MlxDeepseekV4ImageInput>* images,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxDeepseekV4TokenCallback& callback,
        const std::optional<std::vector<std::int64_t>>& eos_token_ids,
        int chunk_size,
        const MlxDeepseekV4PrefillCallback& prefill_callback,
        std::optional<std::size_t> stable_prefix_tokens,
        const MfqTokenConstraintPtr& token_constraint);

    DeepseekV4Config config_;
    MlxEmbedding embedding_;
    std::vector<MlxDeepseekV4Layer> layers_;
    MlxRmsNorm output_norm_;
    MlxLinear output_;
    MlxLinear hc_head_fn_;
    mlx::core::array hc_head_base_;
    mlx::core::array hc_head_scale_;
    std::shared_ptr<MlxMfeOffloadCache>
        expert_offload_;
    std::shared_ptr<MlxMoeSsdExpertCache>
        ssd_expert_cache_;
    ExpertBackend expert_backend_;
    std::optional<MlxDeepseekV4Vision> vision_;
    std::optional<MlxDeepseekV4DSpark> dspark_;
    MlxMtpGenerationStats last_mtp_stats_;
    int max_context_;
    mlx::core::Dtype activation_dtype_;
    std::vector<MlxDeepseekV4LayerState> states_;
    int cache_position_ = 0;
    int cache_batch_ = 0;
    // Exact token sequence represented by states_ when a server request leaves
    // a stable prompt checkpoint.  It is intentionally empty for cache state
    // produced through the public forward/prefill/decode APIs.
    std::vector<std::int64_t> stable_cache_tokens_;
    // Predictor cache matching stable_cache_tokens_. It is adapter state,
    // while proposal sampling and verification remain backend-wide MTP logic.
    std::optional<MlxDeepseekV4DSparkState> stable_dspark_state_;
};

} // namespace mfq::metal
