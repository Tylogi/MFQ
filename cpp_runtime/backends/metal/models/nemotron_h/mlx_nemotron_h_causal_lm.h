#pragma once

#include "nemotron_h_model.h"
#include "mlx_mtp.h"
#include "mlx_nemotron_h_attention.h"
#include "mlx_nemotron_h_linear_attention.h"
#include "mlx_nemotron_h_moe.h"
#include "mlx_nemotron_h_vision.h"
#include "mlx_sampling.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

class MlxNemotronHCausalLm;

using MlxNemotronHLayer = std::variant<
    MlxNemotronHAttentionBlock,
    MlxNemotronHLinearAttentionBlock,
    MlxNemotronHMoeBlock>;

struct MlxNemotronHLayerSnapshot {
    MlxNemotronHRecurrentSnapshot recurrent;
    MlxKvCacheSnapshot attention;
};

struct MlxNemotronHTextSessionState {
    std::vector<std::int64_t> tokens;
    std::vector<MlxNemotronHLayerSnapshot> layers;
    int cache_position = 0;
    int cache_batch = 0;
    std::size_t bytes = 0;
    std::optional<MlxKvCacheSnapshot> mtp_attention;
};

// Nemotron-H multi-token-prediction head: embedding/hidden norms around one
// fusion projection, one attention sub-block and one MoE sub-block, then an
// RMSNorm whose output is both the lm_head input and the next draft step's
// hidden state.
class MlxNemotronHMtpModule {
public:
    static std::optional<MlxNemotronHMtpModule> load_if_present(
        const MfqContainer& model,
        const NemotronHConfig& config);

    MlxNemotronHMtpModule(
        NemotronHConfig config,
        MlxRmsNorm embedding_norm,
        MlxRmsNorm hidden_norm,
        MlxLinear fusion,
        std::vector<MlxNemotronHLayer> blocks,
        MlxRmsNorm output_norm);

    // `hidden` is the target's post-output_norm hidden state; `ids` are the
    // next tokens that follow those rows. The result is the head's finalized
    // output before the lm_head projection.
    mlx::core::array forward(
        const mlx::core::array& hidden,
        const mlx::core::array& ids,
        const MlxEmbedding& embedding,
        bool use_cache = true);

    void reset_cache(int batch = 1, int initial_capacity = 16);
    void trim_cache_to(int position);
    void clear_cache() noexcept;
    int cache_position() const noexcept;
    MlxKvCacheSnapshot snapshot_cache() const;
    void restore_cache(const MlxKvCacheSnapshot& snapshot);
    std::size_t layer_count() const noexcept {
        return blocks_.size();
    }
    const NemotronHConfig& config() const noexcept {
        return config_;
    }

private:
    void validate_components() const;
    NemotronHConfig config_;
    MlxRmsNorm embedding_norm_;
    MlxRmsNorm hidden_norm_;
    MlxLinear fusion_;
    std::vector<MlxNemotronHLayer> blocks_;
    MlxRmsNorm output_norm_;
};

class MlxNemotronHCausalLm {
public:
    static MlxNemotronHCausalLm load(
        const MfqContainer& model,
        int requested_context = 0);

    MlxNemotronHCausalLm(
        NemotronHConfig config,
        MlxEmbedding embedding,
        std::vector<MlxNemotronHLayer> layers,
        MlxRmsNorm output_norm,
        std::optional<MlxLinear> output,
        std::optional<MlxNemotronHMtpModule> mtp = std::nullopt,
        std::optional<MlxNemotronHRadioPromptComponent> vision = std::nullopt);

    // token_ids is [batch,tokens]. Cached calls append at the current cache
    // position; uncached calls leave the cache untouched.
    mlx::core::array forward(
        const mlx::core::array& token_ids,
        bool use_cache = true);
    mlx::core::array forward_embeddings(
        const mlx::core::array& embeddings,
        bool use_cache = true);
    std::pair<mlx::core::array, mlx::core::array>
    forward_embeddings_with_hidden(
        const mlx::core::array& embeddings,
        bool use_cache = true);
    // Verify a confirmed prefix plus a speculative suffix in one window. The
    // second and later calls must be resolved before the cache is reused.
    std::pair<mlx::core::array, mlx::core::array> forward_verified(
        const mlx::core::array& token_ids,
        int confirmed_tokens);
    void commit_speculative();
    void rollback_speculative(int accepted_drafts, int draft_count);

    mlx::core::array embed_tokens(const mlx::core::array& token_ids) const;
    mlx::core::array project_logits(const mlx::core::array& hidden) const;
    mlx::core::array project_normalized(const mlx::core::array& hidden) const;

    void reset_cache(int batch = 1);
    void clear_cache() noexcept;
    int cache_position() const noexcept {
        return cache_position_;
    }
    int cache_batch() const noexcept {
        return cache_batch_;
    }
    std::size_t layer_count() const noexcept {
        return layers_.size();
    }
    const NemotronHConfig& config() const noexcept {
        return config_;
    }
    bool supports_mtp() const noexcept {
        return mtp_.has_value();
    }
    bool supports_multimodal() const noexcept {
        return vision_.has_value();
    }
    MlxPreparedPrompt prepare_multimodal_prompt(
        const std::vector<std::int64_t>& token_ids,
        const MlxGridMediaInput& media) const;
    std::string_view multimodal_input_contract() const noexcept {
        return vision_ ? vision_->input_contract() : std::string_view{};
    }
    const MlxEmbedding& embedding() const noexcept {
        return embedding_;
    }
    const std::optional<MlxNemotronHMtpModule>& mtp() const noexcept {
        return mtp_;
    }
    std::optional<MlxNemotronHMtpModule>& mtp() noexcept {
        return mtp_;
    }
    const std::optional<MlxNemotronHRadioPromptComponent>& vision() const noexcept {
        return vision_;
    }
    bool supports_text_session_state() const noexcept {
        return true;
    }
    const MlxMtpGenerationStats& last_mtp_stats() const noexcept {
        return last_mtp_stats_;
    }

    MlxNemotronHTextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const;
    void restore_text_session_state(const MlxNemotronHTextSessionState& state);

    // Greedy/stochastic generation over the shared Metal sampling and MTP
    // engine. The predictor stays loaded; only sampling.enable_mtp selects
    // whether speculative cycles run.
    std::int32_t generate(
        const std::vector<std::int64_t>& prompt,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxGenerationTokenCallback& callback = {},
        int prefill_chunk_size = 2048,
        const MfqTokenConstraintPtr& token_constraint = {});
    // Prefill a processed or multimodally injected prompt (same backbone,
    // cache and MTP engine as token generation).
    std::int32_t generate_prepared(
        const MlxPreparedPrompt& prepared,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxGenerationTokenCallback& callback = {},
        const std::function<void(std::size_t, double)>& prefill_callback = {},
        const MfqTokenConstraintPtr& token_constraint = {});

private:
    void validate_components() const;
    mlx::core::array embed_ids(const mlx::core::array& token_ids) const;
    std::pair<mlx::core::array, mlx::core::array> forward_embeddings_impl(
        const mlx::core::array& embeddings,
        bool use_cache,
        int speculative_confirmed,
        bool full_logits);
    std::int32_t generate_prepared_impl(
        const MlxPreparedPrompt& prepared,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxGenerationTokenCallback& callback,
        const std::function<void(std::size_t, double)>& prefill_callback,
        int prefill_chunk_size,
        const MfqTokenConstraintPtr& token_constraint);
    std::int32_t generate_mtp(
        const std::vector<std::int64_t>& prompt,
        const mlx::core::array& prompt_ids,
        const mlx::core::array& prefill_hidden,
        const mlx::core::array& initial_logits,
        const MlxSamplingParams& sampling,
        std::int32_t max_tokens,
        const MlxGenerationTokenCallback& callback,
        const MfqTokenConstraintPtr& token_constraint);

    NemotronHConfig config_;
    MlxEmbedding embedding_;
    std::vector<MlxNemotronHLayer> layers_;
    MlxRmsNorm output_norm_;
    std::optional<MlxLinear> output_;
    std::optional<MlxNemotronHMtpModule> mtp_;
    std::optional<MlxNemotronHRadioPromptComponent> vision_;
    MlxMtpGenerationStats last_mtp_stats_;
    mfq::engine::mtp::PolicyState mtp_policy_state_;
    int cache_position_ = 0;
    int cache_batch_ = 0;
    bool speculative_pending_ = false;
};

} // namespace mfq::metal
