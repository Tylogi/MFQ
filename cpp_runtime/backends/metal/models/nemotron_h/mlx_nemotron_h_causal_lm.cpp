#include "mlx_nemotron_h_causal_lm.h"

#include "mlx_nemotron_h_ops.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <variant>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("invalid Nemotron-H ") + name);
    }
    return static_cast<int>(value);
}

array add_branch(const array& residual, const array& branch) {
    const auto value = branch.dtype() == residual.dtype()
        ? branch
        : mlx::core::astype(branch, residual.dtype());
    return residual + value;
}

array take_rows(const array& value, int begin, int end) {
    return mlx::core::slice(
        value, Shape{0, begin, 0},
        Shape{value.shape(0), end, value.shape(2)});
}

array last_row(const array& value) {
    return mlx::core::slice(
        value, Shape{0, value.shape(1) - 1, 0},
        Shape{value.shape(0), value.shape(1), value.shape(2)});
}

bool has_prefix(const MfqContainer& model, const std::string& prefix) {
    for (const auto& entry : model.records()) {
        if (entry.first.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

} // namespace

std::optional<MlxNemotronHMtpModule> MlxNemotronHMtpModule::load_if_present(
    const MfqContainer& model,
    const NemotronHConfig& config) {
    if (!model.contains("predictor.fusion.weight")) {
        if (has_prefix(model, "predictor.")) {
            throw std::runtime_error(
                "Nemotron-H MFQ contains an incomplete MTP head");
        }
        return std::nullopt;
    }
    if (config.num_nextn_predict_layers != 1 || config.mtp_mixers.empty()) {
        throw std::runtime_error(
            "Nemotron-H MTP requires one declared predictor stage");
    }
    std::vector<MlxNemotronHLayer> blocks;
    blocks.reserve(config.mtp_mixers.size());
    for (std::size_t index = 0; index < config.mtp_mixers.size(); ++index) {
        const auto kind = config.mtp_mixers[index];
        if (kind == NemotronHMixer::full_attention) {
            blocks.emplace_back(MlxNemotronHAttentionBlock::load(
                model, config, "predictor", index));
        } else if (kind == NemotronHMixer::moe || kind == NemotronHMixer::mlp) {
            blocks.emplace_back(MlxNemotronHMoeBlock::load(
                model, config, kind, "predictor", index));
        } else {
            throw std::runtime_error(
                "Nemotron-H MTP cannot declare a mamba sub-block");
        }
    }
    return MlxNemotronHMtpModule(
        config,
        load_nemotron_h_rms_norm(
            model, "predictor.embedding_norm.weight", config.rms_norm_eps),
        load_nemotron_h_rms_norm(
            model, "predictor.hidden_norm.weight", config.rms_norm_eps),
        MlxLinear::load(model, "predictor.fusion.weight"),
        std::move(blocks),
        load_nemotron_h_rms_norm(
            model, "predictor.output_norm.weight", config.rms_norm_eps));
}

MlxNemotronHMtpModule::MlxNemotronHMtpModule(
    NemotronHConfig config,
    MlxRmsNorm embedding_norm,
    MlxRmsNorm hidden_norm,
    MlxLinear fusion,
    std::vector<MlxNemotronHLayer> blocks,
    MlxRmsNorm output_norm)
    : config_(std::move(config)),
      embedding_norm_(std::move(embedding_norm)),
      hidden_norm_(std::move(hidden_norm)),
      fusion_(std::move(fusion)),
      blocks_(std::move(blocks)),
      output_norm_(std::move(output_norm)) {
    validate_components();
}

void MlxNemotronHMtpModule::validate_components() const {
    const int hidden = checked_int(config_.hidden_size, "hidden_size");
    if (blocks_.size() != config_.mtp_mixers.size() ||
        fusion_.input_size() != 2 * hidden ||
        fusion_.output_size() != hidden ||
        embedding_norm_.width() != hidden ||
        hidden_norm_.width() != hidden ||
        output_norm_.width() != hidden) {
        throw std::runtime_error(
            "Nemotron-H MTP component dimensions are incompatible");
    }
}

array MlxNemotronHMtpModule::forward(
    const array& hidden,
    const array& ids,
    const MlxEmbedding& embedding,
    bool use_cache) {
    if (hidden.ndim() != 3 || ids.ndim() != 2 ||
        hidden.shape(0) != ids.shape(0) || hidden.shape(1) != ids.shape(1) ||
        hidden.shape(1) <= 0 || hidden.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H MTP inputs must be matching [batch,tokens] hidden "
            "states and next-token IDs");
    }
    const auto embedded = mlx::core::astype(
        embedding(ids, mlx::core::float32), hidden.dtype());
    auto value = fusion_(mlx::core::concatenate(
        {embedding_norm_(embedded), hidden_norm_(hidden)}, -1));
    for (auto& block : blocks_) {
        auto branch = std::visit(
            [&](auto& typed) -> array {
                using Block = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<Block, MlxNemotronHMoeBlock>) {
                    return typed.forward(value);
                } else {
                    return typed.forward(value, use_cache);
                }
            },
            block);
        value = add_branch(value, branch);
    }
    return output_norm_(value);
}

void MlxNemotronHMtpModule::reset_cache(int batch, int initial_capacity) {
    for (auto& block : blocks_) {
        if (auto* attention = std::get_if<MlxNemotronHAttentionBlock>(&block)) {
            attention->reset_cache(batch, initial_capacity);
        }
    }
}

void MlxNemotronHMtpModule::trim_cache_to(int position) {
    for (auto& block : blocks_) {
        if (auto* attention = std::get_if<MlxNemotronHAttentionBlock>(&block)) {
            const int current = attention->cache_position();
            if (position < 0 || position > current) {
                throw std::runtime_error(
                    "Nemotron-H MTP cache trim position is invalid");
            }
            if (position < current) attention->trim_cache(current - position);
        }
    }
}

void MlxNemotronHMtpModule::clear_cache() noexcept {
    for (auto& block : blocks_) {
        if (auto* attention = std::get_if<MlxNemotronHAttentionBlock>(&block)) {
            attention->clear_cache();
        }
    }
}

int MlxNemotronHMtpModule::cache_position() const noexcept {
    for (const auto& block : blocks_) {
        if (const auto* attention =
                std::get_if<MlxNemotronHAttentionBlock>(&block)) {
            return attention->cache_position();
        }
    }
    return 0;
}

MlxKvCacheSnapshot MlxNemotronHMtpModule::snapshot_cache() const {
    for (const auto& block : blocks_) {
        if (const auto* attention =
                std::get_if<MlxNemotronHAttentionBlock>(&block)) {
            return attention->snapshot_cache();
        }
    }
    throw std::runtime_error(
        "Nemotron-H MTP has no attention history to snapshot");
}

void MlxNemotronHMtpModule::restore_cache(const MlxKvCacheSnapshot& snapshot) {
    for (auto& block : blocks_) {
        if (auto* attention = std::get_if<MlxNemotronHAttentionBlock>(&block)) {
            attention->restore_cache(snapshot);
            return;
        }
    }
    throw std::runtime_error(
        "Nemotron-H MTP has no attention history to restore");
}

MlxNemotronHCausalLm MlxNemotronHCausalLm::load(
    const MfqContainer& model,
    int requested_context) {
    auto config = load_nemotron_h_config(model);
    // Validate the declared components before loading any weight, so a graph
    // this backend cannot feed fails as a whole instead of loading partially.
    const auto components = resolve_nemotron_h_components(model);
    if (requested_context > 0) {
        config.max_position_embeddings =
            std::min<std::int64_t>(requested_context, config.max_position_embeddings);
    }
    std::vector<MlxNemotronHLayer> layers;
    layers.reserve(config.mixers.size());
    for (std::size_t index = 0; index < config.mixers.size(); ++index) {
        switch (config.mixers[index]) {
            case NemotronHMixer::linear_attention:
                layers.emplace_back(MlxNemotronHLinearAttentionBlock::load(
                    model, config, "model", index));
                break;
            case NemotronHMixer::full_attention:
                layers.emplace_back(MlxNemotronHAttentionBlock::load(
                    model, config, "model", index));
                break;
            case NemotronHMixer::moe:
            case NemotronHMixer::mlp:
                layers.emplace_back(MlxNemotronHMoeBlock::load(
                    model, config, config.mixers[index], "model", index));
                break;
        }
    }
    std::optional<MlxLinear> output;
    if (model.contains("model.output.weight")) {
        output.emplace(MlxLinear::load(model, "model.output.weight"));
    }
    std::optional<MlxNemotronHMtpModule> mtp;
    if (components.has_predictor) {
        mtp = MlxNemotronHMtpModule::load_if_present(model, config);
    }
    std::optional<MlxNemotronHRadioPromptComponent> vision;
    if (components.has_vision) {
        vision.emplace(MlxNemotronHRadioPromptComponent::load(
            model,
            components.vision,
            components.vision_input_contract,
            components.vision_position_policy));
    }
    return MlxNemotronHCausalLm(
        config,
        MlxEmbedding::load(model, "model.token_embedding.weight"),
        std::move(layers),
        load_nemotron_h_rms_norm(
            model, "model.output_norm.weight", config.rms_norm_eps),
        std::move(output),
        std::move(mtp),
        std::move(vision));
}

MlxNemotronHCausalLm::MlxNemotronHCausalLm(
    NemotronHConfig config,
    MlxEmbedding embedding,
    std::vector<MlxNemotronHLayer> layers,
    MlxRmsNorm output_norm,
    std::optional<MlxLinear> output,
    std::optional<MlxNemotronHMtpModule> mtp,
    std::optional<MlxNemotronHRadioPromptComponent> vision)
    : config_(std::move(config)),
      embedding_(std::move(embedding)),
      layers_(std::move(layers)),
      output_norm_(std::move(output_norm)),
      output_(std::move(output)),
      mtp_(std::move(mtp)),
      vision_(std::move(vision)) {
    validate_components();
}

void MlxNemotronHCausalLm::validate_components() const {
    if (layers_.size() != config_.mixers.size() ||
        embedding_.hidden_size() != config_.hidden_size ||
        embedding_.vocabulary_size() != config_.vocab_size ||
        output_norm_.width() != config_.hidden_size ||
        (output_.has_value() &&
         (output_->input_size() != config_.hidden_size ||
          output_->output_size() != config_.vocab_size))) {
        throw std::runtime_error(
            "Nemotron-H causal LM component dimensions are inconsistent");
    }
}

array MlxNemotronHCausalLm::embed_ids(const array& token_ids) const {
    const auto ids = token_ids.dtype() == mlx::core::int32
        ? token_ids
        : mlx::core::astype(token_ids, mlx::core::int32);
    return embedding_(ids, mlx::core::float32);
}

array MlxNemotronHCausalLm::embed_tokens(const array& token_ids) const {
    return embed_ids(token_ids);
}

array MlxNemotronHCausalLm::project_normalized(const array& hidden) const {
    return output_.has_value()
        ? (*output_)(hidden)
        : embedding_.project(hidden);
}

array MlxNemotronHCausalLm::project_logits(const array& hidden) const {
    return project_normalized(hidden);
}

array MlxNemotronHCausalLm::forward(const array& token_ids, bool use_cache) {
    if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0) {
        throw std::runtime_error(
            "Nemotron-H token ids must have a non-empty [batch,tokens] shape");
    }
    return forward_embeddings(embed_ids(token_ids), use_cache);
}

array MlxNemotronHCausalLm::forward_embeddings(
    const array& embeddings,
    bool use_cache) {
    return forward_embeddings_impl(embeddings, use_cache, 0, true).first;
}

std::pair<array, array> MlxNemotronHCausalLm::forward_embeddings_with_hidden(
    const array& embeddings,
    bool use_cache) {
    return forward_embeddings_impl(embeddings, use_cache, 0, true);
}

std::pair<array, array> MlxNemotronHCausalLm::forward_verified(
    const array& token_ids,
    int confirmed_tokens) {
    if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
        token_ids.shape(1) <= 0) {
        throw std::runtime_error(
            "Nemotron-H verified token ids must have a non-empty [batch,tokens] shape");
    }
    const int tokens = token_ids.shape(1);
    if (confirmed_tokens <= 0 || confirmed_tokens >= tokens) {
        throw std::runtime_error(
            "Nemotron-H verified window needs a confirmed prefix and a draft suffix");
    }
    auto result = forward_embeddings_impl(
        embed_ids(token_ids), true, confirmed_tokens, true);
    speculative_pending_ = true;
    return result;
}

void MlxNemotronHCausalLm::commit_speculative() {
    if (!speculative_pending_) {
        throw std::runtime_error(
            "Nemotron-H has no speculative window to commit");
    }
    for (auto& layer : layers_) {
        if (auto* linear =
                std::get_if<MlxNemotronHLinearAttentionBlock>(&layer)) {
            linear->commit_speculative();
        }
    }
    speculative_pending_ = false;
}

void MlxNemotronHCausalLm::rollback_speculative(
    int accepted_drafts,
    int draft_count) {
    if (!speculative_pending_ || accepted_drafts < 0 ||
        draft_count <= 0 || accepted_drafts >= draft_count ||
        draft_count + 1 > cache_position_) {
        throw std::runtime_error(
            "Nemotron-H speculative rollback range is invalid");
    }
    const int rejected = draft_count - accepted_drafts;
    for (auto& layer : layers_) {
        std::visit(
            [accepted_drafts, rejected](auto& typed) {
                using Block = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<
                                  Block,
                                  MlxNemotronHLinearAttentionBlock>) {
                    typed.rollback_speculative(accepted_drafts);
                } else if constexpr (std::is_same_v<
                                         Block,
                                         MlxNemotronHAttentionBlock>) {
                    typed.trim_cache(rejected);
                }
            },
            layer);
    }
    cache_position_ -= rejected;
    speculative_pending_ = false;
}

std::pair<array, array> MlxNemotronHCausalLm::forward_embeddings_impl(
    const array& embeddings,
    bool use_cache,
    int speculative_confirmed,
    bool full_logits) {
    if (embeddings.ndim() != 3 || embeddings.shape(0) <= 0 ||
        embeddings.shape(1) <= 0 || embeddings.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H embeddings must have a non-empty [batch,tokens,hidden] shape");
    }
    const int batch = embeddings.shape(0);
    const int tokens = embeddings.shape(1);
    if (use_cache) {
        if (cache_batch_ != batch) {
            reset_cache(batch);
        }
        if (cache_position_ + tokens > config_.max_position_embeddings) {
            throw std::runtime_error(
                "Nemotron-H cache would exceed the configured context");
        }
    }
    auto hidden = mlx::core::contiguous(embeddings);
    for (auto& layer : layers_) {
        const auto branch = std::visit(
            [&](auto& typed) -> array {
                using Block = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<
                                  Block,
                                  MlxNemotronHLinearAttentionBlock>) {
                    return speculative_confirmed > 0
                        ? typed.forward_speculative(
                              hidden, cache_position_, speculative_confirmed)
                        : typed.forward(hidden, use_cache);
                } else if constexpr (std::is_same_v<
                                         Block,
                                         MlxNemotronHAttentionBlock>) {
                    return typed.forward(hidden, use_cache);
                } else {
                    return typed.forward(hidden);
                }
            },
            layer);
        hidden = add_branch(hidden, branch);
    }
    auto finalized = output_norm_(hidden);
    auto logits = full_logits
        ? project_logits(finalized)
        : project_logits(last_row(finalized));
    if (use_cache) {
        cache_position_ += tokens;
    }
    return {std::move(logits), std::move(finalized)};
}

void MlxNemotronHCausalLm::reset_cache(int batch) {
    if (batch <= 0) {
        throw std::runtime_error(
            "Nemotron-H cache batch must be positive");
    }
    for (auto& layer : layers_) {
        std::visit(
            [batch](auto& typed) {
                using Block = std::decay_t<decltype(typed)>;
                if constexpr (!std::is_same_v<Block, MlxNemotronHMoeBlock>) {
                    typed.reset_cache(batch);
                }
            },
            layer);
    }
    cache_position_ = 0;
    cache_batch_ = batch;
    speculative_pending_ = false;
}

void MlxNemotronHCausalLm::clear_cache() noexcept {
    for (auto& layer : layers_) {
        std::visit(
            [](auto& typed) {
                using Block = std::decay_t<decltype(typed)>;
                if constexpr (!std::is_same_v<Block, MlxNemotronHMoeBlock>) {
                    typed.clear_cache();
                }
            },
            layer);
    }
    if (mtp_) mtp_->clear_cache();
    cache_position_ = 0;
    cache_batch_ = 0;
    speculative_pending_ = false;
}

MlxPreparedPrompt MlxNemotronHCausalLm::prepare_multimodal_prompt(
    const std::vector<std::int64_t>& token_ids,
    const MlxGridMediaInput& media) const {
    if (!vision_) {
        throw std::runtime_error(
            "Nemotron-H model has no vision component");
    }
    const auto tokens = static_cast<std::int64_t>(token_ids.size());
    std::vector<std::int32_t> values;
    values.reserve(token_ids.size());
    for (const auto token : token_ids) {
        values.push_back(static_cast<std::int32_t>(token));
    }
    const array ids(
        values.begin(), Shape{1, static_cast<int>(tokens)}, mlx::core::int32);
    return vision_->prepare(token_ids, embed_tokens(ids), media);
}

MlxNemotronHTextSessionState MlxNemotronHCausalLm::capture_text_session_state(
    const std::vector<std::int64_t>& tokens) const {
    if (cache_position_ <= 0 || speculative_pending_ ||
        tokens.size() != static_cast<std::size_t>(cache_position_)) {
        throw std::runtime_error(
            "Nemotron-H session token count does not match the cache");
    }
    MlxNemotronHTextSessionState state;
    state.tokens = tokens;
    state.cache_position = cache_position_;
    state.cache_batch = cache_batch_;
    state.layers.reserve(layers_.size());
    for (const auto& layer : layers_) {
        MlxNemotronHLayerSnapshot saved;
        std::visit(
            [&](const auto& typed) {
                using Block = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<
                                  Block,
                                  MlxNemotronHLinearAttentionBlock>) {
                    saved.recurrent = typed.snapshot_cache();
                    state.bytes += saved.recurrent.nbytes();
                } else if constexpr (std::is_same_v<
                                         Block,
                                         MlxNemotronHAttentionBlock>) {
                    saved.attention = typed.snapshot_cache();
                    state.bytes += saved.attention.nbytes();
                }
            },
            layer);
        state.layers.push_back(std::move(saved));
    }
    if (mtp_ && mtp_->cache_position() > 0) {
        state.mtp_attention = mtp_->snapshot_cache();
        state.bytes += state.mtp_attention->nbytes();
    }
    return state;
}

void MlxNemotronHCausalLm::restore_text_session_state(
    const MlxNemotronHTextSessionState& state) {
    if (state.cache_position <= 0 ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_position) ||
        state.layers.size() != layers_.size()) {
        throw std::runtime_error(
            "Nemotron-H session state is incompatible");
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        auto& layer = layers_[index];
        const auto& saved = state.layers[index];
        std::visit(
            [&](auto& typed) {
                using Block = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<
                                  Block,
                                  MlxNemotronHLinearAttentionBlock>) {
                    typed.restore_cache(saved.recurrent);
                } else if constexpr (std::is_same_v<
                                         Block,
                                         MlxNemotronHAttentionBlock>) {
                    typed.restore_cache(saved.attention);
                }
            },
            layer);
    }
    if (mtp_) {
        if (state.mtp_attention) {
            mtp_->restore_cache(*state.mtp_attention);
        } else {
            mtp_->clear_cache();
        }
    }
    cache_position_ = state.cache_position;
    cache_batch_ = state.cache_batch;
    speculative_pending_ = false;
}

std::int32_t MlxNemotronHCausalLm::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MlxGenerationTokenCallback& callback,
    int prefill_chunk_size,
    const MfqTokenConstraintPtr& token_constraint) {
    MlxPreparedPrompt prepared;
    prepared.token_ids = prompt;
    return generate_prepared_impl(
        prepared,
        sampling,
        max_tokens,
        callback,
        {},
        prefill_chunk_size,
        token_constraint);
}

std::int32_t MlxNemotronHCausalLm::generate_prepared(
    const MlxPreparedPrompt& prepared,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MlxGenerationTokenCallback& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint) {
    return generate_prepared_impl(
        prepared,
        sampling,
        max_tokens,
        callback,
        prefill_callback,
        2048,
        token_constraint);
}

std::int32_t MlxNemotronHCausalLm::generate_prepared_impl(
    const MlxPreparedPrompt& prepared,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MlxGenerationTokenCallback& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    int prefill_chunk_size,
    const MfqTokenConstraintPtr& token_constraint) {
    const auto& prompt = prepared.token_ids;
    if (prompt.empty()) {
        throw std::invalid_argument(
            "Nemotron-H generation prompt cannot be empty");
    }
    if (max_tokens < 0) {
        throw std::invalid_argument(
            "Nemotron-H generation max_tokens cannot be negative");
    }
    if (prefill_chunk_size <= 0) {
        throw std::invalid_argument(
            "Nemotron-H prefill chunk size must be positive");
    }
    mlx_validate_token_set(prompt, static_cast<int>(config_.vocab_size));
    const int prompt_count = static_cast<int>(prompt.size());
    if (prompt_count > config_.max_position_embeddings) {
        throw std::invalid_argument(
            "Nemotron-H generation prompt exceeds context capacity");
    }
    if (prepared.embeddings &&
        (prepared.embeddings->ndim() != 3 ||
         prepared.embeddings->shape(0) != 1 ||
         prepared.embeddings->shape(1) != prompt_count ||
         prepared.embeddings->shape(2) != config_.hidden_size)) {
        throw std::invalid_argument(
            "Nemotron-H prepared prompt embeddings disagree with the tokens");
    }
    const int vocab = static_cast<int>(config_.vocab_size);
    std::vector<std::int32_t> values;
    values.reserve(prompt.size());
    for (const auto token : prompt) values.push_back(static_cast<std::int32_t>(token));
    const array ids(values.begin(), Shape{1, prompt_count}, mlx::core::int32);

    reset_cache(1);
    const auto prefill_started = std::chrono::steady_clock::now();
    std::optional<array> prefill_hidden;
    array prompt_logits(0.0f);
    for (int begin = 0; begin < prompt_count; begin += prefill_chunk_size) {
        const int end = std::min(prompt_count, begin + prefill_chunk_size);
        array input = mlx::core::slice(ids, Shape{0, begin}, Shape{1, end});
        if (prepared.embeddings) {
            input = mlx::core::slice(
                *prepared.embeddings,
                Shape{0, begin, 0},
                Shape{1, end, static_cast<int>(config_.hidden_size)});
        } else {
            input = embed_ids(input);
        }
        auto result = forward_embeddings_impl(input, true, 0, true);
        prompt_logits = std::move(result.first);
        prefill_hidden = std::move(result.second);
    }
    if (prefill_callback) {
        const auto milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prefill_started).count();
        prefill_callback(prompt.size(), milliseconds);
    }
    const auto initial_logits = mlx_last_token_logits(prompt_logits, vocab);
    if (max_tokens == 0) {
        last_mtp_stats_ = {mtp_.has_value(), false, 0, 0, 0};
        return 0;
    }
    if (mtp_ && sampling.enable_mtp && max_tokens > 1) {
        return generate_mtp(
            prompt,
            ids,
            *prefill_hidden,
            initial_logits,
            sampling,
            max_tokens,
            callback,
            token_constraint);
    }
    last_mtp_stats_ = {mtp_.has_value(), false, 0, 0, 0};
    MlxSampler sampler(sampling);
    auto counts = mlx_generation_token_counts(sampling, ids, vocab);
    std::int32_t generated = 0;
    auto logits = initial_logits;
    while (generated < max_tokens) {
        auto sampled = counts ? sampler.sample(logits, *counts) : sampler.sample(logits);
        sampled.eval();
        auto token = sampled.data<std::int32_t>()[0];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error(
                "Nemotron-H sampler returned an out-of-range token");
        }
        if (token_constraint && !token_constraint->allows(token)) {
            auto adjusted = counts
                ? sampler.apply_penalties(logits, *counts)
                : logits;
            adjusted = mlx::core::contiguous(
                mlx::core::astype(adjusted, mlx::core::float32));
            adjusted.eval();
            std::vector<float> masked(
                adjusted.data<float>(), adjusted.data<float>() + vocab);
            token_constraint->apply(masked.data(), masked.size());
            const array constrained_logits(
                masked.begin(), Shape{1, vocab}, mlx::core::float32);
            sampled = sampler.sample(constrained_logits);
            sampled.eval();
            token = sampled.data<std::int32_t>()[0];
            if (token < 0 || token >= vocab ||
                !token_constraint->allows(token)) {
                throw std::runtime_error(
                    "Nemotron-H constrained sampler returned an invalid token");
            }
        }
        if (token_constraint) token_constraint->accept(token);
        const array next({token}, Shape{1, 1}, mlx::core::int32);
        if (counts) {
            *counts = sample_token_counts_add(*counts, next);
        }
        ++generated;
        if (callback && !callback(static_cast<std::int64_t>(token))) break;
        if (generated == max_tokens) break;
        logits = mlx_last_token_logits(forward(next, true), vocab);
    }
    return generated;
}

std::int32_t MlxNemotronHCausalLm::generate_mtp(
    const std::vector<std::int64_t>& prompt,
    const array& prompt_ids,
    const array& prefill_hidden,
    const array& initial_logits,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MlxGenerationTokenCallback& callback,
    const MfqTokenConstraintPtr& token_constraint) {
    (void)prompt;
    const int vocab = static_cast<int>(config_.vocab_size);
    const int hidden = static_cast<int>(config_.hidden_size);
    if (!mtp_) {
        throw std::runtime_error("Nemotron-H MTP head is not loaded");
    }
    mtp_->reset_cache(1);
    if (prompt_ids.shape(1) > 1) {
        mlx_prime_mtp_history(
            prefill_hidden,
            prompt_ids,
            [&](const array& hidden_rows, const array& shifted_ids, int) {
                return mtp_->forward(hidden_rows, shifted_ids, embedding_, true);
            });
    }
    const int predictor_start = mtp_->cache_position();
    const auto reserved_hidden = last_row(prefill_hidden);
    std::int32_t predictor_history_position = predictor_start;

    MlxMtpEngineCallbacks callbacks;
    callbacks.predictor = MlxMtpPredictorDescriptor::recurrent(
        kMlxMtpEngineMaximumDraftDepth);
    callbacks.policy_state = &mtp_policy_state_;
    callbacks.target_cache_position = [this] { return cache_position_; };
    callbacks.prepare_draft =
        [&, reserved_hidden](const MlxMtpDraftContext& context,
                             const MlxMtpTokenSelector& select_token) {
            array hidden_rows = reserved_hidden;
            std::vector<std::int32_t> next_ids{context.pending_token};
            if (!context.initial) {
                hidden_rows = mlx_mtp_committed_hidden(context, 1, hidden);
                next_ids.assign(
                    context.next_token_ids.begin(), context.next_token_ids.end());
            }
            mtp_->trim_cache_to(predictor_history_position);
            if (static_cast<std::int64_t>(next_ids.size()) != hidden_rows.shape(1)) {
                if (context.requested_depth == 0) return;
                throw std::runtime_error(
                    "Nemotron-H MTP shifted history is inconsistent");
            }
            const int committed = static_cast<int>(next_ids.size());
            const array committed_ids(
                next_ids.begin(), Shape{1, committed}, mlx::core::int32);
            auto head_hidden =
                mtp_->forward(hidden_rows, committed_ids, embedding_, true);
            predictor_history_position += committed;
            auto head = last_row(head_hidden);
            for (int position = 0; position < context.requested_depth; ++position) {
                auto token = select_token(
                    mlx_last_token_logits(project_logits(head), vocab));
                if (position + 1 == context.requested_depth) break;
                head = mtp_->forward(
                    head,
                    mlx::core::reshape(token, Shape{1, 1}),
                    embedding_,
                    true);
            }
        };
    callbacks.verify_target =
        [this, vocab](std::int32_t pending_token, const array& draft_tokens,
                      int draft_count) {
            auto verify_ids = mlx_mtp_verification_ids(
                pending_token, draft_tokens, draft_count);
            if (draft_count <= 0) {
                auto result = forward_embeddings_impl(
                    embed_ids(verify_ids), true, 0, true);
                return MlxMtpTargetBatch{
                    mlx::core::reshape(result.first, Shape{1, vocab}),
                    std::move(result.second),
                };
            }
            auto result = forward_verified(verify_ids, 1);
            return MlxMtpTargetBatch{
                mlx::core::reshape(
                    result.first, Shape{draft_count + 1, vocab}),
                std::move(result.second),
            };
        };
    callbacks.resolve_target =
        [this](int accepted_drafts, int draft_count) {
            if (draft_count <= 0) return;
            if (accepted_drafts == draft_count) {
                commit_speculative();
            } else {
                rollback_speculative(accepted_drafts, draft_count);
            }
        };
    (void)prompt;
    const auto result = run_mlx_mtp_generation(
        MlxMtpEngineRequest{
            vocab,
            max_tokens,
            static_cast<int>(config_.max_position_embeddings),
            initial_logits,
            sampling,
            mlx_generation_token_counts(sampling, prompt_ids, vocab),
            {},
            callback,
            0u,
            token_constraint,
        },
        callbacks,
        last_mtp_stats_);
    return result;
}

} // namespace mfq::metal
