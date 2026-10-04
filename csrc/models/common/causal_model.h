#pragma once

#include <variant>
#include "step_sequence.h"
#include "causal_forward.h"
#include "causal_metadata.h"
#include <algorithm>
#include <optional>
#include <string_view>
#include <vector>

namespace mfq::models {

template <class Tensor, class Plan> struct CausalForwardInputs {
    Tensor ids, input_embeddings;
    std::optional<Tensor> pos_override, seq_len, attention_mask, cache_positions_override;
    std::vector<Tensor> *block_trace = nullptr;
    Tensor *raw_hidden = nullptr;
    bool advance_cache_with_position_ids = false;
    int64_t confirmed_prefix = 0;
    Plan plan{};
};

template <class Tensor> struct CausalPositions {
    Tensor positions, full_positions;
};

// Backend owns tensors, weights, physical state and numerical operations.
// This class owns the causal model entry points and their logical transitions.
template <class Backend, class Derived> struct CausalModelBase : Backend, CausalState {
    using Tensor = typename Backend::Tensor;
    using ForwardPlan = typename Backend::ForwardPlan;
    using Inputs = CausalForwardInputs<Tensor, ForwardPlan>;
    using SessionState = typename Backend::SessionState;
    using SessionStateKind = typename Backend::SessionStateKind;
    using SessionCodec = typename Backend::SessionCodec;

    template <class Graph, class Source>
    void load_definition(std::string_view payload, const Graph &graph, const Source &source,
                         int64_t context_size = 0) {
        require_model(Derived::accepts_backbone(graph.backbone),
                      "model backbone does not match the requested causal model");
        model().adapter_load_config(payload, graph, source);
        require_model(num_hidden_layers() == graph.topology.text_layers,
                      "model graph/config text-layer topology mismatch");
        if (context_size > 0) {
            require_model(context_size <= max_position_embeddings(),
                          "--ctx-size exceeds max_position_embeddings");
            set_max_position_embeddings(context_size);
        }
    }

    template <class Loader> void load_weights(Loader &loader, bool with_blocks = true) {
        constexpr auto embedding = "model.token_embedding.weight";
        constexpr auto output = "model.output.weight";
        this->embed = loader.embedding(embedding);
        loader.final_state(this->output_norm);
        this->lm_head = tie_word_embeddings() || !loader.has_weight(output)
                            ? loader.tied_output(this->embed, embedding)
                            : loader.output(output);
        if (with_blocks) {
            loader.prepare_blocks();
            this->blocks.reserve(static_cast<size_t>(num_hidden_layers()));
            for (int layer = 0; layer < num_hidden_layers(); ++layer)
                this->blocks.push_back(loader.block(layer, std::string(layer_type(layer))));
        }
    }

    CausalPositions<Tensor> adapter_prepare_positions(Tensor positions, int64_t, int64_t) const {
        return {positions, positions};
    }
    std::optional<Tensor> adapter_attention_mask(std::optional<Tensor> mask, int64_t,
                                                 int64_t) const {
        return mask;
    }
    void adapter_validate_forward(int64_t, int64_t, int64_t, bool, bool, bool) const {}
    bool adapter_allows_speculative_position_override() const noexcept { return false; }
    void adapter_validate_positions(const Tensor &positions, int64_t batch, int64_t tokens,
                                    bool) const {
        require_model((Backend::rank(positions) == 1 && Backend::size(positions, 0) == tokens) ||
                          (Backend::rank(positions) == 2 && Backend::size(positions, 0) == batch &&
                           Backend::size(positions, 1) == tokens),
                      "position_ids must have shape [tokens] or [batch,tokens]");
    }
    bool adapter_pass_cache_positions(bool has_mrope, bool has_override) const noexcept {
        return has_mrope || has_override;
    }
    bool adapter_pass_attention_mask() const noexcept { return false; }
    bool adapter_force_cache_advance() const noexcept { return false; }
    bool adapter_uses_decode_sequence_length() const noexcept { return true; }
    bool adapter_supports_speculation() const noexcept { return false; }
    bool adapter_supports_suffix_speculation() const noexcept { return false; }

    int64_t vocab_size() const noexcept { return this->metadata.vocab_size; }

    int64_t hidden_size() const noexcept { return this->metadata.hidden_size; }

    int64_t num_hidden_layers() const noexcept { return this->metadata.num_hidden_layers; }

    int64_t num_attention_heads() const noexcept { return this->metadata.num_attention_heads; }

    int64_t num_key_value_heads() const noexcept { return this->metadata.num_key_value_heads; }

    int64_t head_dim() const noexcept { return this->metadata.head_dim; }

    int64_t max_position_embeddings() const noexcept {
        return this->metadata.max_position_embeddings;
    }

    int64_t rotary_dim() const noexcept { return this->metadata.rotary_dim; }

    double rope_base() const noexcept { return this->metadata.rope_base; }

    void set_max_position_embeddings(int64_t value) noexcept {
        model().adapter_set_max_position_embeddings(value);
        this->metadata.max_position_embeddings = value;
    }

    double rms_norm_eps() const noexcept { return this->metadata.rms_norm_eps; }

    double norm_weight_offset() const noexcept { return this->metadata.norm_weight_offset; }

    bool tie_word_embeddings() const noexcept { return this->metadata.tie_word_embeddings; }

    int64_t num_experts() const noexcept { return this->metadata.num_experts; }

    int64_t hc_mult() const noexcept { return this->metadata.hc_mult; }

    double hc_eps() const noexcept { return this->metadata.hc_eps; }

    double final_logit_softcapping() const noexcept {
        return this->metadata.final_logit_softcapping;
    }

    double embedding_scale() const noexcept { return this->metadata.embedding_scale; }

    std::string_view model_type() const noexcept { return this->metadata.model_type; }

    std::string_view layer_type(int64_t layer) const {
        return this->metadata.layer_types.at(static_cast<std::size_t>(layer));
    }

    Tensor embed_forward(Tensor ids) const {
        auto scope = this->execution_scope();
        return this->embed_tokens(this->device_ids(std::move(ids)));
    }
    void reset(int64_t batch) {
        reset_model(model(), batch, [](const auto &block) { return Backend::block_scope(block); });
    }
    Tensor hidden_forward(Tensor ids, std::optional<Tensor> positions = {},
                          std::optional<Tensor> lengths = {}, std::vector<Tensor> *trace = nullptr,
                          std::optional<Tensor> cache_positions = {}, Tensor *raw_hidden = nullptr,
                          int64_t confirmed_prefix = 0, ForwardPlan plan = {}) {
        auto scope = this->execution_scope();
        ids = normalize_ids(std::move(ids));
        auto embedded = this->embed_tokens(ids);
        return hidden_forward_inputs(std::move(ids), std::move(embedded), positions, lengths, trace,
                                     {}, false, cache_positions, raw_hidden, confirmed_prefix,
                                     plan);
    }
    Tensor
    hidden_forward_inputs(Tensor ids, Tensor embeddings, std::optional<Tensor> positions = {},
                          std::optional<Tensor> lengths = {}, std::vector<Tensor> *trace = nullptr,
                          std::optional<Tensor> attention_mask = {},
                          bool advance_cache_with_position_ids = false,
                          std::optional<Tensor> cache_positions = {}, Tensor *raw_hidden = nullptr,
                          int64_t confirmed_prefix = 0, ForwardPlan plan = {}) {
        auto scope = this->execution_scope();
        Inputs input{normalize_ids(std::move(ids)),
                     std::move(embeddings),
                     positions,
                     lengths,
                     attention_mask,
                     cache_positions,
                     trace,
                     raw_hidden,
                     advance_cache_with_position_ids,
                     confirmed_prefix,
                     plan};
        auto ops = this->forward_ops(model(), std::move(input));
        return causal_forward(ops);
    }
    Tensor finalize_hidden(Tensor hidden, int64_t batch, int64_t tokens) {
        return model().adapter_finalize_hidden(std::move(hidden), this->output_norm, batch, tokens);
    }
    Tensor apply_final_logit_softcap(Tensor logits) const {
        const auto cap = final_logit_softcapping();
        return cap > 0.0 ? this->softcap(std::move(logits), cap) : std::move(logits);
    }
    Tensor logits_from_hidden(Tensor hidden) {
        return apply_final_logit_softcap(model().adapter_logits(this->lm_head, std::move(hidden)));
    }
    Tensor forward(Tensor ids) { return logits_from_hidden(hidden_forward(std::move(ids))); }
    Tensor forward_inputs(Tensor ids, Tensor embeddings, std::optional<Tensor> positions = {},
                          std::optional<Tensor> lengths = {}) {
        return logits_from_hidden(
            hidden_forward_inputs(std::move(ids), std::move(embeddings), positions, lengths));
    }
    Tensor last_logits(Tensor ids) {
        const auto lengths = decode_lengths(ids);
        auto hidden = hidden_forward(std::move(ids), {}, lengths);
        return apply_final_logit_softcap(
            model().adapter_last_logits(this->lm_head, this->last_hidden(std::move(hidden))));
    }
    Tensor next_token(Tensor ids) {
        const auto lengths = decode_lengths(ids);
        return next_token_from_hidden(hidden_forward(std::move(ids), {}, lengths));
    }
    Tensor next_token_from_hidden(Tensor hidden) {
        return model().adapter_next_token(this->lm_head, this->last_hidden(std::move(hidden)));
    }
    Tensor hidden_forward_static(Tensor ids, Tensor positions, Tensor lengths,
                                 ForwardPlan plan = {}) {
        // Static execution must write KV by the dynamic position tensor too.
        return hidden_forward(std::move(ids), positions, lengths, nullptr, positions, nullptr, 0,
                              plan);
    }
    Tensor last_logits_static(Tensor ids, Tensor positions, Tensor lengths, ForwardPlan plan = {}) {
        auto hidden =
            hidden_forward_static(std::move(ids), std::move(positions), std::move(lengths), plan);
        return apply_final_logit_softcap(
            model().adapter_last_logits(this->lm_head, this->last_hidden(std::move(hidden))));
    }
    Tensor next_token_static(Tensor ids, Tensor positions, Tensor lengths, ForwardPlan plan = {}) {
        return next_token_from_hidden(
            hidden_forward_static(std::move(ids), std::move(positions), std::move(lengths), plan));
    }
    bool supports_speculation() const {
        return model().adapter_supports_speculation() && speculative_blocks();
    }
    bool supports_suffix_speculation() const {
        return model().adapter_supports_suffix_speculation() && speculative_blocks();
    }
    void begin_speculative_suffix(int64_t draft_tokens) {
        models::begin_speculative_suffix(
            model(), draft_tokens, [](const auto &block) { return Backend::block_scope(block); });
    }
    void commit_speculative() {
        finish_speculative(model(), true, 0,
                           [](const auto &block) { return Backend::block_scope(block); });
    }
    void rollback_speculative(int64_t accepted_suffix = 0) {
        finish_speculative(model(), false, accepted_suffix,
                           [](const auto &block) { return Backend::block_scope(block); });
    }
    Tensor hidden_forward_speculative_suffix(Tensor ids, Tensor *raw_hidden = nullptr) {
        require_model(speculative_start >= 0 && speculative_confirmed == 0 &&
                          !speculative_suffix_forward,
                      "speculative suffix is not active");
        speculative_suffix_forward = true;
        try {
            auto result = hidden_forward(std::move(ids), {}, {}, nullptr, {}, raw_hidden);
            speculative_suffix_forward = false;
            return result;
        } catch (...) {
            speculative_suffix_forward = false;
            throw;
        }
    }
    SessionStateKind text_session_state_kind() const { return SessionCodec::kind(model()); }
    bool supports_text_session_state() const {
        return text_session_state_kind() != SessionStateKind::Unsupported;
    }
    bool supports_paged_text_session_state() const { return SessionCodec::supports_paged(model()); }
    mfq::StepSequence<SessionState> capture_text_session_steps(const std::vector<int64_t> &tokens) const {
        if constexpr (requires { SessionCodec::capture_steps(model(), tokens); }) {
            auto sequence = SessionCodec::capture_steps(model(), tokens);
            while (auto step = sequence.next()) {
                if (step.value) step.value->decode_position_delta = decode_position_delta;
                co_yield std::move(step);
            }
        } else {
            auto state = SessionCodec::capture(model(), tokens);
            state.decode_position_delta = decode_position_delta;
            co_yield std::move(state);
        }
    }
    SessionState capture_text_session_state(const std::vector<int64_t> &tokens) const {
        return mfq::finish_steps(capture_text_session_steps(tokens));
    }
    mfq::StepSequence<std::monostate> restore_text_session_steps(const SessionState &state) {
        if constexpr (requires { SessionCodec::restore_steps(model(), state); }) {
            auto sequence = SessionCodec::restore_steps(model(), state);
            while (auto step = sequence.next()) co_yield std::move(step);
        } else SessionCodec::restore(model(), state);
        decode_position_delta = state.decode_position_delta;
        co_yield std::monostate{};
    }
    void restore_text_session_state(const SessionState &state) {
        (void)mfq::finish_steps(restore_text_session_steps(state));
    }

  private:
    Derived &model() { return static_cast<Derived &>(*this); }
    const Derived &model() const { return static_cast<const Derived &>(*this); }
    Tensor normalize_ids(Tensor ids) const {
        const auto rank = this->rank(ids);
        require_model(rank == 1 || rank == 2,
                      "token IDs must have shape [tokens] or [batch,tokens]");
        if (rank == 1)
            ids = this->batch_ids(std::move(ids));
        require_model(this->size(ids, 0) > 0 && this->size(ids, 1) > 0,
                      "token IDs must not be empty");
        return this->device_ids(std::move(ids));
    }
    std::optional<Tensor> decode_lengths(const Tensor &ids) const {
        const auto rank = this->rank(ids);
        require_model(rank == 1 || rank == 2,
                      "token IDs must have shape [tokens] or [batch,tokens]");
        const auto tokens = this->size(ids, rank - 1);
        const auto batch = rank == 1 ? 1 : this->size(ids, 0);
        if (model().adapter_uses_decode_sequence_length() && cache_pos > 0 && tokens == 1)
            return this->sequence_lengths(batch, cache_pos + 1);
        return {};
    }
    bool speculative_blocks() const {
        return !this->blocks.empty() &&
               std::all_of(this->blocks.begin(), this->blocks.end(),
                           [](const auto &block) { return block->supports_speculation(); });
    }
};
} // namespace mfq::models
