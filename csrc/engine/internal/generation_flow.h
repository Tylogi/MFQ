#pragma once

#include "generation_step.h"

#include <algorithm>
#include <iostream>

namespace mfq::engine {

template <class Ops>
Generation generate_prepared(Ops &ops, InferenceRequest &input, InferenceOutput &output,
                             const RequestId& id, bool batched) {
    std::optional<typename Ops::Prepared> prepared;
    if (input.vision && !output.stopped()) {
        auto preparation = ops.prepare(input);
        while (!output.stopped()) {
            auto [step, elapsed] = ops.advance_preparation(preparation);
            output.metrics.multimodal_ms += elapsed;
            if (step.value) {
                prepared = std::move(*step.value);
                break;
            }
            if (!step) throw std::runtime_error("media preparation returned no prompt");
            co_yield step.state;
        }
        if (output.stopped()) co_return;
        const auto elapsed = output.metrics.multimodal_ms;
        co_yield PrefillProgress{{0, 0.0, elapsed, elapsed}};
    }
    if (output.stopped()) co_return;
    auto sequence = ops.generate_text(input, output, std::move(prepared), id, batched);
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr) {
            progress->timing.multimodal_ms = output.metrics.multimodal_ms;
            progress->timing.model_ms += progress->timing.multimodal_ms;
        }
        co_yield std::move(step);
    }
}

template <class Model>
auto capture_session_steps(Model &model, const std::vector<int64_t>& tokens)
    -> mfq::StepSequence<decltype(model.capture_text_session_state(tokens))> {
    if constexpr (requires { model.capture_text_session_steps(tokens); }) {
        auto sequence = model.capture_text_session_steps(tokens);
        while (auto step = sequence.next()) co_yield std::move(step);
    } else co_yield model.capture_text_session_state(tokens);
}

template <class Cache, class Model, class Predictor>
auto restore_session_steps(Cache &cache, Model &model, Predictor *mtp, const std::string &session,
    const std::vector<int64_t>& prompt, size_t limit, const std::string &key)
    -> mfq::StepSequence<decltype(cache.restore_best(model, mtp, session, prompt, limit, key))> {
    if constexpr (requires { cache.restore_steps(model, mtp, session, prompt, limit, key); }) {
        auto sequence = cache.restore_steps(model, mtp, session, prompt, limit, key);
        while (auto step = sequence.next()) co_yield std::move(step);
    } else co_yield cache.restore_best(model, mtp, session, prompt, limit, key);
}

template <class Cache, class State>
Generation store_session_steps(Cache &cache, const std::string &session, State state) {
    if constexpr (requires { cache.store_steps(session, std::move(state)); }) {
        auto sequence = cache.store_steps(session, std::move(state));
        while (auto step = sequence.next()) co_yield step.state;
    } else cache.store(session, std::move(state));
    co_return;
}

inline Generation optional_session_steps(Generation sequence) {
    while (true) {
        decltype(sequence.next()) step;
        try { step = sequence.next(); }
        catch (const std::exception &error) {
            std::cerr << "runtime_session_cache action=skip error=" << error.what() << '\n';
            co_return;
        }
        if (!step) co_return;
        co_yield std::move(step);
    }
}

// Session transactions surround both ordinary and speculative execution.
// Ops owns native prompt/hidden storage and constructs device execution objects.
template <class Ops>
Generation generate_request(Ops ops, InferenceRequest &request, InferenceOutput &output) {
    using Hidden = typename Ops::Hidden;
    auto &model = ops.model;
    auto &session_cache = ops.cache;
    auto *mtp = ops.mtp;
    const auto &prepared = ops.prepared;
    const auto &config = ops.config;
    const auto &prompt = request.prompt;
    auto &sampling = request.sampling;
    const auto &cache_plan = request.cache_plan;
    if (config.generation.prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    if (prepared && prepared->token_ids != prompt)
        throw std::invalid_argument("prepared prompt token IDs disagree");
    const auto occupied =
        static_cast<int64_t>(prompt.size()) +
        (prepared && prepared->transformed() ? prepared->decode_position_delta : 0);
    const auto plan = plan_generation(prompt,
        model.vocab_size(),
        model.max_position_embeddings(),
        sampling.max_tokens,
        cache_plan.stable_prefix_tokens,
        std::max<int64_t>(prompt.size(), occupied));
    sampling.max_tokens = plan.generation_tokens;
    if (output.stopped())
        co_return;
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    const bool caching =
        !ops.batched() && model.supports_text_session_state() &&
        (!prepared || !prepared->transformed() || !input_key.empty()) &&
        plan.stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() || session_cache.persistent_prefix_enabled());
    const bool use_mtp = mtp && sampling.enable_mtp && sampling.max_tokens > 1;
    if (mtp) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    ExecutionCleanup cleanup{output.cleanup_failure, [&] { ops.release(); }};
    typename Ops::Restore restored;
    if (caching && (!use_mtp || mtp->supports_session_state())) {
        auto restore = restore_session_steps(session_cache, model, use_mtp ? mtp : nullptr,
            cache_plan.session_id, prompt, plan.stable_prefix_tokens, input_key);
        while (!output.stopped()) {
            auto step = restore.next();
            if (!step) break;
            if (step.value) restored = std::move(*step.value);
            co_yield step.state;
        }
        if (output.stopped()) co_return;
    }
    if (restored.tokens || use_mtp)
        ops.invalidate_plan();
    if (!restored.tokens) ops.reset();
    std::vector<int64_t> history = prompt;
    Hidden last_target_hidden;
    std::size_t last_snapshot = 0;
    const auto snapshot = [&]() -> Generation {
        const auto position = model.cache_pos;
        if (!caching || output.result.cancelled || position <= 0 ||
            static_cast<std::size_t>(position) > history.size() ||
            static_cast<std::size_t>(position) == last_snapshot)
            co_return;
        std::vector<int64_t> tokens(history.begin(), history.begin() + position);
        auto capture = capture_session_steps(model, tokens);
        while (!output.result.cancelled) {
            auto step = capture.next();
            if (!step) break;
            if (step.value) {
                auto state = std::move(*step.value);
                state.input_key = input_key;
                if (use_mtp) {
                    if (!ops.has_hidden(last_target_hidden) || position <= 1) co_return;
                    state.mtp = mtp->capture_session_state(position, last_target_hidden);
                    state.bytes += state.mtp->bytes;
                    co_yield mfq::StepState::advanced;
                }
                auto store = store_session_steps(session_cache, cache_plan.session_id, std::move(state));
                while (!output.result.cancelled) {
                    auto saved = store.next();
                    if (!saved) break;
                    co_yield std::move(saved);
                }
                if (!output.result.cancelled) last_snapshot = static_cast<std::size_t>(position);
            } else co_yield step.state;
        }
    };
    auto sequence = use_mtp
        ? ops.speculate(request, output, restored.tokens, restored.mtp_last_target_hidden,
                        &last_target_hidden)
        : ops.plain(request, output, restored.tokens, caching ? plan.stable_prefix_tokens : 0);
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr;
            progress && !use_mtp && caching &&
            progress->timing.prompt_tokens + restored.tokens == plan.stable_prefix_tokens)
        {
            auto save = optional_session_steps(snapshot());
            while (auto saved = save.next()) co_yield std::move(saved);
        }
        if (auto *delta = event ? std::get_if<OutputDelta>(&*event) : nullptr)
            history.insert(history.end(), delta->token_ids.begin(), delta->token_ids.end());
        co_yield std::move(step);
    }
    auto save = optional_session_steps(snapshot());
    while (auto saved = save.next()) co_yield std::move(saved);
    cleanup.finish();
}

} // namespace mfq::engine
