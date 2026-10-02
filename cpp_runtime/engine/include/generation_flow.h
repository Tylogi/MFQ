#pragma once

#include "generation_step.h"

#include <algorithm>
#include <iostream>

namespace mfq::engine {

inline bool batch_compatible(const EngineRequest &request, bool persistent_prefix, bool has_mtp) {
    const auto &input = request.input;
    return !input.media && input.cache_plan.session_id.empty() &&
           !((request.token_ids.empty() || input.cache_plan.stable_prefix_tokens) &&
               persistent_prefix) &&
           !(input.sampling.enable_mtp && has_mtp);
}

template <class Ops>
Generation generate_prepared(Ops &ops, InferenceRequest &input, InferenceOutput &output) {
    std::optional<typename Ops::Prepared> prepared;
    if (input.vision && !output.result.cancelled) {
        auto [prompt, elapsed] = ops.prepare(input);
        prepared = std::move(prompt);
        output.metrics.multimodal_ms = elapsed;
        co_yield PrefillProgress{{0, 0.0, elapsed, elapsed}};
    }
    auto sequence = ops.generate_text(input, output, std::move(prepared));
    while (auto event = sequence.next()) {
        if (auto *progress = std::get_if<PrefillProgress>(&*event)) {
            progress->timing.multimodal_ms = output.metrics.multimodal_ms;
            progress->timing.model_ms += progress->timing.multimodal_ms;
        }
        co_yield std::move(*event);
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
        model.supports_text_session_state() &&
        (!prepared || !prepared->transformed() || !input_key.empty()) &&
        plan.stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() || session_cache.persistent_prefix_enabled());
    const bool use_mtp = mtp && sampling.enable_mtp && sampling.max_tokens > 1;
    if (mtp) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    typename Ops::Restore restored;
    if (caching && (!use_mtp || mtp->supports_session_state()))
        restored = session_cache.restore_best(model,
            use_mtp ? mtp : nullptr,
            cache_plan.session_id,
            prompt,
            plan.stable_prefix_tokens,
            input_key);
    if (restored.tokens || use_mtp)
        ops.invalidate_plan();
    if (!restored.tokens) {
        model.reset(1);
        if (mtp)
            mtp->reset(1);
    }
    struct ResetOnFailure {
        Ops &ops;
        bool success = false;
        ~ResetOnFailure() {
            if (!success)
                try {
                    ops.reset();
                } catch (...) {
                }
        }
    } cleanup{ops};
    std::vector<int64_t> history = prompt;
    Hidden last_target_hidden;
    std::size_t last_snapshot = 0;
    const auto snapshot = [&] {
        const auto position = model.cache_pos;
        if (!caching || output.result.cancelled || position <= 0 ||
            static_cast<std::size_t>(position) > history.size() ||
            static_cast<std::size_t>(position) == last_snapshot)
            return;
        try {
            std::vector<int64_t> tokens(history.begin(), history.begin() + position);
            auto state = model.capture_text_session_state(tokens);
            state.input_key = input_key;
            if (use_mtp) {
                if (!ops.has_hidden(last_target_hidden) || position <= 1)
                    return;
                state.mtp = mtp->capture_session_state(position, last_target_hidden);
                state.bytes += state.mtp->bytes;
            }
            session_cache.store(cache_plan.session_id, std::move(state));
            last_snapshot = static_cast<std::size_t>(position);
        } catch (const std::exception &error) {
            std::cerr << "runtime_session_cache action=skip error=" << error.what() << '\n';
        }
    };
    if (use_mtp) {
        auto generation = ops.speculate(
            request, output, restored.tokens, restored.mtp_last_target_hidden, &last_target_hidden);
        while (auto event = generation.next()) {
            if (auto *delta = std::get_if<OutputDelta>(&*event))
                history.insert(history.end(), delta->token_ids.begin(), delta->token_ids.end());
            co_yield std::move(*event);
        }
    } else {
        auto plain = ops.plain(request);
        auto sequence = generate_sequence(plain,
            output,
            prompt.size(),
            restored.tokens,
            caching ? plan.stable_prefix_tokens : 0,
            config.generation.prefill_chunk_size);
        while (auto event = sequence.next()) {
            if (auto *progress = std::get_if<PrefillProgress>(&*event);
                progress && caching &&
                progress->timing.prompt_tokens + restored.tokens == plan.stable_prefix_tokens)
                snapshot();
            if (auto *delta = std::get_if<OutputDelta>(&*event))
                history.insert(history.end(), delta->token_ids.begin(), delta->token_ids.end());
            co_yield std::move(*event);
        }
    }
    snapshot();
    ops.reset();
    cleanup.success = true;
}

} // namespace mfq::engine
