#pragma once
#include "engine/generation.h"

namespace mfq::cuda::diagnostics {
struct GenerationResult {
    std::vector<int64_t> tokens;
    MfqPrefillTiming prefill;
    std::chrono::steady_clock::time_point first_token;
};
inline GenerationResult collect_generation(mfq::engine::Generation generation) {
    GenerationResult result;
    while (auto step = generation.next()) {
        auto& event = step.value;
        if (!event) continue;
        if (auto* progress = std::get_if<mfq::engine::PrefillProgress>(&*event))
            result.prefill = progress->timing;
        if (auto* delta = std::get_if<mfq::engine::OutputDelta>(&*event)) {
            if (result.tokens.empty()) result.first_token = std::chrono::steady_clock::now();
            result.tokens.insert(result.tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        }
    }
    return result;
}
template <class Model>
std::vector<int64_t> check_mtp_steps(Model& model, MtpModule& mtp,
        const std::vector<int64_t>& prompt, const MfqSamplingParams& sampling, int stop_after = 0) {
    mfq::engine::InferenceRequest request;
    request.prompt = prompt; request.sampling = sampling;
    mfq::engine::InferenceOutput output(request, nullptr, "mtp-check");
    model.reset(1);
    mtp.reset(1);
    auto generation = run_mtp_generation(model, mtp, request, output);
    std::vector<int64_t> tokens;
    std::size_t largest_delta = 0;
    while (auto step = generation.next()) {
        auto& event = step.value;
        if (!event) continue;
        if (auto* delta = std::get_if<mfq::engine::OutputDelta>(&*event)) {
            MFQ_RUNTIME_CHECK(model.speculative_start < 0, "MTP published an uncommitted transaction");
            largest_delta = std::max(largest_delta, delta->token_ids.size());
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
            if (stop_after) request.sampling.max_tokens = stop_after;
        }
    }
    if (!stop_after && tokens.size() >= 16 && mtp.last_accepted > 1)
        MFQ_RUNTIME_CHECK(largest_delta > 1, "MTP split a verification delta into per-token events");
    MFQ_RUNTIME_CHECK(output.metrics.mtp.available &&
        output.metrics.mtp.cycles == mtp.last_stats.cycles &&
        output.metrics.mtp.drafted_tokens == mtp.last_stats.drafted_tokens &&
        output.metrics.mtp.accepted_tokens == mtp.last_stats.accepted_tokens,
        "request did not retain its MTP statistics");
    return tokens;
}
inline std::vector<int64_t> check_engine_steps(mfq::engine::Engine& engine,
        std::vector<int64_t> prompt, MfqSamplingParams sampling) {
    mfq::engine::EngineRequest request;
    request.id = "check";
    request.token_ids = std::move(prompt); request.input.sampling = sampling;
    if (engine.admit(std::move(request)) != mfq::engine::Admission::accepted)
        throw std::runtime_error("diagnostic request was not admitted");
    std::vector<int64_t> tokens;
    bool terminal = false;
    while (!terminal) for (auto& event : engine.step({"check"}).events) {
        if (auto* failure = std::get_if<mfq::engine::Failed>(&event.data))
            throw std::runtime_error(failure->message);
        if (auto* delta = std::get_if<mfq::engine::OutputDelta>(&event.data))
            tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        terminal |= mfq::engine::terminal(event.data);
    }
    return tokens;
}
} // namespace mfq::cuda::diagnostics
