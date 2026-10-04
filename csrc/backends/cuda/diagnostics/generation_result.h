#pragma once
#include "engine/generation.h"
#include <thread>

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
inline GenerationResult collect_engine_steps(mfq::engine::Engine& engine,
        std::vector<int64_t> prompt, MfqSamplingParams sampling) {
    mfq::engine::EngineRequest request;
    request.id = "check";
    request.token_ids = std::move(prompt); request.input.sampling = sampling;
    if (engine.admit(std::move(request)) != mfq::engine::Admission::accepted)
        throw std::runtime_error("diagnostic request was not admitted");
    GenerationResult result;
    unsigned terminals = 0;
    while (!terminals) {
        auto step = engine.step({"check"});
        for (auto& event : step.events) {
            if (event.id != "check") throw std::runtime_error("unexpected diagnostic request ID");
            if (auto* failure = std::get_if<mfq::engine::Failed>(&event.data))
                throw std::runtime_error(failure->message);
            if (std::holds_alternative<mfq::engine::Cancelled>(event.data))
                throw std::runtime_error("diagnostic generation cancelled");
            if (auto* progress = std::get_if<mfq::engine::PrefillProgress>(&event.data))
                result.prefill = progress->timing;
            if (auto* delta = std::get_if<mfq::engine::OutputDelta>(&event.data)) {
                if (result.tokens.empty()) result.first_token = std::chrono::steady_clock::now();
                result.tokens.insert(result.tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
            }
            terminals += mfq::engine::terminal(event.data);
        }
        if (!terminals && !step.has_work)
            throw std::runtime_error("diagnostic generation lost its terminal event");
        if (terminals > 1 || (terminals && step.has_work))
            throw std::runtime_error("diagnostic generation retired incorrectly");
        if (!terminals && step.wake_at) std::this_thread::sleep_until(*step.wake_at);
    }
    return result;
}
inline std::vector<int64_t> check_engine_steps(mfq::engine::Engine& engine,
        std::vector<int64_t> prompt, MfqSamplingParams sampling) {
    return collect_engine_steps(engine, std::move(prompt), sampling).tokens;
}
} // namespace mfq::cuda::diagnostics
