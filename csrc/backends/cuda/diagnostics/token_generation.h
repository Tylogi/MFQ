#pragma once

#include "generation_result.h"
#include "mfq/cuda/engine.h"

#include <cuda_profiler_api.h>
#include <iostream>

namespace mfq::cuda::diagnostics {

inline int generate_diagnostic_tokens(CudaLoadOptions options,
        std::vector<int64_t> prompt, int tokens, bool profile) {
    using Clock = std::chrono::steady_clock;
    CudaEngineOptions engine_options;
    static_cast<CudaLoadOptions&>(engine_options) = std::move(options);
    const auto start = Clock::now();
    auto engine = load_cuda_engine(std::move(engine_options));
    const auto loaded = Clock::now();
    MfqSamplingParams sampling;
    sampling.max_tokens = tokens;
    sampling.temperature = 0;
    sampling.top_k = 1;
    if (profile) MFQ_CUDA_CHECK(cudaProfilerStart());
    GenerationResult result;
    try { result = collect_engine_steps(*engine, std::move(prompt), sampling); }
    catch (...) { if (profile) cudaProfilerStop(); throw; }
    if (profile) MFQ_CUDA_CHECK(cudaProfilerStop());
    const auto finished = Clock::now();
    const auto first = result.tokens.empty() ? finished : result.first_token;
    const auto seconds = [](auto end, auto begin) {
        return std::chrono::duration<double>(end - begin).count();
    };
    const auto decode_tokens = result.tokens.empty() ? 0 : result.tokens.size() - 1;
    const auto decode_seconds = seconds(finished, first);
    std::cout << "generation_path=engine_step\n"
        << "load_sec=" << seconds(loaded, start) << '\n'
        << "prefill_sec=" << seconds(first, loaded) << '\n'
        << "decode_tokens=" << decode_tokens << '\n'
        << "decode_sec=" << decode_seconds << '\n';
    if (decode_tokens && decode_seconds > 0)
        std::cout << "decode_tok_per_s=" << decode_tokens / decode_seconds << '\n';
    std::cout << "generated_ids=";
    for (size_t i = 0; i < result.tokens.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << result.tokens[i];
    }
    std::cout << '\n';
    return 0;
}

} // namespace mfq::cuda::diagnostics
