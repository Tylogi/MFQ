#include "runtime_checks.h"

#include "engine/generation.h"
#include "engine/options.h"
#include "engine/runtime_config.h"
#include "engine/text_session_cache.h"
#include "models/qwen35/batch_executor.h"

#include <cuda_profiler_api.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <vector>

namespace mfq::cuda::diagnostics {

using namespace mfq::cuda::internal;

// Fixed synthetic-ID latency probe through the actual runtime dispatch. Model
// loading and the independent serial oracle are outside every timed request.
int run_qwen35_mtp_bench(
        mfq::cuda::Qwen35CausalLm& model, Qwen35Mtp& mtp,
        bool enable_mtp, int generated_tokens, int repetitions) {
    using Clock = std::chrono::steady_clock;
    using Tensor = mfq_tensor_backend::Tensor;
    MFQ_RUNTIME_CHECK(
        generated_tokens >= 2 && repetitions > 0 && repetitions <= 100 &&
            model.max_position_embeddings() >= generated_tokens + 17,
        "MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");
    DecodeGraphCache graph_cache(model.max_position_embeddings());
    const auto runtime_config = resolve_cuda_runtime_config({});
    TextSessionCache session_cache(
        runtime_config.session_cache, runtime_config.prefix_cache);
    std::mutex model_mutex;
    MfqSamplingParams params;
    params.max_tokens = generated_tokens;
    params.temperature = 0.;
    params.top_k = 1;
    params.top_p = 1.;
    params.enable_mtp = enable_mtp;
    params.seed = 20260907;
    const auto options = mfq_tensor_backend::TensorOptions()
        .device(mfq_tensor_backend::kCUDA).dtype(mfq_tensor_backend::kInt64);
    const char* profiler_env = std::getenv("MFQ_CUDA_PROFILER_RANGE");
    const bool profiler_range = profiler_env != nullptr && std::atoi(profiler_env) != 0;
    std::cout << "mtp_bench config mode=" << (enable_mtp ? "mtp" : "ordinary")
        << " full_window_batching=1"
        << " runtime_graph=" << (runtime_config.decode_graph.enabled ? 1 : 0)
        << " gen=" << generated_tokens << " reps=" << repetitions
        << " warmup=1 seed=20260907 synthetic_ids=1 eos_stop=0 session_cache=0\n";
    const auto ms_between = [](Clock::time_point first, Clock::time_point last) {
        return std::chrono::duration<double, std::milli>(last - first).count();
    };
    for (const auto& prompt : std::vector<std::vector<int64_t>>{
            {1, 2, 3}, {100, 200, 300, 400, 500, 600, 700}, std::vector<int64_t>(17, 10)}) {
        model.reset(1);
        Tensor input = mfq_tensor_backend::tensor(prompt, options).reshape({1, -1}).contiguous();
        std::vector<int64_t> reference;
        reference.reserve(generated_tokens);
        for (int step = 0; step < generated_tokens; ++step) {
            auto next = model.next_token(input);
            reference.push_back(next.template item<int64_t>());
            input = next.reshape({1, 1});
        }
        mfq_cuda_synchronize();
        for (int repeat = -1; repeat < repetitions; ++repeat) {
            std::vector<int64_t> output;
            output.reserve(generated_tokens);
            MfqPrefillTiming prefill;
            Clock::time_point first_token;
            mfq_cuda_synchronize();
            // Capture only the actual request; its independent oracle, full
            // warmup, model load and context construction remain unprofiled.
            const bool capture_request = profiler_range && repeat >= 0;
            if (capture_request) {
                std::cout << "mtp_bench profiler_begin prompt=" << prompt.size()
                    << " repeat=" << repeat << '\n';
                MFQ_CUDA_CHECK(cudaProfilerStart());
            }
            const auto started = Clock::now();
            const int produced = generate(model, model_mutex, graph_cache,
                session_cache, runtime_config, prompt, params, [&](int64_t token) {
                    if (output.empty()) first_token = Clock::now();
                    output.push_back(token);
                    return true;
                }, [&](const MfqPrefillTiming& timing) { prefill = timing; }, {}, {}, &mtp);
            mfq_cuda_synchronize();
            const auto finished = Clock::now();
            if (capture_request) {
                MFQ_CUDA_CHECK(cudaProfilerStop());
                std::cout << "mtp_bench profiler_end prompt=" << prompt.size()
                    << " repeat=" << repeat << '\n';
            }
            MFQ_RUNTIME_CHECK(produced == generated_tokens && output == reference,
                "MTP benchmark output differs from ordinary serial oracle");
            MFQ_RUNTIME_CHECK(!enable_mtp || mtp.last_cycles > 0,
                "MTP benchmark did not execute speculative cycles");
            const double total_ms = ms_between(started, finished);
            const double first_ms = ms_between(started, first_token);
            const double decode_ms = ms_between(first_token, finished);
            MFQ_RUNTIME_CHECK(std::isfinite(total_ms) && total_ms > 0. &&
                std::isfinite(decode_ms) && decode_ms > 0., "invalid benchmark clock interval");
            std::cout << "mtp_bench sample mode=" << (enable_mtp ? "mtp" : "ordinary")
                << " prompt=" << prompt.size() << " gen=" << produced
                << " repeat=" << repeat << " warmup=" << (repeat < 0)
                << " total_ms=" << total_ms << " ttft_ms=" << first_ms
                << " decode_ms=" << decode_ms << " prefill_gpu_ms=" << prefill.llm_ms
                << " total_tps=" << produced * 1000. / total_ms
                << " decode_tps=" << (produced - 1) * 1000. / decode_ms
                << " exact=1 cycles=" << (enable_mtp ? mtp.last_cycles : 0)
                << " accepted=" << (enable_mtp ? mtp.last_accepted : 0)
                << " rejected=" << (enable_mtp ? mtp.last_rejected : 0)
                << " graph_captures=" << graph_cache.captures
                << " graph_reuses=" << graph_cache.reuses << '\n';
        }
    }
    std::cout << "mtp_bench PASS samples=" << repetitions * 3 << '\n';
    return 0;
}


int run_cuda_continuous_batching_check(
        mfq::cuda::Qwen35CausalLm& model) {
    return mfq::cuda::qwen35::run_qwen_continuous_batching_check(
        model, resolve_cuda_runtime_config({}));
}

} // namespace mfq::cuda::diagnostics
