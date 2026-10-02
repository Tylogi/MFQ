#include "runtime_checks.h"

#include "engine/cuda_engine.h"
#include "engine/generation.h"
#include "engine/options.h"
#include "cuda_runtime_config.h"
#include "engine/text_session_cache.h"
#include "engine/cuda_batching.h"

#include <cuda_profiler_api.h>

#include <chrono>
#include <cmath>
#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
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
            mfq::engine::InferenceRequest request;
            request.prompt = prompt; request.sampling = params;
            mfq::engine::InferenceOutput parsed(request, nullptr, "bench");
            auto result = collect_generation(generate(model, graph_cache, session_cache,
                runtime_config, request, parsed, &mtp));
            output = std::move(result.tokens); prefill = result.prefill; first_token = result.first_token;
            const int produced = static_cast<int>(output.size());
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


int run_qwen_continuous_batching_check(Qwen35CausalLm& model, const CudaRuntimeConfig& runtime_config) {
    using namespace mfq::engine;
    auto config = runtime_config;
    config.generation.prefill_chunk_size = 64;
    config.continuous_batch.max_sequences = 4;
    MfqSamplingParams sampling;
    sampling.max_tokens = 20; sampling.temperature = 0; sampling.top_k = 1; sampling.enable_mtp = false;
    std::vector<int64_t> first_prompt(193), second_prompt(17);
    for (size_t i = 0; i < first_prompt.size(); ++i) first_prompt[i] = 101 + (i * 37) % 900;
    for (size_t i = 0; i < second_prompt.size(); ++i) second_prompt[i] = 113 + (i * 53) % 880;
    DecodeGraphCache graph(model.max_position_embeddings());
    TextSessionCache cache(config.session_cache, config.prefix_cache);
    auto serial = [&](const std::vector<int64_t>& prompt, MfqPromptCachePlan plan = {}) {
        InferenceRequest request; request.prompt = prompt; request.sampling = sampling; request.cache_plan = std::move(plan);
        InferenceOutput output(request, nullptr, "serial");
        return collect_generation(generate(model, graph, cache, config, request, output)).tokens;
    };
    const auto first_reference = serial(first_prompt), second_reference = serial(second_prompt);
    graph.invalidate(); // Physical batch slots replace the serial graph's storage.
    {
        QwenBatchExecutor batcher(model, *model.execution, config.continuous_batch, 64);
        InferenceRequest first, second;
        first.prompt = first_prompt; first.sampling = sampling;
        second.prompt = second_prompt; second.sampling = sampling;
        InferenceOutput first_output(first, nullptr, "first"), second_output(second, nullptr, "second");
        batcher.admit("first", first, first_output); batcher.admit("second", second, second_output);
        std::vector<int64_t> a, b;
        bool first_done = false, second_done = false;
        int paused_ticks = 0;
        while (!first_done || !second_done) {
            const bool pause = !a.empty() && paused_ticks < 4 && !second_done;
            const auto prior = a.size();
            auto step = batcher.step(pause ? std::vector<std::string>{"second"} : std::vector<std::string>{"first", "second"});
            for (auto& event : step.events) {
                if (auto* failure = std::get_if<Failed>(&event.data)) throw std::runtime_error(failure->message);
                if (auto* delta = std::get_if<OutputDelta>(&event.data)) {
                    auto& tokens = event.id == "first" ? a : b;
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                }
                if (terminal(event.data)) (event.id == "first" ? first_done : second_done) = true;
            }
            if (pause) { ++paused_ticks; MFQ_RUNTIME_CHECK(a.size() == prior, "paused row advanced"); }
        }
        if (a != first_reference || b != second_reference) {
            for (const auto& [name, values] : std::vector<std::pair<const char*, std::vector<int64_t>>>{
                    {"first_serial", first_reference}, {"first_batch", a},
                    {"second_serial", second_reference}, {"second_batch", b}}) {
                std::cerr << name << ':';
                for (auto token : values) std::cerr << ' ' << token;
                std::cerr << '\n';
            }
        }
        MFQ_RUNTIME_CHECK(a == first_reference && b == second_reference && paused_ticks == 4,
            "batched or resumed output differs from serial oracle");
        InferenceOutput output(second, nullptr, "cancel");
        batcher.admit("cancel", second, output);
        while (!output.result.completion_tokens) (void)batcher.step({"cancel"});
        output.result.cancelled = true;
        auto cleanup = batcher.step({});
        MFQ_RUNTIME_CHECK(cleanup.events.size() == 1 && std::holds_alternative<Cancelled>(cleanup.events[0].data),
            "cancelled physical request did not release exactly once");
        for (const auto& [key, value] : batcher.metrics())
            if (key == "paged_kv_live_pages") MFQ_RUNTIME_CHECK(value == 0, "paged KV leaked");
    }
    MfqPromptCachePlan plan{"batch-check", first_prompt.size() - 1};
    MFQ_RUNTIME_CHECK(serial(first_prompt, plan) == first_reference &&
        serial(first_prompt, plan) == first_reference, "session restore differs from serial oracle");
    std::cout << "continuous_batching_check PASS requests=2 paused_ticks=4 cancellation=1 session_reuse=1\n";
    return 0;
}

int run_cuda_continuous_batching_check(
        mfq::cuda::Qwen35CausalLm& model) {
    return run_qwen_continuous_batching_check(
        model, resolve_cuda_runtime_config({}));
}

int run_cuda_engine_isolation_check(CudaEngineOptions options) {
    MFQ_RUNTIME_CHECK(!options.model_path.empty(),
        "engine isolation check requires --model");
    if (options.context_size == 0) options.context_size = 64;
    options.continuous_batching = 0;

    std::unique_ptr<CudaEngine> first;
    std::unique_ptr<CudaEngine> second;
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::atomic<int> load_ready{0};
    auto load = [&](std::unique_ptr<CudaEngine>& engine,
                    std::exception_ptr& error) {
        try {
            load_ready.fetch_add(1, std::memory_order_release);
            while (load_ready.load(std::memory_order_acquire) != 2) {
                std::this_thread::yield();
            }
            engine = std::make_unique<CudaEngine>(load_cuda_engine(options));
        } catch (...) {
            error = std::current_exception();
        }
    };
    std::thread first_load(load, std::ref(first), std::ref(first_error));
    std::thread second_load(load, std::ref(second), std::ref(second_error));
    first_load.join();
    second_load.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);

    MfqSamplingParams sampling;
    sampling.max_tokens = 2;
    sampling.temperature = 0.0;
    sampling.top_k = 1;
    sampling.top_p = 1.0;
    sampling.enable_mtp = false;
    const std::vector<int64_t> first_prompt{101, 138, 175, 212, 249};
    const std::vector<int64_t> second_prompt{113, 166, 219, 272, 325, 378};
    const auto serial = [&](CudaEngine& engine,
                            const std::vector<int64_t>& prompt) {
        std::vector<int64_t> output;
        output = check_engine_steps(engine, prompt, sampling);
        const auto produced = output.size();
        MFQ_RUNTIME_CHECK(
            produced == sampling.max_tokens &&
            output.size() == static_cast<size_t>(sampling.max_tokens),
            "CUDA engine isolation oracle failed");
        return output;
    };
    const auto first_reference = serial(*first, first_prompt);
    const auto second_reference = serial(*second, second_prompt);
    std::vector<int64_t> first_output;
    std::vector<int64_t> second_output;
    int32_t first_produced = 0;
    int32_t second_produced = 0;
    std::atomic<int> ready{0};
    auto generate = [&](CudaEngine& engine,
                        const std::vector<int64_t>& prompt,
                        std::vector<int64_t>& output,
                        int32_t& produced,
                        std::exception_ptr& error) {
        try {
            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) != 2) {
                std::this_thread::yield();
            }
            output = check_engine_steps(engine, prompt, sampling);
            produced = static_cast<int32_t>(output.size());
        } catch (...) {
            error = std::current_exception();
        }
    };
    first_error = nullptr;
    second_error = nullptr;
    std::thread first_generate(
        generate, std::ref(*first), std::cref(first_prompt),
        std::ref(first_output), std::ref(first_produced),
        std::ref(first_error));
    std::thread second_generate(
        generate, std::ref(*second), std::cref(second_prompt),
        std::ref(second_output), std::ref(second_produced),
        std::ref(second_error));
    first_generate.join();
    second_generate.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);

    MFQ_RUNTIME_CHECK(
        first_produced == sampling.max_tokens &&
        second_produced == sampling.max_tokens &&
        first_output == first_reference &&
        second_output == second_reference &&
        first->metadata.model_type == second->metadata.model_type &&
        first->metadata.max_context == options.context_size &&
        second->metadata.max_context == options.context_size &&
        !std::get<mfq::engine::Metrics>(first->control(mfq::engine::RuntimeMetrics{})).empty() &&
        !std::get<mfq::engine::Metrics>(second->control(mfq::engine::RuntimeMetrics{})).empty(),
        "concurrent CUDA engines did not remain isolated");
    std::cout << "cuda_engine_isolation_check PASS engines=2 generated="
              << first_produced + second_produced
              << " context=" << options.context_size << '\n';
    return 0;
}

} // namespace mfq::cuda::diagnostics
