#pragma once

#include "qwen_continuous_workload.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define MFQ_BENCH_CHECK(condition, message) do { \
    if (!(condition)) throw std::runtime_error(message); \
} while (false)

namespace mfq::cuda::continuous {

struct QwenContinuousBatchingBenchmarkResult {
    std::vector<double> baseline_itl_ms;
    std::vector<double> contended_itl_ms;
    double b_ttft_ms = 0.0;
    double prefill_gpu_ms = 0.0;
    double prefill_wall_ms = 0.0;
    double decode_tokens = 0.0;
    double decode_ms = 0.0;
    std::map<std::string, double> metric_deltas;
};

static double qwen_continuous_benchmark_ms(
        std::chrono::steady_clock::time_point first,
        std::chrono::steady_clock::time_point last) {
    return std::chrono::duration<double, std::milli>(last - first).count();
}

static double qwen_continuous_benchmark_percentile(
        std::vector<double> values, double percentile) {
    MFQ_BENCH_CHECK(!values.empty(),
        "continuous batching benchmark has no percentile samples");
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<size_t>(std::ceil(
        percentile * static_cast<double>(values.size())));
    return values[std::max<size_t>(1, rank) - 1];
}

static std::map<std::string, double> qwen_continuous_benchmark_metrics(
        const mfq::cuda::QwenContinuousWorkload & batcher) {
    std::map<std::string, double> result;
    for (const auto & [name, value] : batcher.metrics()) {
        result.emplace(name, value);
    }
    return result;
}

static double qwen_continuous_benchmark_metric(
        const std::map<std::string, double> & values,
        const std::string & name) {
    const auto found = values.find(name);
    return found == values.end() ? 0.0 : found->second;
}

static QwenContinuousBatchingBenchmarkResult
run_qwen_continuous_batching_benchmark_once(
        mfq::cuda::QwenContinuousWorkload & batcher,
        const std::vector<int64_t> & a_prompt,
        const std::vector<int64_t> & b_prompt,
        const MfqSamplingParams & a_params,
        const MfqSamplingParams & b_params,
        const std::vector<int64_t> & a_reference,
        const std::vector<int64_t> & b_reference,
        int baseline_tokens) {
    using Clock = std::chrono::steady_clock;

    std::mutex state_mutex;
    std::condition_variable state_ready;
    std::vector<Clock::time_point> a_timestamps;
    std::vector<int64_t> a_output;
    std::vector<int64_t> b_output;
    std::optional<Clock::time_point> b_prefill_callback;
    std::optional<Clock::time_point> b_first_token;
    std::optional<MfqPrefillTiming> b_prefill_timing;
    Clock::time_point b_submit_started;
    Clock::time_point a_finished;
    std::map<std::string, double> metrics_before;
    std::map<std::string, double> metrics_after;
    std::exception_ptr a_error;
    std::exception_ptr b_error;
    bool a_done = false;

    std::thread a_thread([&] {
        try {
            (void)batcher.submit(
                a_prompt, a_params,
                [&](int64_t token) {
                    const auto now = Clock::now();
                    {
                        std::lock_guard<std::mutex> lock(state_mutex);
                        a_output.push_back(token);
                        a_timestamps.push_back(now);
                    }
                    state_ready.notify_one();
                    return true;
                }, {});
        } catch (...) {
            a_error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            a_finished = Clock::now();
            a_done = true;
        }
        state_ready.notify_one();
    });

    {
        std::unique_lock<std::mutex> lock(state_mutex);
        const bool ready = state_ready.wait_for(
            lock, std::chrono::seconds(60), [&] {
                return a_error || a_done ||
                    a_timestamps.size() >=
                        static_cast<size_t>(baseline_tokens + 3);
            });
        if (!ready || a_error || a_done) {
            lock.unlock();
            a_thread.join();
            if (a_error) std::rethrow_exception(a_error);
            throw std::runtime_error(
                "continuous batching benchmark request A did not reach stable decode");
        }
    }

    std::thread b_thread([&] {
        try {
            metrics_before = qwen_continuous_benchmark_metrics(batcher);
            b_submit_started = Clock::now();
            (void)batcher.submit(
                b_prompt, b_params,
                [&](int64_t token) {
                    const auto now = Clock::now();
                    {
                        std::lock_guard<std::mutex> lock(state_mutex);
                        b_output.push_back(token);
                        if (!b_first_token.has_value()) b_first_token = now;
                    }
                    return true;
                },
                [&](const MfqPrefillTiming & timing) {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    b_prefill_timing = timing;
                    b_prefill_callback = Clock::now();
                });
            metrics_after = qwen_continuous_benchmark_metrics(batcher);
        } catch (...) {
            b_error = std::current_exception();
        }
    });

    b_thread.join();
    a_thread.join();
    if (a_error) std::rethrow_exception(a_error);
    if (b_error) std::rethrow_exception(b_error);
    const auto metrics_final = qwen_continuous_benchmark_metrics(batcher);
    MFQ_BENCH_CHECK(a_output == a_reference && b_output == b_reference,
        "continuous batching benchmark output differs from serial greedy oracle");
    MFQ_BENCH_CHECK(
        b_prefill_callback.has_value() && b_first_token.has_value() &&
            b_prefill_timing.has_value() && b_submit_started < a_finished,
        "continuous batching benchmark did not overlap prefill with active decode");

    QwenContinuousBatchingBenchmarkResult result;
    const size_t stable_begin = 2;
    std::vector<double> available_baseline;
    for (size_t index = stable_begin + 1;
            index < a_timestamps.size(); ++index) {
        const auto first = a_timestamps[index - 1];
        const auto last = a_timestamps[index];
        const double milliseconds = qwen_continuous_benchmark_ms(first, last);
        if (last <= b_submit_started) available_baseline.push_back(milliseconds);
        if (last > b_submit_started && first < *b_prefill_callback) {
            result.contended_itl_ms.push_back(milliseconds);
        }
    }
    const size_t baseline_count = std::min(
        available_baseline.size(), static_cast<size_t>(baseline_tokens));
    result.baseline_itl_ms.insert(
        result.baseline_itl_ms.end(),
        available_baseline.end() - baseline_count,
        available_baseline.end());
    MFQ_BENCH_CHECK(
        result.baseline_itl_ms.size() == static_cast<size_t>(baseline_tokens) &&
            !result.contended_itl_ms.empty(),
        "continuous batching benchmark did not collect both ITL windows");

    result.b_ttft_ms = qwen_continuous_benchmark_ms(
        b_submit_started, *b_first_token);
    result.prefill_gpu_ms = b_prefill_timing->llm_ms;
    result.prefill_wall_ms = qwen_continuous_benchmark_ms(
        b_submit_started, *b_prefill_callback);
    result.decode_tokens = static_cast<double>(a_timestamps.size() - 1);
    result.decode_ms = qwen_continuous_benchmark_ms(
        a_timestamps.front(), a_timestamps.back());
    MFQ_BENCH_CHECK(
        std::isfinite(result.b_ttft_ms) && result.b_ttft_ms > 0.0 &&
            std::isfinite(result.prefill_gpu_ms) &&
            result.prefill_gpu_ms > 0.0 &&
            std::isfinite(result.prefill_wall_ms) &&
            result.prefill_wall_ms > 0.0 &&
            std::isfinite(result.decode_ms) && result.decode_ms > 0.0,
        "continuous batching benchmark produced invalid timing values");

    for (const auto & [name, value] : metrics_after) {
        result.metric_deltas[name] = value -
            qwen_continuous_benchmark_metric(metrics_before, name);
    }
    MFQ_BENCH_CHECK(
        qwen_continuous_benchmark_metric(
            result.metric_deltas,
            "continuous_batching_interleaved_admissions") >= 1.0 &&
        qwen_continuous_benchmark_metric(
            result.metric_deltas,
            "continuous_batching_prefill_yields") >= 1.0,
        "continuous batching benchmark did not exercise contended prefill");
    MFQ_BENCH_CHECK(
        qwen_continuous_benchmark_metric(
            metrics_final, "continuous_batching_active") == 0.0 &&
        qwen_continuous_benchmark_metric(
            metrics_final, "continuous_batching_prefilling") == 0.0 &&
        qwen_continuous_benchmark_metric(
            metrics_final, "continuous_batching_queued") == 0.0,
        "continuous batching benchmark left scheduler work pending");
    if (qwen_continuous_benchmark_metric(
            metrics_final, "continuous_batching_paged_kv") != 0.0) {
        MFQ_BENCH_CHECK(qwen_continuous_benchmark_metric(
            metrics_final, "paged_kv_live_pages") == 0.0,
            "continuous batching benchmark leaked Paged KV pages");
    }
    return result;
}

static int run_qwen_continuous_batching_benchmark(
        mfq::cuda::QwenContinuousWorkload & model,
        int64_t prefill_chunk_size,
        int generated_tokens,
        int64_t long_prefill_tokens,
        int baseline_tokens,
        int repetitions) {
    MFQ_BENCH_CHECK(
        prefill_chunk_size > 0 && long_prefill_tokens > prefill_chunk_size &&
            baseline_tokens >= 2 && repetitions > 0 && repetitions <= 100,
        "continuous batching benchmark requires a split prefill, baseline>=2, and reps1-100");
    const int64_t contended_chunks =
        (long_prefill_tokens + prefill_chunk_size - 1) / prefill_chunk_size;
    MFQ_BENCH_CHECK(
        generated_tokens >= baseline_tokens + contended_chunks + 4 &&
            model.vocab_size() > 1024 &&
            model.max_position_embeddings() >= long_prefill_tokens + 1 &&
            model.max_position_embeddings() >= generated_tokens + 17,
        "continuous batching benchmark requires more decode tokens, vocabulary, or context");

    std::vector<int64_t> a_prompt(17);
    std::vector<int64_t> b_prompt(static_cast<size_t>(long_prefill_tokens));
    for (size_t index = 0; index < a_prompt.size(); ++index) {
        a_prompt[index] = 101 + static_cast<int64_t>((index * 37) % 900);
    }
    for (size_t index = 0; index < b_prompt.size(); ++index) {
        b_prompt[index] = 113 + static_cast<int64_t>((index * 53) % 880);
    }

    MfqSamplingParams a_params;
    a_params.max_tokens = generated_tokens;
    a_params.temperature = 0.0;
    a_params.top_k = 1;
    a_params.top_p = 1.0;
    a_params.enable_mtp = false;
    a_params.seed = 20260907;
    auto b_params = a_params;
    b_params.max_tokens = 1;
    b_params.seed += 1;

    auto serial = [&](const std::vector<int64_t> & prompt,
                      const MfqSamplingParams & params) {
        return model.serial_generate(prompt, params);
    };
    const auto a_reference = serial(a_prompt, a_params);
    const auto b_reference = serial(b_prompt, b_params);
    model.start_batcher();
    auto & batcher = model;
    (void)run_qwen_continuous_batching_benchmark_once(
        batcher, a_prompt, b_prompt, a_params, b_params,
        a_reference, b_reference, baseline_tokens);

    QwenContinuousBatchingBenchmarkResult aggregate;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        auto sample = run_qwen_continuous_batching_benchmark_once(
            batcher, a_prompt, b_prompt, a_params, b_params,
            a_reference, b_reference, baseline_tokens);
        aggregate.baseline_itl_ms.insert(
            aggregate.baseline_itl_ms.end(),
            sample.baseline_itl_ms.begin(), sample.baseline_itl_ms.end());
        aggregate.contended_itl_ms.insert(
            aggregate.contended_itl_ms.end(),
            sample.contended_itl_ms.begin(), sample.contended_itl_ms.end());
        aggregate.b_ttft_ms += sample.b_ttft_ms;
        aggregate.prefill_gpu_ms += sample.prefill_gpu_ms;
        aggregate.prefill_wall_ms += sample.prefill_wall_ms;
        aggregate.decode_tokens += sample.decode_tokens;
        aggregate.decode_ms += sample.decode_ms;
        for (const auto & [name, value] : sample.metric_deltas) {
            aggregate.metric_deltas[name] += value;
        }
    }

    const double divisor = static_cast<double>(repetitions);
    const double prompt_tokens = static_cast<double>(long_prefill_tokens);
    const auto delta = [&](const std::string & name) {
        return qwen_continuous_benchmark_metric(
            aggregate.metric_deltas, name);
    };
    // B stops at its first token, so this is a single-active-decode workload.
    // CUDA Graph counters are diagnostic and are normally zero; this benchmark
    // does not measure batched-decode graph recapture cost.
    std::cout << std::fixed << std::setprecision(3)
        << "continuous_batching_contention_bench"
        << " chunk=" << prefill_chunk_size
        << " reps=" << repetitions
        << " active_decode_batch=1"
        << " a_prompt_tokens=" << a_prompt.size()
        << " b_prompt_tokens=" << b_prompt.size()
        << " baseline_samples=" << aggregate.baseline_itl_ms.size()
        << " contended_samples=" << aggregate.contended_itl_ms.size()
        << " baseline_p50_itl_ms=" << qwen_continuous_benchmark_percentile(
            aggregate.baseline_itl_ms, 0.50)
        << " baseline_p95_itl_ms=" << qwen_continuous_benchmark_percentile(
            aggregate.baseline_itl_ms, 0.95)
        << " baseline_p99_itl_ms=" << qwen_continuous_benchmark_percentile(
            aggregate.baseline_itl_ms, 0.99)
        << " baseline_max_itl_ms=" << *std::max_element(
            aggregate.baseline_itl_ms.begin(), aggregate.baseline_itl_ms.end())
        << " contended_p50_itl_ms=" << qwen_continuous_benchmark_percentile(
            aggregate.contended_itl_ms, 0.50)
        << " contended_p95_itl_ms=" << qwen_continuous_benchmark_percentile(
            aggregate.contended_itl_ms, 0.95)
        << " contended_p99_itl_ms=" << qwen_continuous_benchmark_percentile(
            aggregate.contended_itl_ms, 0.99)
        << " contended_max_itl_ms=" << *std::max_element(
            aggregate.contended_itl_ms.begin(), aggregate.contended_itl_ms.end())
        << " b_ttft_ms=" << aggregate.b_ttft_ms / divisor
        << " prefill_gpu_ms=" << aggregate.prefill_gpu_ms / divisor
        << " prefill_wall_ms=" << aggregate.prefill_wall_ms / divisor
        << " prefill_tps=" << prompt_tokens * 1000.0 /
            (aggregate.prefill_gpu_ms / divisor)
        << " prefill_effective_tps=" << prompt_tokens * 1000.0 /
            (aggregate.prefill_wall_ms / divisor)
        << " decode_tps=" << aggregate.decode_tokens * 1000.0 /
            aggregate.decode_ms
        << " prefill_chunks=" << delta(
            "continuous_batching_prefill_chunks")
        << " prefill_yields=" << delta(
            "continuous_batching_prefill_yields")
        << " interleaved_admissions=" << delta(
            "continuous_batching_interleaved_admissions")
        << " cuda_graph_captures=" << delta(
            "continuous_batching_cuda_graph_captures")
        << " cuda_graph_replays=" << delta(
            "continuous_batching_cuda_graph_replays")
        << " paged_kv_page_table_updates=" << delta(
            "paged_kv_table_updates")
        << " exact=1\n";
    return 0;
}

} // namespace mfq::cuda::continuous

#undef MFQ_BENCH_CHECK
