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
#include <mutex>
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


int run_qwen_continuous_batching_check(
        Qwen35CausalLm& model,
        const CudaRuntimeConfig& runtime_config) {
    auto check_config = runtime_config;
    check_config.generation.prefill_chunk_size = 64;
    check_config.continuous_batch.scheduling.max_sequences = 4;
    check_config.continuous_batch.scheduling.initial_batch_wait =
        std::chrono::milliseconds(100);
    auto& execution = *model.execution;
    MFQ_RUNTIME_CHECK(model.vocab_size() > 1024 &&
        model.max_position_embeddings() >= 208,
        "continuous batching check requires vocab>1024 and context>=208");

    MfqSamplingParams first_params;
    first_params.max_tokens = 20;
    first_params.temperature = 0.0;
    first_params.top_k = 1;
    first_params.top_p = 1.0;
    first_params.enable_mtp = false;
    first_params.seed = 20260907;
    auto second_params = first_params;
    second_params.max_tokens = 18;
    second_params.seed += 1;
    std::vector<int64_t> first_prompt(193);
    std::vector<int64_t> second_prompt(17);
    for (size_t index = 0; index < first_prompt.size(); ++index) {
        first_prompt[index] = 101 +
            static_cast<int64_t>((index * 37) % 900);
    }
    for (size_t index = 0; index < second_prompt.size(); ++index) {
        second_prompt[index] = 113 +
            static_cast<int64_t>((index * 53) % 880);
    }
    auto serial = [&](const std::vector<int64_t> & prompt,
                      const MfqSamplingParams & params) {
        std::vector<int64_t> output;
        std::mutex mutex;
        DecodeGraphCache graph_cache(
            model.max_position_embeddings());
        TextSessionCache session_cache(
            check_config.session_cache, check_config.prefix_cache);
        const int32_t produced = generate(
            model, mutex, graph_cache, session_cache, check_config,
            prompt, params,
            [&](int64_t token) {
                output.push_back(token);
                return true;
            }, {}, {}, {}, nullptr);
        MFQ_RUNTIME_CHECK(
            produced == params.max_tokens &&
                output.size() == static_cast<size_t>(produced),
            "continuous batching serial oracle length mismatch");
        return output;
    };
    const auto first_reference = serial(first_prompt, first_params);
    const auto second_reference = serial(second_prompt, second_params);
    model.reset(1);

    std::mutex model_mutex;
    DecodeGraphCache batch_graph(model.max_position_embeddings());
    TextSessionCache batch_sessions(
        check_config.session_cache, check_config.prefix_cache);
    auto exclusive_generation = [
            &model, &batch_graph, &batch_sessions, &check_config](
            const std::vector<int64_t>& prompt,
            const MfqMultimodalInput* media,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& on_token,
            const MfqPrefillCallback& on_prefill,
            const MfqPromptCachePlan& cache_plan,
            const MfqTokenConstraintPtr& token_constraint,
            const MfqCancellationCheck& cancelled) {
        MFQ_RUNTIME_CHECK(
            media == nullptr && !sampling.enable_mtp,
            "continuous batching check received an unsupported special request");
        std::mutex already_locked_model;
        return generate(
            model, already_locked_model, batch_graph, batch_sessions,
            check_config, prompt, sampling, on_token, on_prefill,
            cache_plan, token_constraint, nullptr, {}, cancelled);
    };
    mfq::cuda::QwenBatchExecutor batcher(
        model, execution, model_mutex, check_config.continuous_batch,
        check_config.generation.prefill_chunk_size,
        std::move(exclusive_generation));
    std::mutex gate_mutex;
    std::condition_variable gate_ready;
    bool first_prefilled = false;
    bool second_delivered = false;
    bool release_first = false;
    std::vector<int64_t> first_output;
    std::vector<int64_t> second_output;
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    int32_t first_produced = 0;
    int32_t second_produced = 0;

    std::thread first_thread([&] {
        try {
            first_produced = batcher.submit(
                first_prompt, first_params,
                [&](int64_t token) {
                    first_output.push_back(token);
                    if (first_output.size() == 1) {
                        std::unique_lock<std::mutex> lock(gate_mutex);
                        first_prefilled = true;
                        gate_ready.notify_one();
                        gate_ready.wait(lock, [&] { return release_first; });
                    }
                    return true;
                }, {}, {}, {}, {});
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    bool first_queued = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
        if (batcher.queued_requests() > 0) {
            first_queued = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::thread second_thread([&] {
        try {
            second_produced = batcher.submit(
                second_prompt, second_params,
                [&](int64_t token) {
                    second_output.push_back(token);
                    if (second_output.size() == 1) {
                        std::lock_guard<std::mutex> lock(gate_mutex);
                        second_delivered = true;
                        gate_ready.notify_one();
                    }
                    return true;
                }, {}, {}, {}, {});
        } catch (...) {
            second_error = std::current_exception();
        }
    });
    bool first_callback_started = false;
    bool callback_isolated = false;
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        first_callback_started = gate_ready.wait_for(
            lock, std::chrono::seconds(10), [&] {
                return first_prefilled;
            });
        callback_isolated = first_callback_started && gate_ready.wait_for(
            lock, std::chrono::seconds(10), [&] {
                return second_delivered;
            });
        release_first = true;
    }
    gate_ready.notify_one();
    first_thread.join();
    second_thread.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);
    MFQ_RUNTIME_CHECK(first_queued && first_callback_started &&
        callback_isolated,
        "a blocked response callback stalled the scheduler");
    MFQ_RUNTIME_CHECK(first_produced == first_params.max_tokens &&
        second_produced == second_params.max_tokens,
        "continuous batching generated token count mismatch");
    const auto print_mismatch = [](const char * name,
            const std::vector<int64_t> & reference,
            const std::vector<int64_t> & actual) {
        if (reference == actual) return;
        std::cerr << "continuous_batching_check mismatch " << name << " reference=";
        for (auto token : reference) std::cerr << token << ',';
        std::cerr << " actual=";
        for (auto token : actual) std::cerr << token << ',';
        std::cerr << '\n';
    };
    print_mismatch("first", first_reference, first_output);
    print_mismatch("second", second_reference, second_output);
    MFQ_RUNTIME_CHECK(first_output == first_reference,
        "continuous batching first request differs from serial greedy oracle");
    MFQ_RUNTIME_CHECK(second_output == second_reference,
        "continuous batching second request differs from serial greedy oracle");
    auto cancel_params = second_params;
    cancel_params.max_tokens = 12;
    int32_t cancellation_callbacks = 0;
    const int32_t cancellation_produced = batcher.submit(
        second_prompt, cancel_params,
        [&](int64_t) {
            ++cancellation_callbacks;
            return false;
        }, {}, {}, {}, {});
    MFQ_RUNTIME_CHECK(cancellation_produced == 1 &&
        cancellation_callbacks == 1,
        "continuous batching callback cancellation did not stop at one token");

    auto prefix_params = first_params;
    prefix_params.max_tokens = 2;
    const auto prefix_reference = serial(first_prompt, prefix_params);
    model.reset(1);
    MfqPromptCachePlan prefix_plan{
        "continuous-batch-check", first_prompt.size() - 1};
    std::vector<int64_t> first_cached;
    std::vector<int64_t> second_cached;
    const auto first_cached_count = batcher.submit(
        first_prompt, prefix_params,
        [&](int64_t token) {
            first_cached.push_back(token);
            return true;
        }, {}, prefix_plan, {}, {});
    const auto second_cached_count = batcher.submit(
        first_prompt, prefix_params,
        [&](int64_t token) {
            second_cached.push_back(token);
            return true;
        }, {}, prefix_plan, {}, {});
    MFQ_RUNTIME_CHECK(
        first_cached_count == prefix_params.max_tokens &&
        second_cached_count == prefix_params.max_tokens &&
        first_cached == prefix_reference &&
        second_cached == prefix_reference,
        "continuous batching prefix reuse differs from serial oracle");

    const auto values = batcher.metrics();
    auto metric = [&](const std::string & name) {
        const auto found = std::find_if(
            values.begin(), values.end(), [&](const auto & item) {
                return item.first == name;
            });
        return found == values.end() ? 0.0 : found->second;
    };
    const auto cache_values = batch_sessions.metrics();
    const auto cache_metric = [&](const std::string& name) {
        const auto found = std::find_if(
            cache_values.begin(), cache_values.end(), [&](const auto& item) {
                return item.first == name;
            });
        return found == cache_values.end() ? 0.0 : found->second;
    };
    std::cout << "continuous_batching_check metrics max_batch="
              << metric("continuous_batching_max_batch")
              << " stable_slot_releases="
              << metric("continuous_batching_stable_slot_releases")
              << " batched_greedy_batches="
              << metric("continuous_batching_batched_greedy_batches")
              << " packed_metadata_batches="
              << metric("continuous_batching_packed_metadata_batches")
              << " cuda_graph_captures="
              << metric("continuous_batching_cuda_graph_captures")
              << " cuda_graph_replays="
              << metric("continuous_batching_cuda_graph_replays")
              << " paged_kv="
              << metric("continuous_batching_paged_kv")
              << " page_size="
              << metric("paged_kv_page_size")
              << " live_pages="
              << metric("paged_kv_live_pages")
              << " peak_pages="
              << metric("paged_kv_peak_live_pages")
              << " capacity_pages="
              << metric("paged_kv_capacity_pages")
              << " page_allocations="
              << metric("paged_kv_page_allocations")
              << " page_reuses="
              << metric("paged_kv_page_reuses")
              << " page_releases="
              << metric("paged_kv_page_releases")
              << " active="
              << metric("continuous_batching_active")
              << " prefilling="
              << metric("continuous_batching_prefilling")
              << " prefill_chunks="
              << metric("continuous_batching_prefill_chunks")
              << " prefill_yields="
              << metric("continuous_batching_prefill_yields")
              << " queued="
              << metric("continuous_batching_queued") << '\n';
    MFQ_RUNTIME_CHECK(metric("continuous_batching_max_batch") >= 2.0 &&
        metric("continuous_batching_compactions") == 0.0 &&
        metric("continuous_batching_stable_slot_releases") >= 1.0 &&
        (!check_config.continuous_batch.greedy ||
            metric("continuous_batching_batched_greedy_batches") >= 1.0) &&
        metric("continuous_batching_packed_metadata_batches") >= 1.0 &&
        (!mfq::cuda::qwen_continuous_batch_cuda_graph_enabled(
                model, check_config.continuous_batch) ||
             (metric("continuous_batching_cuda_graph_captures") >= 1.0 &&
             metric("continuous_batching_cuda_graph_replays") >= 2.0)) &&
        (!check_config.continuous_batch.paged_kv ||
            (metric("continuous_batching_paged_kv") == 1.0 &&
             metric("paged_kv_page_size") ==
                static_cast<double>(batcher.paged_kv_page_size()) &&
             metric("paged_kv_live_pages") == 0.0 &&
             metric("paged_kv_peak_live_pages") > 0.0 &&
             metric("paged_kv_capacity_pages") >=
                metric("paged_kv_peak_live_pages") &&
             metric("paged_kv_page_allocations") ==
                metric("paged_kv_page_releases") &&
             metric("paged_kv_page_reuses") > 0.0)) &&
        metric("continuous_batching_prefill_chunks") >= 4.0 &&
        metric("continuous_batching_prefix_cache_exclusive_requests") == 2.0 &&
        cache_metric("prefix_cache_hits") >= 1.0 &&
        cache_metric("prefix_cache_hit_tokens") >=
            static_cast<double>(first_prompt.size() - 1) &&
        metric("continuous_batching_active") == 0.0 &&
        metric("continuous_batching_prefilling") == 0.0 &&
        metric("continuous_batching_queued") == 0.0,
        "continuous batching check did not exercise join and retire");
    std::cout << "continuous_batching_check PASS concurrent_requests=2"
              << " cancellation_tokens=1 prefix_cache_hits="
              << cache_metric("prefix_cache_hits")
              << " max_batch="
              << metric("continuous_batching_max_batch")
              << " prompt_lengths=193,17 split_k=1"
              << " decode_batches="
              << metric("continuous_batching_decode_batches")
              << " stable_slot_releases="
              << metric("continuous_batching_stable_slot_releases") << '\n';
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
        const auto produced = engine.generate(
            prompt, sampling,
            [&](int64_t token) {
                output.push_back(token);
                return true;
            }, {}, {}, {}, {});
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
            produced = engine.generate(
                prompt, sampling,
                [&](int64_t token) {
                    output.push_back(token);
                    return true;
                }, {}, {}, {}, {});
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
        !first->runtime_metrics().empty() &&
        !second->runtime_metrics().empty(),
        "concurrent CUDA engines did not remain isolated");
    std::cout << "cuda_engine_isolation_check PASS engines=2 generated="
              << first_produced + second_produced
              << " context=" << options.context_size << '\n';
    return 0;
}

} // namespace mfq::cuda::diagnostics
