#include "request_executor.h"
#include "runtime_checks.h"
#include <map>

#include "../engine/cuda_runtime_config.h"
#include "engine/generation.h"
#include "storage/text_session_cache.h"
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
    const auto greedy_sampling = sampling;
    sampling.temperature = .8; sampling.top_k = 32; sampling.top_p = .95;
    sampling.presence_penalty = .2; sampling.frequency_penalty = .1;
    sampling.repetition_penalty = 1.05; sampling.seed = 20261004;
    const auto stochastic_sampling = sampling;
    const auto stochastic_reference = serial(second_prompt);
    sampling = greedy_sampling;
    graph.invalidate(); // Physical batch slots replace the serial graph's storage.
    {
        struct Execution {
            Qwen35CausalLm& model;
            DecodeGraphCache& graph;
            TextSessionCache& cache;
            const CudaRuntimeConfig& config;
            ContinuousBatch<QwenBatchOperations> batcher;
            Execution(Qwen35CausalLm& model, DecodeGraphCache& graph, TextSessionCache& cache,
                      const CudaRuntimeConfig& config)
                : model(model), graph(graph), cache(cache), config(config),
                  batcher(64, config.continuous_batch.prefill_token_budget,
                          model, *model.execution, config.continuous_batch) {}
            bool can_batch(const EngineRequest&) const { return true; }
            bool exclusive() const { return false; }
            bool mtp_available() const { return false; }
            void execute(const std::vector<RequestId>& eligible) { batcher.step(eligible); }
            Generation generate(const RequestId& id, ExecutionRequest& request) {
                return internal::generate(model, graph, cache, config, request.input,
                                          request.output, nullptr, {}, &batcher, id);
            }
        } execution(model, graph, cache, config);
        auto& batcher = execution.batcher;
        RequestExecutor executor(4);
        EngineInfo info;
        info.max_requests = 4;
        info.vocab_size = model.vocab_size();
        info.max_context = model.max_position_embeddings();
        std::map<std::string, std::vector<int64_t>> tokens;
        std::map<std::string, int> terminals, cancellations, prefills;
        const auto admit = [&](const std::string& id, const std::vector<int64_t>& prompt,
                               const MfqSamplingParams* params = nullptr) {
            EngineRequest request;
            request.id = id; request.token_ids = prompt; request.input.sampling = params ? *params : sampling;
            MFQ_RUNTIME_CHECK(executor.admit(std::move(request), nullptr, info, execution) == Admission::accepted,
                              "diagnostic request was not admitted");
        };
        const auto tick = [&](std::vector<RequestId> eligible) {
            auto result = executor.step(eligible, execution);
            for (const auto& event : result.events) {
                if (const auto* error = std::get_if<Failed>(&event.data)) throw std::runtime_error(error->message);
                if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens[event.id].insert(tokens[event.id].end(), delta->token_ids.begin(), delta->token_ids.end());
                prefills[event.id] += std::holds_alternative<PrefillProgress>(event.data);
                cancellations[event.id] += std::holds_alternative<Cancelled>(event.data);
                terminals[event.id] += terminal(event.data);
            }
        };
        admit("first", first_prompt); admit("second", second_prompt);
        int paused_ticks = 0;
        for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks) {
            const bool pause = !tokens["first"].empty() && paused_ticks < 4 && !terminals["second"];
            const auto prior = tokens["first"].size();
            tick(pause ? std::vector<std::string>{"second"} : std::vector<std::string>{"first", "second"});
            if (pause) { ++paused_ticks; MFQ_RUNTIME_CHECK(tokens["first"].size() == prior, "paused row advanced"); }
        }
        MFQ_RUNTIME_CHECK(executor.empty() && tokens["first"] == first_reference &&
            tokens["second"] == second_reference && paused_ticks == 4 &&
            terminals["first"] == 1 && terminals["second"] == 1,
            "batched or resumed output differs from serial oracle");
        admit("cancel", second_prompt);
        for (int ticks = 0; ticks < 100 && tokens["cancel"].empty(); ++ticks) tick({"cancel"});
        MFQ_RUNTIME_CHECK(!tokens["cancel"].empty(), "cancel test never decoded");
        const auto before_cancel = tokens["cancel"].size();
        executor.cancel("cancel"); tick({}); tick({});
        MFQ_RUNTIME_CHECK(executor.empty() && cancellations["cancel"] == 1 && terminals["cancel"] == 1 &&
            tokens["cancel"].size() == before_cancel, "cancelled request published more output");
        // Cancel a partially prefetched request while another row owns a slot.
        admit("survivor", second_prompt);
        for (int ticks = 0; ticks < 100 && tokens["survivor"].empty(); ++ticks) tick({"survivor"});
        admit("partial", first_prompt);
        for (int ticks = 0; ticks < 100 && !prefills["partial"]; ++ticks) tick({"partial"});
        MFQ_RUNTIME_CHECK(prefills["partial"] == 1 && tokens["partial"].empty(), "partial prefill did not yield");
        executor.cancel("partial");
        for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks) tick({"survivor"});
        MFQ_RUNTIME_CHECK(executor.empty() && cancellations["partial"] == 1 && terminals["partial"] == 1 &&
            tokens["partial"].empty() && terminals["survivor"] == 1 && tokens["survivor"] == second_reference,
            "prefill cancellation corrupted a live slot");
        const auto metric = [&](const char* name) {
            for (const auto& [key, value] : batcher.metrics()) if (key == name) return value;
            throw std::runtime_error("missing batching metric");
        };
        admit("anchor", second_prompt);
        for (int ticks = 0; ticks < 100 && tokens["anchor"].size() < 4; ++ticks) tick({"anchor"});
        const auto captures_before_join = metric("continuous_batching_cuda_graph_captures");
        admit("joined", second_prompt);
        for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks) tick({"anchor", "joined"});
        MFQ_RUNTIME_CHECK(executor.empty() && tokens["anchor"] == second_reference &&
            tokens["joined"] == second_reference, "prefill join changed decode output");
        if (config.continuous_batch.greedy &&
            qwen_continuous_batch_cuda_graph_enabled(model, config.continuous_batch))
            MFQ_RUNTIME_CHECK(metric("continuous_batching_cuda_graph_captures") <= captures_before_join + 1,
                "joining prefill repeatedly recaptured decode graphs");
        admit("padded-0", second_prompt); admit("padded-1", second_prompt); admit("padded-2", second_prompt);
        for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks) {
            const bool pause = tokens["padded-0"].size() >= 3 && tokens["padded-1"].size() < 8;
            tick(pause ? std::vector<RequestId>{"padded-1", "padded-2"} :
                std::vector<RequestId>{"padded-0", "padded-1", "padded-2"});
        }
        for (int row = 0; row < 3; ++row)
            MFQ_RUNTIME_CHECK(tokens["padded-" + std::to_string(row)] == second_reference &&
                terminals["padded-" + std::to_string(row)] == 1, "padded batch or paused state differs");
        const auto reads_before = metric("continuous_batching_sampling_readbacks");
        const auto batches_before = metric("continuous_batching_decode_batches");
        const auto started = std::chrono::steady_clock::now();
        for (int run = 0; run < 2; ++run) {
            const auto random_id = "random-" + std::to_string(run);
            const auto greedy_id = "greedy-" + std::to_string(run);
            admit(random_id, second_prompt, &stochastic_sampling);
            admit(greedy_id, first_prompt);
            for (int ticks = 0; ticks < 500 && !executor.empty(); ++ticks)
                tick({random_id, greedy_id});
            MFQ_RUNTIME_CHECK(executor.empty() && tokens[random_id].size() == stochastic_reference.size() &&
                (!run || tokens[random_id] == tokens["random-0"]) &&
                tokens[greedy_id] == first_reference && terminals[random_id] == 1 && terminals[greedy_id] == 1,
                "mixed batch sampling or fixed-seed replay failed");
        }
        MFQ_RUNTIME_CHECK(metric("continuous_batching_physical_decode_rows") <=
            2 * metric("continuous_batching_decode_tokens"), "physical batch exceeded its eligible bucket");
        const auto reads = metric("continuous_batching_sampling_readbacks") - reads_before;
        const auto batches = metric("continuous_batching_decode_batches") - batches_before;
        MFQ_RUNTIME_CHECK(reads == batches, "sampling synchronized more than once per unconstrained batch");
        std::cout << "continuous_batching_sampling_check readbacks=" << reads << " batches=" << batches
            << " mixed_seed_replay=1 tokens_per_sec=" << 4 * sampling.max_tokens /
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
        double captures = 0, replays = 0;
        for (const auto& [key, value] : batcher.metrics()) {
            if (key.starts_with("continuous_batching_graph_b"))
                std::cout << key << "=" << value << '\n';
            if (key == "paged_kv_live_pages") MFQ_RUNTIME_CHECK(value == 0, "paged KV leaked");
            if (key == "continuous_batching_active" || key == "continuous_batching_prefilling")
                MFQ_RUNTIME_CHECK(value == 0, "batch retained finished requests");
            if (key == "continuous_batching_cuda_graph_captures") captures = value;
            if (key == "continuous_batching_cuda_graph_replays") replays = value;
        }
        if (config.continuous_batch.greedy &&
            qwen_continuous_batch_cuda_graph_enabled(model, config.continuous_batch))
            MFQ_RUNTIME_CHECK(captures > 0 && replays > captures, "batch graph was not replayed");
        std::cout << "continuous_batching_state_check paged=" << config.continuous_batch.paged_kv
            << " captures=" << captures << " replays=" << replays
            << " prefill_cancel=1 slot_reuse=1\n";
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

    std::unique_ptr<mfq::engine::Engine> first;
    std::unique_ptr<mfq::engine::Engine> second;
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::atomic<int> load_ready{0};
    auto load = [&](std::unique_ptr<mfq::engine::Engine>& engine,
                    std::exception_ptr& error) {
        try {
            load_ready.fetch_add(1, std::memory_order_release);
            while (load_ready.load(std::memory_order_acquire) != 2) {
                std::this_thread::yield();
            }
            engine = load_cuda_engine(options);
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
    const auto serial = [&](mfq::engine::Engine& engine,
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
    auto generate = [&](mfq::engine::Engine& engine,
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
        first->info().model_type == second->info().model_type &&
        first->info().max_context == options.context_size &&
        second->info().max_context == options.context_size &&
        !std::get<mfq::engine::Metrics>(first->control(mfq::engine::RuntimeMetrics{})).empty() &&
        !std::get<mfq::engine::Metrics>(second->control(mfq::engine::RuntimeMetrics{})).empty(),
        "concurrent CUDA engines did not remain isolated");
    std::cout << "cuda_engine_isolation_check PASS engines=2 generated="
              << first_produced + second_produced
              << " context=" << options.context_size << '\n';
    return 0;
}

} // namespace mfq::cuda::diagnostics
