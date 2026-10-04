#include "../ops/cuda_execution.h"
#include "mfq/cuda/engine.h"

#include <atomic>
#include <barrier>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

static void check_native_resources() {
    using namespace mfq::cuda;
    CudaExecutionContext engines[2];
    std::shared_ptr<Context> contexts[2];
    std::barrier rendezvous(2);
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int i = 0; i < 2; ++i) workers[i] = std::thread([&, i] {
        try {
            CudaExecutionScope scope(engines[i]);
            contexts[i] = default_context(0);
            auto options = TensorOptions().device(kCUDA).dtype(kFloat32);
            auto input = full({32, 32}, double(i + 1), options);
            auto weight = full({32, 32}, 1.0 / 32.0, options);
            auto own_stream = stream_from_pool(false, 0);
            StreamGuard stream(own_stream);
            // Explicitly fence initialization onto this Engine's graph stream.
            engines[i].native_contexts.at(0)->stream().synchronize();
            rendezvous.arrive_and_wait();
            for (int repeat = 0; repeat < 8; ++repeat) {
                auto output = matmul(input, weight);
                if (output.sum().item<float>() != 1024.0f * (i + 1))
                    throw std::runtime_error("concurrent eager matmul changed output");
            }
            Graph graph;
            graph.prepare_memory();
            Tensor output;
            for (int warmup = 0; warmup < 2; ++warmup) { output = matmul(input, weight); output = {}; }
            graph.capture_begin();
            output = matmul(input, weight);
            graph.capture_end();
            for (int replay = 0; replay < 8; ++replay) graph.replay();
            if (output.sum().item<float>() != 1024.0f * (i + 1))
                throw std::runtime_error("concurrent graph matmul changed output");
        } catch (...) { errors[i] = std::current_exception(); rendezvous.arrive_and_drop(); }
    });
    for (auto& worker : workers) worker.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    if (contexts[0] == contexts[1] || contexts[0]->stream().get() == contexts[1]->stream().get() ||
        contexts[0]->blas().get() == contexts[1]->blas().get() ||
        contexts[0]->memory_pool() == contexts[1]->memory_pool())
        throw std::runtime_error("Engine native resources are shared");
    {
        CudaExecutionScope first(engines[0]);
        auto stream = current_stream();
        { CudaExecutionScope second(engines[1]);
          if (current_stream().stream() == stream.stream()) throw std::runtime_error("nested Engine stream shared"); }
        if (current_stream().stream() != stream.stream()) throw std::runtime_error("Engine scope did not restore stream");
    }
    contexts[0].reset();
    engines[0].native_contexts.clear();
    CudaExecutionScope survivor(engines[1]);
    auto value = full({4}, 3.0, TensorOptions().device(kCUDA));
    if (value.sum().item<float>() != 12.0f) throw std::runtime_error("peer Engine teardown changed survivor");
}

static void check_memory_envelope() {
    CudaExecutionContext execution;
    CudaExecutionScope scope(execution);
    auto context = mfq::cuda::default_context(0);
    context->reserve_execution_memory(4096);
    auto* first = context->allocate(3072);
    bool rejected = false;
    try { (void)context->allocate(2048); } catch (const std::bad_alloc&) { rejected = true; }
    if (!rejected) throw std::runtime_error("native allocator exceeded reserved memory");
    context->release(first, 3072);
    auto* second = context->allocate(4096);
    context->release(second, 4096);
    context->trim();
}

static void check_engines(const char* model, const char* tokenizer) {
    mfq::cuda::CudaEngineOptions options;
    options.model_path = model;
    options.tokenizer_model = tokenizer;
    options.context_size = 128;
    options.prefill_chunk_size = 8;
    auto first = mfq::cuda::load_cuda_engine(options);
    auto second = mfq::cuda::load_cuda_engine(options);
    const auto generate = [](mfq::engine::Engine& engine) {
        using namespace mfq::engine;
        EngineRequest request;
        request.id = "isolation"; request.token_ids = {101, 202, 303};
        request.input.sampling.max_tokens = 32;
        request.input.sampling.temperature = 0;
        request.input.sampling.enable_mtp = false;
        if (engine.admit(std::move(request)) != Admission::accepted) throw std::runtime_error("admission failed");
        std::vector<int64_t> tokens;
        for (int tick = 0; tick < 128; ++tick) {
            auto result = engine.step({"isolation"});
            for (auto& event : result.events) {
                if (auto* error = std::get_if<Failed>(&event.data)) throw std::runtime_error(error->message);
                if (auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                if (terminal(event.data)) return tokens;
            }
        }
        throw std::runtime_error("generation did not terminate");
    };
    const auto reference = generate(*first);
    std::vector<int64_t> tokens[2];
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int i = 0; i < 2; ++i) workers[i] = std::thread([&, i] {
        try { tokens[i] = generate(i ? *second : *first); }
        catch (...) { errors[i] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
    if (tokens[0] != reference || tokens[1] != reference) throw std::runtime_error("concurrent Engines changed tokens");
    {
        using namespace mfq::engine;
        const auto await_control = [&](Engine& engine) {
            const auto deadline = Clock::now() + std::chrono::seconds(2);
            while (Clock::now() < deadline) {
                auto result = engine.step({});
                if (result.control) return std::move(*result.control);
                if (result.wake_at) std::this_thread::sleep_until(*result.wake_at);
            }
            throw std::runtime_error("text control did not complete");
        };
        if (!std::holds_alternative<ControlPending>(first->control(DecodeTokens{reference, {}})))
            throw std::runtime_error("decode control ran synchronously");
        auto decoded = await_control(*first);
        if (!std::holds_alternative<std::string>(decoded) || std::get<std::string>(decoded).empty())
            throw std::runtime_error("decode control did not retain its tokenizer");
        PrepareDuplexStep large;
        large.text.assign(2 * 1024 * 1024, 'a');
        first->control(std::move(large));
        const auto start = Clock::now();
        first->control(StopDuplex{});
        if (!std::holds_alternative<Cancelled>(await_control(*first)) ||
            Clock::now() - start > std::chrono::milliseconds(100))
            throw std::runtime_error("long control tokenization delayed cancellation");
        if (first->step({}).control) throw std::runtime_error("duplicate control completion");
    }
    first->reload(256);
    if (generate(*first) != reference) throw std::runtime_error("reload changed tokens");
    first->shutdown();
    if (generate(*second) != reference) throw std::runtime_error("peer shutdown changed tokens");
    const auto metric = [](mfq::engine::Engine& engine, const std::string& name) {
        const auto values = std::get<mfq::engine::Metrics>(engine.control(mfq::engine::RuntimeMetrics{}));
        for (const auto& [key, value] : values) if (key == name) return value;
        throw std::runtime_error("missing memory admission metric");
    };
    options.continuous_batching = 4;
    auto probe = mfq::cuda::load_cuda_engine(options);
    const auto cost = static_cast<std::size_t>(metric(*probe, "admission_device_0_request_bytes"));
    probe.reset();
    options.memory_budget_bytes = cost * 2;
    auto limited = mfq::cuda::load_cuda_engine(options);
    if (limited->info().max_requests != 2) throw std::runtime_error("memory budget did not clamp slot capacity");
    for (int run = 0; run < 4; ++run) {
        using namespace mfq::engine;
        EngineRequest long_request;
        long_request.id = "long"; long_request.token_ids.assign(96, 101);
        long_request.input.sampling.max_tokens = 16; long_request.input.sampling.enable_mtp = false;
        EngineRequest short_request = long_request; short_request.id = "short"; short_request.token_ids = {101, 202, 303};
        limited->admit(std::move(long_request)); limited->admit(std::move(short_request));
        EngineRequest deferred; deferred.id = "deferred"; deferred.token_ids = {101};
        if (limited->status().available || limited->admit(std::move(deferred)) != Admission::deferred || deferred.token_ids.empty())
            throw std::runtime_error("memory admission did not defer without consuming input");
        if (metric(*limited, "admission_device_0_reserved_bytes") != 2 * cost)
            throw std::runtime_error("request memory reservation was not retained");
        for (int tick = 0; tick < 5; ++tick) {
            const auto step = limited->step({"long", "short"});
            for (const auto& event : step.events) if (auto* error = std::get_if<Failed>(&event.data))
                throw std::runtime_error(error->message);
        }
        limited->cancel("long"); limited->cancel("short"); limited->step({});
        if (!limited->status().healthy || limited->status().available != 2 ||
            metric(*limited, "admission_device_0_reserved_bytes") != 0)
            throw std::runtime_error("cancelled requests retained their memory reservation");
        if (generate(*limited) != reference) throw std::runtime_error("budget churn changed output");
        if (metric(*limited, "admission_device_0_reserved_bytes") != 0)
            throw std::runtime_error("completed request retained its memory reservation");
    }
    std::cout << "two Engine isolation/reload/shutdown and memory admission passed\n";
}

int main(int argc, char** argv) try {
#ifdef _WIN32
    _putenv_s("MFQ_DISABLE_NVQ_FUSION", "1");
#else
    setenv("MFQ_DISABLE_NVQ_FUSION", "1", 1);
#endif
    const auto disabled_config = load_cuda_execution_config();
#ifdef _WIN32
    _putenv_s("MFQ_DISABLE_NVQ_FUSION", "");
#else
    unsetenv("MFQ_DISABLE_NVQ_FUSION");
#endif
    const auto enabled_config = load_cuda_execution_config();
    if (disabled_config.nvq_fusion || !enabled_config.nvq_fusion) return 1;

    CudaExecutionContext first;
    CudaExecutionContext second;
    first.tensor_parallel.devices = {0, 1};
    second.expert_parallel.devices = {2, 3};
    first.drop_file_cache = true;
    first.config.nvq_fusion = false;
    second.decode_graph_serial_branches = true;

    std::atomic<int> ready{0};
    std::atomic<bool> isolated{true};
    auto run = [&](CudaExecutionContext& context,
                   bool first_context) {
        context.kl_mmq.dense_calls = first_context ? 11 : 29;
        context.continuous_batch_cache_serial = first_context;
        context.profiler.stats["engine"].calls = first_context ? 3 : 7;
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        const bool valid = first_context
            ? model_parallel_enabled(context) &&
                context.tensor_parallel.primary_device() == 0 &&
                !context.expert_parallel.enabled() &&
                context.kl_mmq.dense_calls == 11 &&
                context.profiler.stats.at("engine").calls == 3 &&
                !context.config.nvq_fusion
            : model_parallel_enabled(context) &&
                !context.tensor_parallel.enabled() &&
                moe_parallel_config(context).primary_device() == 2 &&
                context.kl_mmq.dense_calls == 29 &&
                context.profiler.stats.at("engine").calls == 7 &&
                context.config.nvq_fusion;
        if (!valid) {
            isolated.store(false, std::memory_order_relaxed);
        }
    };
    std::thread first_thread(
        run, std::ref(first), true);
    std::thread second_thread(
        run, std::ref(second), false);
    first_thread.join();
    second_thread.join();

    int devices = 0;
    if (cudaGetDeviceCount(&devices) == cudaSuccess && devices) {
        check_native_resources();
        check_memory_envelope();
        if (argc == 3) check_engines(argv[1], argv[2]);
    }
    return isolated.load(std::memory_order_relaxed) &&
            first.drop_file_cache && !second.drop_file_cache &&
            !first.decode_graph_serial_branches &&
            second.decode_graph_serial_branches &&
            first.continuous_batch_cache_serial &&
            !second.continuous_batch_cache_serial
        ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
