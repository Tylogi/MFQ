#pragma once

#include <chrono>
#include <cstdlib>

namespace mfq::cuda::internal {

// Opt-in diagnostic on the loaded synthetic model, including metadata upload,
// vocabulary projection, sampling and request bookkeeping in the step timer.
template<class Model>
void benchmark_batch_decode(Model& model, CudaContinuousBatchConfig config,
        const std::vector<int64_t>& prompt, const std::vector<int64_t>& reference) {
    if (!std::getenv("MFQ_QSA_BATCH_BENCH"))
        return;
    MFQ_RUNTIME_CHECK(reference.size() >= 13, "batch benchmark requires at least 13 serial tokens");
    using Ops = CudaBatchOperations<Model>;
    using Request = typename Ops::Request;
    for (int rows : {1, 4, 8, 16, 32}) {
        config.max_sequences = rows;
        config.cuda_graph_minimum_tokens = 1;
        config.cuda_graph = true;
        config.greedy = true;
        Ops operations(model, *model.execution, config);
        typename Ops::State state;
        for (int row = 0; row < rows; ++row) {
            mfq::engine::InferenceRequest input;
            input.prompt = prompt;
            input.sampling.max_tokens = reference.size();
            input.sampling.temperature = 0;
            input.sampling.top_k = 1;
            input.sampling.enable_mtp = false;
            auto request = std::make_shared<Request>("bench-" + std::to_string(row), input);
            operations.suspend_decode();
            auto sample = operations.prefill(request, {0, int64_t(prompt.size())});
            MFQ_RUNTIME_CHECK(sample.token == reference[0], "batch benchmark prefill differs from serial");
            request->cache_length = prompt.size();
            request->pending_token = sample.token;
            request->produced = 1;
            request->eligible = true;
            operations.activate(request, sample);
            state.active.push_back(request);
            operations.resume_decode(prompt.size());
        }
        std::vector<double> steps, decode_gpu;
        cudaEvent_t begin, end;
        MFQ_CUDA_CHECK(cudaEventCreate(&begin));
        MFQ_CUDA_CHECK(cudaEventCreate(&end));
        for (int step = 0; step < 12; ++step) {
            const auto start = std::chrono::steady_clock::now();
            MFQ_CUDA_CHECK(cudaEventRecord(begin, mfq_current_cuda_stream()));
            auto decoded = operations.decode(state);
            MFQ_CUDA_CHECK(cudaEventRecord(end, mfq_current_cuda_stream()));
            for (const auto& request : state.active) {
                auto sample = operations.sample(request, decoded);
                MFQ_RUNTIME_CHECK(sample.token == reference[request->produced],
                    "batch benchmark output differs from serial");
                request->pending_token = sample.token;
                ++request->cache_length;
                ++request->produced;
            }
            MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            mfq_cuda_synchronize();
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count();
            float ms = 0;
            MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            if (step >= 4) {
                steps.push_back(us);
                decode_gpu.push_back(double(ms) * 1000);
            }
        }
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));
        MFQ_CUDA_CHECK(cudaEventDestroy(end));
        std::sort(steps.begin(), steps.end());
        std::sort(decode_gpu.begin(), decode_gpu.end());
        std::cout << "batch_decode_bench rows=" << rows << " step_us=" << steps[steps.size() / 2]
                  << " decode_gpu_us=" << decode_gpu[decode_gpu.size() / 2]
                  << " timed_steps=" << steps.size() << " serial=PASS\n";
        operations.retire(state.active, 0);
        state.active.clear();
    }
}

} // namespace mfq::cuda::internal
