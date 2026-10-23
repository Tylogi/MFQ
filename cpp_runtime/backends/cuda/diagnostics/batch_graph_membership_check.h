#pragma once

#include "storage/moe_expert_cache.h"

#include <chrono>
#include <cstring>

namespace mfq::cuda::internal {

inline MfqSamplingParams batch_graph_sampling_params(int row, int tokens) {
    MfqSamplingParams params;
    const int top_k[] = {0, 1, 2, 8, 20};
    params.max_tokens = tokens;
    params.temperature = .7 + .1 * (row % 4);
    params.top_k = top_k[row % 5];
    params.top_p = row % 2 ? .8 : 1.;
    params.presence_penalty = .2;
    params.frequency_penalty = -.01;
    params.repetition_penalty = 1.2;
    params.seed = 73 + row * 173;
    params.enable_mtp = false;
    return params;
}

// Exercise the public batch adapter directly so row order and arrivals are
// deterministic, including a prefill between two replays of the same bucket.
template<class Model>
void check_batch_graph_membership(Model& model, CudaContinuousBatchConfig config,
        const std::vector<int64_t>& prompt, const std::vector<int64_t>& reference,
        const std::vector<std::vector<int64_t>>& sampling_references = {}) {
    namespace tb = mfq_tensor_backend;
    using Ops = CudaBatchOperations<Model>;
    using Request = typename Ops::Request;
    const int initial_rows = config.max_sequences >= 32 ? 17 : 3;
    config.max_sequences = initial_rows > 3 ? 32 : 4;
    config.cuda_graph_minimum_tokens = 1;
    config.cuda_graph = true;
    config.greedy = true;
    const bool stochastic = !sampling_references.empty();
    std::cerr << "batch_membership_bench begin\n";
    {
        Ops operations(model, *model.execution, config);
        typename Ops::State state;
        std::vector<std::shared_ptr<Request>> requests;
        std::vector<std::vector<int64_t>> tokens;
        auto maximum_position = [&] {
            int64_t position = 0;
            for (const auto& request : state.active)
                position = std::max(position, request->cache_length);
            return position;
        };
        auto admit = [&] {
            mfq::engine::InferenceRequest input;
            input.prompt = prompt;
            input.sampling.max_tokens = reference.size();
            input.sampling.temperature = 0;
            input.sampling.top_k = 1;
            input.sampling.enable_mtp = false;
            if (stochastic)
                input.sampling = batch_graph_sampling_params(requests.size(), reference.size());
            auto request = std::make_shared<Request>("row-" + std::to_string(requests.size()), input);
            operations.suspend_decode();
            auto sample = operations.prefill(request, {0, int64_t(prompt.size())});
            request->cache_length = prompt.size();
            request->pending_token = sample.token;
            request->produced = 1;
            request->eligible = true;
            operations.activate(request, sample);
            state.active.push_back(request);
            operations.resume_decode(maximum_position());
            requests.push_back(request);
            tokens.push_back({sample.token});
            return request;
        };
        const auto snapshot = [&](const std::shared_ptr<Request>& request) {
            std::vector<tb::Tensor> result;
            if constexpr (std::is_same_v<Model, Qwen35CausalLm>) {
                for (const auto& block : model.blocks) {
                    if (auto* linear = dynamic_cast<qwen35::LinearAttentionBlock*>(block.get())) {
                        result.push_back(linear->mixer.convolution.narrow(0, request->cache.slot(), 1));
                        result.push_back(linear->mixer.recurrent.narrow(0, request->cache.slot(), 1));
                    }
                }
            } else if constexpr (requires { request->cache->recurrent_state_tensors(); }) {
                result = request->cache->recurrent_state_tensors();
            }
            if (request->counts.defined())
                result.push_back(request->counts);
            return result;
        };
        const auto metric = [&](const char* name) {
            for (const auto& [key, value] : operations.metrics())
                if (key == name)
                    return value;
            return -1.0;
        };
        const auto check_pages = [&] {
            const auto page_size = int64_t(metric("paged_kv_page_size"));
            if (page_size <= 0)
                return;
            int64_t pages = 0;
            for (const auto& request : state.active)
                pages += (request->cache_length + page_size - 1) / page_size;
            MFQ_RUNTIME_CHECK(metric("paged_kv_live_pages") == pages,
                              "inactive slots retained request KV pages");
        };
        std::vector<double> step_ms;
        double routes_per_row = 0;
        const auto expert_routes = [&] {
            for (const auto& [name, value] : moe_expert_memory_metrics(model.execution->moe_expert_cache))
                if (name == "expert_target_multi_route_positions")
                    return value;
            return 0.0;
        };
        const auto decode = [&] {
            const auto routes_before = expert_routes();
            const auto eligible = std::count_if(state.active.begin(), state.active.end(),
                [](const auto& request) { return request->eligible; });
            const auto start = std::chrono::steady_clock::now();
            auto decoded = operations.decode(state);
            for (const auto& request : state.active) {
                if (!request->eligible)
                    continue;
                auto sample = operations.sample(request, decoded);
                const auto index = std::find(requests.begin(), requests.end(), request) - requests.begin();
                tokens[index].push_back(sample.token);
                const auto& expected = stochastic ? sampling_references.at(index) : reference;
                MFQ_RUNTIME_CHECK(tokens[index].size() <= expected.size() &&
                    std::equal(tokens[index].begin(), tokens[index].end(), expected.begin()),
                    "membership change differs from serial generation");
                operations.accept(request, sample);
                request->pending_token = sample.token;
                ++request->cache_length;
                ++request->produced;
            }
            mfq_cuda_synchronize();
            const auto elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            const auto routes = expert_routes() - routes_before;
            if (!routes_per_row && routes > 0)
                routes_per_row = routes / eligible;
            if (routes_per_row)
                MFQ_RUNTIME_CHECK(routes == routes_per_row * eligible,
                                  "padding contributed to expert route statistics");
            check_pages();
            step_ms.push_back(elapsed);
        };
        const auto frozen_decode = [&](const std::shared_ptr<Request>& paused) {
            auto live = snapshot(paused);
            std::vector<tb::Tensor> before;
            for (const auto& value : live)
                before.push_back(value.cpu());
            const auto position = paused->cache_length;
            auto sampler_before = *paused->sampler;
            decode();
            auto sampler_after = *paused->sampler;
            MFQ_RUNTIME_CHECK(sampler_before.next_uniform_float() == sampler_after.next_uniform_float(),
                              "inactive request RNG advanced during a bucket replay");
            for (std::size_t i = 0; i < live.size(); ++i) {
                auto after = live[i].cpu();
                MFQ_RUNTIME_CHECK(after.nbytes() == before[i].nbytes() &&
                     std::memcmp(after.data_ptr(), before[i].data_ptr(), after.nbytes()) == 0,
                    "inactive request state changed during a bucket replay");
            }
            MFQ_RUNTIME_CHECK(paused->cache_length == position, "inactive request position advanced");
        };
        for (int row = 0; row < initial_rows; ++row)
            admit();
        check_pages();
        decode();
        const auto initial_captures = metric("continuous_batching_cuda_graph_captures");
        const auto fourth = admit();
        decode();
        std::rotate(state.active.begin(), state.active.begin() + 1, state.active.end());
        decode();
        fourth->eligible = false;
        frozen_decode(fourth);
        fourth->eligible = true;
        const auto removed_live = snapshot(fourth);
        std::vector<tb::Tensor> removed_before;
        for (const auto& value : removed_live)
            removed_before.push_back(value.cpu());
        std::erase(state.active, fourth);
        operations.retire({fourth}, maximum_position());
        decode();
        for (std::size_t i = 0; i < removed_live.size(); ++i) {
            auto after = removed_live[i].cpu();
            MFQ_RUNTIME_CHECK(std::memcmp(after.data_ptr(), removed_before[i].data_ptr(), after.nbytes()) == 0,
                              "retired request state was modified by a padded row");
        }
        admit();
        for (int step = 0; step < 4; ++step) {
            std::reverse(state.active.begin(), state.active.end());
            state.active.back()->eligible = step % 2;
            if (state.active.back()->eligible)
                decode();
            else
                frozen_decode(state.active.back());
        }
        const auto captures = metric("continuous_batching_cuda_graph_captures");
        const auto reuses = metric("continuous_batching_cuda_graph_bucket_reuses");
        MFQ_RUNTIME_CHECK(reuses >= 0, "missing CUDA graph bucket reuse metric");
        if (initial_captures > 0) {
            MFQ_RUNTIME_CHECK(captures == initial_captures && reuses > 0 &&
                metric("continuous_batching_cuda_graph_membership_captures") == 0,
                "same-bucket membership changes captured another CUDA graph");
        }
        double total_ms = 0;
        for (const auto milliseconds : step_ms)
            total_ms += milliseconds;
        std::cout << (stochastic ? "continuous_batching_sampling_membership_check rows="
                                : "continuous_batching_membership_check rows=")
                  << initial_rows << " steps=" << step_ms.size()
                  << " captures=" << captures << " capture_ms="
                  << metric("continuous_batching_cuda_graph_capture_ms")
                  << " mean_step_ms=" << total_ms / step_ms.size() << " bucket_reuses=" << reuses
                  << " serial=PASS inactive_state_bits=PASS inactive_rng=PASS padding_routes=PASS\n";
        operations.retire(state.active, 0);
    }
    std::cerr << "batch_membership_bench end\n";
}

} // namespace mfq::cuda::internal
