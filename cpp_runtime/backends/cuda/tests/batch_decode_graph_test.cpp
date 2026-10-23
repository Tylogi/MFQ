#include "runtime/batch_decode_graph.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace mfq::cuda;
namespace tb = mfq_tensor_backend;

static void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

int main() {
    try {
        int devices = 0;
        MFQ_CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices)
            return 77;
        for (int maximum : {1, 2, 3, 6, 16, 32}) {
            for (int rows = 1; rows <= maximum; ++rows) {
                const auto bucket = batch_decode_row_bucket(rows, maximum);
                check(bucket >= rows && bucket <= maximum, "invalid decode bucket ceiling");
                check(bucket == maximum || (bucket & (bucket - 1)) == 0, "invalid decode bucket shape");
            }
        }
        BatchDecodeGraphCache cache;
        MfqCudaStreamGuard guard(cache.stream());
        const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
        std::vector<tb::Tensor> states, counts, snapshots;
        std::vector<int> expected(6, 0);
        for (int request = 0; request < 6; ++request) {
            states.push_back(tb::full({1, 64}, request, options));
            counts.push_back(tb::zeros({1}, options.dtype(tb::kInt32)));
        }
        ParallelConfig parallel;
        ModelParallelCollectiveRuntime collectives;
        cache.prepare_pool(parallel, collectives);
        const std::vector<std::vector<int>> schedule{
            {0, 1, 2}, {2, 0, 1, 3}, {3, 2, 0}, {4, 0, 2, 3}, {2, 4, 0},
            {0, 1}, {1}, {5, 4, 3}, {3, 5, 4, 0}};
        std::size_t largest_graph_reserved = 0;
        for (const auto& members : schedule) {
            const auto rows = batch_decode_row_bucket(members.size(), 6);
            auto& entry = cache.get({rows, 512, 256, {}}, 2, 0);
            auto& input = *entry.inputs;
            snapshots.clear();
            for (const auto& state : states)
                snapshots.push_back(state.cpu());
            for (int64_t row = 0; row < rows; ++row) {
                const bool active = row < int64_t(members.size());
                input.row(row, 1, 0, active ? members[row] : -1, active);
                input.state(row, 0, active ? states[members[row]] : states[5]);
                input.state(row, 1, active ? counts[members[row]] : counts[5]);
            }
            input.upload();
            const auto body = [&] {
                // Reuse a sizable temporary across every bucket capture.
                auto scratch = tb::empty({4, 1024 * 1024}, options);
                scratch.zero_();
                for (int64_t row = 0; row < rows; ++row) {
                    auto state = tb::empty_like(states[0]);
                    auto count = tb::empty_like(counts[0]);
                    batch_state_load(input, 0, row, state);
                    batch_state_load(input, 1, row, count);
                    state = state + 1;
                    count = count + 1;
                    batch_state_store(input, 0, row, state);
                    batch_state_store(input, 1, row, count);
                }
            };
            const auto restore = [&] {
                for (int request = 0; request < 6; ++request) {
                    states[request].copy_(snapshots[request]);
                    counts[request].fill_(expected[request]);
                }
            };
            const auto captures = cache.captures;
            if (!entry.valid) {
                entry.window = std::make_unique<DecodeWindow>(cache.stream().stream());
                entry.window->capture(body, restore);
                cache.captured(entry, 0);
            }
            entry.window->run();
            for (const auto member : members)
                ++expected[member];
            cache.replayed(entry);
            if (rows == 4 && captures)
                check(cache.captures == captures, "membership change captured a graph");
            for (int request = 0; request < 6; ++request) {
                const auto value = states[request].cpu();
                check(counts[request].item<int32_t>() == expected[request], "inactive token count changed");
                check(value.data_ptr<float>()[0] == request + expected[request], "request state changed incorrectly");
                if (std::find(members.begin(), members.end(), request) == members.end())
                    check(std::memcmp(value.data_ptr(), snapshots[request].data_ptr(), value.nbytes()) == 0,
                          "removed request state was modified");
            }
            if (!largest_graph_reserved)
                largest_graph_reserved = default_context()->pool_memory_stats().second;
        }
        check(cache.captures == 3 && cache.membership_captures == 0 && cache.bucket_reuses == 6,
              "decode graph reuse counters are incorrect");
        const auto buckets_reserved = default_context()->pool_memory_stats().second;
        check(buckets_reserved <= largest_graph_reserved + (std::size_t(2) << 20),
              "bucket graphs retained separate scratch pools");
        auto& input = *cache.get({4, 512, 256, {}}, 2, 0).inputs;
        input.update_rows({{1, 0, 0}, {1, 0, 1}, {1, 0, 2}});
        input.upload();
        CudaExecutionContext execution;
        std::vector<tb::Tensor> selected{
            tb::zeros({4, 2}, options.dtype(tb::kInt32)), tb::ones({4, 2}, options)};
        BatchDecodeMaskScope mask(execution, input.enabled());
        mask_batch_routes(execution, selected);
        auto routes = selected[0].cpu();
        check(routes.data_ptr<int32_t>()[6] == -1 && routes.data_ptr<int32_t>()[7] == -1,
              "padding selected an expert");
        execution.config.moe_route_stats_path = "padding-test";
        execution.config.moe_route_output_energy = true;
        record_moe_route_stats(execution, 0, selected[0], selected[1], tb::ones({8, 4}, options), 2);
        check(execution.moe_route_stats.at(0).counts.sum().item<double>() == 6,
              "padding contributed to route counts");
        check(execution.moe_route_stats.at(0).output_energy.sum().item<double>() == 24,
              "padding contributed to route energy statistics");
        std::cout << "batch_decode_graph PASS arrivals=1 departures=1 reorder=1 padding=1 state_bits=1"
                  << " captures=" << cache.captures << " bucket_reuses=" << cache.bucket_reuses << '\n';
        std::cout << "batch_decode_graph_pool largest_bytes=" << largest_graph_reserved
                  << " all_buckets_bytes=" << buckets_reserved << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
