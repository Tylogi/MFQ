#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"
#include "moe.h"

#include <cstring>
#include <fstream>
#include <iostream>

// Accepts an ordinary MFQ projection, including one extracted from a real model.
// Compare every output with the per-pool kernel's original two-row reduction.
int main(int argc, char** argv) try {
    using namespace mfq::cuda;
    if (argc != 3 && argc != 4)
        throw std::runtime_error("usage: mfq-mfe-decode-test MODEL TENSOR [OUTPUT_F16]");
    std::ofstream dump;
    if (argc == 4) {
        dump.open(argv[3], std::ios::binary);
        MFQ_RUNTIME_CHECK(bool(dump), "cannot open output dump");
    }
    auto source = mfq::open_model_source(argv[1]);
    CudaExecutionContext execution;
    auto resident = load_mfe_gpu(execution, *source, argv[2]);
    weight_loader::Routed cached;
    if (!execution.config.moe_ssd_cache_dir.empty()) {
        // Real projections exceed this budget: revisit experts after slot eviction.
        execution.moe_expert_cache = make_moe_expert_cache(192 * 1024 * 1024, execution.config);
        weight_loader::validate_load_options(execution);
        cached = weight_loader::routed(execution, *source, argv[2], 0,
            resident.n_experts, resident.out_per_expert, resident.neuron_len, "diagnostic");
        finalize_moe_expert_cache(execution.moe_expert_cache);
    }
    const Device gpu{DeviceType::cuda, 0};
    const int routes = std::min(10, resident.n_experts);
    int cases = 0;
    for (int tokens : {1, 2, 3, 4, 8}) for (bool routed : {false, true}) {
        const int rows = tokens * (routed ? routes : 1);
        std::vector<float> values(rows * resident.neuron_len);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = (static_cast<int>((i * 17) % 257) - 128) / 127.0f;
        auto x = tensor(values).to(gpu, kFloat16);
        x = routed ? x.reshape({tokens, routes, resident.neuron_len})
                   : x.reshape({tokens, resident.neuron_len});
        for (int first = 0; first < resident.n_experts; first += tokens * routes) {
            std::vector<std::int32_t> ids(tokens * routes);
            for (int i = 0; i < tokens * routes; ++i) ids[i] = (first + i) % resident.n_experts;
            auto route = build_moe_route_plan(tensor(ids).reshape({tokens, routes}).to(gpu),
                resident.n_experts);
            execution.force_moe_pool_path = true;
            auto expected = resident.forward(execution, x, route).cpu();
            execution.force_moe_pool_path = false;
            for (bool use_cache : {false, true}) {
                if (use_cache && !cached) continue;
                auto result = use_cache ? cached(execution, x, route)
                                        : resident.forward(execution, x, route);
                if (use_cache)
                    MFQ_RUNTIME_CHECK(route.host_unique_experts,
                        "cached projection did not reuse its caller's route plan");
                MFQ_RUNTIME_CHECK(isfinite(result).all().item<bool>(), "non-finite MFE output");
                auto actual = result.cpu();
                MFQ_RUNTIME_CHECK(actual.sizes() == expected.sizes() &&
                    std::memcmp(actual.data_ptr(), expected.data_ptr(), actual.nbytes()) == 0,
                    "MFE decode differs from per-pool output: tokens=", tokens,
                    " routed=", routed, " first_expert=", first);
                if (dump.is_open()) {
                    dump.write(static_cast<const char*>(actual.data_ptr()), actual.nbytes());
                    MFQ_RUNTIME_CHECK(bool(dump), "cannot write output dump");
                }
                ++cases;
            }
        }
    }
    if (dump.is_open()) {
        dump.flush();
        MFQ_RUNTIME_CHECK(bool(dump), "cannot flush output dump");
    }
    std::cout << "MFE decode full-output bit equality cases=" << cases
              << " experts=" << resident.n_experts << '\n';
    if (cached) print_moe_expert_cache_stats(execution.moe_expert_cache, std::cout);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
