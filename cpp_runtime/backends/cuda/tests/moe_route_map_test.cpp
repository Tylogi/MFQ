// Route-map checks for the compact MoE expert layout.
//
// The parallel expert scan must reproduce the sequential scan it replaced
// exactly: one compact offset array per tile width, cursors reset to zero, and
// every pair placed once inside its expert bucket. The reference kept here is
// that sequential scan (bounds are prefix sums of the per-expert counts or tile
// counts), so these checks pin the legacy layout while leaving the parallel
// implementation free to split the work across threads.
//
// `--route-map-check` (default) runs the layout checks; `--route-map-bench`
// reports the map-stage timings used to justify the scan change.

#include "mfe_weight.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_tensor_backend.h"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace tb = mfq_tensor_backend;

using Ids = std::vector<int32_t>;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string describe(const std::string & label, const char * what) {
    return label + ": " + what;
}

std::vector<int32_t> host_int32(const tb::Tensor & value) {
    require(value.defined(), "route map tensor is undefined");
    const auto host = value.to(tb::kCPU).contiguous();
    require(host.scalar_type() == tb::kInt32 && host.dim() == 1,
        "route map tensor is not a flat int32 tensor");
    const auto * data = host.data_ptr<int32_t>();
    return std::vector<int32_t>(data, data + host.numel());
}

tb::Tensor upload_ids(const Ids & ids, int rows, int routes) {
    return tb::tensor(ids).reshape({rows, routes})
        .to(tb::kInt32).to(tb::kCUDA).contiguous();
}

// Sequential reference scan: `bounds[expert + 1] - bounds[expert]` is the
// number of units (pairs for tile_m == 1, otherwise tiles) owned by an expert.
std::vector<int32_t> reference_bounds(const std::vector<int32_t> & counts, int tile_m) {
    std::vector<int32_t> bounds(counts.size() + 1, 0);
    for (std::size_t expert = 0; expert < counts.size(); ++expert) {
        const int32_t units = (counts[expert] + tile_m - 1) / tile_m;
        bounds[expert + 1] = bounds[expert] + units;
    }
    return bounds;
}

std::vector<int32_t> reference_counts(const Ids & ids, int experts) {
    std::vector<int32_t> counts(static_cast<std::size_t>(experts), 0);
    for (const int32_t expert : ids) {
        require(expert >= 0 && expert < experts, "test ids are out of range");
        ++counts[static_cast<std::size_t>(expert)];
    }
    return counts;
}

std::vector<int32_t> reference_tile_experts(const std::vector<int32_t> & bounds) {
    std::vector<int32_t> experts;
    experts.reserve(static_cast<std::size_t>(bounds.back()));
    for (std::size_t expert = 0; expert + 1 < bounds.size(); ++expert) {
        for (int32_t tile = bounds[expert]; tile < bounds[expert + 1]; ++tile) {
            experts.push_back(static_cast<int32_t>(expert));
        }
    }
    return experts;
}

void check_bounds(
        const std::string & label,
        const char * what,
        const std::vector<int32_t> & expected,
        const tb::Tensor & actual) {
    require(host_int32(actual) == expected, describe(label, what));
}

void check_tile_experts(
        const std::string & label,
        const char * what,
        const std::vector<int32_t> & bounds,
        const tb::Tensor & actual) {
    const auto expected = reference_tile_experts(bounds);
    const auto observed = host_int32(actual);
    require(observed.size() >= expected.size(), describe(label, what));
    require(std::equal(expected.begin(), expected.end(), observed.begin()),
        describe(label, what));
}

// One tile width of the map: compact tile bounds plus the expert owning each
// tile. `tile_experts` is allocated for `pairs` entries, so only the live
// prefix is compared.
void check_tile_width(
        const std::string & label,
        const char * what,
        const std::vector<int32_t> & counts,
        int tile_m,
        const tb::Tensor & bounds,
        const tb::Tensor & tile_experts) {
    const auto expected_bounds = reference_bounds(counts, tile_m);
    check_bounds(label, what, expected_bounds, bounds);
    const std::string experts_label = std::string(what) + "_experts";
    check_tile_experts(label, experts_label.c_str(), expected_bounds, tile_experts);
}

// Every pair must land in exactly one expert bucket, and each bucket must hold
// exactly the pairs routed to that expert.
void check_pair_partition(
        const std::string & label,
        const Ids & ids,
        const std::vector<int32_t> & counts,
        const std::vector<int32_t> & expert_bounds,
        const tb::Tensor & ids_dst) {
    const auto compacted = host_int32(ids_dst);
    require(compacted.size() == ids.size(), describe(label, "ids_dst size"));
    std::vector<char> placed(ids.size(), 0);
    for (std::size_t expert = 0; expert + 1 < expert_bounds.size(); ++expert) {
        const int32_t begin = expert_bounds[expert];
        const int32_t end = expert_bounds[expert + 1];
        require(end - begin == counts[expert], describe(label, "expert bucket size"));
        std::vector<int32_t> bucket(compacted.begin() + begin, compacted.begin() + end);
        std::sort(bucket.begin(), bucket.end());
        std::vector<int32_t> expected;
        for (std::size_t pair = 0; pair < ids.size(); ++pair) {
            if (ids[pair] == static_cast<int32_t>(expert)) {
                expected.push_back(static_cast<int32_t>(pair));
            }
        }
        require(bucket == expected, describe(label, "expert bucket contents"));
        for (const int32_t pair : bucket) {
            require(pair >= 0 && static_cast<std::size_t>(pair) < placed.size(),
                describe(label, "pair index"));
            require(placed[static_cast<std::size_t>(pair)] == 0,
                describe(label, "pair appears twice"));
            placed[static_cast<std::size_t>(pair)] = 1;
        }
    }
    require(std::all_of(placed.begin(), placed.end(),
            [](char value) { return value != 0; }),
        describe(label, "pair is missing from the compact map"));
}

struct MapCase {
    std::string label;
    int rows = 0;
    int routes = 0;
    int experts = 0;
    Ids ids;
};

Ids ids_all_same(std::size_t pairs) {
    return Ids(pairs, 0);
}

Ids ids_spread(std::size_t pairs, int experts) {
    Ids ids(pairs);
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        ids[pair] = static_cast<int32_t>(pair % static_cast<std::size_t>(experts));
    }
    return ids;
}

// Seven of every eight pairs land on expert zero, the rest walk the tail.
Ids ids_skewed(std::size_t pairs, int experts) {
    Ids ids(pairs);
    const std::size_t tail = experts > 1 ? static_cast<std::size_t>(experts - 1) : 1;
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        ids[pair] = pair % 8 == 7
            ? static_cast<int32_t>(1 + (pair / 8) % tail)
            : 0;
    }
    return ids;
}

enum class RoutePattern { AllSame, Spread, Skewed };

MapCase make_case(
        const std::string & label,
        int rows,
        int routes,
        int experts,
        RoutePattern pattern) {
    const auto pairs = static_cast<std::size_t>(rows) * static_cast<std::size_t>(routes);
    MapCase result;
    result.label = label;
    result.rows = rows;
    result.routes = routes;
    result.experts = experts;
    if (pattern == RoutePattern::AllSame) {
        result.ids = ids_all_same(pairs);
    } else if (pattern == RoutePattern::Spread) {
        result.ids = ids_spread(pairs, experts);
    } else {
        result.ids = ids_skewed(pairs, experts);
    }
    return result;
}

// Raw map worker with an explicit tile geometry, independent of the geometry
// `build_moe_route_plan` would pick for the same route count.
void check_map_geometry(
        const MapCase & map,
        const std::string & label,
        int tile_m,
        int secondary_tile_m,
        int tertiary_tile_m) {
    const auto tensor = upload_ids(map.ids, map.rows, map.routes);
    const auto counts = reference_counts(map.ids, map.experts);
    const auto expert_bounds = reference_bounds(counts, 1);
    const auto maps = moe_build_expert_maps_cuda(
        tensor, map.experts, tile_m, secondary_tile_m, tertiary_tile_m);
    require(maps.size() == 9, describe(label, "map tensor count"));
    check_bounds(label, "counts", counts, maps[4]);
    check_bounds(label, "expert_bounds", expert_bounds, maps[1]);
    check_pair_partition(label, map.ids, counts, expert_bounds, maps[0]);
    check_tile_width(label, "tile_bounds", counts, tile_m, maps[2], maps[3]);
    check_tile_width(label, "secondary_tile_bounds", counts, secondary_tile_m,
        maps[5], maps[6]);
    check_tile_width(label, "tertiary_tile_bounds", counts, tertiary_tile_m,
        maps[7], maps[8]);
}

void check_single_width_map(const MapCase & map) {
    const std::string label = map.label + " fine_only";
    const auto tensor = upload_ids(map.ids, map.rows, map.routes);
    const auto counts = reference_counts(map.ids, map.experts);
    const auto expert_bounds = reference_bounds(counts, 1);
    const auto maps = moe_build_expert_map_cuda(tensor, map.experts, 8);
    require(maps.size() == 5, describe(label, "map tensor count"));
    check_bounds(label, "counts", counts, maps[4]);
    check_bounds(label, "expert_bounds", expert_bounds, maps[1]);
    check_pair_partition(label, map.ids, counts, expert_bounds, maps[0]);
    check_tile_width(label, "tile_bounds", counts, 8, maps[2], maps[3]);
}

// Product entry point: geometry selection and the map-free short path for at
// most eight rows. The cursors workspace is not exported, but a scan that left
// it non-zero would scatter pairs outside their buckets, so the partition check
// covers that reset as well.
void check_route_plan(const MapCase & map) {
    const std::string label = map.label + " plan";
    const auto tensor = upload_ids(map.ids, map.rows, map.routes);
    const auto counts = reference_counts(map.ids, map.experts);
    const auto expert_bounds = reference_bounds(counts, 1);
    const auto route = build_moe_route_plan(tensor, map.experts);
    require(route.n_experts == map.experts, describe(label, "expert count"));
    if (map.rows <= 8) {
        require(route.map_ready, describe(label, "short path must stay map-free"));
        require(route.ids_dst.numel() == 0 && route.expert_bounds.numel() == 0,
            describe(label, "short path built a compact map"));
        return;
    }
    require(route.map_ready, describe(label, "compact map was not built"));
    check_bounds(label, "counts", counts, route.counts);
    check_bounds(label, "expert_bounds", expert_bounds, route.expert_bounds);
    check_pair_partition(label, map.ids, counts, expert_bounds, route.ids_dst);
    check_tile_width(label, "tile_bounds", counts, 8,
        route.tile_bounds, route.tile_experts);
    check_tile_width(label, "mma_tile_bounds", counts, route.mma_tile_m,
        route.mma_tile_bounds, route.mma_tile_experts);
    check_tile_width(label, "wide_tile_bounds", counts, route.wide_tile_m,
        route.wide_tile_bounds, route.wide_tile_experts);
}

void check_case(const MapCase & map) {
    require(map.ids.size() ==
        static_cast<std::size_t>(map.rows) * static_cast<std::size_t>(map.routes),
        describe(map.label, "case size"));
    check_route_plan(map);
    check_single_width_map(map);
    check_map_geometry(map, map.label + " widths8-32-128", 8, 32, 128);
    check_map_geometry(map, map.label + " widths1-512-1024", 1, 512, 1024);
}

std::vector<MapCase> route_map_cases() {
    return {
        make_case("rows1-same-512", 1, 8, 512, RoutePattern::AllSame),
        make_case("rows8-spread-512", 8, 8, 512, RoutePattern::Spread),
        make_case("rows9-skewed-512", 9, 8, 512, RoutePattern::Skewed),
        make_case("rows9-same-128", 9, 8, 128, RoutePattern::AllSame),
        make_case("rows9-same-1", 9, 2, 1, RoutePattern::AllSame),
        make_case("rows9-spread-257", 9, 8, 257, RoutePattern::Spread),
        make_case("rows9-spread-1024", 9, 8, 1024, RoutePattern::Spread),
        make_case("rows9-spread-4096", 9, 8, 4096, RoutePattern::Spread),
        make_case("rows16-spread-384", 16, 8, 384, RoutePattern::Spread),
        make_case("rows16-skewed-300", 16, 8, 300, RoutePattern::Skewed),
        make_case("rows32-spread-17", 32, 4, 17, RoutePattern::Spread),
        make_case("rows64-spread-1024", 64, 8, 1024, RoutePattern::Spread),
        make_case("rows65-same-1024", 65, 8, 1024, RoutePattern::AllSame),
        make_case("rows512-skewed-128", 512, 8, 128, RoutePattern::Skewed),
        make_case("rows2048-spread-384", 2048, 8, 384, RoutePattern::Spread),
        make_case("rows4096-spread-128", 4096, 8, 128, RoutePattern::Spread),
        make_case("rows4096-skewed-512", 4096, 8, 512, RoutePattern::Skewed),
    };
}

int run_route_map_checks() {
    const auto cases = route_map_cases();
    for (const auto & map : cases) {
        check_case(map);
    }
    std::cout << "moe_route_map_cases=" << cases.size()
              << " all-identical, spread, skewed, 1..4096 experts, 9 rows, "
                 "large prefill exact PASS\n";
    return 0;
}

// Launch the map stage in batches and keep the fastest batch. The measurement
// host is shared with other builds, so only the minimum batch is comparable
// between the sequential and the parallel scan.
template <typename Body>
double time_map_stage_us(const Body & body, int repetitions, int batches) {
    const auto stream = mfq_current_cuda_stream();
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&begin));
    MFQ_CUDA_CHECK(cudaEventCreate(&end));
    double fastest = 0.0;
    for (int batch = 0; batch < batches; ++batch) {
        MFQ_CUDA_CHECK(cudaEventRecord(begin, stream));
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            body();
        }
        MFQ_CUDA_CHECK(cudaEventRecord(end, stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed_ms = 0.0f;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
        const double per_iteration =
            static_cast<double>(elapsed_ms) * 1000.0 / repetitions;
        if (batch == 0 || per_iteration < fastest) {
            fastest = per_iteration;
        }
    }
    MFQ_CUDA_CHECK(cudaEventDestroy(begin));
    MFQ_CUDA_CHECK(cudaEventDestroy(end));
    return fastest;
}

// Guard against timing an empty path: the coarse map must still report every
// pair exactly once after the timed batches.
void check_bench_output(const MapCase & map) {
    const auto tensor = upload_ids(map.ids, map.rows, map.routes);
    const auto maps = moe_build_expert_maps_cuda(
        tensor, map.experts, 8, 64, 128);
    const auto counts = host_int32(maps[4]);
    const auto total = std::accumulate(counts.begin(), counts.end(), std::int64_t{0});
    require(total == static_cast<std::int64_t>(map.ids.size()),
        describe(map.label, "benchmark map lost pairs"));
}

int run_route_map_bench() {
    constexpr int repetitions = 100;
    constexpr int batches = 9;
    constexpr int warmup = 20;
    std::cout << std::fixed << std::setprecision(3)
              << "moe_route_map_bench_config repetitions=" << repetitions
              << " batches=" << batches
              << " statistic=fastest_batch\n";
    for (const int experts : {128, 384, 512, 1024}) {
        for (const int rows : {9, 64, 512, 4096}) {
            for (const auto pattern : {RoutePattern::Spread, RoutePattern::Skewed}) {
                const std::string label =
                    pattern == RoutePattern::Spread ? "spread" : "skewed";
                const auto map = make_case(label, rows, 8, experts, pattern);
                const auto tensor = upload_ids(map.ids, map.rows, map.routes);
                for (int iteration = 0; iteration < warmup; ++iteration) {
                    (void)build_moe_route_plan(tensor, experts);
                }
                mfq_cuda_synchronize();
                const double plan_us = time_map_stage_us(
                    [&] { (void)build_moe_route_plan(tensor, experts); },
                    repetitions, batches);
                const double fine_us = time_map_stage_us(
                    [&] { (void)moe_build_expert_map_cuda(tensor, experts, 8); },
                    repetitions, batches);
                const double coarse_us = time_map_stage_us(
                    [&] { (void)moe_build_expert_maps_cuda(tensor, experts, 8, 64, 128); },
                    repetitions, batches);
                // Map workspaces only: separates allocator cost from the kernel
                // and launch work left in the stage numbers above.
                const auto options = tensor.options();
                const auto pairs = static_cast<std::int64_t>(map.ids.size());
                const double alloc_us = time_map_stage_us(
                    [&] {
                        return std::vector<tb::Tensor>{
                            tb::empty({experts}, options),
                            tb::empty({experts}, options),
                            tb::empty({pairs}, options),
                            tb::empty({experts + 1}, options),
                            tb::empty({experts + 1}, options),
                            tb::empty({pairs}, options),
                        };
                    },
                    repetitions, batches);
                check_bench_output(map);
                std::cout << "moe_route_map_bench"
                          << " experts=" << experts
                          << " rows=" << rows
                          << " pairs=" << map.ids.size()
                          << " pattern=" << label
                          << " plan_us=" << plan_us
                          << " fine_map_us=" << fine_us
                          << " coarse_map_us=" << coarse_us
                          << " workspace_alloc_us=" << alloc_us << "\n";
            }
        }
    }
    // The same plan call through the map-free short path (at most eight rows)
    // launches no kernel and allocates no workspace, so it prices the plan
    // bookkeeping alone.
    for (const int experts : {128, 512, 1024}) {
        const auto map = make_case("spread", 8, 8, experts, RoutePattern::Spread);
        const auto tensor = upload_ids(map.ids, map.rows, map.routes);
        for (int iteration = 0; iteration < warmup; ++iteration) {
            (void)build_moe_route_plan(tensor, experts);
        }
        mfq_cuda_synchronize();
        const double plan_us = time_map_stage_us(
            [&] { (void)build_moe_route_plan(tensor, experts); },
            repetitions, batches);
        std::cout << "moe_route_map_bench_nomap"
                  << " experts=" << experts
                  << " rows=8"
                  << " plan_us=" << plan_us << "\n";
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1) {
            const std::string mode = argv[1];
            if (mode == "--route-map-check") {
                return run_route_map_checks();
            }
            if (mode == "--route-map-bench") {
                return run_route_map_bench();
            }
            throw std::runtime_error("unknown option: " + mode);
        }
        return run_route_map_checks();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
