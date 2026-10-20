#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include "mfe_ffn_runtime.h"

// Relu-squared banks store Up and Down only. Projection 0 therefore has no
// descriptors, and the RAM expert pipeline publishes the same Up image in both
// the Gate and Up cohorts. The resident-plan classification and the cached
// resident dispatch must keep that invariant: a bank that reads the empty Gate
// projection as a missing cohort classifies every route as a transfer that no
// producer staged, which silently drops all routed experts.
static void mfe_test_require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

namespace mfe_cached_dispatch_test_detail {
struct Option {
    const char* name;
    std::string previous;
    bool present = std::getenv(name) != nullptr;
    Option(const char* key, const char* value) : name(key) {
        if (present) previous = std::getenv(key);
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
    ~Option() {
#ifdef _WIN32
        _putenv_s(name, present ? previous.c_str() : "");
#else
        if (present) setenv(name, previous.c_str(), 1);
        else unsetenv(name);
#endif
    }
};
}  // namespace mfe_cached_dispatch_test_detail

static MixedMoePool mfe_test_nint_pool(int experts, int rows, int width, int group_size,
                                       int payload) {
    namespace tb = mfq_tensor_backend;
    const auto options = tb::TensorOptions().device(tb::kCUDA);
    MixedMoePool pool;
    pool.family = MixedMoeFamily::Nint;
    pool.local_experts = experts;
    auto& weight = pool.nint;
    const int groups = (width + group_size - 1) / group_size;
    weight.out = rows;
    weight.neuron_len = width;
    weight.ng = groups;
    weight.gs = group_size;
    // One byte per output row keeps the fabricated payload expert-stride exact.
    weight.bits = 8;
    weight.q_expert_stride = rows;
    weight.uniform_qbits = 8;
    weight.shape = {rows, width};
    std::vector<uint8_t> packed(static_cast<std::size_t>(rows) * experts);
    for (std::size_t i = 0; i < packed.size(); ++i)
        packed[i] = static_cast<uint8_t>((i * 7 + payload) % 61);
    // Row metadata is stored for every local expert, like a released bank.
    const std::size_t local_rows = static_cast<std::size_t>(experts) * rows;
    std::vector<uint8_t> row_bits(local_rows, 8), sub_scale(local_rows * groups, 200),
        sub_min(local_rows * groups, 40);
    std::vector<int64_t> offsets(local_rows);
    for (std::size_t row = 0; row < local_rows; ++row) offsets[row] = int64_t(row % rows) * 8;
    std::vector<int32_t> locals(static_cast<std::size_t>(experts));
    for (int expert = 0; expert < experts; ++expert) locals[std::size_t(expert)] = expert;
    weight.q_packed = tb::tensor(packed).to(tb::kCUDA);
    weight.row_q_bits = tb::tensor(row_bits).to(tb::kCUDA);
    weight.row_q_bit_offsets = tb::tensor(offsets).to(tb::kCUDA);
    weight.sub_scale = tb::tensor(sub_scale).to(tb::kCUDA);
    weight.sub_min = tb::tensor(sub_min).to(tb::kCUDA);
    weight.neuron_scale = tb::full({int64_t(local_rows)}, 0.01, options.dtype(tb::kFloat32));
    weight.neuron_min = tb::full({int64_t(local_rows)}, -0.02, options.dtype(tb::kFloat32));
    pool.expert_local = tb::tensor(locals).to(tb::kCUDA);
    return pool;
}

static std::vector<int64_t> mfe_test_nint_bytes(const MixedMoePool& pool) {
    const auto& weight = pool.nint;
    const int64_t rows = weight.out, groups = weight.ng;
    return {weight.q_expert_stride, rows, rows * 8, rows * groups, rows * groups, rows * 4, rows * 4};
}

static void check_mfe_cached_dispatch_geometry() {
    namespace tb = mfq_tensor_backend;
    using mfe_cached_dispatch_test_detail::Option;
    constexpr int experts = 3, width = 16, intermediate = 8, tokens = 2, routes = 2;
    // The fixed-quota server workers let the device classify the resident plan.
    Option overlap("MFQ_MFE_RESIDENT_PLAN_OVERLAP", "1");
    const auto sigmoid = moe_swiglu_sigmoid_table_cuda();
    std::array<MixedMoeRuntime, 3> projections;
    projections[0].n_experts = experts;
    projections[1].n_experts = experts;
    projections[1].out_per_expert = intermediate;
    projections[1].neuron_len = width;
    projections[1].pools.push_back(mfe_test_nint_pool(experts, intermediate, width, 8, 3));
    projections[2].n_experts = experts;
    projections[2].out_per_expert = width;
    projections[2].neuron_len = intermediate;
    projections[2].pools.push_back(mfe_test_nint_pool(experts, width, intermediate, 8, 5));
    const auto gpu_options = tb::TensorOptions().device(tb::kCUDA);
    std::vector<float> input_values(std::size_t(tokens) * width);
    for (std::size_t i = 0; i < input_values.size(); ++i)
        input_values[i] = std::sin(float(i) * 0.11f) * 0.4f;
    auto input = tb::tensor(input_values).reshape({tokens, width}).to(tb::kFloat16).to(tb::kCUDA);
    std::vector<int32_t> route_ids(std::size_t(tokens) * routes);
    for (std::size_t i = 0; i < route_ids.size(); ++i) route_ids[i] = int32_t(i % experts);
    auto ids = tb::tensor(route_ids).reshape({tokens, routes}).to(tb::kCUDA);
    auto weights = tb::full({tokens, routes}, 1.0 / routes, gpu_options.dtype(tb::kFloat32));
    // [plan, transfer, cpu_ready, abort, transfer index * experts, kinds * experts]
    const std::size_t words = 4 + 2 * std::size_t(experts);
    mfq::cuda::HostBuffer mapped_state(words * sizeof(int32_t), true);
    void* mapped_device = nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_device, mapped_state.data(), 0));
    auto* host_words = static_cast<int32_t*>(mapped_state.data());
    auto* device_words = static_cast<int32_t*>(mapped_device);
    const auto mapped_flag = [&](std::size_t index) {
        return reinterpret_cast<uint32_t*>(host_words + index);
    };
    const auto mapped_device_flag = [&](std::size_t index) {
        return reinterpret_cast<const uint32_t*>(device_words + index);
    };
    // Published routes have their own mapped image; the plan words already carry
    // the host kind for every expert.
    mfq::cuda::HostBuffer published_ids(std::size_t(tokens) * routes * sizeof(int32_t), true);
    void* published_device = nullptr;
    MFQ_CUDA_CHECK(cudaHostGetDevicePointer(&published_device, published_ids.data(), 0));
    auto cpu_pairs = tb::zeros({tokens, routes, width}, gpu_options.dtype(tb::kFloat16));
    MfeFfnRuntime runtime({nullptr, &projections[1], &projections[2]}, input, ids, weights, sigmoid,
                          {}, {}, 0, 0, nullptr, false, false, 0, 0, true, false);
    mfe_test_require(runtime.batch().expert_activation == 1,
                     "relu-squared bank did not reach the fused runtime");
    mfe_test_require(runtime.batch().projection_counts[0] == 0 &&
                         runtime.batch().projection_counts[1] == 1 &&
                         runtime.batch().projection_counts[2] == 1,
                     "ungated fused runtime did not keep the empty Gate projection");
    const auto bind = [&](bool overlap) {
        Option value("MFQ_MFE_RESIDENT_PLAN_OVERLAP", overlap ? "1" : "0");
        runtime.asynchronous(device_words + 4 + experts, mapped_device_flag(0),
                             mapped_device_flag(1), mapped_device_flag(2), mapped_device_flag(3),
                             device_words + 4, cpu_pairs.data_ptr());
    };
    const auto run = [&](bool overlap, bool cached) {
        bind(overlap);
        for (int index = 0; index < 4; ++index)
            *mapped_flag(std::size_t(index)) = index == 3 ? 0 : 1;
        for (int expert = 0; expert < experts; ++expert) {
            host_words[4 + expert] = -1;
            host_words[4 + experts + expert] = 1;
        }
        if (cached)
            runtime.prepare(static_cast<int32_t*>(published_device),
                            const_cast<uint32_t*>(mapped_device_flag(0)));
        else runtime.prepare();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        // The device classification must be read before the transfer fence
        // republishes the host plan over it.
        std::vector<int32_t> kinds(static_cast<std::size_t>(experts));
        MFQ_CUDA_CHECK(cudaMemcpy(kinds.data(), runtime.batch().kinds,
                                  sizeof(int32_t) * kinds.size(), cudaMemcpyDeviceToHost));
        mfe_test_require(std::all_of(kinds.begin(), kinds.end(), [](int kind) { return kind == 1; }),
                         "ungated resident bank was not classified as GPU resident");
        runtime.resident();
        runtime.wait_transfer();
        runtime.transferred();
        runtime.wait_cpu();
        runtime.down_reduce_compute();
        MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        auto host = runtime.output().to(tb::kFloat16).contiguous().cpu();
        return std::vector<unsigned char>(
            static_cast<const unsigned char*>(host.data_ptr()),
            static_cast<const unsigned char*>(host.data_ptr()) + host.numel() * host.element_size());
    };
    // A host-published resident plan is the reference for the fixed-quota
    // device-classified plan the server's streaming workers run.
    const auto reference = run(false, false);
    const auto device_plan = run(true, false);
    mfe_test_require(device_plan == reference,
                     "device-classified ungated residency changed the routed expert output");
    // The cached resident dispatch receives the RAM pipeline's cohort bytes,
    // where the ungated bank reports the same Up cohorts for Gate and Up.
    std::array<std::vector<std::vector<int64_t>>, 3> sizes;
    sizes[0].push_back(mfe_test_nint_bytes(projections[1].pools[0]));
    sizes[1].push_back(mfe_test_nint_bytes(projections[1].pools[0]));
    sizes[2].push_back(mfe_test_nint_bytes(projections[2].pools[0]));
    auto cached_table =
        tb::zeros({int64_t(experts) * int64_t(sizeof(mfq::cuda::MfeCachedExpert))},
                  gpu_options.dtype(tb::kUInt8));
    std::vector<mfq::cuda::MfeCachedExpert> cached_host(static_cast<std::size_t>(experts));
    for (auto& expert : cached_host) expert.ready_mask = 7;
    MFQ_CUDA_CHECK(cudaMemcpy(cached_table.data_ptr(), cached_host.data(),
                              sizeof(mfq::cuda::MfeCachedExpert) * cached_host.size(),
                              cudaMemcpyHostToDevice));
    runtime.enable_cached_dispatch(
        static_cast<const mfq::cuda::MfeCachedExpert*>(cached_table.data_ptr()), sizes,
        const_cast<uint32_t*>(mapped_device_flag(0)));
    mfe_test_require(runtime.batch().projection_counts[0] == 0 &&
                         runtime.batch().projection_offsets[0] == 0,
                     "cached resident dispatch installed descriptors for the missing Gate projection");
    mfe_test_require(runtime.batch().projection_counts[1] == 1 + experts &&
                         runtime.batch().projection_counts[2] == 1 + experts,
                     "cached resident dispatch lost the Up or Down per-expert slots");
    const auto cached = run(true, true);
    mfe_test_require(cached == reference,
                     "cached resident dispatch changed the ungated expert output");
    std::cout << "MFE ungated resident plan and cached dispatch keep Up/Down geometry PASS\n";
}
