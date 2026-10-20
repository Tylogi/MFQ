#include "hip/hip_runtime.h"
#include "backend_checks.h"

#include "mfq_hip_activation_ops.h"
#include "mfq_tensor_backend.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef MFQ_CUDA_CHECK
#define MFQ_CUDA_CHECK(expr) do { \
    hipError_t err__ = (expr); \
    if (err__ != hipSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + hipGetErrorString(err__)); \
    } \
} while (0)
#endif

int run_backend_bf16_add_check(
        int64_t elements,
        int repetitions) {
    MFQ_RUNTIME_CHECK(
        elements > 0 && elements <= (1LL << 30),
        "backend BF16 add element count is out of range");
    MFQ_RUNTIME_CHECK(
        repetitions > 0,
        "backend BF16 add repetitions must be positive");
    auto sequence = mfq_tensor_backend::arange(
        elements,
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kFloat32));
    auto left = ((sequence.remainder(257) - 128.0) / 64.0)
        .to(mfq_tensor_backend::kBFloat16).contiguous();
    auto right = ((sequence.remainder(131) - 65.0) / 32.0)
        .to(mfq_tensor_backend::kBFloat16).contiguous();

    mfq_tensor_backend::Tensor eager;
    for (int warmup = 0; warmup < 30; ++warmup) {
        eager = (left + right).contiguous();
    }
    mfq_hip_synchronize();
    hipEvent_t eager_start = nullptr;
    hipEvent_t eager_stop = nullptr;
    MFQ_CUDA_CHECK(hipEventCreate(&eager_start));
    MFQ_CUDA_CHECK(hipEventCreate(&eager_stop));
    const auto eager_stream = mfq_get_current_cuda_stream();
    MFQ_CUDA_CHECK(hipEventRecord(eager_start, eager_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        eager = (left + right).contiguous();
    }
    MFQ_CUDA_CHECK(hipEventRecord(eager_stop, eager_stream.stream()));
    MFQ_CUDA_CHECK(hipEventSynchronize(eager_stop));
    float eager_ms = 0.0f;
    MFQ_CUDA_CHECK(hipEventElapsedTime(&eager_ms, eager_start, eager_stop));
    MFQ_CUDA_CHECK(hipEventDestroy(eager_start));
    MFQ_CUDA_CHECK(hipEventDestroy(eager_stop));

    const auto graph_stream = mfq_get_stream_from_pool(false);
    MfqCudaStreamGuard graph_guard(graph_stream);
    MfqCudaGraph graph;
    mfq_prepare_cuda_graph_memory(graph);
    mfq_tensor_backend::Tensor graph_result = (left + right).contiguous();
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));
    graph_result = mfq_tensor_backend::Tensor();
    graph.capture_begin();
    graph_result = (left + right).contiguous();
    graph.capture_end();
    graph.replay();
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));

    hipEvent_t graph_start = nullptr;
    hipEvent_t graph_stop = nullptr;
    MFQ_CUDA_CHECK(hipEventCreate(&graph_start));
    MFQ_CUDA_CHECK(hipEventCreate(&graph_stop));
    MFQ_CUDA_CHECK(hipEventRecord(graph_start, graph_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        graph.replay();
    }
    MFQ_CUDA_CHECK(hipEventRecord(graph_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(hipEventSynchronize(graph_stop));
    float graph_ms = 0.0f;
    MFQ_CUDA_CHECK(hipEventElapsedTime(&graph_ms, graph_start, graph_stop));
    MFQ_CUDA_CHECK(hipEventDestroy(graph_start));
    MFQ_CUDA_CHECK(hipEventDestroy(graph_stop));

    const auto difference = (
        graph_result.to(mfq_tensor_backend::kFloat32) -
        eager.to(mfq_tensor_backend::kFloat32)).abs();
    const double maximum = difference.max().item<double>();
    MFQ_RUNTIME_CHECK(
        maximum == 0.0 &&
            mfq_tensor_backend::isfinite(graph_result).all().item<bool>(),
        "backend BF16 add graph result differs from eager output");
    graph_result = mfq_tensor_backend::Tensor();
    graph.reset();

    constexpr int chain_nodes = 512;
    const int chain_repetitions =
        std::max(100, repetitions / 50);
    MfqCudaGraph chain_graph;
    mfq_prepare_cuda_graph_memory(chain_graph);
    mfq_tensor_backend::Tensor chain_result;
    for (int node = 0; node < chain_nodes; ++node) {
        chain_result = (left + right).contiguous();
    }
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));
    chain_result = mfq_tensor_backend::Tensor();
    chain_graph.capture_begin();
    for (int node = 0; node < chain_nodes; ++node) {
        chain_result = (left + right).contiguous();
    }
    chain_graph.capture_end();
    chain_graph.replay();
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));

    hipEvent_t chain_start = nullptr;
    hipEvent_t chain_stop = nullptr;
    MFQ_CUDA_CHECK(hipEventCreate(&chain_start));
    MFQ_CUDA_CHECK(hipEventCreate(&chain_stop));
    MFQ_CUDA_CHECK(hipEventRecord(chain_start, graph_stream.stream()));
    for (int repetition = 0;
            repetition < chain_repetitions;
            ++repetition) {
        chain_graph.replay();
    }
    MFQ_CUDA_CHECK(hipEventRecord(chain_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(hipEventSynchronize(chain_stop));
    float chain_ms = 0.0f;
    MFQ_CUDA_CHECK(hipEventElapsedTime(
        &chain_ms, chain_start, chain_stop));
    MFQ_CUDA_CHECK(hipEventDestroy(chain_start));
    MFQ_CUDA_CHECK(hipEventDestroy(chain_stop));
    const auto chain_difference = (
        chain_result.to(mfq_tensor_backend::kFloat32) -
        eager.to(mfq_tensor_backend::kFloat32)).abs();
    const double chain_maximum =
        chain_difference.max().item<double>();
    MFQ_RUNTIME_CHECK(
        chain_maximum == 0.0 &&
            mfq_tensor_backend::isfinite(chain_result).all().item<bool>(),
        "backend BF16 add chain Graph differs from eager output");
    const double chain_graph_us =
        1000.0 * static_cast<double>(chain_ms) /
        chain_repetitions;
    chain_result = mfq_tensor_backend::Tensor();
    chain_graph.reset();

    auto shared_reference = silu_mul_cuda(left, right);
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));
    MfqCudaGraph shared_graph;
    mfq_prepare_cuda_graph_memory(shared_graph);
    mfq_tensor_backend::Tensor shared_result;
    for (int node = 0; node < chain_nodes; ++node) {
        shared_result = silu_mul_cuda(left, right);
    }
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));
    shared_result = mfq_tensor_backend::Tensor();
    shared_graph.capture_begin();
    for (int node = 0; node < chain_nodes; ++node) {
        shared_result = silu_mul_cuda(left, right);
    }
    shared_graph.capture_end();
    shared_graph.replay();
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));

    hipEvent_t shared_start = nullptr;
    hipEvent_t shared_stop = nullptr;
    MFQ_CUDA_CHECK(hipEventCreate(&shared_start));
    MFQ_CUDA_CHECK(hipEventCreate(&shared_stop));
    MFQ_CUDA_CHECK(hipEventRecord(shared_start, graph_stream.stream()));
    for (int repetition = 0;
            repetition < chain_repetitions;
            ++repetition) {
        shared_graph.replay();
    }
    MFQ_CUDA_CHECK(hipEventRecord(shared_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(hipEventSynchronize(shared_stop));
    float shared_ms = 0.0f;
    MFQ_CUDA_CHECK(hipEventElapsedTime(
        &shared_ms, shared_start, shared_stop));
    MFQ_CUDA_CHECK(hipEventDestroy(shared_start));
    MFQ_CUDA_CHECK(hipEventDestroy(shared_stop));
    const auto shared_difference = (
        shared_result.to(mfq_tensor_backend::kFloat32) -
        shared_reference.to(mfq_tensor_backend::kFloat32)).abs();
    const double shared_maximum =
        shared_difference.max().item<double>();
    MFQ_RUNTIME_CHECK(
        shared_maximum == 0.0 &&
            mfq_tensor_backend::isfinite(shared_result).all().item<bool>(),
        "shared BF16 SwiGLU chain Graph differs from eager output");
    const double shared_graph_us =
        1000.0 * static_cast<double>(shared_ms) /
        chain_repetitions;
    std::cout
        << "backend_bf16_add elements=" << elements
        << " repetitions=" << repetitions
        << " eager_us="
        << 1000.0 * static_cast<double>(eager_ms) / repetitions
        << " graph_us="
        << 1000.0 * static_cast<double>(graph_ms) / repetitions
        << " max_abs=" << maximum
        << " chain_nodes=" << chain_nodes
        << " chain_repetitions=" << chain_repetitions
        << " chain_graph_us=" << chain_graph_us
        << " chain_node_us=" << chain_graph_us / chain_nodes
        << " chain_max_abs=" << chain_maximum
        << " shared_chain_graph_us=" << shared_graph_us
        << " shared_chain_node_us=" << shared_graph_us / chain_nodes
        << " shared_chain_max_abs=" << shared_maximum << '\n';
    return 0;
}

int run_backend_argmax_check(
        int64_t elements,
        int repetitions) {
    MFQ_RUNTIME_CHECK(
        elements > 0 && elements <= (1LL << 30),
        "backend argmax element count is out of range");
    MFQ_RUNTIME_CHECK(
        repetitions > 0,
        "backend argmax repetitions must be positive");
    const int64_t expected = elements * 3 / 5;
    auto sequence = mfq_tensor_backend::arange(
        elements,
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kFloat32));
    auto logits = ((sequence - static_cast<double>(expected)).abs() * -1.0)
        .to(mfq_tensor_backend::kBFloat16)
        .contiguous()
        .view({1, elements});

    mfq_tensor_backend::Tensor eager;
    for (int warmup = 0; warmup < 30; ++warmup) {
        eager = mfq_tensor_backend::argmax(logits, -1);
    }
    mfq_hip_synchronize();
    hipEvent_t eager_start = nullptr;
    hipEvent_t eager_stop = nullptr;
    MFQ_CUDA_CHECK(hipEventCreate(&eager_start));
    MFQ_CUDA_CHECK(hipEventCreate(&eager_stop));
    const auto eager_stream = mfq_get_current_cuda_stream();
    MFQ_CUDA_CHECK(hipEventRecord(eager_start, eager_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        eager = mfq_tensor_backend::argmax(logits, -1);
    }
    MFQ_CUDA_CHECK(hipEventRecord(eager_stop, eager_stream.stream()));
    MFQ_CUDA_CHECK(hipEventSynchronize(eager_stop));
    float eager_ms = 0.0f;
    MFQ_CUDA_CHECK(hipEventElapsedTime(&eager_ms, eager_start, eager_stop));
    MFQ_CUDA_CHECK(hipEventDestroy(eager_start));
    MFQ_CUDA_CHECK(hipEventDestroy(eager_stop));

    const auto graph_stream = mfq_get_stream_from_pool(false);
    MfqCudaStreamGuard graph_guard(graph_stream);
    MfqCudaGraph graph;
    mfq_prepare_cuda_graph_memory(graph);
    mfq_tensor_backend::Tensor graph_result =
        mfq_tensor_backend::argmax(logits, -1);
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));
    graph_result = mfq_tensor_backend::Tensor();
    graph.capture_begin();
    graph_result = mfq_tensor_backend::argmax(logits, -1);
    graph.capture_end();
    graph.replay();
    MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream.stream()));

    hipEvent_t graph_start = nullptr;
    hipEvent_t graph_stop = nullptr;
    MFQ_CUDA_CHECK(hipEventCreate(&graph_start));
    MFQ_CUDA_CHECK(hipEventCreate(&graph_stop));
    MFQ_CUDA_CHECK(hipEventRecord(graph_start, graph_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        graph.replay();
    }
    MFQ_CUDA_CHECK(hipEventRecord(graph_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(hipEventSynchronize(graph_stop));
    float graph_ms = 0.0f;
    MFQ_CUDA_CHECK(hipEventElapsedTime(&graph_ms, graph_start, graph_stop));
    MFQ_CUDA_CHECK(hipEventDestroy(graph_start));
    MFQ_CUDA_CHECK(hipEventDestroy(graph_stop));

    const int64_t eager_index = eager.item<int64_t>();
    const int64_t graph_index = graph_result.item<int64_t>();
    MFQ_RUNTIME_CHECK(
        eager_index == expected && graph_index == expected,
        "backend argmax returned an unexpected index");
    std::cout << "backend_argmax_check"
              << " elements=" << elements
              << " repetitions=" << repetitions
              << " expected=" << expected
              << " eager_index=" << eager_index
              << " graph_index=" << graph_index
              << " eager_us=" << eager_ms * 1000.0f / repetitions
              << " graph_us=" << graph_ms * 1000.0f / repetitions
              << "\n";
    return 0;
}
