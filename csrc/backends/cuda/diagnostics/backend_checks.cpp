#include "backend_checks.h"

#include "../kernels/mfq_cuda_activation_ops.h"
#include "../kernels/mfq_cuda_sampling_ops.h"
#include "../native/tensor_backend.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MFQ_CUDA_CHECK
#define MFQ_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
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
    mfq_cuda_synchronize();
    cudaEvent_t eager_start = nullptr;
    cudaEvent_t eager_stop = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&eager_start));
    MFQ_CUDA_CHECK(cudaEventCreate(&eager_stop));
    const auto eager_stream = mfq_get_current_cuda_stream();
    MFQ_CUDA_CHECK(cudaEventRecord(eager_start, eager_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        eager = (left + right).contiguous();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(eager_stop, eager_stream.stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(eager_stop));
    float eager_ms = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&eager_ms, eager_start, eager_stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(eager_start));
    MFQ_CUDA_CHECK(cudaEventDestroy(eager_stop));

    const auto graph_stream = mfq_get_stream_from_pool(false);
    MfqCudaStreamGuard graph_guard(graph_stream);
    MfqCudaGraph graph;
    mfq_prepare_cuda_graph_memory(graph);
    mfq_tensor_backend::Tensor graph_result = (left + right).contiguous();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    graph_result = mfq_tensor_backend::Tensor();
    graph.capture_begin();
    graph_result = (left + right).contiguous();
    graph.capture_end();
    graph.replay();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    cudaEvent_t graph_start = nullptr;
    cudaEvent_t graph_stop = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&graph_start));
    MFQ_CUDA_CHECK(cudaEventCreate(&graph_stop));
    MFQ_CUDA_CHECK(cudaEventRecord(graph_start, graph_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        graph.replay();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(graph_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(graph_stop));
    float graph_ms = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&graph_ms, graph_start, graph_stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(graph_start));
    MFQ_CUDA_CHECK(cudaEventDestroy(graph_stop));

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
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    chain_result = mfq_tensor_backend::Tensor();
    chain_graph.capture_begin();
    for (int node = 0; node < chain_nodes; ++node) {
        chain_result = (left + right).contiguous();
    }
    chain_graph.capture_end();
    chain_graph.replay();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    cudaEvent_t chain_start = nullptr;
    cudaEvent_t chain_stop = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&chain_start));
    MFQ_CUDA_CHECK(cudaEventCreate(&chain_stop));
    MFQ_CUDA_CHECK(cudaEventRecord(chain_start, graph_stream.stream()));
    for (int repetition = 0;
            repetition < chain_repetitions;
            ++repetition) {
        chain_graph.replay();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(chain_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(chain_stop));
    float chain_ms = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(
        &chain_ms, chain_start, chain_stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(chain_start));
    MFQ_CUDA_CHECK(cudaEventDestroy(chain_stop));
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
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    MfqCudaGraph shared_graph;
    mfq_prepare_cuda_graph_memory(shared_graph);
    mfq_tensor_backend::Tensor shared_result;
    for (int node = 0; node < chain_nodes; ++node) {
        shared_result = silu_mul_cuda(left, right);
    }
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    shared_result = mfq_tensor_backend::Tensor();
    shared_graph.capture_begin();
    for (int node = 0; node < chain_nodes; ++node) {
        shared_result = silu_mul_cuda(left, right);
    }
    shared_graph.capture_end();
    shared_graph.replay();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    cudaEvent_t shared_start = nullptr;
    cudaEvent_t shared_stop = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&shared_start));
    MFQ_CUDA_CHECK(cudaEventCreate(&shared_stop));
    MFQ_CUDA_CHECK(cudaEventRecord(shared_start, graph_stream.stream()));
    for (int repetition = 0;
            repetition < chain_repetitions;
            ++repetition) {
        shared_graph.replay();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(shared_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(shared_stop));
    float shared_ms = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(
        &shared_ms, shared_start, shared_stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(shared_start));
    MFQ_CUDA_CHECK(cudaEventDestroy(shared_stop));
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

static int run_backend_argmax_dtype_check(
        int64_t elements,
        int repetitions,
        mfq_tensor_backend::ScalarType dtype) {
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
        .to(dtype)
        .contiguous()
        .view({1, elements});

    // A non-last axis with one output row forces the generic serial kernel
    // over exactly the same logits. Keep its timing short for large vocabularies.
    auto generic_input = logits.transpose(0, 1);
    cudaEvent_t generic_start = nullptr, generic_stop = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&generic_start));
    MFQ_CUDA_CHECK(cudaEventCreate(&generic_stop));
    constexpr int generic_repetitions = 8;
    mfq_tensor_backend::Tensor generic_result;
    const auto generic_stream = mfq_get_current_cuda_stream();
    MFQ_CUDA_CHECK(cudaEventRecord(generic_start, generic_stream.stream()));
    for (int i = 0; i < generic_repetitions; ++i)
        generic_result = generic_input.argmax(0, false);
    MFQ_CUDA_CHECK(cudaEventRecord(generic_stop, generic_stream.stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(generic_stop));
    float generic_ms = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&generic_ms, generic_start, generic_stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(generic_start));
    MFQ_CUDA_CHECK(cudaEventDestroy(generic_stop));
    MFQ_RUNTIME_CHECK(generic_result.item<int64_t>() == expected,
        "generic argmax reference returned an unexpected index");

    mfq_tensor_backend::Tensor eager;
    for (int warmup = 0; warmup < 30; ++warmup) {
        eager = mfq_tensor_backend::argmax(logits, -1);
    }
    mfq_cuda_synchronize();
    cudaEvent_t eager_start = nullptr;
    cudaEvent_t eager_stop = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&eager_start));
    MFQ_CUDA_CHECK(cudaEventCreate(&eager_stop));
    const auto eager_stream = mfq_get_current_cuda_stream();
    MFQ_CUDA_CHECK(cudaEventRecord(eager_start, eager_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        eager = mfq_tensor_backend::argmax(logits, -1);
    }
    MFQ_CUDA_CHECK(cudaEventRecord(eager_stop, eager_stream.stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(eager_stop));
    float eager_ms = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&eager_ms, eager_start, eager_stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(eager_start));
    MFQ_CUDA_CHECK(cudaEventDestroy(eager_stop));

    const auto graph_stream = mfq_get_stream_from_pool(false);
    MfqCudaStreamGuard graph_guard(graph_stream);
    MfqCudaGraph graph;
    mfq_prepare_cuda_graph_memory(graph);
    mfq_tensor_backend::Tensor graph_result =
        mfq_tensor_backend::argmax(logits, -1);
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    graph_result = mfq_tensor_backend::Tensor();
    graph.capture_begin();
    graph_result = mfq_tensor_backend::argmax(logits, -1);
    graph.capture_end();
    graph.replay();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    cudaEvent_t graph_start = nullptr;
    cudaEvent_t graph_stop = nullptr;
    MFQ_CUDA_CHECK(cudaEventCreate(&graph_start));
    MFQ_CUDA_CHECK(cudaEventCreate(&graph_stop));
    MFQ_CUDA_CHECK(cudaEventRecord(graph_start, graph_stream.stream()));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        graph.replay();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(graph_stop, graph_stream.stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(graph_stop));
    float graph_ms = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&graph_ms, graph_start, graph_stop));
    MFQ_CUDA_CHECK(cudaEventDestroy(graph_start));
    MFQ_CUDA_CHECK(cudaEventDestroy(graph_stop));

    const int64_t eager_index = eager.item<int64_t>();
    const int64_t graph_index = graph_result.item<int64_t>();
    MFQ_RUNTIME_CHECK(
        eager_index == expected && graph_index == expected,
        "backend argmax returned an unexpected index");
    std::cout << "backend_argmax_check"
              << " dtype=" << static_cast<int>(dtype)
              << " elements=" << elements
              << " repetitions=" << repetitions
              << " expected=" << expected
              << " eager_index=" << eager_index
              << " graph_index=" << graph_index
              << " generic_us=" << generic_ms * 1000.0f / generic_repetitions
              << " eager_us=" << eager_ms * 1000.0f / repetitions
              << " graph_us=" << graph_ms * 1000.0f / repetitions
              << "\n";
    return 0;
}

int run_backend_argmax_check(int64_t elements, int repetitions) {
    namespace tb = mfq_tensor_backend;
    int cases = 0;
    for (auto dtype : {tb::kBFloat16, tb::kFloat16, tb::kFloat32}) {
        for (int64_t width : {1LL, 3LL, 31LL, 255LL, 256LL, 257LL, 8193LL, 248320LL}) {
            std::vector<float> values(static_cast<size_t>(3 * width), -7.0f);
            std::vector<int64_t> expected(3);
            for (int row = 0; row < 3; ++row) {
                const auto first = std::min<int64_t>(row * 17, width - 1);
                values[row * width + first] = 5.0f;
                values[row * width + width - 1] = 5.0f;
                expected[row] = first;
            }
            auto input = tb::tensor(values).reshape({3, width}).to(dtype).to(tb::kCUDA);
            const auto check = [&](const tb::Tensor& indices) {
                const auto host = indices.cpu().contiguous();
                MFQ_RUNTIME_CHECK(host.scalar_type() == tb::kInt64 && host.numel() == 3,
                    "argmax result dtype or size changed");
                for (int row = 0; row < 3; ++row)
                    MFQ_RUNTIME_CHECK(host.data_ptr<int64_t>()[row] == expected[row],
                        "argmax failed first-maximum or batched-row check");
            };
            check(input.argmax(-1, false));
            check(input.argmax(1, true));
            check(input.transpose(0, 1).contiguous().argmax(0, false));
            ++cases;
        }
        if (dtype != tb::kBFloat16) {
            // Match the existing generic floating reduction on NaNs, infinities,
            // signed zero and rows without a value greater than lowest().
            constexpr int64_t width = 513;
            std::vector<float> values(4 * width, std::numeric_limits<float>::quiet_NaN());
            std::fill(values.begin() + width, values.begin() + 2 * width,
                -std::numeric_limits<float>::infinity());
            values[width + 257] = values[width + 512] = std::numeric_limits<float>::infinity();
            std::fill(values.begin() + 2 * width, values.begin() + 3 * width, -0.0f);
            values[2 * width + 32] = 0.0f;
            std::fill(values.begin() + 3 * width, values.end(), std::numeric_limits<float>::lowest());
            auto input = tb::tensor(values).reshape({4, width}).to(dtype).to(tb::kCUDA);
            auto reference = input.transpose(0, 1).contiguous().argmax(0, false).cpu().contiguous();
            auto actual = input.argmax(-1, false).cpu().contiguous();
            MFQ_RUNTIME_CHECK(std::memcmp(reference.data_ptr(), actual.data_ptr(), 4 * sizeof(int64_t)) == 0,
                "parallel argmax changed generic nonfinite or signed-zero behavior");
            ++cases;
        }
        run_backend_argmax_dtype_check(elements, repetitions, dtype);
        for (int64_t width : {1LL, 3LL, 257LL, 16383LL, 16384LL, 16385LL, 248320LL, 248321LL}) {
            constexpr int rows = 6;
            std::vector<float> values(rows * width, -7.0f);
            for (int row = 0; row < rows; ++row) {
                auto begin = values.begin() + row * width;
                if (row == 0) {
                    values[std::min<int64_t>(4095, width - 1)] = 5.0f;
                    values[width - 1] = 5.0f;
                } else if (row == 1) std::fill(begin, begin + width, std::numeric_limits<float>::quiet_NaN());
                else if (row == 2) std::fill(begin, begin + width, -std::numeric_limits<float>::infinity());
                else if (row == 3) {
                    std::fill(begin, begin + width, std::numeric_limits<float>::quiet_NaN());
                    begin[std::min<int64_t>(4096, width - 1)] = std::numeric_limits<float>::infinity();
                    begin[width - 1] = std::numeric_limits<float>::infinity();
                } else if (row == 4) {
                    std::fill(begin, begin + width, -0.0f);
                    begin[width - 1] = 0.0f;
                } else std::fill(begin, begin + width, std::numeric_limits<float>::lowest());
            }
            auto input = tb::tensor(values).reshape({rows, width}).to(dtype).to(tb::kCUDA);
            auto quantized = input.to(tb::kCPU, tb::kFloat32).contiguous();
            auto actual = sample_greedy_cuda(input).cpu().contiguous();
            for (int row = 0; row < rows; ++row) {
                float best = std::numeric_limits<float>::lowest();
                int64_t expected = 0;
                for (int64_t column = 0; column < width; ++column) {
                    const float value = quantized.data_ptr<float>()[row * width + column];
                    if (value > best) { best = value; expected = column; }
                }
                MFQ_RUNTIME_CHECK(actual.data_ptr<int64_t>()[row] == expected,
                    "greedy sampler changed tail, tie, NaN, infinity or fallback behavior");
            }
            ++cases;
        }
        for (int rows : {1, 6}) {
            auto input = tb::zeros({rows, elements}, tb::TensorOptions().device(tb::kCUDA).dtype(dtype));
            for (int i = 0; i < 20; ++i) (void)sample_greedy_cuda(input);
            mfq_cuda_synchronize();
            cudaEvent_t begin, end;
            MFQ_CUDA_CHECK(cudaEventCreate(&begin)); MFQ_CUDA_CHECK(cudaEventCreate(&end));
            const auto stream = mfq_current_cuda_stream();
            MFQ_CUDA_CHECK(cudaEventRecord(begin, stream));
            for (int i = 0; i < repetitions; ++i) (void)sample_greedy_cuda(input);
            MFQ_CUDA_CHECK(cudaEventRecord(end, stream)); MFQ_CUDA_CHECK(cudaEventSynchronize(end));
            float ms = 0; MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            MFQ_CUDA_CHECK(cudaEventDestroy(begin)); MFQ_CUDA_CHECK(cudaEventDestroy(end));
            std::cout << "backend_greedy rows=" << rows << " dtype=" << static_cast<int>(dtype)
                      << " eager_us=" << ms * 1000 / repetitions << "\n";
        }
    }
    std::cout << "backend_argmax_cases=" << cases
              << " BF16/FP16/FP32, batched rows, tails, ties, keepdim, strided fallback, nonfinite PASS\n";
    return 0;
}
