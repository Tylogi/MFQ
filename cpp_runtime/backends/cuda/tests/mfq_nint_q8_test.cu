#include "mfq_cuda_ops.h"

#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace mfq::cuda;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void exact_half(const Tensor &actual, const Tensor &expected) {
    require(actual.sizes() == expected.sizes() &&
            actual.scalar_type() == kFloat16 &&
            expected.scalar_type() == kFloat16, "NINT output contract mismatch");
    auto a = actual.to(kCPU).contiguous();
    auto b = expected.to(kCPU).contiguous();
    require(std::memcmp(a.data_ptr(), b.data_ptr(),
            static_cast<std::size_t>(a.numel()) * sizeof(std::uint16_t)) == 0,
            "NINT specialized output differs from the original generic path");
}

void check_case(int gs, int ng, int m, int width_tail, int storage_offset,
                int mode, int &cases, int &graphs) {
    const Device gpu{DeviceType::cuda, 0};
    const auto options = TensorOptions().device(gpu);
    constexpr int n = 5;  // Exercises a partial four-warp output block.
    const int k = gs * ng;
    const int stride = (k + 3) / 4 * 4;
    std::vector<std::uint8_t> bytes(n * stride + storage_offset, 0);
    std::vector<std::int64_t> offsets(n);
    for (int row = 0; row < n; ++row) {
        offsets[row] = static_cast<std::int64_t>(row) * stride * 8;
        for (int col = 0; col < k; ++col) {
            bytes[storage_offset + row * stride + col] =
                static_cast<std::uint8_t>((row * 97 + col * 31) % 256);
        }
    }
    auto q = tensor(bytes).to(gpu).narrow(0, storage_offset, n * stride);
    auto bits = tensor(std::vector<std::uint8_t>(n, 8)).to(gpu);
    auto off = tensor(offsets).to(gpu);
    std::vector<std::uint8_t> scales(n * ng), minima(n * ng);
    for (int i = 0; i < n * ng; ++i) {
        scales[i] = static_cast<std::uint8_t>(i % 13);
        minima[i] = static_cast<std::uint8_t>(i % 7);
    }
    auto s = tensor(scales).reshape({n, ng}).to(gpu);
    auto mn = tensor(minima).reshape({n, ng}).to(gpu);
    auto ns = tensor<float>({0.0001f, 0.0002f, 0.0003f, 0.0004f, 0.0005f}).to(gpu);
    auto nm = tensor<float>({0.01f, 0.02f, 0.03f, 0.04f, 0.05f}).to(gpu);
    std::vector<float> input(m * (k - width_tail));
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 37) - 18) / 8;
    }
    auto x = tensor(input).reshape({m, k - width_tail}).to(gpu, kFloat16);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 53) - 26) / 2;
    }
    auto gate = tensor(input).reshape({m, k - width_tail}).to(gpu, kFloat16);
    auto qx = empty({m, k}, options.dtype(kInt8));
    auto xs = empty({m, ng}, options.dtype(kFloat32));
    auto run = [&](bool fast) {
        if (mode == 0) {
            return fast
                ? nint_matmul_q8_ws_cuda(q, bits, off, s, mn, ns, nm, x, gs, qx, xs)
                : nint_matmul_ws_cuda(q, bits, off, s, mn, ns, nm, x, gs, qx, xs);
        }
        return fast
            ? nint_matmul_input_mul_q8_ws_cuda(
                q, bits, off, s, mn, ns, nm, x, gate, mode, gs, qx, xs)
            : nint_matmul_input_mul_ws_cuda(
                q, bits, off, s, mn, ns, nm, x, gate, mode, gs, qx, xs);
    };
    const auto expected = run(false);
    const auto actual = run(true);
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
    exact_half(actual, expected);
    ++cases;

    // Compare both specialization branches during capture/replay. Odd group
    // widths and offset storage also exercise the existing generic fallback.
    if (ng == 3 && width_tail == 0 && storage_offset == 0 && (m == 1 || m == 8)) {
        const auto stream = mfq_get_stream_from_pool(false);
        MfqCudaStreamGuard guard(stream);
        MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);
        Tensor result = run(true);
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        result = Tensor();
        graph.capture_begin();
        result = run(true);
        graph.capture_end();
        for (int repeat = 0; repeat < 3; ++repeat) {
            graph.replay();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            exact_half(result, expected);
        }
        ++graphs;
    }
}

void check_q46_case(int width, int gs, int ng, int mode, int width_tail,
                    int storage_offset, int &cases, int &graphs) {
    const Device gpu{DeviceType::cuda, 0};
    const auto options = TensorOptions().device(gpu);
    constexpr int n = 5;
    const int k = gs * ng;
    const std::uint64_t row_bits = static_cast<std::uint64_t>(k) * width;
    require(row_bits % 8 == 0, "q4/q6 test row must be byte aligned");
    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(n * row_bits / 8) + storage_offset + 8, 0);
    std::vector<std::int64_t> offsets(n);
    for (int row = 0; row < n; ++row) {
        offsets[row] = static_cast<std::int64_t>(row * row_bits);
        for (int col = 0; col < k; ++col) {
            const std::uint32_t code = static_cast<std::uint32_t>(
                (row * 97 + col * 31) & ((1 << width) - 1));
            const std::uint64_t bit = row * row_bits +
                static_cast<std::uint64_t>(col) * width;
            for (int component = 0; component < width; ++component) {
                bytes[storage_offset + (bit + component) / 8] |=
                    static_cast<std::uint8_t>(
                        ((code >> component) & 1u) << ((bit + component) & 7u));
            }
        }
    }
    auto q = tensor(bytes).to(gpu).narrow(
        0, storage_offset, static_cast<std::int64_t>(n * row_bits / 8 + 8));
    auto bits = tensor(std::vector<std::uint8_t>(n, static_cast<std::uint8_t>(width))).to(gpu);
    auto off = tensor(offsets).to(gpu);
    std::vector<std::uint8_t> scales(n * ng), minima(n * ng);
    for (int i = 0; i < n * ng; ++i) {
        scales[i] = static_cast<std::uint8_t>(i % 13);
        minima[i] = static_cast<std::uint8_t>(i % 7);
    }
    auto s = tensor(scales).reshape({n, ng}).to(gpu);
    auto mn = tensor(minima).reshape({n, ng}).to(gpu);
    auto ns = tensor<float>({0.0001f, 0.0002f, 0.0003f, 0.0004f, 0.0005f}).to(gpu);
    auto nm = tensor<float>({0.01f, 0.02f, 0.03f, 0.04f, 0.05f}).to(gpu);
    std::vector<float> input(k - width_tail);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 37) - 18) / 8;
    }
    auto x = tensor(input).reshape({1, k - width_tail}).to(gpu, kFloat16);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 53) - 26) / 2;
    }
    auto gate = tensor(input).reshape({1, k - width_tail}).to(gpu, kFloat16);
    auto qx = empty({1, k}, options.dtype(kInt8));
    auto xs = empty({1, ng}, options.dtype(kFloat32));
    auto run = [&](bool specialized) {
        if (mode == 0) {
            return specialized
                ? nint_matmul_q46_ws_cuda(q, bits, off, s, mn, ns, nm,
                    x, gs, qx, xs, width)
                : nint_matmul_ws_cuda(q, bits, off, s, mn, ns, nm,
                    x, gs, qx, xs);
        }
        return specialized
            ? nint_matmul_input_mul_q46_ws_cuda(
                q, bits, off, s, mn, ns, nm, x, gate, mode, gs, qx, xs, width)
            : nint_matmul_input_mul_ws_cuda(
                q, bits, off, s, mn, ns, nm, x, gate, mode, gs, qx, xs);
    };
    const auto expected = run(false);
    const auto actual = run(true);
    MFQ_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());
    exact_half(actual, expected);
    ++cases;
    if (gs == 24 && ng == 3 && width_tail == 0 && storage_offset == 0) {
        const auto stream = mfq_get_stream_from_pool(false);
        MfqCudaStreamGuard guard(stream);
        MfqCudaGraph graph;
        mfq_prepare_cuda_graph_memory(graph);
        Tensor result = run(true);
        MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        result = Tensor();
        graph.capture_begin();
        result = run(true);
        graph.capture_end();
        for (int repeat = 0; repeat < 3; ++repeat) {
            graph.replay();
            MFQ_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
            exact_half(result, expected);
        }
        ++graphs;
    }
}
}  // namespace

int main() {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
                (status == cudaSuccess && devices == 0)) {
            return 77;
        }
        MFQ_NATIVE_CUDA_CHECK(status);
        int cases = 0, graphs = 0;
        for (int gs : {4, 12, 28, 32, 48, 64, 5, 47, 63}) {
            for (int ng : {1, 3, 22}) {
                for (int m = 1; m <= 8; ++m) {
                    for (int tail : {0, 1}) {
                        for (int offset : {0, 1}) {
                            for (int mode : {0, 1, 2}) {
                                check_case(gs, ng, m, tail, offset, mode, cases, graphs);
                            }
                        }
                    }
                }
            }
        }
        int q46_cases = 0, q46_graphs = 0;
        for (int width : {4, 6}) {
            for (int gs : {4, 12, 24, 48, 64}) {
                for (int ng : {1, 3, 22}) {
                    for (int mode : {0, 1, 2}) {
                        for (int tail : {0, 1}) {
                            for (int offset : {0, 1}) {
                                check_q46_case(width, gs, ng, mode, tail,
                                    offset, q46_cases, q46_graphs);
                            }
                        }
                    }
                }
            }
        }
        std::cout << "NINT q8 bit-exact FP16 cases=" << cases
                  << " graph_cases=" << graphs << " replays_per_case=3\n"
                  << "NINT q4/q6 bit-exact FP16 cases=" << q46_cases
                  << " graph_cases=" << q46_graphs << " replays_per_case=3\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
