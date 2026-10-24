#include "mfq_hip_context.h"
#include "mfq_hip_sampling_ops.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace mfq::cuda;

const char* dtype_name(ScalarType dtype) {
    return dtype == kFloat16 ? "f16" : dtype == kBFloat16 ? "bf16" : "f32";
}

void fill_logits(Tensor& host, int pattern) {
    const int rows = int(host.size(0));
    const int vocab = int(host.size(1));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (int row = 0; row < rows; ++row) {
        float* values = host.data_ptr<float>() + size_t(row) * vocab;
        for (int i = 0; i < vocab; ++i)
            values[i] = float((i * 37 + row * 19) % 113 - 56) * .125f;
        if (pattern == 1) {
            std::fill_n(values, vocab, -1.f);
            values[std::min(4095, vocab - 1)] = 7.f;
            values[std::min(4096, vocab - 1)] = 7.f;
            values[vocab - 1] = 7.f;
        } else if (pattern == 2) {
            values[vocab - 1 - row % vocab] = 20.f;
        } else if (pattern == 3) {
            for (int i = 0; i < vocab; ++i) values[i] = i & 1 ? 0.f : -0.f;
        } else if (pattern == 4) {
            std::fill_n(values, vocab, nan);
        } else if (pattern == 5) {
            std::fill_n(values, vocab, -inf);
            values[0] = nan;
        } else if (pattern == 6) {
            std::fill_n(values, vocab, -FLT_MAX);
            values[0] = nan;
        } else if (pattern == 7) {
            values[std::min(4096, vocab - 1)] = inf;
            values[vocab - 1] = inf;
        } else if (pattern == 8) {
            for (int i = 0; i < vocab; i += 3) values[i] = nan;
            values[vocab - 1] = 30.f;
        } else if (pattern == 9) {
            std::fill_n(values, vocab, 1.f);
            values[vocab - 1] = 1.00001f;
        }
    }
}

void check_output(const Tensor& logits, const Tensor& output, const std::string& label) {
    auto rounded = logits.to(kFloat32).cpu();
    auto actual = output.cpu();
    const int rows = int(logits.size(0));
    const int vocab = int(logits.size(1));
    for (int row = 0; row < rows; ++row) {
        const float* values = rounded.data_ptr<float>() + size_t(row) * vocab;
        float best = -FLT_MAX;
        int expected = 0;
        for (int i = 0; i < vocab; ++i) {
            if (values[i] > best) {
                best = values[i];
                expected = i;
            }
        }
        if (actual.data_ptr<int64_t>()[row] != expected) {
            throw std::runtime_error(label + " row=" + std::to_string(row) +
                " expected=" + std::to_string(expected) +
                " actual=" + std::to_string(actual.data_ptr<int64_t>()[row]));
        }
    }
}

Tensor make_logits(int rows, int vocab, ScalarType dtype, int offset) {
    auto storage = empty({int64_t(rows) * vocab + offset},
        TensorOptions().device(kCUDA).dtype(dtype));
    return storage.narrow(0, offset, int64_t(rows) * vocab).reshape({rows, vocab});
}

std::string describe(int rows, int vocab, ScalarType dtype, int offset) {
    return std::string(dtype_name(dtype)) + " rows=" + std::to_string(rows) +
        " vocab=" + std::to_string(vocab) + " offset=" + std::to_string(offset);
}

int check_eager(int rows, int vocab, ScalarType dtype, int offset) {
    auto host = empty({rows, vocab}, TensorOptions().dtype(kFloat32));
    auto logits = make_logits(rows, vocab, dtype, offset);
    const auto label = describe(rows, vocab, dtype, offset);
    for (int pattern = 0; pattern < 10; ++pattern) {
        fill_logits(host, pattern);
        logits.copy_(host.to(kCUDA).to(dtype));
        auto output = sample_greedy_cuda(logits);
        check_output(logits, output, label + " pattern=" + std::to_string(pattern));
    }
    return 10;
}

int check_graph(int rows, int vocab, ScalarType dtype, int offset) {
    auto stream = stream_from_pool(false);
    StreamGuard guard(stream);
    auto host = empty({rows, vocab}, TensorOptions().dtype(kFloat32));
    auto logits = make_logits(rows, vocab, dtype, offset);
    fill_logits(host, 0);
    logits.copy_(host.to(kCUDA).to(dtype));
    Graph graph;
    graph.prepare_memory();
    auto output = sample_greedy_cuda(logits);
    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream.stream()));
    output = {};
    graph.capture_begin();
    output = sample_greedy_cuda(logits);
    graph.capture_end();
    for (int pattern : {2, 1, 5, 7}) {
        fill_logits(host, pattern);
        logits.copy_(host.to(kCUDA).to(dtype));
        output.fill_(-1);
        graph.replay();
        graph.replay();
        MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream.stream()));
        check_output(logits, output,
            "graph " + describe(rows, vocab, dtype, offset) +
            " pattern=" + std::to_string(pattern));
    }
    return 4;
}

void benchmark(int rows, int vocab, ScalarType dtype) {
    auto stream = stream_from_pool(false);
    StreamGuard guard(stream);
    auto host = empty({rows, vocab}, TensorOptions().dtype(kFloat32));
    fill_logits(host, 0);
    auto logits = host.to(kCUDA).to(dtype);
    Graph graph;
    graph.prepare_memory();
    auto output = sample_greedy_cuda(logits);
    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream.stream()));
    output = {};
    graph.capture_begin();
    output = sample_greedy_cuda(logits);
    graph.capture_end();
    for (int i = 0; i < 8; ++i) graph.replay();
    Event begin(hipEventDefault), end(hipEventDefault);
    constexpr int repeats = 512;
    begin.record(stream.stream());
    for (int i = 0; i < repeats; ++i) graph.replay();
    end.record(stream.stream());
    end.synchronize();
    float milliseconds = 0;
    MFQ_NATIVE_CUDA_CHECK(hipEventElapsedTime(&milliseconds, begin.get(), end.get()));
    check_output(logits, output, "benchmark");
    std::cout << "greedy " << describe(rows, vocab, dtype, 0)
              << " us=" << milliseconds * 1000 / repeats << '\n';
}
} // namespace

int main(int argc, char** argv) try {
    int devices = 0;
    MFQ_NATIVE_CUDA_CHECK(hipGetDeviceCount(&devices));
    if (devices == 0) return 77;
    if (argc > 1) {
        if (argc != 2 || std::string(argv[1]) != "--bench")
            throw std::runtime_error("expected --bench or no arguments");
        for (auto dtype : {kFloat32, kFloat16, kBFloat16})
            for (int rows : {1, 4})
                for (int vocab : {4096, 16384, 32768, 151936, 248320})
                    benchmark(rows, vocab, dtype);
        return 0;
    }
    int eager = 0, graph = 0;
    for (auto dtype : {kFloat32, kFloat16, kBFloat16}) {
        for (int rows : {1, 3}) {
            for (int offset : {0, 1}) {
                for (int vocab : {1, 31, 256, 4095, 4096, 4097, 16383,
                                  16384, 16385, 32768, 151936, 248320, 248321})
                    eager += check_eager(rows, vocab, dtype, offset);
                for (int vocab : {31, 16383, 16384, 16385, 248320, 248321})
                    graph += check_graph(rows, vocab, dtype, offset);
            }
        }
    }
    std::cout << "greedy eager_cases=" << eager << " graph_cases=" << graph
              << " replays_per_case=2 passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
