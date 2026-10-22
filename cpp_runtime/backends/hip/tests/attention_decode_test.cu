#include "mfq_hip_attention_ops.h"
#include "hip_execution.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mfq_tensor_backend;

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

Tensor random_tensor(std::initializer_list<int64_t> shape, ScalarType dtype,
                     std::mt19937& generator) {
    auto host = empty(shape, TensorOptions{}.dtype(kFloat32));
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (int64_t i = 0; i < host.numel(); ++i) host.data_ptr<float>()[i] = distribution(generator);
    return host.to(kCUDA).to(dtype);
}

struct Fixture {
    Tensor q, k, v, lengths, po, pm, pl;
    Fixture(int batch, int ratio, int dim, int capacity, ScalarType dtype, std::mt19937& generator) {
        q = random_tensor({batch, 2 * ratio, 1, dim}, dtype, generator);
        k = random_tensor({batch, 2, capacity, dim}, dtype, generator);
        v = random_tensor({batch, 2, capacity, dim}, dtype, generator);
        const auto options = TensorOptions{}.device(kCUDA);
        lengths = zeros({batch}, options.dtype(kInt64));
        po = empty({batch * 2 * ratio, 64, dim}, options.dtype(kFloat32));
        pm = empty({batch * 2 * ratio, 64}, options.dtype(kFloat32));
        pl = empty_like(pm);
    }
    Tensor run(int mode, int parts = 64) {
        const double scale = 1. / std::sqrt(double(q.size(3)));
        if (mode == 0) return attention_cache_decode_cuda(q, k, v, lengths, scale);
        if (mode == 1) return attention_cache_decode_split_cuda(q, k, v, lengths, scale, po, pm, pl, parts);
        return attention_cache_decode_dynamic_cuda(q, k, v, lengths, scale, po, pm, pl, parts);
    }
    std::vector<double> reference() {
        auto qh = q.to(kFloat32).cpu(), kh = k.to(kFloat32).cpu(), vh = v.to(kFloat32).cpu();
        auto lh = lengths.cpu();
        const int batch = q.size(0), heads = q.size(1), dim = q.size(3), capacity = k.size(2);
        std::vector<double> expected(q.numel(), 0.);
        for (int b = 0; b < batch; ++b) for (int h = 0; h < heads; ++h) {
            const int count = std::max<int64_t>(0, std::min<int64_t>(capacity, lh.data_ptr<int64_t>()[b]));
            if (!count) continue;
            const auto kv_base = size_t(b * 2 + h / (heads / 2)) * capacity * dim;
            std::vector<double> scores(count);
            double maximum = -INFINITY, sum = 0.;
            for (int s = 0; s < count; ++s) {
                double dot = 0.;
                for (int d = 0; d < dim; ++d)
                    dot += double(qh.data_ptr<float>()[(b * heads + h) * dim + d]) *
                           kh.data_ptr<float>()[kv_base + size_t(s) * dim + d];
                scores[s] = dot / std::sqrt(double(dim));
                maximum = std::max(maximum, scores[s]);
            }
            for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
            for (int d = 0; d < dim; ++d) for (int s = 0; s < count; ++s)
                expected[(b * heads + h) * dim + d] +=
                    scores[s] / sum * vh.data_ptr<float>()[kv_base + size_t(s) * dim + d];
        }
        return expected;
    }
    void compare(Tensor output, const std::vector<double>& expected) {
        auto host = output.to(kFloat32).cpu();
        const double tolerance = q.scalar_type() == kBFloat16 ? .008 :
                                 q.scalar_type() == kFloat16 ? .001 : .00003;
        double error = 0., norm = 0.;
        for (size_t i = 0; i < expected.size(); ++i) {
            const double actual = host.data_ptr<float>()[i], difference = actual - expected[i];
            if (!std::isfinite(actual) || std::abs(difference) > tolerance * (1. + std::abs(expected[i])))
                throw std::runtime_error("decode attention differs from FP64 reference");
            error += difference * difference; norm += expected[i] * expected[i];
        }
        if (std::sqrt(error / std::max(norm, 1e-20)) > tolerance * 2.)
            throw std::runtime_error("decode attention relative error exceeds tolerance");
    }
};

void numerical() {
    std::mt19937 generator(20261021);
    int cases = 0;
    for (auto dtype : {kFloat16, kBFloat16, kFloat32}) for (int dim : {64, 128, 256, 512, 80})
        for (int ratio : {1, 4, 8}) for (int kind = 0; kind < 3; ++kind) {
            const std::vector<int64_t> lengths = kind == 0 ? std::vector<int64_t>{33} :
                kind == 1 ? std::vector<int64_t>{-7, 0, 281} : std::vector<int64_t>{191, 193};
            Fixture f(lengths.size(), ratio, dim, 263, dtype, generator);
            f.lengths.copy_(tensor(lengths).to(kCUDA));
            const auto reference = f.reference();
            for (int mode = 0; mode < 3; ++mode) { f.compare(f.run(mode, 7), reference); ++cases; }
        }
    for (auto dtype : {kFloat16, kBFloat16, kFloat32}) {
        Fixture long_context(1, 4, 256, 8193, dtype, generator);
        long_context.lengths.copy_(tensor(std::vector<int64_t>{8193}).to(kCUDA));
        const auto long_reference = long_context.reference();
        for (int mode : {1, 2}) {
            long_context.compare(long_context.run(mode), long_reference); ++cases;
        }
        Fixture extreme(2, 4, 128, 257, dtype, generator);
        extreme.q.copy_(extreme.q * 64.);
        extreme.lengths.copy_(tensor(std::vector<int64_t>{193, 257}).to(kCUDA));
        const auto extreme_reference = extreme.reference();
        for (int mode = 0; mode < 3; ++mode) {
            extreme.compare(extreme.run(mode, 7), extreme_reference); ++cases;
        }
        Fixture masked(1, 4, 256, 131, dtype, generator);
        masked.q.copy_(zeros_like(masked.q));
        masked.lengths.copy_(tensor(std::vector<int64_t>{13}).to(kCUDA));
        for (auto cache : {masked.k, masked.v}) {
            auto host = cache.to(kFloat32).cpu();
            for (int h = 0; h < 2; ++h) for (int t = 13; t < 131; ++t)
                for (int d = 0; d < 256; ++d)
                    host.data_ptr<float>()[(h * 131 + t) * 256 + d] = NAN;
            cache.copy_(host.to(kCUDA).to(dtype));
        }
        const auto masked_reference = masked.reference();
        for (int mode = 0; mode < 3; ++mode) {
            masked.compare(masked.run(mode, 7), masked_reference); ++cases;
        }
        Fixture f(2, 4, 256, 521, dtype, generator);
        const auto stream = mfq_get_stream_from_pool(false);
        MfqCudaStreamGuard guard(stream);
        MfqCudaGraph graph; mfq_prepare_cuda_graph_memory(graph);
        auto output = f.run(2); check(hipStreamSynchronize(stream.stream()));
        output = Tensor{};
        graph.capture_begin(); output = f.run(2); graph.capture_end();
        for (auto lengths : {std::vector<int64_t>{0, 17}, {129, 521}, {574, -2}}) {
            f.q.copy_(random_tensor({2, 8, 1, 256}, dtype, generator));
            f.k.copy_(random_tensor({2, 2, 521, 256}, dtype, generator));
            f.v.copy_(random_tensor({2, 2, 521, 256}, dtype, generator));
            f.lengths.copy_(tensor(lengths).to(kCUDA));
            graph.replay(); check(hipStreamSynchronize(stream.stream()));
            f.compare(output, f.reference()); ++cases;
        }
    }
    std::cout << "attention_decode FP64-reference cases=" << cases << " changing_graph_replays=9 PASS\n";
}

void benchmark() {
    std::mt19937 generator(20261021);
    Fixture f(1, 4, 256, 16384, kFloat16, generator);
    for (int length : {32, 2048, 8192}) for (int limit : {1, 16, 64}) {
        f.lengths.copy_(tensor(std::vector<int64_t>{length}).to(kCUDA));
        const int parts = std::min(limit, std::max(2, (length + 127) / 128));
        const auto stream = mfq_get_stream_from_pool(false); MfqCudaStreamGuard guard(stream);
        MfqCudaGraph graph; mfq_prepare_cuda_graph_memory(graph);
        auto run = [&] { return f.run(limit == 1 ? 0 : 1, parts); };
        std::vector<Tensor> outputs;
        for (int i = 0; i < 32; ++i) outputs.push_back(run());
        check(hipStreamSynchronize(stream.stream())); outputs.clear();
        graph.capture_begin(); for (int i = 0; i < 32; ++i) outputs.push_back(run()); graph.capture_end();
        graph.replay(); check(hipStreamSynchronize(stream.stream()));
        hipEvent_t start, stop; check(hipEventCreate(&start)); check(hipEventCreate(&stop));
        std::vector<double> samples;
        for (int sample = 0; sample < 7; ++sample) {
            check(hipEventRecord(start, stream.stream()));
            for (int i = 0; i < 16; ++i) graph.replay();
            check(hipEventRecord(stop, stream.stream())); check(hipEventSynchronize(stop));
            float ms; check(hipEventElapsedTime(&ms, start, stop)); samples.push_back(ms * 1000. / 512.);
        }
        check(hipEventDestroy(start)); check(hipEventDestroy(stop));
        auto sorted = samples; std::sort(sorted.begin(), sorted.end());
        std::cout << "{\"length\":" << length << ",\"parts_limit\":" << limit
                  << ",\"parts\":" << (limit == 1 ? 1 : parts) << ",\"median_us\":" << sorted[3]
                  << ",\"samples_us\":[";
        for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << "]}\n" << std::flush;
    }
}
}
int main(int argc, char** argv) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || !count) return 77;
    try {
        if (argc == 2 && std::string(argv[1]) == "--bench") benchmark(); else numerical();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
