#include "mfq_hip_attention_ops.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace mfq_tensor_backend;

Tensor input(std::initializer_list<int64_t> shape, ScalarType dtype,
             std::mt19937& generator) {
    auto host = empty(shape, TensorOptions{}.dtype(kFloat32));
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int64_t i = 0; i < host.numel(); ++i) {
        host.data_ptr<float>()[i] = distribution(generator);
    }
    return host.to(kCUDA).to(dtype);
}

void check_attention(int B, int T, int ratio) {
    constexpr int Hk = 2, D = 256;
    const int Hq = Hk * ratio;
    std::mt19937 generator(20261013 + T);
    auto q = input({B, Hq, T, D}, kFloat32, generator);
    auto k = input({B, Hk, T, D}, kFloat16, generator);
    auto v = input({B, Hk, T, D}, kFloat16, generator);
    auto output = mfq_attention_mma256_cuda(q, k, v, 1.0 / std::sqrt(double(D)));
    MFQ_CUDA_CHECK(hipDeviceSynchronize());
    auto actual = output.cpu();
    auto q_host = q.cpu();
    auto k_host = k.to(kFloat32).cpu();
    auto v_host = v.to(kFloat32).cpu();
    const float* q_values = q_host.data_ptr<float>();
    const float* k_values = k_host.data_ptr<float>();
    const float* v_values = v_host.data_ptr<float>();
    const float* got = actual.data_ptr<float>();
    double error_squared = 0.0, reference_squared = 0.0, max_error = 0.0;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < Hq; ++h) {
            const int kv_head = b * Hk + h / ratio;
            for (int t = 0; t < T; ++t) {
                std::vector<double> weights(t + 1);
                double max_score = -std::numeric_limits<double>::infinity();
                const float* query = q_values + ((b * Hq + h) * T + t) * D;
                for (int s = 0; s <= t; ++s) {
                    const float* key = k_values + (kv_head * T + s) * D;
                    double score = 0.0;
                    for (int d = 0; d < D; ++d) {
                        score += double(query[d]) * key[d];
                    }
                    weights[s] = score / std::sqrt(double(D));
                    max_score = std::max(max_score, weights[s]);
                }
                double sum = 0.0;
                for (double& weight : weights) {
                    weight = std::exp(weight - max_score);
                    sum += weight;
                }
                for (int d = 0; d < D; ++d) {
                    double expected = 0.0;
                    for (int s = 0; s <= t; ++s) {
                        expected += weights[s] *
                            v_values[(kv_head * T + s) * D + d] / sum;
                    }
                    const double difference =
                        got[((b * T + t) * Hq + h) * D + d] - expected;
                    if (!std::isfinite(difference)) {
                        throw std::runtime_error("attention MMA output is not finite");
                    }
                    error_squared += difference * difference;
                    reference_squared += expected * expected;
                    max_error = std::max(max_error, std::abs(difference));
                }
            }
        }
    }
    const double relative_error = std::sqrt(error_squared / reference_squared);
    std::cout << "B=" << B << " T=" << T << " GQA=" << ratio
              << " relative_error=" << relative_error
              << " max_error=" << max_error << '\n';
    if (relative_error > 0.003 || max_error > 0.003) {
        throw std::runtime_error("attention MMA numerical mismatch");
    }
}

void check_unsupported_config() {
    auto options = TensorOptions{}.device(kCUDA);
    auto q = empty({1, 8, 256, 512}, options.dtype(kFloat32));
    auto k = empty({1, 1, 256, 512}, options.dtype(kFloat16));
    auto v = empty({1, 1, 256, 512}, options.dtype(kFloat16));
    try {
        mfq_attention_mma512_cuda(q, k, v, 1.0 / std::sqrt(512.0));
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("unsupported HIP tile") != std::string::npos) {
            return;
        }
        throw;
    }
    throw std::runtime_error("unsupported attention MMA tile was accepted");
}
}  // namespace

int main() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return 77;
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 77;
    const std::string arch(properties.gcnArchName);
    if (arch.rfind("gfx908", 0) != 0 && arch.rfind("gfx90a", 0) != 0 &&
        arch.rfind("gfx942", 0) != 0 && arch.rfind("gfx950", 0) != 0) return 77;
    try {
        check_unsupported_config();
        for (int ratio : {4, 8}) {
            for (int tokens : {1, 8, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 257}) {
                check_attention(1, tokens, ratio);
            }
            check_attention(2, 65, ratio);
            check_attention(2, 129, ratio);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
