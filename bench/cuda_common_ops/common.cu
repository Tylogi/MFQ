#include "common.h"

#include "mfq_tensor_backend.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace mfq::bench {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(
        message +
        "\nusage: mfq-cuda-common-ops-bench "
        "<acc|acc-rms-norm|silu-mul|rms-norm|rope> "
        "[--rows N] [--width N] [--rotary-width N] "
        "[--dtype f16|bf16|f32] [--warmup N] "
        "[--iterations N] [--samples N] [--seed N] [--epsilon F]");
}

mfq::cuda::ScalarType scalar_type(const std::string& dtype) {
    if (dtype == "f16") return mfq::cuda::kFloat16;
    if (dtype == "bf16") return mfq::cuda::kBFloat16;
    return mfq::cuda::kFloat32;
}

std::vector<float> host_float_values(const mfq::cuda::Tensor& tensor) {
    auto host = tensor.to(mfq::cuda::kCPU, mfq::cuda::kFloat32).contiguous();
    return std::vector<float>(
        host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

double maximum_error(
    const mfq::cuda::Tensor& actual,
    const mfq::cuda::Tensor& expected) {
    if (actual.sizes() != expected.sizes()) {
        throw std::runtime_error("correctness comparison shape mismatch");
    }
    const auto actual_values = host_float_values(actual);
    const auto expected_values = host_float_values(expected);
    double result = 0.0;
    for (std::size_t index = 0; index < actual_values.size(); ++index) {
        if (!std::isfinite(actual_values[index]) ||
            !std::isfinite(expected_values[index])) {
            throw std::runtime_error(
                "correctness comparison found a non-finite value");
        }
        result = std::max(
            result,
            std::abs(
                static_cast<double>(actual_values[index]) -
                static_cast<double>(expected_values[index])));
    }
    return result;
}

bool within_tolerance(
    const mfq::cuda::Tensor& actual,
    const mfq::cuda::Tensor& expected,
    double absolute_tolerance,
    double relative_tolerance) {
    if (actual.sizes() != expected.sizes()) {
        throw std::runtime_error("correctness comparison shape mismatch");
    }
    const auto actual_values = host_float_values(actual);
    const auto expected_values = host_float_values(expected);
    for (std::size_t index = 0; index < actual_values.size(); ++index) {
        const double actual_value = actual_values[index];
        const double expected_value = expected_values[index];
        if (!std::isfinite(actual_value) || !std::isfinite(expected_value) ||
            std::abs(actual_value - expected_value) >
                absolute_tolerance +
                    relative_tolerance * std::abs(expected_value)) {
            return false;
        }
    }
    return true;
}

std::vector<float> measure(
    const std::function<void()>& operation,
    int warmup,
    int iterations,
    int samples) {
    const cudaStream_t stream = mfq_current_cuda_stream();
    for (int index = 0; index < warmup; ++index) operation();
    check_cuda(cudaStreamSynchronize(stream), "benchmark warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    try {
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>(samples));
        for (int sample = 0; sample < samples; ++sample) {
            check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
            for (int iteration = 0; iteration < iterations; ++iteration) {
                operation();
            }
            check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            float milliseconds = 0.0f;
            check_cuda(
                cudaEventElapsedTime(&milliseconds, start, stop),
                "cudaEventElapsedTime");
            values.push_back(milliseconds * 1000.0f / iterations);
        }
        check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
        check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
        return values;
    } catch (...) {
        if (stop != nullptr) (void)cudaEventDestroy(stop);
        (void)cudaEventDestroy(start);
        throw;
    }
}

namespace {

float percentile(std::vector<float> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(std::ceil(
        fraction * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

std::string json_escape(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        if (character == '"' || character == '\\') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

}  // namespace

void print_result(
    const Options& options,
    const cudaDeviceProp& properties,
    const Correctness& correctness,
    const std::vector<float>& samples) {
    std::cout << std::fixed << std::setprecision(6)
              << "{\"schema_version\":1"
              << ",\"device\":\"" << json_escape(properties.name) << '"'
              << ",\"compute_capability\":\"" << properties.major << '.'
              << properties.minor << '"'
              << ",\"operation\":\"" << options.operation << '"'
              << ",\"parameters\":{\"rows\":" << options.rows
              << ",\"width\":" << options.width;
    if (options.operation == "rope") {
        std::cout << ",\"rotary_width\":" << options.rotary_width;
    }
    std::cout << ",\"dtype\":\"" << options.dtype << '"'
              << ",\"warmup\":" << options.warmup
              << ",\"iterations\":" << options.iterations
              << ",\"samples\":" << options.samples
              << ",\"seed\":" << options.seed
              << ",\"epsilon\":" << options.epsilon << '}'
              << ",\"correctness\":{\"passed\":true"
              << ",\"maximum_sum_error\":" << correctness.maximum_sum_error
              << ",\"maximum_output_error\":"
              << correctness.maximum_output_error
              << ",\"tolerance\":" << correctness.tolerance
              << ",\"relative_tolerance\":"
              << correctness.relative_tolerance << '}'
              << ",\"timing_us\":{\"samples\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << samples[index];
    }
    std::cout << "]"
              << ",\"median\":" << percentile(samples, 0.5)
              << ",\"p95\":" << percentile(samples, 0.95)
              << "}}\n";
}

}  // namespace mfq::bench
