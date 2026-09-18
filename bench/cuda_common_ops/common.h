#pragma once

#include "mfq_native_tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mfq::bench {

struct Options {
    std::string operation;
    std::string dtype = "f16";
    std::int64_t rows = 1;
    std::int64_t width = 4096;
    std::int64_t rotary_width = 0;
    int warmup = 20;
    int iterations = 100;
    int samples = 7;
    std::int64_t seed = 20260918;
    double epsilon = 1.0e-6;
};

struct Correctness {
    double maximum_sum_error = 0.0;
    double maximum_output_error = 0.0;
    double tolerance = 0.0;
    double relative_tolerance = 0.0;
};

void check_cuda(cudaError_t status, const char* operation);
[[noreturn]] void usage_error(const std::string& message);

mfq::cuda::ScalarType scalar_type(const std::string& dtype);
std::vector<float> host_float_values(const mfq::cuda::Tensor& tensor);
double maximum_error(
    const mfq::cuda::Tensor& actual,
    const mfq::cuda::Tensor& expected);
bool within_tolerance(
    const mfq::cuda::Tensor& actual,
    const mfq::cuda::Tensor& expected,
    double absolute_tolerance,
    double relative_tolerance);
std::vector<float> measure(
    const std::function<void()>& operation,
    int warmup,
    int iterations,
    int samples);
void print_result(
    const Options& options,
    const cudaDeviceProp& properties,
    const Correctness& correctness,
    const std::vector<float>& samples);

void run_acc(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties);
void run_acc_rms_norm(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties);
void run_silu_mul(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties);
void run_rms_norm(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties);
void run_rope(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties);

}  // namespace mfq::bench
