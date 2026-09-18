#include "common.h"

#include "mfq_tensor_backend.h"

#include <stdexcept>
#include <vector>

mfq_tensor_backend::Tensor acc_cuda(
    mfq_tensor_backend::Tensor a,
    mfq_tensor_backend::Tensor b);
std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_cuda(
    mfq_tensor_backend::Tensor a,
    mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);
std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_f16_cuda(
    mfq_tensor_backend::Tensor a,
    mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);
std::vector<mfq_tensor_backend::Tensor> acc_rms_norm_bf16_cuda(
    mfq_tensor_backend::Tensor a,
    mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

namespace mfq::bench {

void run_acc(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties) {
    const auto dtype = scalar_type(options.dtype);
    const auto fp32 = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kFloat32);
    mfq::cuda::manual_seed(options.seed);
    auto a = mfq::cuda::randn({options.rows, options.width}, fp32)
        .to(dtype).contiguous();
    auto b = mfq::cuda::randn({options.rows, options.width}, fp32)
        .to(dtype).contiguous();

    auto reference = a + b;
    auto output = acc_cuda(a, b);
    Correctness correctness;
    correctness.maximum_output_error = maximum_error(output, reference);
    if (correctness.maximum_output_error != 0.0) {
        throw std::runtime_error(
            "acc output differs from the native Tensor reference");
    }

    auto invoke = [&] { output = acc_cuda(a, b); };
    const auto samples = measure(
        invoke, options.warmup, options.iterations, options.samples);
    print_result(options, properties, correctness, samples);
}

void run_acc_rms_norm(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties) {
    const auto dtype = scalar_type(options.dtype);
    const auto fp32 = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kFloat32);
    mfq::cuda::manual_seed(options.seed);
    auto a = mfq::cuda::randn({options.rows, options.width}, fp32)
        .to(dtype).contiguous();
    auto b = mfq::cuda::randn({options.rows, options.width}, fp32)
        .to(dtype).contiguous();
    auto weight = mfq::cuda::randn({options.width}, fp32).contiguous();

    const auto call = [&]() {
        if (dtype == mfq::cuda::kFloat16) {
            return acc_rms_norm_f16_cuda(
                a, b, weight, options.epsilon, 0.0);
        }
        if (dtype == mfq::cuda::kBFloat16) {
            return acc_rms_norm_bf16_cuda(
                a, b, weight, options.epsilon, 0.0);
        }
        return acc_rms_norm_cuda(a, b, weight, options.epsilon, 0.0);
    };

    auto reference_sum = a + b;
    auto working = reference_sum.to(mfq::cuda::kFloat32);
    auto inverse = mfq::cuda::rsqrt(
        working.square().mean(-1, true) + options.epsilon);
    auto reference_norm = working * inverse * weight;
    if (dtype == mfq::cuda::kFloat16) {
        reference_norm = reference_norm.to(mfq::cuda::kFloat16);
    } else if (dtype == mfq::cuda::kBFloat16) {
        auto normalized = (working * inverse).to(mfq::cuda::kBFloat16);
        auto scale = weight.to(mfq::cuda::kBFloat16);
        reference_norm = (normalized.to(mfq::cuda::kFloat32) *
                          scale.to(mfq::cuda::kFloat32))
                             .to(mfq::cuda::kBFloat16);
    }

    auto output = call();
    Correctness correctness;
    correctness.maximum_sum_error = maximum_error(output[0], reference_sum);
    correctness.maximum_output_error = maximum_error(output[1], reference_norm);
    correctness.tolerance = dtype == mfq::cuda::kFloat32 ? 2.0e-5 :
        dtype == mfq::cuda::kFloat16 ? 2.0e-3 : 3.2e-2;
    correctness.relative_tolerance = correctness.tolerance;
    if (correctness.maximum_sum_error != 0.0 ||
        !within_tolerance(
            output[1], reference_norm, correctness.tolerance,
            correctness.relative_tolerance)) {
        throw std::runtime_error(
            "acc-rms-norm output error " +
            std::to_string(correctness.maximum_output_error) +
            " exceeds tolerance " +
            std::to_string(correctness.tolerance));
    }

    auto invoke = [&] { output = call(); };
    const auto samples = measure(
        invoke, options.warmup, options.iterations, options.samples);
    print_result(options, properties, correctness, samples);
}

}  // namespace mfq::bench
