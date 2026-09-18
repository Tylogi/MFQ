#include "common.h"

#include "mfq_tensor_backend.h"

#include <stdexcept>

mfq_tensor_backend::Tensor rms_norm_cuda(
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor weight,
    double eps);
mfq_tensor_backend::Tensor rms_norm_f16_cuda(
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);
mfq_tensor_backend::Tensor qwen_rms_norm_bf16_cuda(
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

namespace mfq::bench {

void run_rms_norm(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties) {
    const auto dtype = scalar_type(options.dtype);
    const auto fp32 = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kFloat32);
    mfq::cuda::manual_seed(options.seed);
    auto input = mfq::cuda::randn({options.rows, options.width}, fp32)
        .to(dtype).contiguous();
    auto weight = mfq::cuda::randn({options.width}, fp32).contiguous();

    const auto call = [&]() {
        if (dtype == mfq::cuda::kFloat16) {
            return rms_norm_f16_cuda(
                input, weight, options.epsilon, 0.0);
        }
        if (dtype == mfq::cuda::kBFloat16) {
            return qwen_rms_norm_bf16_cuda(
                input, weight, options.epsilon, 0.0);
        }
        return rms_norm_cuda(input, weight, options.epsilon);
    };

    auto working = input.to(mfq::cuda::kFloat32);
    auto inverse = mfq::cuda::rsqrt(
        working.square().mean(-1, true) + options.epsilon);
    auto reference = working * inverse * weight;
    if (dtype == mfq::cuda::kFloat16) {
        reference = reference.to(mfq::cuda::kFloat16);
    } else if (dtype == mfq::cuda::kBFloat16) {
        auto normalized = (working * inverse).to(mfq::cuda::kBFloat16);
        auto scale = weight.to(mfq::cuda::kBFloat16);
        reference = (normalized.to(mfq::cuda::kFloat32) *
                     scale.to(mfq::cuda::kFloat32))
                        .to(mfq::cuda::kBFloat16);
    }

    auto output = call();
    Correctness correctness;
    correctness.maximum_output_error = maximum_error(output, reference);
    correctness.tolerance = dtype == mfq::cuda::kFloat32 ? 1.0e-5 :
        dtype == mfq::cuda::kFloat16 ? 2.0e-3 : 3.2e-2;
    correctness.relative_tolerance = correctness.tolerance;
    if (!within_tolerance(
            output, reference, correctness.tolerance,
            correctness.relative_tolerance)) {
        throw std::runtime_error(
            "rms-norm output exceeds the correctness tolerance");
    }

    auto invoke = [&] { output = call(); };
    const auto samples = measure(
        invoke, options.warmup, options.iterations, options.samples);
    print_result(options, properties, correctness, samples);
}

}  // namespace mfq::bench
