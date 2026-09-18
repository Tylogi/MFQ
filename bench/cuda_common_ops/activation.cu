#include "common.h"

#include "mfq_tensor_backend.h"

#include <stdexcept>

mfq_tensor_backend::Tensor silu_mul_cuda(
    mfq_tensor_backend::Tensor gate,
    mfq_tensor_backend::Tensor up);

namespace mfq::bench {

void run_silu_mul(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties) {
    const auto dtype = scalar_type(options.dtype);
    const auto fp32 = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kFloat32);
    mfq::cuda::manual_seed(options.seed);
    auto gate = mfq::cuda::randn({options.rows, options.width}, fp32)
        .to(dtype).contiguous();
    auto up = mfq::cuda::randn({options.rows, options.width}, fp32)
        .to(dtype).contiguous();

    auto gate_fp32 = gate.to(mfq::cuda::kFloat32);
    auto up_fp32 = up.to(mfq::cuda::kFloat32);
    auto activated = mfq::cuda::silu(gate_fp32);
    mfq::cuda::Tensor reference;
    if (dtype == mfq::cuda::kBFloat16) {
        reference = (activated.to(mfq::cuda::kBFloat16)
                         .to(mfq::cuda::kFloat32) * up_fp32)
                        .to(mfq::cuda::kBFloat16);
    } else {
        reference = activated * up_fp32;
        if (dtype == mfq::cuda::kFloat16) {
            reference = reference.to(mfq::cuda::kFloat16);
        }
    }

    auto output = silu_mul_cuda(gate, up);
    Correctness correctness;
    correctness.maximum_output_error = maximum_error(output, reference);
    correctness.tolerance = dtype == mfq::cuda::kFloat32 ? 2.0e-6 :
        dtype == mfq::cuda::kFloat16 ? 1.0e-3 : 1.5625e-2;
    correctness.relative_tolerance = correctness.tolerance;
    if (!within_tolerance(
            output, reference, correctness.tolerance,
            correctness.relative_tolerance)) {
        throw std::runtime_error(
            "silu-mul output exceeds the correctness tolerance");
    }

    auto invoke = [&] { output = silu_mul_cuda(gate, up); };
    const auto samples = measure(
        invoke, options.warmup, options.iterations, options.samples);
    print_result(options, properties, correctness, samples);
}

}  // namespace mfq::bench
