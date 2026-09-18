#include "common.h"

#include "mfq_tensor_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

mfq_tensor_backend::Tensor rope_table_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor pos,
    mfq_tensor_backend::Tensor cos,
    mfq_tensor_backend::Tensor sin,
    int64_t rotary_dim,
    mfq_tensor_backend::Tensor sections);

namespace mfq::bench {

void run_rope(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties) {
    if (options.dtype != "f32") {
        usage_error("rope currently supports --dtype f32");
    }
    const std::int64_t rotary_width = options.rotary_width == 0
        ? options.width : options.rotary_width;
    if (rotary_width <= 0 || rotary_width > options.width ||
        rotary_width % 2 != 0) {
        usage_error(
            "rope requires an even --rotary-width no larger than --width");
    }

    const auto fp32 = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kFloat32);
    const auto int64 = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kInt64);
    mfq::cuda::manual_seed(options.seed);
    auto input = mfq::cuda::randn({options.rows, options.width}, fp32)
        .contiguous();

    const std::int64_t half = rotary_width / 2;
    const std::int64_t table_length = std::max<std::int64_t>(16, options.rows);
    std::vector<std::int64_t> positions(static_cast<std::size_t>(options.rows));
    std::vector<float> cosine(
        static_cast<std::size_t>(table_length * half));
    std::vector<float> sine(cosine.size());
    for (std::int64_t row = 0; row < options.rows; ++row) {
        positions[static_cast<std::size_t>(row)] = row % table_length;
    }
    for (std::int64_t position = 0; position < table_length; ++position) {
        for (std::int64_t pair = 0; pair < half; ++pair) {
            const double frequency = std::pow(
                1'000'000.0,
                -2.0 * static_cast<double>(pair) /
                    static_cast<double>(rotary_width));
            const double angle = static_cast<double>(position) * frequency;
            const auto index = static_cast<std::size_t>(position * half + pair);
            cosine[index] = static_cast<float>(std::cos(angle));
            sine[index] = static_cast<float>(std::sin(angle));
        }
    }

    auto pos = mfq::cuda::tensor(positions, int64).contiguous();
    auto cos = mfq::cuda::tensor(cosine, fp32)
        .reshape({table_length, half}).contiguous();
    auto sin = mfq::cuda::tensor(sine, fp32)
        .reshape({table_length, half}).contiguous();
    auto sections = mfq::cuda::empty(
        {0}, mfq::cuda::TensorOptions().dtype(mfq::cuda::kInt64));

    const auto input_values = host_float_values(input);
    std::vector<float> expected_values(input_values);
    for (std::int64_t row = 0; row < options.rows; ++row) {
        const auto table_offset = static_cast<std::size_t>(
            positions[static_cast<std::size_t>(row)] * half);
        const auto row_offset = static_cast<std::size_t>(row * options.width);
        for (std::int64_t pair = 0; pair < half; ++pair) {
            const auto first = row_offset + static_cast<std::size_t>(pair);
            const auto second = first + static_cast<std::size_t>(half);
            const float cs = cosine[
                table_offset + static_cast<std::size_t>(pair)];
            const float sn = sine[
                table_offset + static_cast<std::size_t>(pair)];
            const float x0 = input_values[first];
            const float x1 = input_values[second];
            expected_values[first] = x0 * cs - x1 * sn;
            expected_values[second] = x1 * cs + x0 * sn;
        }
    }
    auto reference = mfq::cuda::tensor(expected_values, fp32)
        .reshape({options.rows, options.width}).contiguous();

    auto call = [&] {
        return rope_table_cuda(
            input, pos, cos, sin, rotary_width, sections);
    };
    auto output = call();
    Correctness correctness;
    correctness.maximum_output_error = maximum_error(output, reference);
    correctness.tolerance = 2.0e-6;
    correctness.relative_tolerance = 2.0e-6;
    if (!within_tolerance(
            output, reference, correctness.tolerance,
            correctness.relative_tolerance)) {
        throw std::runtime_error(
            "rope output exceeds the correctness tolerance");
    }

    auto invoke = [&] { output = call(); };
    const auto samples = measure(
        invoke, options.warmup, options.iterations, options.samples);
    print_result(options, properties, correctness, samples);
}

}  // namespace mfq::bench
