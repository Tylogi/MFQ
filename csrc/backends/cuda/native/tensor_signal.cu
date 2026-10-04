#include "tensor_ops_common.cuh"

#include <random>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {
using namespace native_ops_detail;
namespace {
template <typename Value>
__global__ void conv1d_kernel(
    TensorView output,
    TensorView input,
    TensorView weight,
    TensorView bias,
    bool has_bias,
    std::int64_t elements,
    std::int64_t output_length,
    std::int64_t stride,
    std::int64_t padding,
    std::int64_t dilation,
    std::int64_t groups) {
    const auto input_channels = input.sizes[1];
    const auto output_channels = output.sizes[1];
    const auto input_length = input.sizes[2];
    const auto kernel = weight.sizes[2];
    const auto input_per_group = input_channels / groups;
    const auto output_per_group = output_channels / groups;
    const auto* x = static_cast<const Value*>(input.data);
    const auto* w = static_cast<const Value*>(weight.data);
    const auto* b = has_bias ? static_cast<const Value*>(bias.data) : nullptr;
    auto* y = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        const auto position = residual % output_length;
        residual /= output_length;
        const auto channel = residual % output_channels;
        const auto batch = residual / output_channels;
        const auto group = channel / output_per_group;
        using Accumulator = std::conditional_t<std::is_same_v<Value, double>, double, float>;
        Accumulator accumulator = has_bias
            ? static_cast<Accumulator>(load_number(b, channel))
            : Accumulator{0};
        for (std::int64_t local_channel = 0; local_channel < input_per_group;
             ++local_channel) {
            const auto input_channel = group * input_per_group + local_channel;
            for (std::int64_t tap = 0; tap < kernel; ++tap) {
                const auto source_position = position * stride - padding + tap * dilation;
                if (source_position < 0 || source_position >= input_length) continue;
                const auto input_offset =
                    (batch * input_channels + input_channel) * input_length + source_position;
                const auto weight_offset =
                    (channel * input_per_group + local_channel) * kernel + tap;
                accumulator += static_cast<Accumulator>(load_number(x, input_offset)) *
                    static_cast<Accumulator>(load_number(w, weight_offset));
            }
        }
        y[linear] = store_number<Value>(accumulator);
    }
}

template <typename Value>
__global__ void conv2d_kernel(
    TensorView output,
    TensorView input,
    TensorView weight,
    TensorView bias,
    bool has_bias,
    std::int64_t elements,
    std::int64_t stride_h,
    std::int64_t stride_w,
    std::int64_t padding_h,
    std::int64_t padding_w,
    std::int64_t dilation_h,
    std::int64_t dilation_w,
    std::int64_t groups) {
    const auto input_channels = input.sizes[1];
    const auto output_channels = output.sizes[1];
    const auto input_h = input.sizes[2];
    const auto input_w = input.sizes[3];
    const auto output_h = output.sizes[2];
    const auto output_w = output.sizes[3];
    const auto kernel_h = weight.sizes[2];
    const auto kernel_w = weight.sizes[3];
    const auto input_per_group = input_channels / groups;
    const auto output_per_group = output_channels / groups;
    const auto* x = static_cast<const Value*>(input.data);
    const auto* w = static_cast<const Value*>(weight.data);
    const auto* b = has_bias ? static_cast<const Value*>(bias.data) : nullptr;
    auto* y = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        auto residual = linear;
        const auto output_x = residual % output_w;
        residual /= output_w;
        const auto output_y = residual % output_h;
        residual /= output_h;
        const auto channel = residual % output_channels;
        const auto batch = residual / output_channels;
        const auto group = channel / output_per_group;
        using Accumulator = std::conditional_t<std::is_same_v<Value, double>, double, float>;
        Accumulator accumulator = has_bias
            ? static_cast<Accumulator>(load_number(b, channel))
            : Accumulator{0};
        for (std::int64_t local_channel = 0; local_channel < input_per_group;
             ++local_channel) {
            const auto input_channel = group * input_per_group + local_channel;
            for (std::int64_t ky = 0; ky < kernel_h; ++ky) {
                const auto source_y = output_y * stride_h - padding_h + ky * dilation_h;
                if (source_y < 0 || source_y >= input_h) continue;
                for (std::int64_t kx = 0; kx < kernel_w; ++kx) {
                    const auto source_x = output_x * stride_w - padding_w + kx * dilation_w;
                    if (source_x < 0 || source_x >= input_w) continue;
                    const auto input_offset =
                        ((batch * input_channels + input_channel) * input_h + source_y) *
                            input_w + source_x;
                    const auto weight_offset =
                        ((channel * input_per_group + local_channel) * kernel_h + ky) *
                            kernel_w + kx;
                    accumulator += static_cast<Accumulator>(load_number(x, input_offset)) *
                        static_cast<Accumulator>(load_number(w, weight_offset));
                }
            }
        }
        y[linear] = store_number<Value>(accumulator);
    }
}

template <typename Value>
__global__ void avg_pool1d_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    std::int64_t kernel,
    std::int64_t stride,
    std::int64_t padding) {
    const auto output_length = output.sizes[2];
    const auto input_length = input.sizes[2];
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto output_position = linear % output_length;
        const auto row = linear / output_length;
        using Accumulator = std::conditional_t<std::is_same_v<Value, double>, double, float>;
        Accumulator accumulator = 0;
        for (std::int64_t tap = 0; tap < kernel; ++tap) {
            const auto position = output_position * stride - padding + tap;
            if (position >= 0 && position < input_length) {
                accumulator += static_cast<Accumulator>(
                    load_number(source, row * input_length + position));
            }
        }
        destination[linear] = store_number<Value>(
            accumulator / static_cast<Accumulator>(kernel));
    }
}

template <typename Value>
__global__ void cumsum_kernel(
    TensorView output,
    TensorView input,
    std::int64_t outer,
    std::int64_t length,
    std::int64_t inner) {
    const auto rows = outer * inner;
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t row =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto outer_index = row / inner;
        const auto inner_index = row % inner;
        using Accumulator = std::conditional_t<
            std::is_same_v<Value, double>, double,
            std::conditional_t<std::is_integral_v<Value>, std::int64_t, float>>;
        Accumulator accumulator = 0;
        for (std::int64_t index = 0; index < length; ++index) {
            const auto offset = (outer_index * length + index) * inner + inner_index;
            if constexpr (std::is_integral_v<Value>) {
                accumulator += static_cast<Accumulator>(source[offset]);
            } else {
                accumulator += static_cast<Accumulator>(load_number(source, offset));
            }
            if constexpr (std::is_integral_v<Value>) {
                destination[offset] = static_cast<Value>(accumulator);
            } else {
                destination[offset] = store_number<Value>(accumulator);
            }
        }
    }
}

template <typename Value>
__global__ void multinomial_one_kernel(
    const Value* probabilities,
    std::int64_t* output,
    const double* uniforms,
    std::int64_t rows,
    std::int64_t columns) {
    for (std::int64_t row =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < rows;
         row += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        double total = 0.0;
        for (std::int64_t column = 0; column < columns; ++column) {
            total += ::fmax(0.0, load_number(probabilities, row * columns + column));
        }
        const double target = uniforms[row] * total;
        double cumulative = 0.0;
        std::int64_t selected = columns - 1;
        for (std::int64_t column = 0; column < columns; ++column) {
            cumulative += ::fmax(0.0, load_number(probabilities, row * columns + column));
            if (target < cumulative) {
                selected = column;
                break;
            }
        }
        output[row] = selected;
    }
}

}  // namespace

Tensor conv1d(
    const Tensor& input_source,
    const Tensor& weight_source,
    const Tensor& bias_source,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups) {
    if (input_source.dim() != 3 || weight_source.dim() != 3 || groups <= 0 ||
        stride.size() != 1 || padding.size() != 1 || dilation.size() != 1) {
        throw std::invalid_argument("conv1d geometry is invalid");
    }
    auto input = input_source.contiguous();
    auto weight = weight_source.to(input.device(), input.scalar_type()).contiguous();
    auto bias = bias_source.defined()
        ? bias_source.to(input.device(), input.scalar_type()).contiguous()
        : Tensor{};
    if (!floating(input.scalar_type()) || input.size(1) % groups != 0 ||
        weight.size(0) % groups != 0 || weight.size(1) != input.size(1) / groups) {
        throw std::invalid_argument("conv1d channel geometry is invalid");
    }
    const auto output_length =
        (input.size(2) + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) /
            stride[0] + 1;
    if (output_length < 0) throw std::invalid_argument("conv1d output is empty");
    auto output = empty(
        {input.size(0), weight.size(0), output_length}, input.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        conv1d_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(), weight.view_descriptor(),
            bias.defined() ? bias.view_descriptor() : TensorView{}, bias.defined(),
            output.numel(), output_length, stride[0], padding[0], dilation[0], groups);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor conv2d(
    const Tensor& input_source,
    const Tensor& weight_source,
    const Tensor& bias_source,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding,
    std::span<const std::int64_t> dilation,
    std::int64_t groups) {
    if (input_source.dim() != 4 || weight_source.dim() != 4 || groups <= 0 ||
        stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
        throw std::invalid_argument("conv2d geometry is invalid");
    }
    auto input = input_source.contiguous();
    auto weight = weight_source.to(input.device(), input.scalar_type()).contiguous();
    auto bias = bias_source.defined()
        ? bias_source.to(input.device(), input.scalar_type()).contiguous()
        : Tensor{};
    if (!floating(input.scalar_type()) || input.size(1) % groups != 0 ||
        weight.size(0) % groups != 0 || weight.size(1) != input.size(1) / groups) {
        throw std::invalid_argument("conv2d channel geometry is invalid");
    }
    const auto output_h =
        (input.size(2) + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) /
            stride[0] + 1;
    const auto output_w =
        (input.size(3) + 2 * padding[1] - dilation[1] * (weight.size(3) - 1) - 1) /
            stride[1] + 1;
    if (output_h < 0 || output_w < 0) {
        throw std::invalid_argument("conv2d output is empty");
    }
    auto output = empty(
        {input.size(0), weight.size(0), output_h, output_w}, input.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        conv2d_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(), weight.view_descriptor(),
            bias.defined() ? bias.view_descriptor() : TensorView{}, bias.defined(),
            output.numel(), stride[0], stride[1], padding[0], padding[1],
            dilation[0], dilation[1], groups);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor avg_pool1d(
    const Tensor& input_source,
    std::span<const std::int64_t> kernel,
    std::span<const std::int64_t> stride,
    std::span<const std::int64_t> padding) {
    if (input_source.dim() != 3 || kernel.size() != 1 || stride.size() != 1 ||
        (!padding.empty() && padding.size() != 1)) {
        throw std::invalid_argument("avg_pool1d geometry is invalid");
    }
    auto input = input_source.contiguous();
    const auto selected_padding = padding.empty() ? 0 : padding[0];
    const auto output_length =
        (input.size(2) + 2 * selected_padding - kernel[0]) / stride[0] + 1;
    auto output = empty(
        {input.size(0), input.size(1), output_length}, input.options());
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        avg_pool1d_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(), output.numel(),
            kernel[0], stride[0], selected_padding);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor cumsum(const Tensor& input_source, std::int64_t dimension) {
    auto input = input_source.contiguous();
    const auto selected = normalize_dimension(dimension, input.dim());
    std::int64_t outer = 1;
    std::int64_t inner = 1;
    for (std::size_t axis = 0; axis < selected; ++axis) outer *= input.size(axis);
    for (std::size_t axis = selected + 1; axis < static_cast<std::size_t>(input.dim()); ++axis) {
        inner *= input.size(axis);
    }
    auto output = empty(input.sizes(), input.options());
    const auto [blocks, threads] = launch_geometry(outer * inner);
    const auto stream_value = current_stream(input.get_device()).stream();
    auto launch = [&]<typename Value>() {
        cumsum_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            output.view_descriptor(), input.view_descriptor(),
            outer, input.size(selected), inner);
    };
    dispatch_numeric(input.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor multinomial(
    const Tensor& probabilities_source,
    std::int64_t samples,
    bool replacement) {
    if (samples != 1 || replacement) {
        throw std::invalid_argument(
            "native inference multinomial currently supports one sample without replacement");
    }
    auto probabilities = probabilities_source.contiguous();
    if (!probabilities.is_cuda() || !floating(probabilities.scalar_type()) ||
        probabilities.dim() < 1 || probabilities.size(-1) <= 0) {
        throw std::invalid_argument("multinomial probabilities are invalid");
    }
    const auto columns = probabilities.size(-1);
    const auto rows = probabilities.numel() / columns;
    auto shape = probabilities.sizes().vec();
    shape.back() = 1;
    auto output = empty(shape, probabilities.options().dtype(kInt64));
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        native_random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::vector<double> host_uniforms(static_cast<std::size_t>(rows));
    for (auto& value : host_uniforms) value = distribution(generator);
    auto uniforms = tensor(
        host_uniforms,
        TensorOptions{}.dtype(kFloat64).device(probabilities.device()));
    const auto [blocks, threads] = launch_geometry(rows);
    const auto stream_value = current_stream(probabilities.get_device()).stream();
    auto launch = [&]<typename Value>() {
        multinomial_one_kernel<Value><<<blocks, threads, 0, stream_value>>>(
            probabilities.data_ptr<Value>(), output.data_ptr<std::int64_t>(),
            uniforms.data_ptr<double>(), rows, columns);
    };
    dispatch_numeric(probabilities.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

}  // namespace mfq::cuda
