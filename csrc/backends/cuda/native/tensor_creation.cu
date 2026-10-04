#include "tensor_ops_common.cuh"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace mfq::cuda::native_ops_detail {
std::atomic<std::int64_t> native_random_seed{0};
}

namespace mfq::cuda {
using namespace native_ops_detail;
namespace {
template <typename Value>
__global__ void arange_kernel(
    Value* output,
    std::int64_t elements,
    double start,
    double step) {
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        output[index] = store_number<Value>(start + step * static_cast<double>(index));
    }
}

template <typename Value>
__global__ void where_kernel(
    TensorView output,
    TensorView condition,
    TensorView yes,
    TensorView no,
    std::int64_t elements) {
    auto* destination = static_cast<Value*>(output.data);
    const auto* selected = static_cast<const bool*>(condition.data);
    const auto* yes_values = static_cast<const Value*>(yes.data);
    const auto* no_values = static_cast<const Value*>(no.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        destination[linear] = selected[tensor_offset(condition, linear)]
            ? yes_values[tensor_offset(yes, linear)]
            : no_values[tensor_offset(no, linear)];
    }
}

}  // namespace

void manual_seed(std::int64_t seed) {
    native_random_seed.store(seed, std::memory_order_relaxed);
}

Tensor cat(std::span<const Tensor> tensors, std::int64_t dimension) {
    if (tensors.empty()) throw std::invalid_argument("cat requires at least one tensor");
    const auto selected = normalize_dimension(dimension, tensors.front().dim());
    auto shape = tensors.front().sizes().vec();
    shape[selected] = 0;
    for (const auto& tensor : tensors) {
        if (tensor.dim() != tensors.front().dim()) {
            throw std::invalid_argument("cat tensor ranks differ");
        }
        for (std::int64_t axis = 0; axis < tensor.dim(); ++axis) {
            if (axis != static_cast<std::int64_t>(selected) &&
                tensor.size(axis) != tensors.front().size(axis)) {
                throw std::invalid_argument("cat non-concatenated extents differ");
            }
        }
        shape[selected] += tensor.size(static_cast<std::int64_t>(selected));
    }
    auto output = empty(shape, tensors.front().options());
    std::int64_t offset = 0;
    for (const auto& tensor : tensors) {
        auto value = tensor.device() == output.device() &&
                     tensor.scalar_type() == output.scalar_type()
            ? tensor
            : tensor.to(output.device(), output.scalar_type());
        output.narrow(
            static_cast<std::int64_t>(selected), offset,
            value.size(static_cast<std::int64_t>(selected))).copy_(value);
        offset += value.size(static_cast<std::int64_t>(selected));
    }
    return output;
}

Tensor cat(std::initializer_list<Tensor> tensors, std::int64_t dimension) {
    return cat(std::span<const Tensor>(tensors.begin(), tensors.size()), dimension);
}

Tensor cat(const std::vector<Tensor>& tensors, std::int64_t dimension) {
    return cat(std::span<const Tensor>(tensors), dimension);
}

Tensor stack(std::span<const Tensor> tensors, std::int64_t dimension) {
    if (tensors.empty()) throw std::invalid_argument("stack requires at least one tensor");
    auto selected = dimension;
    if (selected < 0) selected += tensors.front().dim() + 1;
    if (selected < 0 || selected > tensors.front().dim()) {
        throw std::out_of_range("stack dimension is out of range");
    }
    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());
    for (const auto& tensor : tensors) expanded.push_back(tensor.unsqueeze(selected));
    return cat(expanded, selected);
}

Tensor stack(std::initializer_list<Tensor> tensors, std::int64_t dimension) {
    return stack(std::span<const Tensor>(tensors.begin(), tensors.size()), dimension);
}

Tensor stack(const std::vector<Tensor>& tensors, std::int64_t dimension) {
    return stack(std::span<const Tensor>(tensors), dimension);
}

Tensor arange(
    std::int64_t start,
    std::int64_t end,
    std::int64_t step,
    const TensorOptions& options) {
    if (step == 0) throw std::invalid_argument("arange step cannot be zero");
    const auto distance = end - start;
    const auto elements = distance == 0 || (distance > 0) != (step > 0)
        ? 0
        : (std::llabs(distance) + std::llabs(step) - 1) / std::llabs(step);
    auto output = empty({elements}, options);
    if (!output.is_cuda()) {
        for (std::int64_t index = 0; index < elements; ++index) {
            switch (output.scalar_type()) {
                case kInt64: output.data_ptr<std::int64_t>()[index] = start + index * step; break;
                case kInt32: output.data_ptr<std::int32_t>()[index] = static_cast<std::int32_t>(start + index * step); break;
                case kFloat32: output.data_ptr<float>()[index] = static_cast<float>(start + index * step); break;
                case kFloat64: output.data_ptr<double>()[index] = static_cast<double>(start + index * step); break;
                default: throw std::invalid_argument("CPU arange dtype is unsupported");
            }
        }
        return output;
    }
    const auto [blocks, threads] = launch_geometry(elements);
    const auto stream = current_stream(output.get_device()).stream();
    auto launch = [&]<typename Value>() {
        arange_kernel<Value><<<blocks, threads, 0, stream>>>(
            output.data_ptr<Value>(), elements,
            static_cast<double>(start), static_cast<double>(step));
    };
    dispatch_numeric(output.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor arange(std::int64_t start, std::int64_t end, const TensorOptions& options) {
    return arange(start, end, 1, options);
}

Tensor arange(std::int64_t end, const TensorOptions& options) {
    return arange(0, end, 1, options);
}

Tensor randn(std::span<const std::int64_t> shape, const TensorOptions& options) {
    const auto elements = std::accumulate(
        shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>());
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        native_random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(elements));
    for (auto& value : values) value = distribution(generator);
    auto result = tensor(
        values,
        TensorOptions{}.dtype(kFloat32).device(options.target_device()));
    return result.to(options.scalar_type()).reshape(shape);
}

Tensor randn(std::initializer_list<std::int64_t> shape, const TensorOptions& options) {
    return randn(std::span<const std::int64_t>(shape.begin(), shape.size()), options);
}

Tensor randint(
    std::int64_t low,
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options) {
    if (high <= low) throw std::invalid_argument("randint range is empty");
    const auto elements = std::accumulate(
        shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>());
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        native_random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::uniform_int_distribution<std::int64_t> distribution(low, high - 1);
    std::vector<std::int64_t> values(static_cast<std::size_t>(elements));
    for (auto& value : values) value = distribution(generator);
    auto result = tensor(
        values,
        TensorOptions{}.dtype(kInt64).device(options.target_device()));
    return result.to(options.scalar_type()).reshape(shape);
}

Tensor randint(
    std::int64_t high,
    std::span<const std::int64_t> shape,
    const TensorOptions& options) {
    return randint(0, high, shape, options);
}

Tensor randperm(std::int64_t size, const TensorOptions& options) {
    std::vector<std::int64_t> values(static_cast<std::size_t>(size));
    std::iota(values.begin(), values.end(), 0);
    std::mt19937_64 generator(static_cast<std::uint64_t>(
        native_random_seed.fetch_add(1, std::memory_order_relaxed)));
    std::shuffle(values.begin(), values.end(), generator);
    return tensor(values, TensorOptions{}.dtype(kInt64).device(options.target_device()))
        .to(options.scalar_type());
}

Tensor where(
    const Tensor& condition_source,
    const Tensor& yes_source,
    const Tensor& no_source) {
    auto shape = broadcast_shape(yes_source, no_source);
    auto type = promote(yes_source.scalar_type(), no_source.scalar_type());
    auto yes = yes_source.to(yes_source.device(), type);
    auto no = no_source.to(yes.device(), type);
    auto condition = condition_source.to(yes.device(), kBool);
    auto output = empty(shape, yes.options());
    const auto yes_view = align_for_broadcast(yes, shape);
    const auto no_view = align_for_broadcast(no, shape);
    const auto condition_view = align_for_broadcast(condition, shape);
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(output.get_device()).stream();
    auto launch = [&]<typename Value>() {
        where_kernel<Value><<<blocks, threads, 0, stream>>>(
            output.view_descriptor(), condition_view, yes_view, no_view, output.numel());
    };
    dispatch_numeric(type, launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor where(const Tensor& condition, const Tensor& yes, double no) {
    return where(condition, yes, full_like(yes, no));
}

Tensor where(const Tensor& condition, double yes, const Tensor& no) {
    return where(condition, full_like(no, yes), no);
}

}  // namespace mfq::cuda
