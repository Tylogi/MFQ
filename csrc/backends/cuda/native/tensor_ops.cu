#include "tensor_ops_common.cuh"

#include <cmath>
#include <limits>

namespace mfq::cuda {
using namespace native_ops_detail;
namespace {
template <typename Value>
__global__ void unary_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    int operation,
    double first,
    double second) {
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        double result = value;
        switch (operation) {
            case 0: result = ::fabs(value); break;
            case 1: result = value * value; break;
            case 2: result = ::exp(value); break;
            case 3: result = ::sqrt(value); break;
            case 4: result = ::fmin(::fmax(value, first), second); break;
            case 5: result = ::fmax(value, first); break;
            case 6: result = ::fmin(value, first); break;
            case 7: result = value - ::floor(value / first) * first; break;
            case 9: result = ::sin(value); break;
            case 10: result = ::cos(value); break;
            case 11: result = 1.0 / (1.0 + ::exp(-value)); break;
            case 12: result = ::tanh(value); break;
            case 13: result = ::fmax(value, 0.0); break;
            case 14: result = 1.0 / ::sqrt(value); break;
            case 15: result = 1.0 / value; break;
            case 16: result = ::ceil(value); break;
            case 17: result = value > 20.0 ? value : ::log1p(::exp(value)); break;
            case 18: result = ::log(value); break;
            case 19: result = ::log1p(value); break;
            case 20: result = ::log2(value); break;
            case 21: result = ::exp2(value); break;
            case 22: result = value / (1.0 + ::exp(-value)); break;
            case 23: {
                constexpr double inv_sqrt_two = 0.7071067811865475244;
                result = 0.5 * value * (1.0 + ::erf(value * inv_sqrt_two));
                break;
            }
            case 24: result = ::pow(value, first); break;
        }
        destination[tensor_offset(output, linear)] = store_number<Value>(result);
    }
}

template <typename Input>
__global__ void unary_bool_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    int operation) {
    const auto* source = static_cast<const Input*>(input.data);
    auto* destination = static_cast<bool*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        bool result = false;
        if (operation == 8) result = value == 0.0;
        if (operation == 25) result = ::isfinite(value);
        if (operation == 26) result = ::isinf(value) && value < 0.0;
        destination[tensor_offset(output, linear)] = result;
    }
}

template <typename Value>
__global__ void binary_kernel(
    TensorView output,
    TensorView left,
    TensorView right,
    std::int64_t elements,
    int operation) {
    const auto* a = static_cast<const Value*>(left.data);
    const auto* b = static_cast<const Value*>(right.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double x = load_number(a, tensor_offset(left, linear));
        const double y = load_number(b, tensor_offset(right, linear));
        double result = 0.0;
        if (operation == 0) result = x + y;
        if (operation == 1) result = x - y;
        if (operation == 2) result = x * y;
        if (operation == 3) result = x / y;
        if (operation == 8) result = static_cast<double>(
            static_cast<std::int64_t>(x) & static_cast<std::int64_t>(y));
        if (operation == 9) result = static_cast<double>(
            static_cast<std::int64_t>(x) | static_cast<std::int64_t>(y));
        if (operation == 12) result = ::pow(x, y);
        destination[tensor_offset(output, linear)] = store_number<Value>(result);
    }
}

template <typename Value>
__global__ void comparison_kernel(
    TensorView output,
    TensorView left,
    TensorView right,
    std::int64_t elements,
    int operation) {
    const auto* a = static_cast<const Value*>(left.data);
    const auto* b = static_cast<const Value*>(right.data);
    auto* destination = static_cast<bool*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double x = load_number(a, tensor_offset(left, linear));
        const double y = load_number(b, tensor_offset(right, linear));
        bool result = false;
        if (operation == 4) result = x < y;
        if (operation == 5) result = x <= y;
        if (operation == 6) result = x > y;
        if (operation == 7) result = x >= y;
        if (operation == 10) result = x == y;
        if (operation == 11) result = x != y;
        destination[tensor_offset(output, linear)] = result;
    }
}

template <typename Value>
__global__ void scalar_binary_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    double scalar,
    int operation,
    bool scalar_first) {
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<Value*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        const double x = scalar_first ? scalar : value;
        const double y = scalar_first ? value : scalar;
        double result = 0.0;
        if (operation == 0) result = x + y;
        if (operation == 1) result = x - y;
        if (operation == 2) result = x * y;
        if (operation == 3) result = x / y;
        if (operation == 8) result = static_cast<double>(
            static_cast<std::int64_t>(x) & static_cast<std::int64_t>(y));
        if (operation == 9) result = static_cast<double>(
            static_cast<std::int64_t>(x) | static_cast<std::int64_t>(y));
        if (operation == 12) result = ::pow(x, y);
        destination[tensor_offset(output, linear)] = store_number<Value>(result);
    }
}

// Shared arithmetic for contiguous elementwise kernels.  Direct linear
// indexing removes TensorView coordinate reconstruction while keeping one
// logical element per CUDA thread.  That preserves parallel coverage across
// devices instead of reducing the grid by a dtype-dependent packing factor.
__device__ float apply_arithmetic(float x, float y, int operation) {
    if (operation == 0) return x + y;
    if (operation == 1) return x - y;
    if (operation == 2) return x * y;
    return x / y;
}

template <typename Value>
__device__ float to_float_value(Value value) {
    if constexpr (std::is_same_v<Value, __half>) {
        return __half2float(value);
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else {
        return value;
    }
}

template <typename Value>
__global__ void contiguous_binary_kernel(
    Value* destination,
    const Value* left,
    const Value* right,
    std::int64_t elements,
    int operation) {
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += stride) {
        destination[index] = store_number<Value>(static_cast<double>(
            apply_arithmetic(
                to_float_value(left[index]),
                to_float_value(right[index]), operation)));
    }
}

template <typename Value>
__global__ void contiguous_scalar_kernel(
    Value* destination,
    const Value* source,
    std::int64_t elements,
    double scalar,
    int operation,
    bool scalar_first) {
    const float scalar_value = static_cast<float>(scalar);
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += stride) {
        const float value = to_float_value(source[index]);
        destination[index] = store_number<Value>(static_cast<double>(
            scalar_first
                ? apply_arithmetic(scalar_value, value, operation)
                : apply_arithmetic(value, scalar_value, operation)));
    }
}

// Unary operations eligible for the contiguous fast path: exp(2), sqrt(3),
// sigmoid(11), tanh(12), relu(13), rsqrt(14), log(18), silu(22). All are
// parameter-free and produce numeric (non-boolean) output. relu is exact;
// the transcendentals carry the usual one-ulp float-to-reduced-precision
// difference covered by the benchmark tolerances.
bool contiguous_unary_operation(int operation) {
    return operation == 2 || operation == 3 || operation == 11 ||
        operation == 12 || operation == 13 || operation == 14 ||
        operation == 18 || operation == 22;
}

__device__ float apply_unary(float value, int operation) {
    switch (operation) {
        case 2: return ::expf(value);
        case 3: return ::sqrtf(value);
        case 11: return 1.0f / (1.0f + ::expf(-value));
        case 12: return ::tanhf(value);
        case 13: return ::fmaxf(value, 0.0f);
        case 14: return 1.0f / ::sqrtf(value);
        case 18: return ::logf(value);
        case 22: return value / (1.0f + ::expf(-value));
        default: return value;
    }
}

template <typename Value>
__global__ void contiguous_unary_kernel(
    Value* destination,
    const Value* source,
    std::int64_t elements,
    int operation) {
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += stride) {
        destination[index] = store_number<Value>(static_cast<double>(
            apply_unary(to_float_value(source[index]), operation)));
    }
}

template <typename Value>
__global__ void scalar_comparison_kernel(
    TensorView output,
    TensorView input,
    std::int64_t elements,
    double scalar,
    int operation,
    bool scalar_first) {
    const auto* source = static_cast<const Value*>(input.data);
    auto* destination = static_cast<bool*>(output.data);
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const double value = load_number(source, tensor_offset(input, linear));
        const double x = scalar_first ? scalar : value;
        const double y = scalar_first ? value : scalar;
        bool result = false;
        if (operation == 4) result = x < y;
        if (operation == 5) result = x <= y;
        if (operation == 6) result = x > y;
        if (operation == 7) result = x >= y;
        if (operation == 10) result = x == y;
        if (operation == 11) result = x != y;
        destination[tensor_offset(output, linear)] = result;
    }
}

}  // namespace

Tensor unary_cuda(
    const Tensor& source,
    int operation,
    double first = 0.0,
    double second = 0.0) {
    if (!source.is_cuda()) {
        throw std::invalid_argument("native unary CUDA op requires a CUDA tensor");
    }
    const bool boolean_output = operation == 8 || operation == 25 || operation == 26;
    auto options = source.options().dtype(boolean_output ? kBool : source.scalar_type());
    auto output = empty(source.sizes(), options);
    const auto [blocks, threads] = launch_geometry(source.numel());
    const auto stream = current_stream(source.get_device()).stream();
    const bool contiguous_unary = contiguous_unary_operation(operation) &&
        (source.scalar_type() == kFloat32 ||
         source.scalar_type() == kFloat16 ||
         source.scalar_type() == kBFloat16) &&
        source.is_contiguous() && output.is_contiguous();
    auto launch = [&]<typename Value>() {
        if (contiguous_unary) {
            contiguous_unary_kernel<Value><<<blocks, threads, 0, stream>>>(
                static_cast<Value*>(output.data_ptr()),
                static_cast<const Value*>(source.data_ptr()),
                source.numel(), operation);
        } else if (boolean_output) {
            unary_bool_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(), source.numel(), operation);
        } else {
            unary_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), source.view_descriptor(), source.numel(),
                operation, first, second);
        }
    };
    dispatch_numeric(source.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor binary_cuda(const Tensor& left_source, const Tensor& right_source, int operation) {
    if (!left_source.is_cuda() || !right_source.is_cuda()) {
        throw std::invalid_argument("native binary CUDA op requires CUDA tensors");
    }
    const auto type = promote(left_source.scalar_type(), right_source.scalar_type());
    auto left = left_source.scalar_type() == type ? left_source : left_source.to(type);
    auto right = right_source.scalar_type() == type ? right_source : right_source.to(type);
    const auto shape = broadcast_shape(left, right);
    const bool comparison = operation >= 4 && operation <= 7 ||
                            operation == 10 || operation == 11;
    auto output = empty(shape, left.options().dtype(comparison ? kBool : type));
    const auto left_view = align_for_broadcast(left, shape);
    const auto right_view = align_for_broadcast(right, shape);
    const auto [blocks, threads] = launch_geometry(output.numel());
    const auto stream = current_stream(left.get_device()).stream();
    const bool contiguous_binary = operation >= 0 && operation <= 3 &&
        (type == kFloat32 || type == kFloat16 || type == kBFloat16) &&
        same_shape(left, right) &&
        left.is_contiguous() && right.is_contiguous() &&
        output.is_contiguous();
    auto launch = [&]<typename Value>() {
        if (contiguous_binary) {
            contiguous_binary_kernel<Value><<<blocks, threads, 0, stream>>>(
                static_cast<Value*>(output.data_ptr()),
                static_cast<const Value*>(left.data_ptr()),
                static_cast<const Value*>(right.data_ptr()),
                output.numel(), operation);
        } else if (comparison) {
            comparison_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left_view, right_view, output.numel(), operation);
        } else {
            binary_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left_view, right_view, output.numel(), operation);
        }
    };
    dispatch_numeric(type, launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor scalar_binary_cuda(
    const Tensor& left,
    double right,
    int operation,
    bool scalar_first = false) {
    if (!left.is_cuda()) {
        throw std::invalid_argument("native scalar CUDA op requires a CUDA tensor");
    }
    const bool comparison = operation >= 4 && operation <= 7 ||
                            operation == 10 || operation == 11;
    auto output = empty(
        left.sizes(), left.options().dtype(comparison ? kBool : left.scalar_type()));
    const auto [blocks, threads] = launch_geometry(left.numel());
    const auto stream = current_stream(left.get_device()).stream();
    const bool contiguous_scalar = operation >= 0 && operation <= 3 &&
        (left.scalar_type() == kFloat32 ||
         left.scalar_type() == kFloat16 ||
         left.scalar_type() == kBFloat16) &&
        left.is_contiguous() && output.is_contiguous() &&
        static_cast<double>(static_cast<float>(right)) == right;
    auto launch = [&]<typename Value>() {
        if (contiguous_scalar) {
            contiguous_scalar_kernel<Value><<<blocks, threads, 0, stream>>>(
                static_cast<Value*>(output.data_ptr()),
                static_cast<const Value*>(left.data_ptr()),
                left.numel(), right, operation, scalar_first);
        } else if (comparison) {
            scalar_comparison_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left.view_descriptor(), left.numel(),
                right, operation, scalar_first);
        } else {
            scalar_binary_kernel<Value><<<blocks, threads, 0, stream>>>(
                output.view_descriptor(), left.view_descriptor(), left.numel(),
                right, operation, scalar_first);
        }
    };
    dispatch_numeric(left.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor operator+(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 0); }
Tensor operator+(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 0); }
Tensor operator+(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 0, true); }
Tensor operator-(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 1); }
Tensor operator-(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 1); }
Tensor operator-(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 1, true); }
Tensor operator-(const Tensor& value) { return scalar_binary_cuda(value, -1.0, 2); }
Tensor operator*(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 2); }
Tensor operator*(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 2); }
Tensor operator*(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 2, true); }
Tensor operator/(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 3); }
Tensor operator/(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 3); }
Tensor operator/(double left, const Tensor& right) { return scalar_binary_cuda(right, left, 3, true); }
Tensor operator<(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 4); }
Tensor operator<(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 4); }
Tensor operator<=(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 5); }
Tensor operator<=(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 5); }
Tensor operator>(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 6); }
Tensor operator>(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 6); }
Tensor operator>=(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 7); }
Tensor operator>=(const Tensor& left, double right) { return scalar_binary_cuda(left, right, 7); }
Tensor operator&(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 8); }
Tensor operator|(const Tensor& left, const Tensor& right) { return binary_cuda(left, right, 9); }
Tensor operator~(const Tensor& value) { return value.logical_not(); }
Tensor operator==(const Tensor& left, const Tensor& right) { return left.eq(right); }
Tensor operator==(const Tensor& left, double right) { return left.eq(right); }
Tensor operator!=(const Tensor& left, const Tensor& right) { return left.ne(right); }
Tensor operator!=(const Tensor& left, double right) { return left.ne(right); }

Tensor sin(const Tensor& input) { return unary_cuda(input, 9); }
Tensor cos(const Tensor& input) { return unary_cuda(input, 10); }
Tensor sigmoid(const Tensor& input) { return unary_cuda(input, 11); }
Tensor tanh(const Tensor& input) { return unary_cuda(input, 12); }
Tensor relu(const Tensor& input) { return unary_cuda(input, 13); }
Tensor rsqrt(const Tensor& input) { return unary_cuda(input, 14); }
Tensor reciprocal(const Tensor& input) { return unary_cuda(input, 15); }
Tensor ceil(const Tensor& input) { return unary_cuda(input, 16); }
Tensor softplus(const Tensor& input) { return unary_cuda(input, 17); }
Tensor log(const Tensor& input) { return unary_cuda(input, 18); }
Tensor log1p(const Tensor& input) { return unary_cuda(input, 19); }
Tensor log2(const Tensor& input) { return unary_cuda(input, 20); }
Tensor exp2(const Tensor& input) { return unary_cuda(input, 21); }
Tensor silu(const Tensor& input) { return unary_cuda(input, 22); }
Tensor gelu(const Tensor& input, const std::string&) { return unary_cuda(input, 23); }
Tensor exp(const Tensor& input) { return input.exp(); }
Tensor sqrt(const Tensor& input) { return input.sqrt(); }
Tensor pow(const Tensor& input, double exponent) { return unary_cuda(input, 24, exponent); }
Tensor pow(const Tensor& input, const Tensor& exponent) { return binary_cuda(input, exponent, 12); }
Tensor remainder(const Tensor& input, double divisor) { return input.remainder(divisor); }
Tensor clamp(const Tensor& input, double minimum, double maximum) { return input.clamp(minimum, maximum); }
Tensor clamp_min(const Tensor& input, double minimum) { return input.clamp_min(minimum); }
Tensor clamp_max(const Tensor& input, double maximum) { return input.clamp_max(maximum); }
Tensor isfinite(const Tensor& input) { return unary_cuda(input, 25); }
Tensor isneginf(const Tensor& input) { return unary_cuda(input, 26); }
}  // namespace mfq::cuda
