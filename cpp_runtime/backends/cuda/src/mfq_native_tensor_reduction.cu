#include "mfq_native_tensor_ops_common.cuh"

#include <limits>

namespace mfq::cuda {
using namespace native_ops_detail;
namespace {
template <typename Value, typename Output>
__global__ void reduce_kernel(
    TensorView output,
    TensorView input,
    std::int64_t outer_elements,
    std::int64_t reduced,
    int dimension,
    int operation,
    bool keep_dimension) {
    auto* destination = static_cast<Output*>(output.data);
    const auto* source = static_cast<const Value*>(input.data);
    for (std::int64_t output_linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_linear < outer_elements;
         output_linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        std::array<std::int64_t, kMaximumTensorRank> coordinates{};
        auto residual = output_linear;
        for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
            const auto axis = reverse - 1;
            coordinates[axis] = residual % output.sizes[axis];
            residual /= output.sizes[axis];
        }
        std::int64_t base = 0;
        std::size_t output_axis = 0;
        for (std::size_t input_axis = 0; input_axis < input.rank; ++input_axis) {
            if (static_cast<int>(input_axis) == dimension) continue;
            const auto coordinate_axis = keep_dimension ? input_axis : output_axis++;
            base += coordinates[coordinate_axis] * input.strides[input_axis];
        }
        using Accumulator = std::conditional_t<
            std::is_same_v<Value, double>, double,
            std::conditional_t<
                std::is_same_v<Value, float> || std::is_same_v<Value, __half> ||
                    std::is_same_v<Value, __nv_bfloat16>,
                float,
                std::int64_t>>;
        Accumulator accumulator = operation == 2 || operation == 3
            ? std::numeric_limits<Accumulator>::lowest()
            : Accumulator{0};
        std::int64_t best = 0;
        bool boolean = operation == 4;
        if (operation == 5) boolean = false;
        for (std::int64_t index = 0; index < reduced; ++index) {
            const auto source_offset = base + index * input.strides[dimension];
            const auto value = [&] {
                if constexpr (std::is_integral_v<Value>) {
                    return static_cast<Accumulator>(source[source_offset]);
                } else {
                    return static_cast<Accumulator>(load_number(source, source_offset));
                }
            }();
            if (operation == 0 || operation == 1) accumulator += value;
            if (operation == 2 || operation == 3) {
                if (value > accumulator) {
                    accumulator = value;
                    best = index;
                }
            }
            if (operation == 4) boolean = boolean && value != 0.0;
            if (operation == 5) boolean = boolean || value != 0.0;
        }
        if (operation == 1) accumulator /= static_cast<Accumulator>(reduced);
        if (operation == 3) accumulator = static_cast<Accumulator>(best);
        if (operation == 4 || operation == 5) accumulator = boolean ? 1.0 : 0.0;
        if constexpr (std::is_integral_v<Output>) {
            destination[output_linear] = static_cast<Output>(accumulator);
        } else {
            destination[output_linear] = store_number<Output>(accumulator);
        }
    }
}

template <typename Value, int Threads>
__global__ void row_mean_last_contiguous_kernel(
    const Value* __restrict__ input,
    Value* __restrict__ output,
    std::int64_t rows,
    std::int64_t columns) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    float accumulator = 0.0f;
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        accumulator += static_cast<float>(
            load_number(input, row * columns + column));
    }
    __shared__ float partial[Threads];
    partial[threadIdx.x] = accumulator;
    __syncthreads();
    for (int offset = Threads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = store_number<Value>(
            partial[0] / static_cast<float>(columns));
    }
}

template <typename Value>
void launch_row_mean(
    const Value* input,
    Value* output,
    std::int64_t rows,
    std::int64_t columns,
    cudaStream_t stream) {
    const auto launch = [&]<int Threads>() {
        row_mean_last_contiguous_kernel<Value, Threads>
            <<<static_cast<unsigned int>(rows), Threads, 0, stream>>>(
                input, output, rows, columns);
    };
    if (columns <= 32) launch.template operator()<32>();
    else if (columns <= 64) launch.template operator()<64>();
    else if (columns <= 128) launch.template operator()<128>();
    else launch.template operator()<256>();
}

template <int Threads>
__global__ void argmax_last_contiguous_bf16_kernel(
    const __nv_bfloat16* input,
    std::int64_t* output,
    std::int64_t rows,
    std::int64_t columns) {
    __shared__ float values[Threads];
    __shared__ std::int64_t indices[Threads];
    constexpr std::int64_t invalid_index = 0x7fffffffffffffffLL;
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) return;
    const auto* row_input = input + row * columns;
    float best_value = 0.0f;
    std::int64_t best_index = invalid_index;
    for (std::int64_t column = threadIdx.x;
         column < columns;
         column += Threads) {
        const float value = __bfloat162float(row_input[column]);
        if (best_index == invalid_index ||
            value > best_value ||
            (value == best_value && column < best_index)) {
            best_value = value;
            best_index = column;
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int offset = Threads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const auto other_index = indices[threadIdx.x + offset];
            const auto other_value = values[threadIdx.x + offset];
            if (other_index != invalid_index &&
                (indices[threadIdx.x] == invalid_index ||
                 other_value > values[threadIdx.x] ||
                 (other_value == values[threadIdx.x] &&
                  other_index < indices[threadIdx.x]))) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = indices[0];
}

ScalarType reduction_type(ScalarType input, int operation) {
    if (operation == 3) return kInt64;
    if (operation == 4 || operation == 5) return kBool;
    if (operation == 0 && !floating(input)) return kInt64;
    return input;
}

template <typename Input>
void launch_reduction_output(
    Tensor& output,
    const Tensor& input,
    std::int64_t outer,
    std::int64_t reduced,
    int dimension,
    int operation,
    bool keep_dimension,
    cudaStream_t stream) {
    const auto [blocks, threads] = launch_geometry(outer);
    auto launch = [&]<typename Output>() {
        reduce_kernel<Input, Output><<<blocks, threads, 0, stream>>>(
            output.view_descriptor(), input.view_descriptor(), outer,
            reduced, dimension, operation, keep_dimension);
    };
    dispatch_numeric(output.scalar_type(), launch);
}

}  // namespace

Tensor reduce_cuda(
    const Tensor& source,
    std::int64_t dimension,
    bool keep_dimension,
    int operation) {
    if (!source.is_cuda() || source.dim() == 0) {
        throw std::invalid_argument("native reduction requires a ranked CUDA tensor");
    }
    if (operation == 1 && !floating(source.scalar_type())) {
        throw std::invalid_argument("native mean requires a floating dtype");
    }
    const auto selected = normalize_dimension(dimension, source.dim());
    const auto reduced = source.size(static_cast<std::int64_t>(selected));
    if (reduced == 0) throw std::invalid_argument("cannot reduce an empty dimension");
    auto shape = source.sizes().vec();
    if (keep_dimension) shape[selected] = 1;
    else shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(selected));
    auto output = empty(shape, source.options().dtype(
        reduction_type(source.scalar_type(), operation)));
    const auto outer = output.numel();
    const auto stream = current_stream(source.get_device()).stream();
    bool row_mean_enabled = true;
    if (source.scalar_type() == kFloat32) {
        const char* value =
            std::getenv("MFQ_DISABLE_NATIVE_PARALLEL_F32_MEAN");
        row_mean_enabled = value == nullptr || value[0] != '1';
    }
    if (operation == 1 &&
        (source.scalar_type() == kFloat32 ||
         source.scalar_type() == kFloat16 ||
         source.scalar_type() == kBFloat16) &&
        source.is_contiguous() &&
        selected + 1 == static_cast<std::size_t>(source.dim()) &&
        outer > 0 &&
        outer <= std::numeric_limits<unsigned int>::max() &&
        row_mean_enabled) {
        if (source.scalar_type() == kFloat32) {
            launch_row_mean(
                source.data_ptr<float>(), output.data_ptr<float>(),
                outer, reduced, stream);
        } else if (source.scalar_type() == kFloat16) {
            launch_row_mean(
                    static_cast<const __half*>(source.data_ptr()),
                    static_cast<__half*>(output.data_ptr()),
                    outer, reduced, stream);
        } else {
            launch_row_mean(
                    static_cast<const __nv_bfloat16*>(source.data_ptr()),
                    static_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer, reduced, stream);
        }
        MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
        return output;
    }
    if (operation == 3 &&
        source.scalar_type() == kBFloat16 &&
        source.is_contiguous() &&
        outer <= std::numeric_limits<unsigned int>::max() &&
        selected + 1 == static_cast<std::size_t>(source.dim())) {
        constexpr int threads = 256;
        argmax_last_contiguous_bf16_kernel<threads>
            <<<static_cast<unsigned int>(outer), threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(source.data_ptr()),
                output.data_ptr<std::int64_t>(), outer, reduced);
        MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
        return output;
    }
    auto launch = [&]<typename Input>() {
        launch_reduction_output<Input>(
            output, source, outer, reduced, static_cast<int>(selected), operation,
            keep_dimension, stream);
    };
    dispatch_numeric(source.scalar_type(), launch);
    MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
    return output;
}

Tensor sum(const Tensor& input, std::int64_t dimension, bool keep_dimension) {
    return input.sum(dimension, keep_dimension);
}
Tensor mean(const Tensor& input, std::int64_t dimension, bool keep_dimension) {
    return input.mean(dimension, keep_dimension);
}
Tensor argmax(const Tensor& input, std::int64_t dimension, bool keep_dimension) {
    return input.argmax(dimension, keep_dimension);
}

}  // namespace mfq::cuda
