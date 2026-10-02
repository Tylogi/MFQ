#include "mfq_native_tensor_ops_common.cuh"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mfq::cuda {
using namespace native_ops_detail;
namespace {
cudaDataType_t cuda_data_type(ScalarType type) {
    switch (type) {
        case kFloat16: return CUDA_R_16F;
        case kBFloat16: return CUDA_R_16BF;
        case kFloat32: return CUDA_R_32F;
        case kFloat64: return CUDA_R_64F;
        default: throw std::invalid_argument("cuBLAS matmul requires a floating dtype");
    }
}

std::vector<std::int64_t> matmul_batch_shape(const Tensor& left, const Tensor& right) {
    const auto left_rank = static_cast<std::size_t>(left.dim() - 2);
    const auto right_rank = static_cast<std::size_t>(right.dim() - 2);
    const auto rank = std::max(left_rank, right_rank);
    std::vector<std::int64_t> shape(rank, 1);
    for (std::size_t reverse = 0; reverse < rank; ++reverse) {
        const auto left_axis = static_cast<std::int64_t>(left_rank) - 1 -
            static_cast<std::int64_t>(reverse);
        const auto right_axis = static_cast<std::int64_t>(right_rank) - 1 -
            static_cast<std::int64_t>(reverse);
        const auto l = left_axis >= 0 ? left.size(left_axis) : 1;
        const auto r = right_axis >= 0 ? right.size(right_axis) : 1;
        if (l != r && l != 1 && r != 1) {
            throw std::invalid_argument("matmul batch dimensions cannot be broadcast");
        }
        shape[rank - 1 - reverse] = std::max(l, r);
    }
    return shape;
}

std::vector<std::int64_t> batch_coordinates(
    std::int64_t linear,
    std::span<const std::int64_t> shape) {
    std::vector<std::int64_t> coordinates(shape.size(), 0);
    for (std::size_t reverse = shape.size(); reverse > 0; --reverse) {
        const auto axis = reverse - 1;
        coordinates[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    return coordinates;
}

std::int64_t matmul_batch_offset(
    const Tensor& tensor,
    std::span<const std::int64_t> output_shape,
    std::span<const std::int64_t> coordinates) {
    const auto input_rank = static_cast<std::size_t>(tensor.dim() - 2);
    const auto offset = output_shape.size() - input_rank;
    std::int64_t result = 0;
    for (std::size_t axis = 0; axis < input_rank; ++axis) {
        const auto extent = tensor.size(static_cast<std::int64_t>(axis));
        if (extent != 1) {
            result += coordinates[offset + axis] *
                tensor.stride(static_cast<std::int64_t>(axis));
        }
    }
    return result;
}

__global__ void scale_causal_bf16_kernel(
    const __nv_bfloat16* source,
    __nv_bfloat16* destination,
    std::int64_t elements,
    std::int64_t query_rows,
    std::int64_t keys,
    double factor) {
    for (std::int64_t linear =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < elements;
         linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto key = linear % keys;
        const auto query = (linear / keys) % query_rows;
        destination[linear] = key <= query
            ? store_number<__nv_bfloat16>(load_number(source, linear) * factor)
            : store_number<__nv_bfloat16>(
                -std::numeric_limits<double>::infinity());
    }
}

bool matrix_layout_supported(const Tensor& tensor) {
    return (tensor.stride(-1) == 1 &&
            (tensor.size(-2) <= 1 || tensor.stride(-2) >= tensor.size(-1))) ||
        (tensor.stride(-2) == 1 &&
            (tensor.size(-1) <= 1 || tensor.stride(-1) >= tensor.size(-2)));
}

int matrix_leading_dimension(const Tensor& tensor, bool row_major) {
    const auto minimum = std::max<std::int64_t>(1, tensor.size(row_major ? -1 : -2));
    const auto leading = tensor.size(row_major ? -2 : -1) <= 1
        ? minimum : tensor.stride(row_major ? -2 : -1);
    if (leading > std::numeric_limits<int>::max()) {
        throw std::overflow_error("matmul stride exceeds cuBLAS integer ABI");
    }
    return static_cast<int>(leading);
}

// Flatten only batches whose addresses form an arithmetic progression. Mixed
// broadcasting (e.g. [B,1] by [1,H]) keeps the general per-batch path.
std::optional<std::int64_t> regular_batch_stride(
    const Tensor& tensor, const std::vector<std::int64_t>& batch_shape) {
    std::optional<std::int64_t> step;
    std::int64_t extent = 1;
    const auto padding = static_cast<std::int64_t>(batch_shape.size()) - tensor.dim() + 2;
    for (auto axis = static_cast<std::int64_t>(batch_shape.size()); axis-- > 0;) {
        if (batch_shape[axis] <= 1) continue;
        const auto source_axis = axis - padding;
        const auto stride = source_axis < 0 || tensor.size(source_axis) == 1
            ? 0 : tensor.stride(source_axis);
        if (!step) step = stride;
        if (stride != *step * extent) return std::nullopt;
        extent *= batch_shape[axis];
    }
    return step.value_or(0);
}

struct ParallelBatchMatmulContext {
    static constexpr std::size_t kStreams = 4;

    explicit ParallelBatchMatmulContext(int device) : ready() {
        streams.reserve(kStreams);
        handles.reserve(kStreams);
        done.reserve(kStreams);
        for (std::size_t index = 0; index < kStreams; ++index) {
            streams.emplace_back(device);
            handles.emplace_back(device);
            handles.back().set_stream(streams.back().get());
            done.emplace_back();
        }
    }

    std::mutex mutex;
    Event ready;
    std::vector<Stream> streams;
    std::vector<BlasHandle> handles;
    std::vector<Event> done;
};

std::mutex parallel_batch_matmul_contexts_mutex;
std::unordered_map<int, std::unique_ptr<ParallelBatchMatmulContext>>
    parallel_batch_matmul_contexts;

ParallelBatchMatmulContext& parallel_batch_matmul_context(int device) {
    std::lock_guard<std::mutex> lock(parallel_batch_matmul_contexts_mutex);
    auto& context = parallel_batch_matmul_contexts[device];
    if (!context) {
        DeviceGuard guard(device);
        context = std::make_unique<ParallelBatchMatmulContext>(device);
    }
    return *context;
}

}  // namespace

Tensor matmul(const Tensor& left_source, const Tensor& right_source) {
    if (!left_source.defined() || !right_source.defined() ||
        !left_source.is_cuda() || !right_source.is_cuda()) {
        throw std::invalid_argument("native matmul requires CUDA tensors");
    }
    if (left_source.device() != right_source.device()) {
        throw std::invalid_argument("native matmul requires one CUDA device");
    }
    if (left_source.dim() < 1 || right_source.dim() < 1) {
        throw std::invalid_argument("matmul requires tensors with at least one dimension");
    }
    if (left_source.scalar_type() != right_source.scalar_type() ||
        !floating(left_source.scalar_type())) {
        throw std::invalid_argument("native matmul requires matching floating dtypes");
    }
    const bool left_vector = left_source.dim() == 1;
    const bool right_vector = right_source.dim() == 1;
    auto left = left_vector ? left_source.unsqueeze(0) : left_source;
    auto right = right_vector ? right_source.unsqueeze(-1) : right_source;
    if (left.size(-1) != right.size(-2)) {
        throw std::invalid_argument("matmul contraction dimensions do not match");
    }
    if (!matrix_layout_supported(left)) left = left.contiguous();
    if (!matrix_layout_supported(right)) right = right.contiguous();

    const auto batch_shape = matmul_batch_shape(left, right);
    const auto batches = std::accumulate(
        batch_shape.begin(), batch_shape.end(), std::int64_t{1}, std::multiplies<>());
    const auto rows = left.size(-2);
    const auto contraction = left.size(-1);
    const auto columns = right.size(-1);
    auto output_shape = batch_shape;
    output_shape.push_back(rows);
    output_shape.push_back(columns);
    auto output = empty(output_shape, left.options());
    if (output.numel() == 0) {
        if (left_vector) output = output.squeeze(-2);
        if (right_vector) output = output.squeeze(-1);
        return output;
    }

    DeviceGuard guard(left.get_device());
    auto context = default_context(left.get_device());
    const auto stream = current_stream(left.get_device()).stream();
    context->blas().set_stream(stream);
    const auto handle = context->blas().get();
    const auto data_type = cuda_data_type(left.scalar_type());

    const bool left_row_major = left.stride(-1) == 1;
    const bool right_row_major = right.stride(-1) == 1;
    const auto left_operation = left_row_major ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto right_operation = right_row_major ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (rows > std::numeric_limits<int>::max() ||
        columns > std::numeric_limits<int>::max() ||
        contraction > std::numeric_limits<int>::max()) {
        throw std::overflow_error("matmul dimension exceeds cuBLAS integer ABI");
    }
    const int left_leading = matrix_leading_dimension(left, left_row_major);
    const int right_leading = matrix_leading_dimension(right, right_row_major);

    if (contraction == 0) {
        output.zero_();
        if (left_vector) output = output.squeeze(-2);
        if (right_vector) output = output.squeeze(-1);
        return output;
    }
    const auto left_batch_stride = regular_batch_stride(left, batch_shape);
    const auto right_batch_stride = regular_batch_stride(right, batch_shape);
    const char* strided_disabled = std::getenv("MFQ_DISABLE_NATIVE_STRIDED_BATCH_MATMUL");
    if (batches > 1 && batches <= std::numeric_limits<int>::max() &&
        left_batch_stride && right_batch_stride &&
        (strided_disabled == nullptr || strided_disabled[0] != '1')) {
        const float alpha = 1.0f, beta = 0.0f;
        const double alpha64 = 1.0, beta64 = 0.0;
        const bool fp64 = left.scalar_type() == kFloat64;
        MFQ_NATIVE_CUDA_CHECK(cublasGemmStridedBatchedEx(
            handle, right_operation, left_operation,
            static_cast<int>(columns), static_cast<int>(rows), static_cast<int>(contraction),
            fp64 ? static_cast<const void*>(&alpha64) : &alpha,
            right.data_ptr(), data_type, right_leading, *right_batch_stride,
            left.data_ptr(), data_type, left_leading, *left_batch_stride,
            fp64 ? static_cast<const void*>(&beta64) : &beta,
            output.data_ptr(), data_type, static_cast<int>(columns), rows * columns,
            static_cast<int>(batches),
            fp64 ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        if (left_vector) output = output.squeeze(-2);
        if (right_vector) output = output.squeeze(-1);
        return output;
    }

    const char* parallel_batch_disabled =
        std::getenv("MFQ_DISABLE_NATIVE_PARALLEL_BATCH_MATMUL");
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool parallel_batch_eligible =
        batches >= static_cast<std::int64_t>(
            ParallelBatchMatmulContext::kStreams) &&
        rows >= 32 &&
        (parallel_batch_disabled == nullptr ||
         parallel_batch_disabled[0] != '1');
    if (parallel_batch_eligible) {
        MFQ_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    }
    const bool parallel_batch_enabled =
        parallel_batch_eligible &&
        capture_status == cudaStreamCaptureStatusNone;
    if (parallel_batch_enabled) {
        auto& parallel = parallel_batch_matmul_context(left.get_device());
        std::lock_guard<std::mutex> lock(parallel.mutex);
        parallel.ready.record(stream);
        for (auto& worker : parallel.streams) {
            worker.wait(parallel.ready);
        }
        for (std::int64_t batch = 0; batch < batches; ++batch) {
            const auto coordinates = batch_coordinates(batch, batch_shape);
            const auto left_offset =
                matmul_batch_offset(left, batch_shape, coordinates);
            const auto right_offset =
                matmul_batch_offset(right, batch_shape, coordinates);
            const auto* left_pointer =
                static_cast<const std::byte*>(left.data_ptr()) +
                left_offset * left.element_size();
            const auto* right_pointer =
                static_cast<const std::byte*>(right.data_ptr()) +
                right_offset * right.element_size();
            auto* output_pointer = static_cast<std::byte*>(output.data_ptr()) +
                batch * rows * columns * output.element_size();
            const auto worker = static_cast<std::size_t>(batch) %
                ParallelBatchMatmulContext::kStreams;
            const auto worker_handle = parallel.handles[worker].get();
            if (left.scalar_type() == kFloat64) {
                const double alpha = 1.0;
                const double beta = 0.0;
                MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                    worker_handle,
                    right_operation, left_operation,
                    static_cast<int>(columns), static_cast<int>(rows),
                    static_cast<int>(contraction),
                    &alpha,
                    right_pointer, data_type, right_leading,
                    left_pointer, data_type, left_leading,
                    &beta,
                    output_pointer, data_type, static_cast<int>(columns),
                    CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
            } else {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                    worker_handle,
                    right_operation, left_operation,
                    static_cast<int>(columns), static_cast<int>(rows),
                    static_cast<int>(contraction),
                    &alpha,
                    right_pointer, data_type, right_leading,
                    left_pointer, data_type, left_leading,
                    &beta,
                    output_pointer, data_type, static_cast<int>(columns),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
        for (std::size_t worker = 0;
             worker < ParallelBatchMatmulContext::kStreams; ++worker) {
            parallel.done[worker].record(parallel.streams[worker].get());
            MFQ_NATIVE_CUDA_CHECK(cudaStreamWaitEvent(
                stream, parallel.done[worker].get(), 0));
        }
        if (left_vector) output = output.squeeze(-2);
        if (right_vector) output = output.squeeze(-1);
        return output;
    }

    for (std::int64_t batch = 0; batch < batches; ++batch) {
        const auto coordinates = batch_coordinates(batch, batch_shape);
        const auto left_offset = matmul_batch_offset(left, batch_shape, coordinates);
        const auto right_offset = matmul_batch_offset(right, batch_shape, coordinates);
        const auto* left_pointer = static_cast<const std::byte*>(left.data_ptr()) +
            left_offset * left.element_size();
        const auto* right_pointer = static_cast<const std::byte*>(right.data_ptr()) +
            right_offset * right.element_size();
        auto* output_pointer = static_cast<std::byte*>(output.data_ptr()) +
            batch * rows * columns * output.element_size();
        if (left.scalar_type() == kFloat64) {
            const double alpha = 1.0;
            const double beta = 0.0;
            MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                handle,
                right_operation, left_operation,
                static_cast<int>(columns), static_cast<int>(rows),
                static_cast<int>(contraction),
                &alpha,
                right_pointer, data_type, right_leading,
                left_pointer, data_type, left_leading,
                &beta,
                output_pointer, data_type, static_cast<int>(columns),
                CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
        } else {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            MFQ_NATIVE_CUDA_CHECK(cublasGemmEx(
                handle,
                right_operation, left_operation,
                static_cast<int>(columns), static_cast<int>(rows),
                static_cast<int>(contraction),
                &alpha,
                right_pointer, data_type, right_leading,
                left_pointer, data_type, left_leading,
                &beta,
                output_pointer, data_type, static_cast<int>(columns),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    }
    if (left_vector) output = output.squeeze(-2);
    if (right_vector) output = output.squeeze(-1);
    return output;
}

Tensor bmm(const Tensor& left, const Tensor& right) {
    if (left.dim() != 3 || right.dim() != 3 || left.size(0) != right.size(0)) {
        throw std::invalid_argument("bmm requires equally batched rank-three tensors");
    }
    return matmul(left, right);
}

Tensor baddbmm(const Tensor& input, const Tensor& left, const Tensor& right) {
    return input + bmm(left, right);
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
    if (weight.dim() != 2) throw std::invalid_argument("linear weight must be a matrix");
    auto output = matmul(input, weight.transpose(0, 1));
    if (bias.has_value() && bias->defined()) output = output + *bias;
    return output;
}

Tensor scaled_dot_product_attention(
    const Tensor& query_source,
    const Tensor& key_source,
    const Tensor& value_source,
    const std::optional<Tensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention) {
    if (dropout != 0.0) {
        throw std::invalid_argument("native inference attention does not support dropout");
    }
    if (query_source.dim() < 3 || key_source.dim() != query_source.dim() ||
        value_source.dim() != query_source.dim()) {
        throw std::invalid_argument("attention tensors have incompatible ranks");
    }
    if (causal && mask.has_value()) {
        throw std::invalid_argument("attention cannot combine explicit and causal masks");
    }
    // Bound score storage to one query tile. Each tile still uses GEMM and the
    // existing dtype-specific softmax, preserving reduced-precision semantics.
    const char* tiled_disabled = std::getenv("MFQ_DISABLE_NATIVE_TILED_SDPA");
    if (query_source.dim() == 4 && query_source.size(-2) > 128 &&
        key_source.size(0) == query_source.size(0) &&
        value_source.size(0) == query_source.size(0) &&
        (tiled_disabled == nullptr || tiled_disabled[0] != '1')) {
        auto shape = query_source.sizes().vec();
        shape.back() = value_source.size(-1);
        auto output = empty(shape, query_source.options());
        for (std::int64_t begin = 0; begin < query_source.size(-2); begin += 128) {
            const auto length = std::min<std::int64_t>(128, query_source.size(-2) - begin);
            std::optional<Tensor> local_mask = mask;
            if (causal) {
                const auto indices = TensorOptions{}.dtype(kInt64).device(query_source.device());
                local_mask = arange(key_source.size(-2), indices).unsqueeze(0) <=
                    (arange(length, indices) + begin).unsqueeze(-1);
            } else if (mask && mask->dim() >= 2 && mask->size(-2) > 1) {
                local_mask = mask->narrow(-2, begin, length);
            }
            output.narrow(-2, begin, length).copy_(scaled_dot_product_attention(
                query_source.narrow(-2, begin, length), key_source, value_source,
                local_mask, dropout, false, scale, enable_grouped_query_attention));
        }
        return output;
    }
    auto query = query_source;
    auto key = key_source;
    auto value = value_source;
    bool grouped_view = false;
    const auto head_dimension = query_source.dim() - 3;
    if (query_source.size(head_dimension) != key.size(head_dimension)) {
        if (!enable_grouped_query_attention ||
            query_source.size(head_dimension) % key.size(head_dimension) != 0) {
            throw std::invalid_argument("attention head counts are incompatible");
        }
        const auto repeat = query_source.size(head_dimension) / key.size(head_dimension);
        if (query_source.dim() == 4 && key.size(1) == value.size(1)) {
            query = query_source.reshape({query_source.size(0), key.size(1), repeat,
                query_source.size(2), query_source.size(3)});
            key = key.unsqueeze(2);
            value = value.unsqueeze(2);
            grouped_view = true;
        } else {
            key = key.repeat_interleave(repeat, head_dimension).contiguous();
            value = value.repeat_interleave(repeat, head_dimension).contiguous();
        }
    }
    const auto factor = scale.value_or(
        1.0 / std::sqrt(static_cast<double>(query_source.size(-1))));
    auto scores = matmul(query, key.transpose(-2, -1));
    if (grouped_view) scores = scores.reshape({scores.size(0), query_source.size(1),
        query_source.size(-2), key_source.size(-2)});
    const char* fused_causal_scale_disabled =
        std::getenv("MFQ_DISABLE_NATIVE_FUSED_CAUSAL_SCALE");
    const bool fused_causal_scale = causal && !mask.has_value() &&
        scores.scalar_type() == kBFloat16 && scores.is_contiguous() &&
        scores.dim() >= 2 && scores.size(-2) > 0 && scores.size(-1) > 0 &&
        (fused_causal_scale_disabled == nullptr ||
         fused_causal_scale_disabled[0] != '1');
    if (fused_causal_scale) {
        auto fused_scores = empty(scores.sizes(), scores.options());
        const auto [blocks, threads] = launch_geometry(scores.numel());
        const auto stream = current_stream(scores.get_device()).stream();
        scale_causal_bf16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scores.data_ptr()),
            static_cast<__nv_bfloat16*>(fused_scores.data_ptr()),
            scores.numel(), scores.size(-2), scores.size(-1), factor);
        MFQ_NATIVE_CUDA_CHECK(cudaGetLastError());
        scores = std::move(fused_scores);
    } else {
        scores = scores * factor;
    }
    if (causal && !fused_causal_scale) {
        if (mask.has_value()) {
            throw std::invalid_argument("attention cannot combine explicit and causal masks");
        }
        auto rows = arange(
            scores.size(-2), TensorOptions{}.dtype(kInt64).device(scores.device()))
            .unsqueeze(-1);
        auto columns = arange(
            scores.size(-1), TensorOptions{}.dtype(kInt64).device(scores.device()))
            .unsqueeze(0);
        scores = where(
            columns <= rows,
            scores,
            -std::numeric_limits<double>::infinity());
    } else if (mask.has_value()) {
        auto selected = mask->to(scores.device());
        if (selected.scalar_type() == kBool) {
            scores = where(
                selected,
                scores,
                -std::numeric_limits<double>::infinity());
        } else {
            scores = scores + selected.to(scores.scalar_type());
        }
    }
    auto probabilities = softmax(scores, -1).to(value.scalar_type());
    if (grouped_view) {
        probabilities = probabilities.reshape({scores.size(0), key.size(1),
            query.size(2), query_source.size(-2), key_source.size(-2)});
        return matmul(probabilities, value).reshape({scores.size(0), query_source.size(1),
            query_source.size(-2), value_source.size(-1)});
    }
    return matmul(probabilities, value);
}

}  // namespace mfq::cuda
