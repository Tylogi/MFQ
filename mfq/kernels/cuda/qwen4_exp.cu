#include "qwen4_exp.h"

#include "selected_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mfq_qwen4_exp {
namespace tb = mfq_tensor_backend;
using tb::Tensor;

#ifdef MFQ_NATIVE_CUDA_RUNTIME
namespace {
template <typename Branch>
__global__ void gated_residual_post_kernel(const Branch* branch, const float* residual,
    const float* injection, float* output, int64_t count, int64_t hidden, int64_t streams) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t row = i / (hidden * streams);
        const int64_t stream = (i / hidden) % streams;
        // Match Metal's separate multiply/add rounding; FMA changes residual bits.
        const float update = __fmul_rn(float(branch[row * hidden + i % hidden]),
            injection[row * streams + stream]);
        output[i] = __fadd_rn(residual[i], update);
    }
}

template <int Threads>
__global__ void grouped_rms_norm_kernel(const float* input, const float* weight,
    float* output, int64_t group, int64_t width, double eps) {
    const int64_t base = int64_t(blockIdx.x) * group;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group; i += Threads) {
        const float value = input[base + i];
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    // Match native mean's tree and each materialized FP32 rounding step.
    // Warp reductions, FMA and rsqrtf can change the final output bits.
    __shared__ float partial[Threads];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = Threads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset)
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    // Native scalar arithmetic keeps non-FP32 scalars (including eps) in FP64.
    const float variance = float(double(partial[0] / float(group)) + eps);
    const float inverse = 1.0f / sqrtf(variance);
    for (int64_t i = threadIdx.x; i < group; i += Threads) {
        const float normalized = __fmul_rn(input[base + i], inverse);
        output[base + i] = __fmul_rn(normalized,
            __fadd_rn(1.0f, weight[(base + i) % width]));
    }
}
} // namespace
#endif

Tensor grouped_rms_norm(const Tensor& value, const Tensor& weight,
                        int64_t group, double eps) {
    mfq_selected_attention::values({&value, &weight});
    MFQ_RUNTIME_CHECK(value.dim() >= 1 && group > 0 && value.size(-1) > 0 &&
        value.size(-1) % group == 0 && weight.dim() == 1 && weight.size(0) == value.size(-1),
        "Qwen4 grouped RMSNorm dimensions disagree");
    MfqCudaGuard guard(value.device());
    auto shape = value.sizes().vec();
    shape.back() /= group;
    shape.push_back(group);
    auto source = value.to(tb::kFloat32).reshape(shape);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    const auto rows = source.numel() / group;
    const char* disable_mean = std::getenv("MFQ_DISABLE_NATIVE_PARALLEL_F32_MEAN");
    if (rows > 0 && rows <= std::numeric_limits<unsigned int>::max() &&
        (!disable_mean || disable_mean[0] != '1')) {
        source = source.contiguous();
        auto scale = weight.to(tb::kFloat32).contiguous();
        auto output = tb::empty(value.sizes(), source.options());
        const auto launch = [&]<int Threads>() {
            grouped_rms_norm_kernel<Threads><<<unsigned(rows), Threads, 0,
                mfq_current_cuda_stream()>>>(source.data_ptr<float>(), scale.data_ptr<float>(),
                    output.data_ptr<float>(), group, value.size(-1), eps);
        };
        if (group <= 32) launch.template operator()<32>();
        else if (group <= 64) launch.template operator()<64>();
        else if (group <= 128) launch.template operator()<128>();
        else launch.template operator()<256>();
        MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        return output.to(value.scalar_type());
    }
#endif
    auto normalized = source * tb::rsqrt((source * source).mean(-1, true) + eps);
    return (normalized.reshape(value.sizes()) * (1.0 + weight.to(tb::kFloat32)))
        .to(value.scalar_type());
}

std::vector<Tensor> gated_residual_pre(
    const Tensor& input, const Tensor& norm, const Tensor& down, const Tensor& up,
    const std::optional<Tensor>& inject, int64_t hidden, int64_t streams, double eps) {
    mfq_selected_attention::values({&input, &norm, &down, &up});
    MFQ_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 &&
        down.dim() == 2 && up.dim() == 2 && down.size(1) == input.size(-1) &&
        up.size(0) == input.size(-1) && up.size(1) == down.size(0),
        "Qwen4 gated-residual projection dimensions disagree");
    if (inject) {
        mfq_selected_attention::values({&input, &*inject});
        MFQ_RUNTIME_CHECK(inject->dim() == 2 && inject->size(0) == streams &&
            inject->size(1) == input.size(-1), "Qwen4 injection projection dimensions disagree");
    }
    MfqCudaGuard guard(input.device());
    auto normalized = grouped_rms_norm(input, norm, hidden, eps);
    auto low = mfq_selected_attention::promoted_matmul(normalized, down.transpose(-1, -2)) / streams;
    low = low * tb::sigmoid(low);
    auto mixing = tb::sigmoid(mfq_selected_attention::promoted_matmul(low, up.transpose(-1, -2)));
    auto shape = input.sizes().vec();
    shape.back() = streams;
    shape.push_back(hidden);
    auto mixed = (mixing.reshape(shape) * normalized.reshape(shape)).mean(-2);
    auto injection = inject
        ? 2.0 * tb::sigmoid(mfq_selected_attention::promoted_matmul(normalized, inject->transpose(-1, -2)) / streams)
        : Tensor{};
    return {mixed, input, injection};
}

Tensor gated_residual_post(const Tensor& branch, const Tensor& residual,
                           const Tensor& injection, int64_t streams) {
    mfq_selected_attention::values({&branch, &residual, &injection});
    MFQ_RUNTIME_CHECK(branch.dim() >= 1 && streams > 0 && branch.dim() == residual.dim() &&
        branch.dim() == injection.dim(), "Qwen4 residual ranks disagree");
    auto shape = branch.sizes().vec();
    shape.back() *= streams;
    MFQ_RUNTIME_CHECK(residual.sizes().vec() == shape, "Qwen4 residual width disagrees");
    shape.back() = streams;
    MFQ_RUNTIME_CHECK(injection.sizes().vec() == shape, "Qwen4 injection shape disagrees");
    MfqCudaGuard guard(branch.device());
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    if (residual.scalar_type() == tb::kFloat32 && injection.scalar_type() == tb::kFloat32) {
        auto b = branch.contiguous(), r = residual.contiguous(), g = injection.contiguous();
        auto output = tb::empty(residual.sizes(), residual.options());
        if (output.numel()) {
            const auto blocks = unsigned(std::min<int64_t>((output.numel() + 255) / 256, 65535));
            const auto launch = [&]<typename Branch>() {
                gated_residual_post_kernel<Branch><<<blocks, 256, 0, mfq_current_cuda_stream()>>>(
                    b.data_ptr<Branch>(), r.data_ptr<float>(), g.data_ptr<float>(), output.data_ptr<float>(),
                    output.numel(), branch.size(-1), streams);
            };
            if (branch.scalar_type() == tb::kFloat32) launch.template operator()<float>();
            else if (branch.scalar_type() == tb::kFloat16) launch.template operator()<__half>();
            else launch.template operator()<__nv_bfloat16>();
            MFQ_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return output;
    }
#endif
    return residual + (branch.unsqueeze(-2) * injection.unsqueeze(-1)).reshape(residual.sizes());
}

Tensor block_scores(const Tensor& query, const Tensor& pooled) {
    mfq_selected_attention::values({&query, &pooled});
    MfqCudaGuard guard(query.device());
    return mfq_selected_attention::dots(query, pooled).sum(-2) /
        std::sqrt(double(query.size(-1)));
}

std::vector<Tensor> ple_dilated_conv_silu(const Tensor& input,
    const Tensor& weight, const std::optional<Tensor>& state, int64_t dilation) {
    mfq_selected_attention::values({&input, &weight});
    MFQ_RUNTIME_CHECK(input.dim() == 3, "Qwen4 PLE input must have [B,T,C] shape");
    auto w = weight;
    if (w.dim() == 3) {
        MFQ_RUNTIME_CHECK(w.size(0) == input.size(2) && w.size(1) == 1,
            "Qwen4 PLE packed weight dimensions disagree");
        w = w.select(1, 0);
    }
    MFQ_RUNTIME_CHECK(w.dim() == 2 && w.size(0) == input.size(2) &&
        w.size(1) > 0 && dilation > 0 && w.size(1) - 1 <=
            std::numeric_limits<int64_t>::max() / dilation,
        "Qwen4 PLE kernel/dilation dimensions disagree");
    const auto length = (w.size(1) - 1) * dilation;
    const std::vector<int64_t> state_shape{input.size(0), length, input.size(2)};
    if (state) {
        mfq_selected_attention::values({&input, &*state});
        MFQ_RUNTIME_CHECK(state->sizes().vec() == state_shape, "Qwen4 PLE state shape disagrees");
    }
    MfqCudaGuard guard(input.device());
    auto previous = state ? state->to(input.scalar_type()) : tb::zeros(state_shape, input.options());
    auto combined = tb::cat({previous, input}, 1);
    auto output = tb::zeros(input.sizes(), input.options().dtype(tb::kFloat32));
    for (int64_t tap = 0; tap < w.size(1); ++tap) {
        output = output + combined.narrow(1, tap * dilation, input.size(1)).to(tb::kFloat32) *
            w.select(1, tap).to(tb::kFloat32).unsqueeze(0).unsqueeze(0);
    }
    output = output * tb::sigmoid(output);
    auto next = combined.narrow(1, length ? combined.size(1) - length : 0, length).contiguous();
    return {output.to(input.scalar_type()), next};
}

Tensor dense_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, int64_t offset) {
    return mfq_selected_attention::dense(
        q, k, v, offset, 1.0 / std::sqrt(double(q.size(3))));
}

Tensor sparse_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& indices) {
    return mfq_selected_attention::sparse(
        q, k, v, indices, 1.0 / std::sqrt(double(q.size(3))), false);
}

} // namespace mfq_qwen4_exp
