#include <cuda_fp16.h>
#include "deepseek_v4_hc.h"
#include "native/tensor/backend.h"
#include "deepseek_v41.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kHyperConnections = 4;
constexpr int kHyperMixWidth =
    (2 + kHyperConnections) * kHyperConnections;

__device__ __forceinline__ float dsv4_sigmoid(float value) {
    return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-value)));
}

__device__ __forceinline__ float dsv4_sum4(
    float x0, float x1, float x2, float x3) {
    return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3);
}

__device__ __forceinline__ float dsv4_row_sum4(
    float x0, float x1, float x2, float x3) {
    return __fadd_rn(
        __fadd_rn(x0, x1),
        __fadd_rn(x2, x3));
}

__device__ __forceinline__ float dsv4_softmax_sum4(
    float x0, float x1, float x2, float x3) {
    return __fadd_rn(
        __fadd_rn(x0, x2),
        __fadd_rn(x1, x3));
}

__global__ void dsv4_hc_pre_finalize_kernel(
    const half * __restrict__ x,
    const float * __restrict__ mixes,
    const float * __restrict__ scale,
    const float * __restrict__ base,
    half * __restrict__ reduced,
    float * __restrict__ post,
    float * __restrict__ combination,
    float * __restrict__ next_pre,
    int rows,
    int hidden,
    int iterations,
    float eps,
    const float* partials, int parts, const float* previous, const float* norm, float rms_epsilon)
{
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= rows) return;

    __shared__ float shared_pre[kHyperConnections];
    __shared__ float shared_comb[
        kHyperConnections * kHyperConnections];

    __shared__ float fused_mixes[24];
    __shared__ float fused_inverse[2];
    float sum=0;
    if(partials) {
        if(thread<26) for(int part=0;part<parts;++part)
            sum=__fadd_rn(sum,partials[(int64_t(row)*parts+part)*26+thread]);
        if(thread==24)fused_inverse[0]=rsqrtf(sum/(4.f*hidden)+rms_epsilon);
        if(thread==25)fused_inverse[1]=rsqrtf(sum/hidden+rms_epsilon);
        __syncthreads();
        if(thread<24)fused_mixes[thread]=__fmul_rn(sum,fused_inverse[0]);
        __syncthreads();
    }
    const float* row_mixes=partials?fused_mixes:mixes+int64_t(row)*kHyperMixWidth;
    if (thread < 32) {
        if (thread < kHyperConnections) {
            const float pre_affine = __fadd_rn(
                __fmul_rn(
                    row_mixes[thread],
                    scale[0]),
                base[thread]);
            shared_pre[thread] = __fadd_rn(
                dsv4_sigmoid(pre_affine), eps);
            if (next_pre)
                next_pre[static_cast<int64_t>(row) * kHyperConnections + thread] = shared_pre[thread];

            const int post_index = kHyperConnections + thread;
            const float post_affine = __fadd_rn(
                __fmul_rn(
                    row_mixes[post_index],
                    scale[1]),
                base[post_index]);
            post[static_cast<int64_t>(row) * kHyperConnections + thread] =
                __fmul_rn(2.0f, dsv4_sigmoid(post_affine));
        }
        if (thread < kHyperConnections * kHyperConnections) {
            const int mix_index = 2 * kHyperConnections + thread;
            shared_comb[thread] = __fadd_rn(
                __fmul_rn(
                    row_mixes[mix_index],
                    scale[2]),
                base[mix_index]);
        }
        __syncwarp();

        if (thread < kHyperConnections) {
            const int offset = thread * kHyperConnections;
            const float maximum = fmaxf(
                fmaxf(shared_comb[offset], shared_comb[offset + 1]),
                fmaxf(shared_comb[offset + 2], shared_comb[offset + 3]));
            const float e0 = expf(shared_comb[offset] - maximum);
            const float e1 = expf(shared_comb[offset + 1] - maximum);
            const float e2 = expf(shared_comb[offset + 2] - maximum);
            const float e3 = expf(shared_comb[offset + 3] - maximum);
            const float sum = dsv4_softmax_sum4(e0, e1, e2, e3);
            shared_comb[offset] = __fadd_rn(__fdiv_rn(e0, sum), eps);
            shared_comb[offset + 1] =
                __fadd_rn(__fdiv_rn(e1, sum), eps);
            shared_comb[offset + 2] =
                __fadd_rn(__fdiv_rn(e2, sum), eps);
            shared_comb[offset + 3] =
                __fadd_rn(__fdiv_rn(e3, sum), eps);
        }
        __syncwarp();

        if (thread < kHyperConnections) {
            const int column = thread;
            const float denominator = __fadd_rn(
                dsv4_sum4(
                    shared_comb[column],
                    shared_comb[kHyperConnections + column],
                    shared_comb[2 * kHyperConnections + column],
                    shared_comb[3 * kHyperConnections + column]),
                eps);
#pragma unroll
            for (int source = 0; source < kHyperConnections; ++source) {
                const int index =
                    source * kHyperConnections + column;
                shared_comb[index] =
                    __fdiv_rn(shared_comb[index], denominator);
            }
        }
        __syncwarp();

#pragma unroll
        for (int iteration = 1;
             iteration < iterations;
             ++iteration) {
            if (thread < kHyperConnections) {
                const int offset = thread * kHyperConnections;
                const float denominator = __fadd_rn(
                    dsv4_row_sum4(
                        shared_comb[offset],
                        shared_comb[offset + 1],
                        shared_comb[offset + 2],
                        shared_comb[offset + 3]),
                    eps);
                shared_comb[offset] =
                    __fdiv_rn(shared_comb[offset], denominator);
                shared_comb[offset + 1] =
                    __fdiv_rn(shared_comb[offset + 1], denominator);
                shared_comb[offset + 2] =
                    __fdiv_rn(shared_comb[offset + 2], denominator);
                shared_comb[offset + 3] =
                    __fdiv_rn(shared_comb[offset + 3], denominator);
            }
            __syncwarp();
            if (thread < kHyperConnections) {
                const int column = thread;
                const float denominator = __fadd_rn(
                    dsv4_sum4(
                        shared_comb[column],
                        shared_comb[kHyperConnections + column],
                        shared_comb[2 * kHyperConnections + column],
                        shared_comb[3 * kHyperConnections + column]),
                    eps);
#pragma unroll
                for (int source = 0;
                     source < kHyperConnections;
                     ++source) {
                    const int index =
                        source * kHyperConnections + column;
                    shared_comb[index] =
                        __fdiv_rn(shared_comb[index], denominator);
                }
            }
            __syncwarp();
        }

        if (thread < kHyperConnections * kHyperConnections) {
            combination[
                static_cast<int64_t>(row) *
                    kHyperConnections * kHyperConnections +
                thread] = shared_comb[thread];
        }
    }
    __syncthreads();

    if (!reduced) return;

    for (int feature = thread; feature < hidden; feature += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int source = 0; source < kHyperConnections; ++source) {
            value = __fadd_rn(
                value,
                __fmul_rn(
                    previous ? previous[row*4+source] : shared_pre[source],
                    __half2float(
                        x[(static_cast<int64_t>(row) *
                               kHyperConnections +
                           source) *
                              hidden +
                          feature])));
        }
        if(partials)value=__fmul_rn(__fmul_rn(value,fused_inverse[1]),norm[feature]);
        reduced[static_cast<int64_t>(row) * hidden + feature] =
            __float2half_rn(value);
    }
}

__global__ void dsv4_hc_post_kernel(
    const half * __restrict__ x,
    const half * __restrict__ residual,
    const float * __restrict__ post,
    const float * __restrict__ combination,
    half * __restrict__ output,
    int rows,
    int hidden)
{
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int destination = blockIdx.z;
    if (feature >= hidden || row >= rows) return;

    float residual_sum = 0.0f;
#pragma unroll
    for (int source = 0; source < kHyperConnections; ++source) {
        residual_sum = __fadd_rn(
            residual_sum,
            __fmul_rn(
                combination[
                    (static_cast<int64_t>(row) *
                         kHyperConnections +
                     source) *
                        kHyperConnections +
                    destination],
                __half2float(
                    residual[
                        (static_cast<int64_t>(row) *
                             kHyperConnections +
                         source) *
                            hidden +
                        feature])));
    }
    const float direct = __fmul_rn(
        post[static_cast<int64_t>(row) * kHyperConnections + destination],
        __half2float(x[static_cast<int64_t>(row) * hidden + feature]));
    output[
        (static_cast<int64_t>(row) * kHyperConnections + destination) *
            hidden +
        feature] = __float2half_rn(__fadd_rn(direct, residual_sum));
}

} // namespace

std::vector<mfq_tensor_backend::Tensor> dsv4_hc_pre_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor mixes,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor base,
    int64_t iterations,
    double eps)
{
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 &&
            x.dim() == 4 && x.size(2) == kHyperConnections &&
            x.size(3) > 0 && x.size(3) <= INT_MAX,
        "dsv4_hc_pre: x must be contiguous CUDA f16 [B,T,4,H]");
    MFQ_RUNTIME_CHECK(
        mixes.is_cuda() && mixes.is_contiguous() &&
            mixes.scalar_type() == mfq_tensor_backend::kFloat32 &&
            mixes.dim() == 3 && mixes.size(0) == x.size(0) &&
            mixes.size(1) == x.size(1) &&
            mixes.size(2) == kHyperMixWidth,
        "dsv4_hc_pre: mixes must be contiguous CUDA f32 [B,T,24]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
            scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
            scale.numel() == 3 &&
            base.is_cuda() && base.is_contiguous() &&
            base.scalar_type() == mfq_tensor_backend::kFloat32 &&
            base.numel() == kHyperMixWidth &&
            iterations > 0 && iterations <= INT_MAX &&
            std::isfinite(eps) && eps > 0.0,
        "dsv4_hc_pre: invalid scale, base, iterations, or epsilon");
    const int64_t rows64 = x.size(0) * x.size(1);
    MFQ_RUNTIME_CHECK(
        rows64 > 0 && rows64 <= INT_MAX,
        "dsv4_hc_pre: row count is out of range");
    const int rows = static_cast<int>(rows64);
    const int hidden = static_cast<int>(x.size(3));
    auto reduced = mfq_tensor_backend::empty(
        {x.size(0), x.size(1), hidden}, x.options());
    auto post = mfq_tensor_backend::empty(
        {x.size(0), x.size(1), kHyperConnections},
        mixes.options());
    auto combination = mfq_tensor_backend::empty(
        {x.size(0), x.size(1),
         kHyperConnections, kHyperConnections},
        mixes.options());
    dsv4_hc_pre_finalize_kernel<<<
        rows, 256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<mfq_half>()),
        mixes.data_ptr<float>(), scale.data_ptr<float>(),
        base.data_ptr<float>(),
        reinterpret_cast<half *>(reduced.data_ptr<mfq_half>()),
        post.data_ptr<float>(), combination.data_ptr<float>(),
        nullptr,
        rows, hidden, static_cast<int>(iterations), static_cast<float>(eps),
        nullptr, 0, nullptr, nullptr, 0.f);
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_hc_pre launch failed: ", cudaGetErrorString(status));
    return {reduced, post, combination};
}

std::vector<mfq_tensor_backend::Tensor> dsv4_hc_coefficients_cuda(
    mfq_tensor_backend::Tensor mixes,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor base,
    int64_t iterations,
    double eps) {
    MFQ_RUNTIME_CHECK(mixes.is_cuda() && mixes.is_contiguous() &&
                          mixes.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          mixes.dim() == 3 && mixes.size(2) == kHyperMixWidth,
                      "dsv4_hc_coefficients: expected CUDA f32 [B,T,24]");
    MFQ_RUNTIME_CHECK(scale.is_cuda() && scale.is_contiguous() &&
                          scale.scalar_type() == mfq_tensor_backend::kFloat32 && scale.numel() == 3 &&
                          base.is_cuda() && base.is_contiguous() &&
                          base.scalar_type() == mfq_tensor_backend::kFloat32 &&
                          base.numel() == kHyperMixWidth && iterations > 0 && iterations <= INT_MAX &&
                          std::isfinite(eps) && eps > 0,
                      "dsv4_hc_coefficients: invalid coefficient parameters");
    const auto rows = mixes.size(0) * mixes.size(1);
    MFQ_RUNTIME_CHECK(rows > 0 && rows <= INT_MAX,
                      "dsv4_hc_coefficients: row count is out of range");
    auto pre = mfq_tensor_backend::empty({mixes.size(0), mixes.size(1), kHyperConnections}, mixes.options());
    auto post = mfq_tensor_backend::empty_like(pre);
    auto combination = mfq_tensor_backend::empty(
        {mixes.size(0), mixes.size(1), kHyperConnections, kHyperConnections}, mixes.options());
    dsv4_hc_pre_finalize_kernel<<<static_cast<int>(rows), 32, 0, mfq_current_cuda_stream()>>>(
        nullptr, mixes.data_ptr<float>(), scale.data_ptr<float>(), base.data_ptr<float>(), nullptr,
        post.data_ptr<float>(), combination.data_ptr<float>(), pre.data_ptr<float>(),
        static_cast<int>(rows), 0, static_cast<int>(iterations), static_cast<float>(eps),
        nullptr, 0, nullptr, nullptr, 0.f);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {pre, post, combination};
}

mfq_tensor_backend::Tensor dsv4_hc_post_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor post,
    mfq_tensor_backend::Tensor combination)
{
    MFQ_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == mfq_tensor_backend::kFloat16 &&
            x.dim() == 3 && x.size(2) > 0 && x.size(2) <= INT_MAX,
        "dsv4_hc_post: x must be contiguous CUDA f16 [B,T,H]");
    MFQ_RUNTIME_CHECK(
        residual.is_cuda() && residual.is_contiguous() &&
            residual.scalar_type() == mfq_tensor_backend::kFloat16 &&
            residual.dim() == 4 &&
            residual.size(0) == x.size(0) &&
            residual.size(1) == x.size(1) &&
            residual.size(2) == kHyperConnections &&
            residual.size(3) == x.size(2),
        "dsv4_hc_post: residual shape mismatch");
    MFQ_RUNTIME_CHECK(
        post.is_cuda() && post.is_contiguous() &&
            post.scalar_type() == mfq_tensor_backend::kFloat32 &&
            post.dim() == 3 &&
            post.size(0) == x.size(0) &&
            post.size(1) == x.size(1) &&
            post.size(2) == kHyperConnections &&
            combination.is_cuda() && combination.is_contiguous() &&
            combination.scalar_type() == mfq_tensor_backend::kFloat32 &&
            combination.dim() == 4 &&
            combination.size(0) == x.size(0) &&
            combination.size(1) == x.size(1) &&
            combination.size(2) == kHyperConnections &&
            combination.size(3) == kHyperConnections,
        "dsv4_hc_post: post or combination shape mismatch");
    const int64_t rows64 = x.size(0) * x.size(1);
    MFQ_RUNTIME_CHECK(
        rows64 > 0 && rows64 <= 65535,
        "dsv4_hc_post: row count is out of range");
    const int rows = static_cast<int>(rows64);
    const int hidden = static_cast<int>(x.size(2));
    auto output = mfq_tensor_backend::empty(
        {x.size(0), x.size(1), kHyperConnections, hidden},
        residual.options());
    dsv4_hc_post_kernel<<<
        dim3((hidden + 255) / 256, rows, kHyperConnections),
        256, 0, mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half *>(x.data_ptr<mfq_half>()),
        reinterpret_cast<const half *>(residual.data_ptr<mfq_half>()),
        post.data_ptr<float>(), combination.data_ptr<float>(),
        reinterpret_cast<half *>(output.data_ptr<mfq_half>()),
        rows, hidden);
    const cudaError_t status = cudaGetLastError();
    MFQ_RUNTIME_CHECK(
        status == cudaSuccess,
        "dsv4_hc_post launch failed: ", cudaGetErrorString(status));
    return output;
}

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_finalize_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous,
    mfq_tensor_backend::Tensor norm, mfq_tensor_backend::Tensor partials,
    mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor base,
    int64_t iterations, double rms_epsilon, double hc_epsilon) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(residual.is_cuda() && residual.is_contiguous() && residual.scalar_type()==tb::kFloat16 &&
        residual.dim()==4 && residual.size(2)==4 && residual.size(3)>0 && residual.size(3)<=INT_MAX/4,
        "invalid mHC finalize residual");
    const auto rows=residual.size(0)*residual.size(1),hidden=residual.size(3),parts=(4*hidden+63)/64;
    MFQ_RUNTIME_CHECK(rows>0 && rows<=INT_MAX && partials.is_contiguous() && partials.is_cuda() &&
        partials.scalar_type()==tb::kFloat32 && partials.dim()==3 && partials.size(0)==rows &&
        partials.size(1)==parts && partials.size(2)==26 && previous.is_cuda() && previous.is_contiguous() &&
        previous.scalar_type()==tb::kFloat32 && previous.dim()==3 && previous.size(0)==residual.size(0) &&
        previous.size(1)==residual.size(1) && previous.size(2)==4 && norm.is_cuda() && norm.is_contiguous() &&
        norm.scalar_type()==tb::kFloat32 && norm.numel()==hidden && scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type()==tb::kFloat32 && scale.numel()==3 && base.is_cuda() && base.is_contiguous() &&
        base.scalar_type()==tb::kFloat32 && base.numel()==24 && iterations>0 && iterations<=INT_MAX &&
        std::isfinite(rms_epsilon) && rms_epsilon>0 && std::isfinite(hc_epsilon) && hc_epsilon>0 &&
        previous.device()==residual.device() && norm.device()==residual.device() && partials.device()==residual.device() &&
        scale.device()==residual.device() && base.device()==residual.device(),"invalid mHC finalize inputs");
    auto branch=tb::empty({residual.size(0),residual.size(1),hidden},residual.options());
    auto pre=tb::empty({residual.size(0),residual.size(1),4},previous.options()),post=tb::empty_like(pre);
    auto combination=tb::empty({residual.size(0),residual.size(1),4,4},previous.options());
    dsv4_hc_pre_finalize_kernel<<<int(rows),256,0,mfq_current_cuda_stream()>>>(
        reinterpret_cast<const half*>(residual.data_ptr<mfq_half>()),nullptr,scale.data_ptr<float>(),base.data_ptr<float>(),
        reinterpret_cast<half*>(branch.data_ptr<mfq_half>()),post.data_ptr<float>(),combination.data_ptr<float>(),
        pre.data_ptr<float>(),int(rows),int(hidden),int(iterations),float(hc_epsilon),partials.data_ptr<float>(),int(parts),
        previous.data_ptr<float>(),norm.data_ptr<float>(),float(rms_epsilon));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();return {branch,pre,post,combination};
}
