// GLU activation helpers used by materialized prefill paths.  This unit holds
// only device bodies and the TensorView + cudaStream_t launchers; the
// allocating wrappers are in ops/activation.cpp, which keeps this include
// closure free of any Tensor backend header.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "glu.cuh"
#include "kernels.h"

// Calling libdevice explicitly keeps standalone FP32 GeGLU within its FP32
// tolerance even when Torch compiles the extension with --use_fast_math.
// Ordinary tanhf is rewritten to the lower-accuracy tanh.approx instruction.
extern "C" __device__ float __nv_tanhf(float);

__global__ void clipped_silu_mul_kernel(const half* gate,const half* up,half* output,
        int64_t count,float limit) {
    const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) {
        const float g=fminf(__half2float(gate[i]),limit);
        const float u=fminf(fmaxf(__half2float(up[i]),-limit),limit);
        output[i]=__float2half_rn((g/(1.0f+expf(-g)))*u);
    }
}


template <typename T>
__device__ __forceinline__ float activation_as_float(T value) {
    if constexpr (std::is_same_v<T, half>) return __half2float(value);
    else return static_cast<float>(value);
}

template <typename T>
__device__ __forceinline__ T activation_from_float(float value) {
    if constexpr (std::is_same_v<T, half>) return __float2half_rn(value);
    else return static_cast<T>(value);
}

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void silu_mul_f16_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        out[i] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

__global__ void silu_mul_f16x2_kernel(
    const half2* __restrict__ gate,
    const half2* __restrict__ up,
    half2* __restrict__ out,
    size_t n)
{
    const size_t pairs = n / 2;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < pairs;
         i += (size_t)gridDim.x * blockDim.x) {
        const float2 g = __half22float2(gate[i]);
        const float2 u = __half22float2(up[i]);
        const float x = (g.x / (1.0f + expf(-g.x))) * u.x;
        const float y = (g.y / (1.0f + expf(-g.y))) * u.y;
        out[i] = __floats2half2_rn(x, y);
    }
    if ((n & 1) != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        const half* gate_tail = reinterpret_cast<const half*>(gate);
        const half* up_tail = reinterpret_cast<const half*>(up);
        half* out_tail = reinterpret_cast<half*>(out);
        const float g = __half2float(gate_tail[n - 1]);
        const float u = __half2float(up_tail[n - 1]);
        out_tail[n - 1] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        const __nv_bfloat16 activated = __float2bfloat16(
            g / (1.0f + expf(-g)));
        out[i] = __float2bfloat16(
            __bfloat162float(activated) * u);
    }
}

template <typename scalar_t>
__global__ void gelu_mul_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    size_t n)
{
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float g = activation_as_float(gate[i]);
        const float u = activation_as_float(up[i]);
        float value;
        if constexpr (std::is_same_v<scalar_t, float>) {
            constexpr float kGeluA = 0.044715f;
            constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876f;
            value = u * (0.5f * g *
                (1.0f + __nv_tanhf(kSqrt2OverPi * g * (1.0f + kGeluA * g * g))));
        } else {
            value = mfq_glu<true>(g, u);
        }
        if constexpr (!std::is_same_v<scalar_t, float>) {
            value = fminf(65504.0f, fmaxf(-65504.0f, value));
        }
        out[i] = activation_from_float<scalar_t>(value);
    }
}

template <typename scalar_t>
__global__ void linear_gate_beta_kernel(
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    float* __restrict__ gate_t,
    float* __restrict__ beta_t,
    int64_t B, int64_t T, int64_t V,
    int64_t as0, int64_t as1, int64_t as2,
    int64_t bs0, int64_t bs1, int64_t bs2, bool stable_softplus)
{
    size_t n = (size_t)B * (size_t)T * (size_t)V;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        int64_t v = (int64_t)(idx % (size_t)V);
        int64_t t = (int64_t)((idx / (size_t)V) % (size_t)T);
        int64_t b = (int64_t)(idx / ((size_t)T * (size_t)V));
        float a = activation_as_float(alpha[b * as0 + t * as1 + v * as2]) + dt_bias[v];
        float sp = stable_softplus ? fmaxf(a, 0.0f) + log1pf(expf(-fabsf(a)))
                                  : (a > 20.0f ? a : log1pf(expf(a)));
        float g = sp * -expf(a_log[v]);
        float bv = activation_as_float(beta[b * bs0 + t * bs1 + v * bs2]);
        float sig = 1.0f / (1.0f + expf(-bv));
        size_t out = ((size_t)b * (size_t)V + (size_t)v) * (size_t)T + (size_t)t;
        gate_t[out] = g;
        beta_t[out] = sig;
    }
}

namespace mfq::cuda::kernels {

void clipped_silu_mul(const TensorView& gate, const TensorView& up,
    const TensorView& output, double limit, cudaStream_t stream) {
    if (!gate.is_contiguous() || !up.is_contiguous() || !output.is_contiguous() ||
        gate.numel() != up.numel() || gate.numel() != output.numel() ||
        gate.scalar_type != ScalarType::float16 || up.scalar_type != ScalarType::float16 ||
        output.scalar_type != ScalarType::float16 || !(limit > 0))
        throw std::invalid_argument("native clipped_silu_mul tensor mismatch");
    if (gate.numel()) clipped_silu_mul_kernel<<<unsigned((gate.numel()+255)/256),256,0,stream>>>(
        gate.data_as<half>(), up.data_as<half>(), output.data_as<half>(), gate.numel(), float(limit));
}

void silu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream) {
    if (!gate.is_contiguous() || !up.is_contiguous() || !output.is_contiguous() ||
        gate.numel() != up.numel() || gate.numel() != output.numel() ||
        gate.scalar_type != up.scalar_type || gate.scalar_type != output.scalar_type) {
        throw std::invalid_argument("native silu_mul tensor mismatch");
    }
    const auto elements = static_cast<std::size_t>(gate.numel());
    if (elements == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>(std::min<std::size_t>(
        4096, (elements + block - 1) / block));
    if (gate.scalar_type == ScalarType::float32) {
        silu_mul_f32_kernel<<<grid, block, 0, stream>>>(
            gate.data_as<float>(), up.data_as<float>(), output.data_as<float>(), elements);
    } else if (gate.scalar_type == ScalarType::float16) {
        // Half2 reduces work for large tensors, but scalar launch geometry is
        // faster for the small decode-sized cases.
        constexpr std::size_t vector_threshold = 1U << 18;
        const auto gate_address = reinterpret_cast<std::uintptr_t>(gate.data);
        const auto up_address = reinterpret_cast<std::uintptr_t>(up.data);
        const auto output_address = reinterpret_cast<std::uintptr_t>(output.data);
        const bool half2_aligned =
            ((gate_address | up_address | output_address) & (alignof(half2) - 1)) == 0;
        if (elements >= vector_threshold && half2_aligned) {
            const auto pairs = elements / 2;
            const int vector_grid = static_cast<int>(std::min<std::size_t>(
                4096, (pairs + block - 1) / block));
            silu_mul_f16x2_kernel<<<vector_grid, block, 0, stream>>>(
                reinterpret_cast<const half2*>(gate.data_as<half>()),
                reinterpret_cast<const half2*>(up.data_as<half>()),
                reinterpret_cast<half2*>(output.data_as<half>()), elements);
        } else {
            silu_mul_f16_kernel<<<grid, block, 0, stream>>>(
                gate.data_as<half>(), up.data_as<half>(),
                output.data_as<half>(), elements);
        }
    } else if (gate.scalar_type == ScalarType::bfloat16) {
        silu_mul_bf16_kernel<<<grid, block, 0, stream>>>(
            gate.data_as<__nv_bfloat16>(), up.data_as<__nv_bfloat16>(),
            output.data_as<__nv_bfloat16>(), elements);
    } else {
        throw std::invalid_argument("native silu_mul dtype must be f32, f16, or bf16");
    }
}

void gelu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream) {
    if (!gate.is_contiguous() || !up.is_contiguous() || !output.is_contiguous() ||
        gate.numel() != up.numel() || gate.numel() != output.numel() ||
        gate.scalar_type != up.scalar_type || gate.scalar_type != output.scalar_type) {
        throw std::invalid_argument("native gelu_mul tensor mismatch");
    }
    const auto elements = static_cast<std::size_t>(gate.numel());
    if (elements == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>(std::min<std::size_t>(
        4096, (elements + block - 1) / block));
    if (gate.scalar_type == ScalarType::float32) {
        gelu_mul_kernel<float><<<grid, block, 0, stream>>>(
            gate.data_as<float>(), up.data_as<float>(), output.data_as<float>(), elements);
    } else if (gate.scalar_type == ScalarType::float16) {
        gelu_mul_kernel<half><<<grid, block, 0, stream>>>(
            gate.data_as<half>(), up.data_as<half>(), output.data_as<half>(), elements);
    } else {
        throw std::invalid_argument("native gelu_mul dtype must be f32 or f16");
    }
}

void linear_gate_beta(
    const TensorView& alpha,
    const TensorView& beta,
    const TensorView& dt_bias,
    const TensorView& a_log,
    const TensorView& gate_t,
    const TensorView& beta_t,
    cudaStream_t stream, bool stable_softplus) {
    if (alpha.rank != 3 || beta.rank != 3 || alpha.scalar_type != beta.scalar_type ||
        alpha.sizes != beta.sizes || dt_bias.scalar_type != ScalarType::float32 ||
        a_log.scalar_type != ScalarType::float32 ||
        gate_t.scalar_type != ScalarType::float32 || beta_t.scalar_type != ScalarType::float32) {
        throw std::invalid_argument("native linear_gate_beta tensor mismatch");
    }
    const auto batch = alpha.sizes[0];
    const auto tokens = alpha.sizes[1];
    const auto width = alpha.sizes[2];
    if (dt_bias.numel() != width || a_log.numel() != width ||
        gate_t.numel() != alpha.numel() || beta_t.numel() != alpha.numel()) {
        throw std::invalid_argument("native linear_gate_beta shape mismatch");
    }
    const auto elements = static_cast<std::size_t>(alpha.numel());
    if (elements == 0) return;
    constexpr int block = 256;
    const int grid = static_cast<int>(std::min<std::size_t>(
        4096, (elements + block - 1) / block));
    if (alpha.scalar_type == ScalarType::float32) {
        linear_gate_beta_kernel<float><<<grid, block, 0, stream>>>(
            alpha.data_as<float>(), beta.data_as<float>(),
            dt_bias.data_as<float>(), a_log.data_as<float>(),
            gate_t.data_as<float>(), beta_t.data_as<float>(),
            batch, tokens, width,
            alpha.strides[0], alpha.strides[1], alpha.strides[2],
            beta.strides[0], beta.strides[1], beta.strides[2], stable_softplus);
    } else if (alpha.scalar_type == ScalarType::float16) {
        linear_gate_beta_kernel<half><<<grid, block, 0, stream>>>(
            alpha.data_as<half>(), beta.data_as<half>(),
            dt_bias.data_as<float>(), a_log.data_as<float>(),
            gate_t.data_as<float>(), beta_t.data_as<float>(),
            batch, tokens, width,
            alpha.strides[0], alpha.strides[1], alpha.strides[2],
            beta.strides[0], beta.strides[1], beta.strides[2], stable_softplus);
    } else {
        throw std::invalid_argument("native linear_gate_beta dtype must be f32 or f16");
    }
}

}  // namespace mfq::cuda::kernels
