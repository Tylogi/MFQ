// Tensor adapter for the activation kernels: validation, allocation, view conversion
// and stream acquisition.  Compiled by the native ops target and by the Python
// extension, so these checks exist once while the kernel unit stays backend-free.

#include "kernels/activation_ops.h"
#include "kernels/attention/linear_ops.h"
#include "kernels/kernels.h"

#include <array>
#include <cstdint>
#include <vector>

mfq_tensor_backend::Tensor clipped_silu_mul_cuda(mfq_tensor_backend::Tensor gate,
    mfq_tensor_backend::Tensor up, double limit) {
    MFQ_RUNTIME_CHECK(gate.is_cuda() && up.device()==gate.device() &&
        gate.scalar_type()==mfq_tensor_backend::kFloat16 && up.scalar_type()==mfq_tensor_backend::kFloat16 &&
        gate.sizes()==up.sizes() && gate.is_contiguous() && up.is_contiguous() && limit>0,
        "clipped SwiGLU requires matching contiguous CUDA FP16 tensors and a positive limit");
    auto output=mfq_tensor_backend::empty_like(gate);
    const auto shape=gate.sizes().vec();
    const auto dtype=mfq::cuda::ScalarType::float16;
    const auto device=mfq::cuda::Device{mfq::cuda::DeviceType::cuda,gate.get_device()};
    mfq::cuda::kernels::clipped_silu_mul(
        mfq::cuda::make_contiguous_view(gate.data_ptr(),shape,dtype,device),
        mfq::cuda::make_contiguous_view(up.data_ptr(),shape,dtype,device),
        mfq::cuda::make_contiguous_view(output.data_ptr(),shape,dtype,device),
        limit,mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

mfq_tensor_backend::Tensor silu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up)
{
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous(), "silu_mul: gate must be cuda contiguous");
    MFQ_RUNTIME_CHECK(up.is_cuda() && up.is_contiguous(), "silu_mul: up must be cuda contiguous");
    MFQ_RUNTIME_CHECK(gate.sizes() == up.sizes(), "silu_mul: gate/up shapes must match");
    MFQ_RUNTIME_CHECK(gate.scalar_type() == up.scalar_type(), "silu_mul: gate/up dtype must match");
    MFQ_RUNTIME_CHECK(
        gate.scalar_type() == mfq_tensor_backend::kFloat32 ||
        gate.scalar_type() == mfq_tensor_backend::kFloat16 ||
        gate.scalar_type() == mfq_tensor_backend::kBFloat16,
        "silu_mul: dtype must be f32, f16, or bf16");

    auto out = mfq_tensor_backend::empty_like(gate);
    const auto dtype = gate.scalar_type() == mfq_tensor_backend::kFloat32
        ? mfq::cuda::ScalarType::float32
        : gate.scalar_type() == mfq_tensor_backend::kFloat16
            ? mfq::cuda::ScalarType::float16
            : mfq::cuda::ScalarType::bfloat16;
    const auto shape = gate.sizes().vec();
    mfq::cuda::kernels::silu_mul(
        mfq::cuda::make_contiguous_view(
            gate.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, gate.get_device()}),
        mfq::cuda::make_contiguous_view(
            up.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, up.get_device()}),
        mfq::cuda::make_contiguous_view(
            out.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, out.get_device()}),
        mfq_current_cuda_stream());
    return out;
}


mfq_tensor_backend::Tensor gelu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up)
{
    MFQ_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous(), "gelu_mul: gate must be cuda contiguous");
    MFQ_RUNTIME_CHECK(up.is_cuda() && up.is_contiguous(), "gelu_mul: up must be cuda contiguous");
    MFQ_RUNTIME_CHECK(gate.sizes() == up.sizes(), "gelu_mul: gate/up shapes must match");
    MFQ_RUNTIME_CHECK(gate.scalar_type() == up.scalar_type(), "gelu_mul: gate/up dtype must match");
    MFQ_RUNTIME_CHECK(gate.scalar_type() == mfq_tensor_backend::kFloat32 || gate.scalar_type() == mfq_tensor_backend::kFloat16,
                "gelu_mul: dtype must be f32 or f16");

    auto out = mfq_tensor_backend::empty_like(gate);
    const auto dtype = gate.scalar_type() == mfq_tensor_backend::kFloat32
        ? mfq::cuda::ScalarType::float32
        : mfq::cuda::ScalarType::float16;
    const auto shape = gate.sizes().vec();
    mfq::cuda::kernels::gelu_mul(
        mfq::cuda::make_contiguous_view(
            gate.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, gate.get_device()}),
        mfq::cuda::make_contiguous_view(
            up.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, up.get_device()}),
        mfq::cuda::make_contiguous_view(
            out.data_ptr(), shape, dtype,
            mfq::cuda::Device{mfq::cuda::DeviceType::cuda, out.get_device()}),
        mfq_current_cuda_stream());
    return out;
}


std::vector<mfq_tensor_backend::Tensor> linear_gate_beta_cuda(
    mfq_tensor_backend::Tensor alpha, mfq_tensor_backend::Tensor beta, mfq_tensor_backend::Tensor dt_bias, mfq_tensor_backend::Tensor a_log,
    bool stable_softplus)
{
    MFQ_RUNTIME_CHECK(alpha.is_cuda() && beta.is_cuda(), "linear_gate_beta: alpha/beta must be cuda tensors");
    MFQ_RUNTIME_CHECK(dt_bias.is_cuda() && a_log.is_cuda(), "linear_gate_beta: dt_bias/a_log must be cuda tensors");
    MFQ_RUNTIME_CHECK(alpha.dim() == 3 && beta.dim() == 3, "linear_gate_beta: alpha/beta must be [B,T,V]");
    MFQ_RUNTIME_CHECK(alpha.sizes() == beta.sizes(), "linear_gate_beta: alpha/beta shape mismatch");
    MFQ_RUNTIME_CHECK(alpha.scalar_type() == beta.scalar_type(), "linear_gate_beta: alpha/beta dtype mismatch");
    MFQ_RUNTIME_CHECK(alpha.scalar_type() == mfq_tensor_backend::kFloat32 || alpha.scalar_type() == mfq_tensor_backend::kFloat16,
                "linear_gate_beta: alpha/beta dtype must be f32 or f16");
    MFQ_RUNTIME_CHECK(dt_bias.is_contiguous() && a_log.is_contiguous(), "linear_gate_beta: dt_bias/a_log must be contiguous");
    MFQ_RUNTIME_CHECK(dt_bias.scalar_type() == mfq_tensor_backend::kFloat32 && a_log.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_gate_beta: dt_bias/a_log must be f32");
    int64_t B = alpha.size(0);
    int64_t T = alpha.size(1);
    int64_t V = alpha.size(2);
    MFQ_RUNTIME_CHECK(dt_bias.numel() == V && a_log.numel() == V, "linear_gate_beta: parameter size mismatch");

    auto opts = mfq_tensor_backend::TensorOptions().device(alpha.device()).dtype(mfq_tensor_backend::kFloat32);
    auto gate_t = mfq_tensor_backend::empty({B, V, T}, opts);
    auto beta_t = mfq_tensor_backend::empty({B, V, T}, opts);
    const auto dtype = alpha.scalar_type() == mfq_tensor_backend::kFloat32
        ? mfq::cuda::ScalarType::float32
        : mfq::cuda::ScalarType::float16;
    const auto device = mfq::cuda::Device{mfq::cuda::DeviceType::cuda, alpha.get_device()};
    mfq::cuda::TensorView alpha_view;
    alpha_view.data = alpha.data_ptr();
    alpha_view.rank = 3;
    alpha_view.scalar_type = dtype;
    alpha_view.device = device;
    mfq::cuda::TensorView beta_view = alpha_view;
    beta_view.data = beta.data_ptr();
    for (int index = 0; index < 3; ++index) {
        alpha_view.sizes[index] = alpha.size(index);
        alpha_view.strides[index] = alpha.stride(index);
        beta_view.sizes[index] = beta.size(index);
        beta_view.strides[index] = beta.stride(index);
    }
    const std::array<std::int64_t, 1> parameter_shape = {V};
    const std::array<std::int64_t, 3> output_shape = {B, V, T};
    mfq::cuda::kernels::linear_gate_beta(
        alpha_view,
        beta_view,
        mfq::cuda::make_contiguous_view(
            dt_bias.data_ptr(), parameter_shape, mfq::cuda::ScalarType::float32, device),
        mfq::cuda::make_contiguous_view(
            a_log.data_ptr(), parameter_shape, mfq::cuda::ScalarType::float32, device),
        mfq::cuda::make_contiguous_view(
            gate_t.data_ptr(), output_shape, mfq::cuda::ScalarType::float32, device),
        mfq::cuda::make_contiguous_view(
            beta_t.data_ptr(), output_shape, mfq::cuda::ScalarType::float32, device),
        mfq_current_cuda_stream(), stable_softplus);
    return {gate_t, beta_t};
}
