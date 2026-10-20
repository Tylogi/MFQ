#pragma once
#include "native/tensor/backend.h"

struct NintWeight;
int nint_float_projection_resident_blocks(int device);

mfq_tensor_backend::Tensor try_gr_bf16_down_cuda(
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& weight,int64_t streams);
mfq_tensor_backend::Tensor try_gr_bf16_up_mix_cuda(
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized,
    const mfq_tensor_backend::Tensor& weight,int64_t streams);

// Direct packed affine projection, converting each activation to FP32 at load.
// No activation quantization, half output, or dense weight expansion.
mfq_tensor_backend::Tensor nint_float_projection_cuda(
    const NintWeight&, const mfq_tensor_backend::Tensor&,bool parallel_groups=true,bool native_input=true,bool fixed_groups=true);
mfq_tensor_backend::Tensor nint_float_projection_activation_cuda(
    const NintWeight&, const mfq_tensor_backend::Tensor&,int64_t streams,bool injection,
    bool parallel_groups=true,bool native_input=true,bool fixed_groups=true);
mfq_tensor_backend::Tensor nint_float_projection_mix_cuda(
    const NintWeight&, const mfq_tensor_backend::Tensor&,
    const mfq_tensor_backend::Tensor& normalized,int64_t streams,bool native_input=true,bool fixed_groups=true,bool compact=false);
