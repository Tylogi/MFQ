#pragma once
#include "mfq_tensor_backend.h"

struct NintWeight;
int nint_float_projection_resident_blocks(int device);

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
