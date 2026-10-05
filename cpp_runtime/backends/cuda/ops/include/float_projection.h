#pragma once
#include "mfq_tensor_backend.h"

struct NintWeight;

// Direct packed affine projection, with FP32 activations and accumulation.
// No activation quantization, half output, or dense weight expansion.
mfq_tensor_backend::Tensor nint_float_projection_cuda(
    const NintWeight&, const mfq_tensor_backend::Tensor&);
