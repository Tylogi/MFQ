#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

mfq_tensor_backend::Tensor silu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up);
mfq_tensor_backend::Tensor gelu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up);
mfq_tensor_backend::Tensor clipped_silu_mul_cuda(mfq_tensor_backend::Tensor gate,
    mfq_tensor_backend::Tensor up,double limit);
// relu(x)^2 for f32/f16/bf16 input; the output keeps the input dtype.
mfq_tensor_backend::Tensor relu_squared_cuda(mfq_tensor_backend::Tensor input);
