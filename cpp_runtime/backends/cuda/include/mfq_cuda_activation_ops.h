#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

mfq_tensor_backend::Tensor silu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up);
mfq_tensor_backend::Tensor gelu_mul_cuda(mfq_tensor_backend::Tensor gate, mfq_tensor_backend::Tensor up);
