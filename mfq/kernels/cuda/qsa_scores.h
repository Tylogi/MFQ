#pragma once

#include "mfq_tensor_backend.h"

namespace mfq_qsa {
// FP32 scores [B,T,P]: sum_h relu(dot(query[b,t,h], keys[b,p])) / sqrt(D).
// Strided F32/F16/BF16 inputs retain their precision; unsupported tiles use GEMM.
mfq_tensor_backend::Tensor block_scores(const mfq_tensor_backend::Tensor& query,
    const mfq_tensor_backend::Tensor& keys);
} // namespace mfq_qsa
