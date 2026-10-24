#pragma once

#include "mfq_tensor_backend.h"

namespace mfq::cuda {
// Shared QSA prefill scores with FP32 head reduction kept on chip.
Tensor qsa_block_scores(const Tensor& query, const Tensor& pooled_keys);
} // namespace mfq::cuda
