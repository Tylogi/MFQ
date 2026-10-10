#pragma once
#include "mfq_tensor_backend.h"
#include <optional>

// One BF16 scalar weight row, Half input, 1..8 tokens. Unsupported views
// return nullopt so the caller keeps its ordinary dense projection.
std::optional<mfq_tensor_backend::Tensor> try_shared_gate_sigmoid_cuda(
    const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& weight);
