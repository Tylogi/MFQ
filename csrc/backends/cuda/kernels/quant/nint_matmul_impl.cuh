#pragma once

// Shared by the NINT8-0 inference unit and the tool-only backward unit: the
// symmetric-quantization metadata both sides check before launching.

#include "native/tensor/backend.h"

namespace {

void validate_nint8_zero(
        const mfq_tensor_backend::Tensor & quantized,
        const mfq_tensor_backend::Tensor & scale) {
    MFQ_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == mfq_tensor_backend::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 scale must be contiguous CUDA fp16 [N,G]");
    MFQ_RUNTIME_CHECK(
        quantized.device() == scale.device(),
        "NINT8-0 tensors must share one CUDA device");
}

}  // namespace
