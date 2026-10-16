#pragma once

// Shared by the NEPQ-A inference unit and the tool-only backward unit: the
// record-layout checks both sides run before launching, and the warp constant
// their launch geometry uses.

#include "csrc/backends/cuda/native/tensor/backend.h"

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;

void validate_residual(
    const mfq_tensor_backend::Tensor & dictionary,
    const mfq_tensor_backend::Tensor & first,
    const mfq_tensor_backend::Tensor & second,
    int64_t width,
    int64_t position_bits,
    int64_t block_vectors) {
    MFQ_RUNTIME_CHECK(
        dictionary.is_cuda() && dictionary.scalar_type() == mfq_tensor_backend::kFloat16 &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8,
        "NEPQ-A dictionary must be CUDA contiguous fp16 [1024,8]");
    MFQ_RUNTIME_CHECK(
        first.is_cuda() && first.scalar_type() == mfq_tensor_backend::kInt16 &&
        first.is_contiguous() && first.dim() == 2,
        "NEPQ-A first records must be CUDA contiguous int16 rank-2");
    MFQ_RUNTIME_CHECK(
        second.is_cuda() && second.scalar_type() == mfq_tensor_backend::kInt16 &&
        second.is_contiguous() && second.sizes() == first.sizes(),
        "NEPQ-A second records must be int16 and match the first-record layout");
    MFQ_RUNTIME_CHECK(
        dictionary.device() == first.device() &&
        dictionary.device() == second.device(),
        "NEPQ-A residual tensors must share one CUDA device");
    MFQ_RUNTIME_CHECK(width > 0 && width % 8 == 0, "NEPQ-A width must be divisible by 8");
    MFQ_RUNTIME_CHECK(
        position_bits >= 1 && position_bits <= 5 &&
        block_vectors >= 2 && block_vectors <= 32,
        "invalid NEPQ-A residual profile");
    const int64_t expected_blocks = (width / 8 + block_vectors - 1) / block_vectors;
    MFQ_RUNTIME_CHECK(first.size(1) == expected_blocks, "NEPQ-A block count mismatch");
}

}  // namespace
