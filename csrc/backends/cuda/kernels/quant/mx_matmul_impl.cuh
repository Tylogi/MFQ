#pragma once

// Shared by the MXFP8/MXFP4 inference unit and the tool-only backward unit:
// the format decoders and the weight geometry checks both sides run.

#include "native/tensor/backend.h"

#include <cstdint>
#include <limits>

namespace {

__device__ __forceinline__ float decode_e8m0(std::uint8_t raw) {
    if (raw == 255u) {
        return __int_as_float(0x7fffffff);
    }
    // E8M0 stores the unbiased power-of-two exponent with the same bias
    // as FP32.  Constructing the FP32 exponent is exact and avoids an
    // expensive per-block ldexpf call.  raw=0 denotes 2^-127, the largest
    // FP32 subnormal power of two.
    return raw == 0u
        ? __int_as_float(0x00400000)
        : __int_as_float(unsigned(raw) << 23u);
}

__device__ __forceinline__ float decode_e4m3fn(std::uint8_t raw) {
    const unsigned exponent = (unsigned(raw) >> 3u) & 15u;
    const unsigned mantissa = unsigned(raw) & 7u;
    if (exponent == 15u && mantissa == 7u) {
        return __int_as_float(0x7fffffff);
    }
    // Normal E4M3FN numbers map exactly to FP32 by rebiasing the exponent
    // and shifting the three mantissa bits.  Subnormals are integer
    // multiples of 2^-9.  This removes one transcendental instruction per
    // weight while preserving every represented value exactly.
    const float magnitude = exponent == 0u
        ? float(mantissa) * 0.001953125f
        : __int_as_float(
            ((exponent + 120u) << 23u) |
            (mantissa << 20u));
    return (raw & 128u) == 0u ? magnitude : -magnitude;
}

__device__ __forceinline__ float decode_mxfp4_e2m1(std::uint8_t code) {
    constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float value = magnitude[code & 7u];
    return (code & 8u) == 0u ? value : -value;
}

void validate_mxfp8(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales) {
    MFQ_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP8 values and scales must be CUDA tensors");
    MFQ_RUNTIME_CHECK(values.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    scales.scalar_type() == mfq_tensor_backend::kUInt8,
                "MXFP8 values and scales must be uint8");
    MFQ_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP8 values and scales must be contiguous");
    MFQ_RUNTIME_CHECK(values.dim() == 2 && scales.dim() == 2,
                "MXFP8 values and scales must be rank-2");
    MFQ_RUNTIME_CHECK(values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() &&
                    values.size(1) % 32 == 0,
                "MXFP8 input width must be divisible by 32");
    const bool block128=values.size(1)%128==0 && scales.size(0)==(values.size(0)+127)/128 && scales.size(1)==values.size(1)/128;
    const bool block32=scales.size(0)==(values.size(0)+31)/32 && scales.size(1)==values.size(1)/32;
    const bool row32=scales.size(0)==values.size(0) && scales.size(1)==values.size(1)/32;
    MFQ_RUNTIME_CHECK(block128 || block32 || row32,"MXFP8 scale geometry mismatch");
}

void validate_mxfp4_dense(
        const mfq_tensor_backend::Tensor & values,
        const mfq_tensor_backend::Tensor & scales) {
    MFQ_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP4 values and scales must be CUDA tensors");
    MFQ_RUNTIME_CHECK(values.scalar_type() == mfq_tensor_backend::kUInt8 &&
                    scales.scalar_type() == mfq_tensor_backend::kUInt8,
                "MXFP4 values and scales must be uint8");
    MFQ_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP4 values and scales must be contiguous");
    MFQ_RUNTIME_CHECK(values.dim() == 2 && scales.dim() == 2 &&
                    values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() / 2 &&
                    values.size(1) % 16 == 0 &&
                    scales.size(0) == values.size(0) &&
                    scales.size(1) == values.size(1) / 16,
                "MXFP4 weight geometry mismatch");
}

}  // namespace
