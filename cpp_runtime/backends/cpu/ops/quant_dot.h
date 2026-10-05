#pragma once

#include <cstdint>

namespace mfq::cpu {
// Operates on one decoded vector group, with the original FP32 activation.
// Applying scale before the dot matches FP32 weight reconstruction.
using ScaledI8Dot = float (*)(const std::int8_t*, const float*, int count, float scale);
ScaledI8Dot scaled_i8_dot_kernel();
float scaled_i8_dot_scalar(const std::int8_t*, const float*, int count, float scale);
namespace detail {
float scaled_i8_dot_avx2(const std::int8_t*, const float*, int count, float scale);
}
} // namespace mfq::cpu
