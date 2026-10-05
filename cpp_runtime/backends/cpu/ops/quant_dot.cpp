#include "quant_dot.h"
#include "packed_nint.h"

namespace mfq::cpu {
float scaled_i8_dot_scalar(const std::int8_t* weights, const float* input, int count, float scale) {
    float result=0;
    for (int i=0; i<count; ++i) {
        const float decoded=scale*weights[i];
        result += decoded*input[i];
    }
    return result;
}

ScaledI8Dot scaled_i8_dot_kernel() {
#ifdef MFQ_CPU_PACKED_AVX2
    if (packed_nint_has_avx2()) return detail::scaled_i8_dot_avx2;
#endif
    return scaled_i8_dot_scalar;
}
} // namespace mfq::cpu
