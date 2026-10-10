#include "quant_dot.h"
#include <immintrin.h>

namespace mfq::cpu::detail {
float scaled_i8_dot_avx2(const std::int8_t* weights, const float* input, int count, float scale) {
    __m256 sum=_mm256_setzero_ps();
    const auto multiplier=_mm256_set1_ps(scale);
    int i=0;
    for (; i+8<=count; i+=8) {
        const auto values=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights+i))));
        const auto decoded=_mm256_mul_ps(multiplier,values);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(decoded,_mm256_loadu_ps(input+i)));
    }
    auto reduced=_mm_add_ps(_mm256_castps256_ps128(sum),_mm256_extractf128_ps(sum,1));
    reduced=_mm_add_ps(reduced,_mm_movehl_ps(reduced,reduced));
    reduced=_mm_add_ss(reduced,_mm_shuffle_ps(reduced,reduced,1));
    float result=_mm_cvtss_f32(reduced);
    for (; i<count; ++i) result += (scale*weights[i])*input[i];
    return result;
}
} // namespace mfq::cpu::detail
