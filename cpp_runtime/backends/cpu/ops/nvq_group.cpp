#include "nvq_group.h"
#include "packed_nint.h"
#include <cstdlib>
#include <cstring>

namespace mfq::cpu {
#if defined(MFQ_CPU_PACKED_AVX2)
namespace detail {
NvqGroupDot nvq_group_dot_avx2(int format) noexcept;
NvqRowsDot nvq_rows_dot_avx2(int format) noexcept;
NvqRowsDot nvq_rows_dot_single_avx2(int format) noexcept;
NvqRowsDot nvq_dense_rows_dot_avx2(int format) noexcept;
}
#endif
NvqGroupDot nvq_group_dot_kernel(int format) noexcept {
#if defined(MFQ_CPU_PACKED_AVX2)
    if (packed_nint_has_avx2()) return detail::nvq_group_dot_avx2(format);
#endif
    return nullptr;
}
NvqRowsDot nvq_rows_dot_kernel(int format) noexcept {
#if defined(MFQ_CPU_PACKED_AVX2)
    if (packed_nint_has_avx2()) {
        const auto* single=std::getenv("MFQ_CPU_NVQ_SINGLE_ROW");
        // The compressed single-row path preserves the original FP32 bits.
        // NPQ0 retains its measured faster implementation; other batches are
        // handled by the existing matrix kernel inside the single-row entry.
        if(format!=7 && (!single || std::strcmp(single,"1")==0))return detail::nvq_rows_dot_single_avx2(format);
        return detail::nvq_rows_dot_avx2(format);
    }
#endif
    return nullptr;
}
NvqRowsDot nvq_dense_rows_dot_kernel(int format) noexcept {
#if defined(MFQ_CPU_PACKED_AVX2)
    if (packed_nint_has_avx2()) return detail::nvq_dense_rows_dot_avx2(format);
#endif
    return nullptr;
}
} // namespace mfq::cpu
