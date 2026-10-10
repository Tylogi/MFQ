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
        // NPQ0 retains the original implementation after standalone hot and
        // rotating-weight measurements showed a single-row regression.
        if(format!=7 && single && std::strcmp(single,"1")==0)return detail::nvq_rows_dot_single_avx2(format);
        return detail::nvq_rows_dot_avx2(format);
    }
#endif
    return nullptr;
}
} // namespace mfq::cpu
