#include "nvq_group.h"
#include "packed_nint.h"

namespace mfq::cpu {
#if defined(MFQ_CPU_PACKED_AVX2)
namespace detail {
NvqGroupDot nvq_group_dot_avx2(int format) noexcept;
NvqRowsDot nvq_rows_dot_avx2(int format) noexcept;
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
    if (packed_nint_has_avx2()) return detail::nvq_rows_dot_avx2(format);
#endif
    return nullptr;
}
} // namespace mfq::cpu
