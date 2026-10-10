#include "host_memory_copy.h"
#include "packed_nint.h"
#include <cstring>

namespace mfq::cpu {
namespace detail {
void host_memory_copy_avx2(void*,const void*,std::size_t) noexcept;
void host_memory_copy_finish_avx2() noexcept;
}
namespace {
void copy_cached(void* destination,const void* source,std::size_t bytes) noexcept {
    if(bytes)std::memcpy(destination,source,bytes);
}
}
HostMemoryCopyBatch::HostMemoryCopyBatch(bool streaming) noexcept:copy_(copy_cached) {
#if defined(MFQ_CPU_PACKED_AVX2)
    if(streaming && packed_nint_has_avx2()) {
        copy_=detail::host_memory_copy_avx2;
        finish_=detail::host_memory_copy_finish_avx2;
    }
#else
    (void)streaming;
#endif
}
} // namespace mfq::cpu
