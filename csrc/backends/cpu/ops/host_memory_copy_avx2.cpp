#include "host_memory_copy.h"
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace mfq::cpu::detail {
void host_memory_copy_avx2(void* destination_ptr,const void* source_ptr,size_t count) noexcept {
    auto* destination=static_cast<uint8_t*>(destination_ptr);
    const auto* source=static_cast<const uint8_t*>(source_ptr);
    if(!count)return;
    if(count<4096 || count>=(2u<<20)) {std::memcpy(destination,source,count);return;}
    const size_t prefix=std::min(count,size_t((-reinterpret_cast<uintptr_t>(destination))&63u));
    std::memcpy(destination,source,prefix);destination+=prefix;source+=prefix;count-=prefix;
    size_t at=0;
    for(;at+256<=count;at+=256) {
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+32));
        const auto c=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+64));
        const auto d=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+96));
        const auto e=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+128));
        const auto f=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+160));
        const auto g=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+192));
        const auto h=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+224));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at),a);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+32),b);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+64),c);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+96),d);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+128),e);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+160),f);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+192),g);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+224),h);
    }
    for(;at+64<=count;at+=64) {
        const auto first=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at));
        const auto second=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+at+32));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at),first);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(destination+at+32),second);
    }
    std::memcpy(destination+at,source+at,count-at);
}
void host_memory_copy_finish_avx2() noexcept {_mm_sfence();}
} // namespace mfq::cpu::detail
