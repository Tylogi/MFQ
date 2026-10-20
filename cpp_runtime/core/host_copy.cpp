#include "mfq/host_copy.h"
#include "mfq/host_parallel.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace mfq {
namespace {
void copy_upload(void* target,const void* origin,std::size_t bytes) {
#if defined(__x86_64__) || defined(_M_X64)
    // These buffers are consumed by DMA, so retaining a medium-sized upload
    // in the CPU cache only adds write-allocate traffic. Larger copies already
    // use the platform memcpy's bulk path; preserve its wider-vector loop.
    if(bytes>=(256u<<10) && bytes<(2u<<20)) {
        auto* dst=static_cast<unsigned char*>(target);
        auto* src=static_cast<const unsigned char*>(origin);
        const auto head=std::min(bytes,(64-(reinterpret_cast<std::uintptr_t>(dst)&63))&63);
        if(head)std::memcpy(dst,src,head);
        dst+=head;src+=head;auto remaining=bytes-head;
        while(remaining>=128) {
            const auto a=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            const auto b=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+16));
            const auto c=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+32));
            const auto d=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+48));
            const auto e=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+64));
            const auto f=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+80));
            const auto g=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+96));
            const auto h=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src+112));
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst),a);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst+16),b);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst+32),c);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst+48),d);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst+64),e);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst+80),f);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst+96),g);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst+112),h);
            dst+=128;src+=128;remaining-=128;
        }
        if(remaining>=64) {
            for(int i=0;i<4;++i)_mm_stream_si128(reinterpret_cast<__m128i*>(dst+16*i),
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src+16*i)));
            dst+=64;src+=64;remaining-=64;
        }
        if(remaining)std::memcpy(dst,src,remaining);
        // Each participating worker must finish its own write-combining
        // stores before the caller can submit DMA from these destinations.
        _mm_sfence();
        static std::atomic<bool> reported{false};
        if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
            std::fprintf(stderr,"host_streaming_copy scope=first_medium_upload enabled=1 bytes=%zu\n",bytes);
        return;
    }
#endif
    if(bytes)std::memcpy(target,origin,bytes);
}
}
void copy_host_range(const HostCopy& copy) {
    if(copy.bytes && (!copy.destination || !copy.source))
        throw std::invalid_argument("host copy range has a null endpoint");
    copy_upload(copy.destination,copy.source,copy.bytes);
}
void copy_host_ranges(const std::vector<HostCopy>& copies) {
    std::size_t bytes=0;
    int large=0;
    for(const auto& copy:copies) {
        if(copy.bytes && (!copy.destination || !copy.source))
            throw std::invalid_argument("host copy range has a null endpoint");
        if(copy.bytes>std::numeric_limits<std::size_t>::max()-bytes)
            throw std::overflow_error("host copy batch size overflow");
        bytes+=copy.bytes;
        if(copy.bytes>=(1u<<20))large=std::min(4,large+1);
    }
    const auto copy=[&](std::int64_t begin,std::int64_t end) {
        for(auto i=begin;i<end;++i) {
            const auto& part=copies[std::size_t(i)];
            copy_upload(part.destination,part.source,part.bytes);
        }
    };
    const int threads=std::min(large,int(std::max(1u,std::thread::hardware_concurrency())));
    if(bytes<(4u<<20) || threads<2) {
        copy(0,std::int64_t(copies.size()));return;
    }
    static HostParallelPool pool({},false);
    static std::atomic<bool> reported{false};
    const bool observe=!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed);
    HostParallelProfile profile;
    pool.run(0,std::int64_t(copies.size()),1,threads,copy,observe?&profile:nullptr);
    if(observe)std::fprintf(stderr,
        "host_parallel_copy scope=first_large_batch enabled=1 ranges=%zu bytes=%zu participants=%d\n",
        copies.size(),bytes,profile.participants);
}
}
