#include "mfq/host_copy.h"
#include "mfq/host_parallel.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace mfq {
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
            if(part.bytes)std::memcpy(part.destination,part.source,part.bytes);
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
