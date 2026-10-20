#pragma once
#include <cstddef>
#include <vector>

namespace mfq {
struct HostCopy {
    void* destination;
    const void* source;
    std::size_t bytes;
};

// Synchronous completion; destinations must be mutually disjoint. Several
// large ranges can share persistent CPU workers before a DMA submission.
void copy_host_ranges(const std::vector<HostCopy>& copies);
}
