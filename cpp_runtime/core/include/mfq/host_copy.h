#pragma once
#include <cstddef>
#include <vector>

namespace mfq {
struct HostCopy {
    void* destination;
    const void* source;
    std::size_t bytes;
};

// Copy on the calling thread, without serializing independent I/O workers.
// Uses the same DMA-friendly stores as the batch helper.
void copy_host_range(const HostCopy& copy);

// Synchronous completion; destinations must be mutually disjoint. Several
// large ranges can share persistent CPU workers before a DMA submission.
void copy_host_ranges(const std::vector<HostCopy>& copies);
}
