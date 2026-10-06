#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace mfq::cuda {

struct CudaResourcePlan {
    std::vector<std::size_t> request_bytes;
    std::size_t snapshot_bytes = 0;

    explicit CudaResourcePlan(std::size_t devices) : request_bytes(devices) {}
    void add(int device, std::initializer_list<std::size_t> factors);
};

} // namespace mfq::cuda
