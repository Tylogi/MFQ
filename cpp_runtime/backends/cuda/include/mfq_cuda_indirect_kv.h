#pragma once

#include <cstdint>

namespace mfq::cuda {

// A view of BatchDecodeInputs, never a second request-pointer allocation.
// Each field has [row, {address, sequence capacity}] entries.
struct IndirectKvView {
    const int64_t* states = nullptr;
    const int64_t* enabled = nullptr;
    int rows = 0;
    int field = 0;
#ifdef __CUDACC__
    __device__ int64_t capacity(int row, int offset = 0) const {
        return states[2 * (int64_t(field + offset) * rows + row) + 1];
    }
    __device__ void* address(int row, int offset = 0) const {
        return reinterpret_cast<void*>(states[2 * (int64_t(field + offset) * rows + row)]);
    }
    __device__ bool active(int row) const {
        return enabled[row] && address(row);
    }
#endif
};

} // namespace mfq::cuda
