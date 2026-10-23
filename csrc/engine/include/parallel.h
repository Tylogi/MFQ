#pragma once
#include <cstdint>

namespace mfq {
struct TensorParallelSlice {
    int device = 0;
    int64_t begin = 0;
    int64_t end = 0;

    int64_t size() const {
        return end - begin;
    }
};

} // namespace mfq
