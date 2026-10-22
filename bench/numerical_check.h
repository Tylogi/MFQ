#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace mfq::bench {
inline float max_absolute_difference(
    const float* reference, const float* candidate, std::size_t size) {
    float maximum = 0.0f;
    for (std::size_t index = 0; index < size; ++index) {
        const float difference = std::abs(reference[index] - candidate[index]);
        if (!std::isfinite(reference[index]) || !std::isfinite(candidate[index]) ||
            !std::isfinite(difference))
            throw std::runtime_error("benchmark comparison contains non-finite values");
        maximum = std::max(maximum, difference);
    }
    return maximum;
}
} // namespace mfq::bench
