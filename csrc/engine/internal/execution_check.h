#pragma once

#include <stdexcept>

namespace mfq::engine {
inline void require_execution(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace mfq::engine
