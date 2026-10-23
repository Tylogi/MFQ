#pragma once

#include <stdexcept>

// Physical allocation failure; execution can retry after reclaiming residency.
class MfqResourceExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace mfq::engine {

// Admission or reservation exceeds the Engine budget.
struct ResourceExhausted : std::runtime_error { using std::runtime_error::runtime_error; };

} // namespace mfq::engine
