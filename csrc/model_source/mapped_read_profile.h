#pragma once
#include <array>
#include <cstdint>

namespace mfq {
// Process-local totals enabled by MFQ_PROFILE_SSD_EXPERTS: prefetch ns,
// memcpy ns, bytes copied, range reads, and batch prefetch calls.
std::array<std::uint64_t,5> mapped_tensor_read_profile();
}
