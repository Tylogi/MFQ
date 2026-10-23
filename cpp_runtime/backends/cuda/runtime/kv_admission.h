#pragma once

#include <kv_admission.h>
#include <functional>

namespace mfq::cuda {

// Cache ownership stays in the engine; the scheduler sees only reclaimable
// device bytes and asks the owner to release them before allocating KV.
struct BatchMemoryProvider {
    std::function<mfq::engine::KvAdmissionBytes()> evictable;
    std::function<void(const mfq::engine::KvAdmissionBytes&)> reclaim;
};

} // namespace mfq::cuda
