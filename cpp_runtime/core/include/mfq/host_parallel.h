#pragma once

#include <cstdint>
#include <functional>

namespace mfq {

// Persistent host workers. Concurrent callers are serialized; nested work
// executes on its calling worker so it cannot wait for the same pool.
void host_parallel_for(std::int64_t begin, std::int64_t end,
    std::int64_t grain, int threads,
    const std::function<void(std::int64_t, std::int64_t)>& function);

} // namespace mfq
