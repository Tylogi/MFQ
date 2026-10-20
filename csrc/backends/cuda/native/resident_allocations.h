#pragma once
#include <cstddef>
#include <optional>

namespace mfq::cuda {
// Remaining WDDM shared-memory budget; absent on platforms without WDDM.
std::optional<std::size_t> shared_memory_available(int device);
// Keep long-lived request state out of the temporary async allocator pool.
class ResidentAllocationScope {
public:
    explicit ResidentAllocationScope(bool enabled=true) noexcept;
    ~ResidentAllocationScope() noexcept { finish(); }
    ResidentAllocationScope(const ResidentAllocationScope&)=delete;
    ResidentAllocationScope& operator=(const ResidentAllocationScope&)=delete;
    void finish() noexcept;
private:
    bool previous_,active_=true;
};
}
