#pragma once

namespace mfq::cuda {
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
