#pragma once

namespace mfq {
// File hints follow the operation, not the reader's lifetime. Bulk prefill
// opts in on each read worker; latency-sensitive demand reads use memcpy.
class MappedPrefetchScope {
public:
    explicit MappedPrefetchScope(bool enabled, bool release_after_read=false);
    ~MappedPrefetchScope();
    MappedPrefetchScope(const MappedPrefetchScope&)=delete;
    MappedPrefetchScope& operator=(const MappedPrefetchScope&)=delete;
private:
    bool previous_;
    bool previous_release_;
};
}
