#pragma once
#include "file_range_reader.h"
#include "read_span.h"
#include <vector>

namespace mfq {
// Sector deduplication and range coalescing precede concurrent file submission.
class AsyncRangeReader {
public:
    explicit AsyncRangeReader(const std::filesystem::path& path);
    ~AsyncRangeReader();
    void read(const std::vector<ReadSpan>& spans) const;
    FileReadStats stats() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
