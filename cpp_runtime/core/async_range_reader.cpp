#include "mfq/async_range_reader.h"
#include <atomic>

namespace mfq {
struct AsyncRangeReader::Impl {
    FileRangeReader file;
    mutable std::atomic<std::uint64_t> logical{0};
    explicit Impl(const std::filesystem::path& path,FileReadMode mode):file(path,mode) {}
};
AsyncRangeReader::AsyncRangeReader(const std::filesystem::path& path,FileReadMode mode)
    :impl_(std::make_unique<Impl>(path,mode)) {}
AsyncRangeReader::~AsyncRangeReader()=default;

void AsyncRangeReader::read(const std::vector<ReadSpan>& spans) const {
    // Coalescing uses the file reader's aligned pool. Its direct-I/O requests
    // can fill that storage in place, followed by one scatter to the fields.
    impl_->file.read_coalesced(spans);
    std::uint64_t bytes=0;
    for(const auto& span:spans)bytes+=span.size;
    impl_->logical.fetch_add(bytes,std::memory_order_relaxed);
}
FileReadStats AsyncRangeReader::stats() const noexcept {
    auto result=impl_->file.stats();
    result.logical_bytes=impl_->logical.load(std::memory_order_relaxed);
    return result;
}
}
