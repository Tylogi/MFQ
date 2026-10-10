#include "mfq/async_range_reader.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mfq {
struct AsyncRangeReader::Impl {
    FileRangeReader file;
    std::uint64_t size;
    mutable std::atomic<std::uint64_t> logical{0}, staging{0}, peak{0};
    explicit Impl(const std::filesystem::path& path)
        : file(path, FileReadMode::Direct), size(std::filesystem::file_size(path)) {}
};
AsyncRangeReader::AsyncRangeReader(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}
AsyncRangeReader::~AsyncRangeReader() = default;

void AsyncRangeReader::read(const std::vector<ReadSpan>& spans) const {
    struct Interval { std::uint64_t first, last; };
    auto& state = *impl_;
    const auto alignment = state.file.alignment();
    std::vector<Interval> intervals;
    for (const auto& span : spans) {
        if (span.offset > state.size || span.size > state.size - span.offset)
            throw std::out_of_range("batched range exceeds model shard");
        if (!span.size) continue;
        if (!span.destination) throw std::invalid_argument("batched range destination is null");
        const auto last = span.offset + span.size;
        const auto remainder = (alignment - last % alignment) % alignment;
        intervals.push_back({span.offset / alignment * alignment,
            last + std::min<std::uint64_t>(remainder, state.size - last)});
    }
    if (intervals.empty()) return;
    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        return a.first < b.first;
    });
    std::vector<Interval> merged;
    for (const auto interval : intervals) {
        if (!merged.empty() && interval.first <= merged.back().last)
            merged.back().last = std::max(merged.back().last, interval.last);
        else merged.push_back(interval);
    }
    std::uint64_t bytes = 0;
    for (const auto interval : merged) {
        const auto count = interval.last - interval.first;
        if (count > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("batched range storage exceeds address space");
        bytes += count;
    }
    std::vector<std::byte> storage(static_cast<std::size_t>(bytes));
    struct StorageCharge {
        Impl& owner;
        std::uint64_t bytes;
        ~StorageCharge() { owner.staging.fetch_sub(bytes, std::memory_order_relaxed); }
    } charge{state, bytes};
    const auto current = state.staging.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto peak = state.peak.load(std::memory_order_relaxed);
    while (peak < current && !state.peak.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {}
    std::vector<ReadSpan> reads;
    std::size_t offset = 0;
    for (const auto interval : merged) {
        const auto count = static_cast<std::size_t>(interval.last - interval.first);
        reads.push_back({interval.first, storage.data() + offset, count});
        offset += count;
    }
    state.file.read_batch(reads);
    for (const auto& span : spans) {
        if (!span.size) continue;
        const auto found = std::upper_bound(merged.begin(), merged.end(), span.offset,
            [](std::uint64_t at, const Interval& interval) { return at < interval.first; });
        const auto index = static_cast<std::size_t>(found - merged.begin() - 1);
        std::memcpy(span.destination, reads[index].destination + span.offset - merged[index].first, span.size);
        state.logical.fetch_add(span.size, std::memory_order_relaxed);
    }
}
FileReadStats AsyncRangeReader::stats() const noexcept {
    auto result = impl_->file.stats();
    result.logical_bytes = impl_->logical.load(std::memory_order_relaxed);
    result.staging_bytes += impl_->staging.load(std::memory_order_relaxed);
    result.staging_peak_bytes += impl_->peak.load(std::memory_order_relaxed);
    return result;
}
}
