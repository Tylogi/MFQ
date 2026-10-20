#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>
#include "read_span.h"

namespace mfq {

enum class FileReadMode { Buffered, Direct };
FileReadMode model_file_read_mode();
const char* file_read_mode_name(FileReadMode mode) noexcept;

struct FileReadStats {
    FileReadMode mode = FileReadMode::Buffered;
    std::uint64_t files = 0;
    std::uint64_t calls = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_bytes = 0;
    std::uint64_t errors = 0;
    std::uint64_t staging_bytes = 0;
    // Summing this across files is a conservative bound, not a simultaneous peak.
    std::uint64_t staging_peak_bytes = 0;
    std::uint64_t read_nanoseconds = 0;
    std::uint64_t staging_cached_bytes = 0;
    std::uint64_t staging_allocations = 0;
};

// Owning positional reads. Direct mode never silently reverts to buffered I/O.
// Arbitrary byte ranges use aligned temporary storage. Direct readers
// retain at most 8 MiB of idle arenas, released with the reader. Only storage is
// reused: each request still reads the file, without a shared file-position lock.
// Linux batches use a per-worker native AIO queue when supported by the file
// system; otherwise the same Direct descriptor is read on that SSD worker.
class FileRangeReader final {
public:
    FileRangeReader(const std::filesystem::path& path, FileReadMode mode);
    ~FileRangeReader();
    FileRangeReader(const FileRangeReader&) = delete;
    FileRangeReader& operator=(const FileRangeReader&) = delete;
    void read(std::uint64_t offset, std::byte* destination, std::size_t size) const;
    // Submit independent ranges together; all requests finish before returning.
    void read_batch(const std::vector<ReadSpan>& spans) const;
    FileReadStats stats() const noexcept;
    std::size_t alignment() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq
