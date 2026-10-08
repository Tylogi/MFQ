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
};

// Owning positional reads. Direct mode never silently reverts to buffered I/O.
// Arbitrary byte ranges use aligned temporary storage, released after the read;
// no payload cache or file-position lock is shared between concurrent requests.
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
