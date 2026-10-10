#include "mfq/file_range_reader.h"
#include "mfq/host_parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mfq {

FileReadMode model_file_read_mode() {
    const char* value = std::getenv("MFQ_MODEL_FILE_IO");
    if (!value || !*value || std::strcmp(value, "buffered") == 0) return FileReadMode::Buffered;
    if (std::strcmp(value, "direct") == 0) return FileReadMode::Direct;
    throw std::invalid_argument("MFQ_MODEL_FILE_IO must be buffered or direct");
}

const char* file_read_mode_name(FileReadMode mode) noexcept {
    return mode == FileReadMode::Direct ? "direct" : "buffered";
}

struct FileRangeReader::Impl {
    FileReadMode mode;
    std::size_t alignment = 1;
    std::uint64_t size = 0;
    mutable std::atomic<std::uint64_t> calls{0}, logical{0}, physical{0}, errors{0}, staging{0}, peak{0}, nanos{0};
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~Impl() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#else
    int handle = -1;
    ~Impl() { if (handle >= 0) ::close(handle); }
#endif
    explicit Impl(FileReadMode value) : mode(value) {}
    void acquire(std::size_t bytes) const {
        const auto current = staging.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        auto previous = peak.load(std::memory_order_relaxed);
        while (previous < current && !peak.compare_exchange_weak(previous, current, std::memory_order_relaxed)) {}
    }
    std::size_t transfer(std::uint64_t offset, void* buffer, std::size_t bytes) const {
#if defined(_WIN32)
        struct Event {
            HANDLE value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            ~Event() { if (value) CloseHandle(value); }
        } event;
        if (!event.value) throw std::system_error(GetLastError(), std::system_category(), "create file-read event");
        OVERLAPPED request{};
        request.Offset = static_cast<DWORD>(offset);
        request.OffsetHigh = static_cast<DWORD>(offset >> 32);
        request.hEvent = event.value;
        DWORD actual = 0;
        if (!ReadFile(handle, buffer, static_cast<DWORD>(bytes), nullptr, &request) && GetLastError() != ERROR_IO_PENDING)
            throw std::system_error(GetLastError(), std::system_category(), "read model file");
        if (!GetOverlappedResult(handle, &request, &actual, TRUE))
            throw std::system_error(GetLastError(), std::system_category(), "complete model-file read");
        return actual;
#else
        ssize_t actual;
        do { actual = ::pread(handle, buffer, bytes, static_cast<off_t>(offset)); } while (actual < 0 && errno == EINTR);
        if (actual < 0) throw std::system_error(errno, std::generic_category(), "read model file");
        return static_cast<std::size_t>(actual);
#endif
    }
};

FileRangeReader::FileRangeReader(const std::filesystem::path& path, FileReadMode mode)
    : impl_(std::make_unique<Impl>(mode)) {
#if defined(_WIN32)
    const DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS |
        (mode == FileReadMode::Direct ? FILE_FLAG_NO_BUFFERING : 0);
    impl_->handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
    if (impl_->handle == INVALID_HANDLE_VALUE)
        throw std::system_error(GetLastError(), std::system_category(), "open model file");
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(impl_->handle, &size) || size.QuadPart < 0)
        throw std::system_error(GetLastError(), std::system_category(), "size model file");
    impl_->size = static_cast<std::uint64_t>(size.QuadPart);
    if (mode == FileReadMode::Direct) {
        FILE_STORAGE_INFO storage{};
        if (!GetFileInformationByHandleEx(impl_->handle, FileStorageInfo, &storage, sizeof(storage)))
            throw std::system_error(GetLastError(), std::system_category(), "query model-file sector alignment");
        impl_->alignment = std::max(storage.LogicalBytesPerSector, storage.PhysicalBytesPerSectorForPerformance);
    }
#else
    int flags = O_RDONLY;
    if (mode == FileReadMode::Direct) {
#if defined(O_DIRECT)
        flags |= O_DIRECT;
#elif !defined(F_NOCACHE)
        throw std::runtime_error("direct model-file I/O is unsupported on this platform");
#endif
    }
    impl_->handle = ::open(path.c_str(), flags);
    if (impl_->handle < 0) throw std::system_error(errno, std::generic_category(), "open model file");
    struct stat info{};
    if (::fstat(impl_->handle, &info) || info.st_size < 0)
        throw std::system_error(errno, std::generic_category(), "size model file");
    impl_->size = static_cast<std::uint64_t>(info.st_size);
    if (mode == FileReadMode::Direct) {
        impl_->alignment = std::max<std::size_t>(sizeof(void*), info.st_blksize);
#if defined(F_NOCACHE)
        if (::fcntl(impl_->handle, F_NOCACHE, 1)) throw std::system_error(errno, std::generic_category(), "disable model-file cache");
#endif
    }
#endif
    if (!impl_->alignment || (impl_->alignment & (impl_->alignment - 1)))
        throw std::runtime_error("model-file alignment must be a power of two");
}

FileRangeReader::~FileRangeReader() = default;
std::size_t FileRangeReader::alignment() const noexcept { return impl_->alignment; }

void FileRangeReader::read(std::uint64_t offset, std::byte* destination, std::size_t size) const {
    if (offset > impl_->size || size > impl_->size - offset) throw std::out_of_range("model-file range is out of bounds");
    if (!size) return;
    if (!destination) throw std::invalid_argument("model-file destination is null");
    const auto started = std::chrono::steady_clock::now();
    impl_->calls.fetch_add(1, std::memory_order_relaxed);
    try {
        const auto alignment = impl_->alignment;
#if defined(_WIN32)
        const auto limit = std::size_t(std::numeric_limits<DWORD>::max()) / alignment * alignment;
#else
        const auto limit = std::size_t(std::numeric_limits<ssize_t>::max()) / alignment * alignment;
#endif
        while (size) {
            const auto aligned_offset = offset / alignment * alignment;
            const auto prefix = static_cast<std::size_t>(offset - aligned_offset);
            const auto copy_bytes = std::min(size, limit - prefix);
            const auto span = prefix + copy_bytes;
            const auto transfer_bytes = span + (alignment - span % alignment) % alignment;
            const bool bounce = prefix || copy_bytes != transfer_bytes ||
                reinterpret_cast<std::uintptr_t>(destination) % alignment;
            struct Buffer {
                const Impl& owner;
                void* data = nullptr;
                std::size_t bytes = 0;
                Buffer(const Impl& value, std::size_t count, bool allocate) : owner(value) {
                    if (!allocate) return;
#if defined(_WIN32)
                    data = _aligned_malloc(count, owner.alignment);
#else
                    if (::posix_memalign(&data, std::max(owner.alignment, sizeof(void*)), count)) data = nullptr;
#endif
                    if (!data) throw std::bad_alloc();
                    bytes = count;
                    owner.acquire(bytes);
                }
                ~Buffer() {
                    if (!data) return;
#if defined(_WIN32)
                    _aligned_free(data);
#else
                    std::free(data);
#endif
                    owner.staging.fetch_sub(bytes, std::memory_order_relaxed);
                }
            } buffer(*impl_, transfer_bytes, bounce);
            const auto actual = impl_->transfer(aligned_offset, bounce ? buffer.data : destination, transfer_bytes);
            impl_->physical.fetch_add(actual, std::memory_order_relaxed);
            if (actual < span) throw std::runtime_error("model file was truncated during range read");
            if (bounce) std::memcpy(destination, static_cast<std::byte*>(buffer.data) + prefix, copy_bytes);
            destination += copy_bytes;
            offset += copy_bytes;
            size -= copy_bytes;
            impl_->logical.fetch_add(copy_bytes, std::memory_order_relaxed);
        }
    } catch (...) {
        impl_->errors.fetch_add(1, std::memory_order_relaxed);
        throw;
    }
    impl_->nanos.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count(), std::memory_order_relaxed);
}

FileReadStats FileRangeReader::stats() const noexcept {
    const auto& p = *impl_;
    return {p.mode, 1, p.calls.load(), p.logical.load(), p.physical.load(), p.errors.load(),
            p.staging.load(), p.peak.load(), p.nanos.load()};
}

void FileRangeReader::read_batch(const std::vector<ReadSpan>& spans) const {
    for (const auto& span : spans) {
        if (span.offset > impl_->size || span.size > impl_->size - span.offset)
            throw std::out_of_range("batched model-file range is out of bounds");
        if (span.size && !span.destination)
            throw std::invalid_argument("batched model-file destination is null");
    }
#if defined(_WIN32)
    struct Request {
        const Impl& owner;
        OVERLAPPED native{};
        void* storage = nullptr;
        std::byte* destination = nullptr;
        std::size_t prefix = 0, bytes = 0, allocation = 0;
        bool submitted = false, finished = false;
        explicit Request(const Impl& value) : owner(value) {}
        ~Request() {
            if (submitted && !finished) {
                DWORD ignored = 0;
                GetOverlappedResult(owner.handle, &native, &ignored, TRUE);
            }
            if (native.hEvent) CloseHandle(native.hEvent);
            if (storage) {
                _aligned_free(storage);
                owner.staging.fetch_sub(allocation, std::memory_order_relaxed);
            }
        }
    };
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<Request>> requests;
    try {
        const auto alignment = impl_->alignment;
        const auto limit = std::size_t(std::numeric_limits<DWORD>::max()) / alignment * alignment;
        for (const auto& span : spans) {
            std::size_t copied = 0;
            while (copied < span.size) {
                const auto offset = span.offset + copied;
                const auto base = offset / alignment * alignment;
                auto request = std::make_unique<Request>(*impl_);
                request->prefix = static_cast<std::size_t>(offset - base);
                request->bytes = std::min(span.size - copied, limit - request->prefix);
                const auto required = request->prefix + request->bytes;
                request->allocation = required + (alignment - required % alignment) % alignment;
                request->destination = span.destination + copied;
                request->storage = _aligned_malloc(request->allocation, std::max(alignment, sizeof(void*)));
                if (!request->storage) throw std::bad_alloc();
                impl_->acquire(request->allocation);
                request->native.Offset = static_cast<DWORD>(base);
                request->native.OffsetHigh = static_cast<DWORD>(base >> 32);
                request->native.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                if (!request->native.hEvent)
                    throw std::system_error(GetLastError(), std::system_category(), "create batch-read event");
                copied += request->bytes;
                requests.push_back(std::move(request));
            }
        }
        // Keep every IO request outstanding before waiting for completion.
        for (auto& request : requests) {
            impl_->calls.fetch_add(1, std::memory_order_relaxed);
            const auto ok = ReadFile(impl_->handle, request->storage,
                static_cast<DWORD>(request->allocation), nullptr, &request->native);
            const auto error = ok ? ERROR_SUCCESS : GetLastError();
            if (!ok && error != ERROR_IO_PENDING)
                throw std::system_error(error, std::system_category(), "submit batch read");
            request->submitted = true;
        }
        for (auto& request : requests) {
            DWORD actual = 0;
            const auto ok = GetOverlappedResult(impl_->handle, &request->native, &actual, TRUE);
            request->finished = true;
            if (!ok) throw std::system_error(GetLastError(), std::system_category(), "complete batch read");
            impl_->physical.fetch_add(actual, std::memory_order_relaxed);
            if (actual < request->prefix + request->bytes)
                throw std::runtime_error("model file truncated during batch read");
        }
        for (auto& request : requests) {
            std::memcpy(request->destination,
                static_cast<std::byte*>(request->storage) + request->prefix, request->bytes);
            impl_->logical.fetch_add(request->bytes, std::memory_order_relaxed);
        }
    } catch (...) {
        impl_->errors.fetch_add(1, std::memory_order_relaxed);
        throw;
    }
    impl_->nanos.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count(), std::memory_order_relaxed);
#else
    static HostParallelPool io_workers;
    const auto threads = std::max(1u, std::thread::hardware_concurrency());
    io_workers.run(0, static_cast<std::int64_t>(spans.size()), 1, threads,
        [&](std::int64_t first, std::int64_t last) {
            for (auto index = first; index < last; ++index) {
                const auto& span = spans[static_cast<std::size_t>(index)];
                read(span.offset, span.destination, span.size);
            }
        });
#endif
}

} // namespace mfq
