#include "mfq/file_range_reader.h"
#include "mfq/host_parallel.h"
#include "mfq/host_copy.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <mutex>
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
#if defined(__linux__)
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#endif
#endif

namespace mfq {
namespace {
std::size_t direct_chunk_limit(std::size_t alignment) {
    return std::max<std::size_t>(1,(1u<<20)/alignment)*alignment;
}
void report_direct_chunking(std::size_t bytes,std::size_t limit) {
    if(bytes<=limit)return;
    static std::atomic<bool> reported{false};
    if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "model_direct_read_chunks scope=first_large_range enabled=1 max_request_bytes=%zu logical_bytes=%zu\n",
        limit,bytes);
}
void copy_read_buffer(void* destination,const void* source,std::size_t bytes) {
    // Large packed fields are normally consumed by a later DMA. Avoid
    // write-allocating medium fields while preserving independent I/O workers.
    copy_host_range({destination,source,bytes});
    static std::atomic<bool> reported{false};
    if(bytes>=(256u<<10) && bytes<(2u<<20) && !reported.load(std::memory_order_relaxed) &&
            !reported.exchange(true,std::memory_order_relaxed))
        std::fprintf(stderr,"model_read_copy scope=first_medium_bounce enabled=1 strategy=host_copy bytes=%zu\n",bytes);
}
}

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
    std::shared_ptr<std::atomic<std::uint64_t>> cached_staging=
        std::make_shared<std::atomic<std::uint64_t>>(0);
    mutable std::atomic<std::uint64_t> staging_allocations{0};
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int handle = -1;
#endif
    struct Slab {
        void* data=nullptr;std::size_t bytes=0;std::uint64_t order=0;
        std::shared_ptr<std::atomic<std::uint64_t>> charge;
    };
    static void free_staging(void* data) noexcept {
#if defined(_WIN32)
        _aligned_free(data);
#else
        std::free(data);
#endif
    }
    struct StagingPool {
        static constexpr std::size_t budget=128u<<20,maximum_slab=8u<<20;
        std::mutex mutex;
        std::array<Slab,32> slabs{};
        std::size_t idle=0;
        std::uint64_t order=0;
        void erase(Slab& slab) noexcept {
            if(!slab.data)return;
            idle-=slab.bytes;slab.charge->fetch_sub(slab.bytes,std::memory_order_relaxed);
            free_staging(slab.data);slab={};
        }
        ~StagingPool(){for(auto& slab:slabs)erase(slab);}
    };
    static StagingPool& staging_pool() {
        // All shard readers use the same bounded idle storage. Active reads
        // own their slabs exclusively, and never retain a file handle here.
        static StagingPool pool;return pool;
    }
    ~Impl() {
#if defined(_WIN32)
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (handle >= 0) ::close(handle);
#endif
    }
    Slab borrow_staging(std::size_t bytes) const {
        Slab result;
        auto& pool=staging_pool();
        {
            std::lock_guard lock(pool.mutex);
            auto best=pool.slabs.end();
            for(auto it=pool.slabs.begin();it!=pool.slabs.end();++it)
                if(it->data && it->bytes>=bytes && it->bytes-bytes<=std::max<std::size_t>(bytes,65536) &&
                        reinterpret_cast<std::uintptr_t>(it->data)%alignment==0 &&
                        (best==pool.slabs.end() || it->bytes<best->bytes))best=it;
            if(best!=pool.slabs.end()) {
                result=std::move(*best);*best={};pool.idle-=result.bytes;
                result.charge->fetch_sub(result.bytes,std::memory_order_relaxed);result.charge.reset();
            }
        }
        if(result.data) {
            static std::atomic<bool> reported{false};
            if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
                std::fprintf(stderr,"model_direct_staging_pool scope=first_reuse shared=1 max_idle_bytes=%zu bytes=%zu\n",
                    StagingPool::budget,result.bytes);
        }
        if(!result.data) {
            // Sector-aligned size classes reuse ordinary expert images across
            // nearby formats, without rounding small metadata up to 64 KiB.
            const auto granule=std::max<std::size_t>(alignment,65536);
            if(bytes>=granule && bytes<=StagingPool::maximum_slab)bytes=(bytes+granule-1)/granule*granule;
            void* data=nullptr;
#if defined(_WIN32)
            data=_aligned_malloc(bytes,std::max(alignment,sizeof(void*)));
#else
            if(::posix_memalign(&data,std::max(alignment,sizeof(void*)),bytes))data=nullptr;
#endif
            result={data,bytes};
            if(!result.data)throw std::bad_alloc();
            staging_allocations.fetch_add(1,std::memory_order_relaxed);
        }
        acquire(result.bytes);return result;
    }
    void return_staging(Slab slab) const noexcept {
        bool retained=false;
        if(slab.bytes<=StagingPool::maximum_slab) {
            auto& pool=staging_pool();std::lock_guard lock(pool.mutex);
            for(;;) {
                auto free=pool.slabs.end(),oldest=pool.slabs.end();
                for(auto it=pool.slabs.begin();it!=pool.slabs.end();++it) {
                    if(!it->data)free=it;
                    else if(oldest==pool.slabs.end() || it->order<oldest->order)oldest=it;
                }
                if(free!=pool.slabs.end() && slab.bytes<=StagingPool::budget-pool.idle) {
                    slab.order=++pool.order;slab.charge=cached_staging;*free=slab;
                    pool.idle+=slab.bytes;cached_staging->fetch_add(slab.bytes,std::memory_order_relaxed);
                    retained=true;break;
                }
                if(oldest==pool.slabs.end())break;
                pool.erase(*oldest);
            }
        }
        staging.fetch_sub(slab.bytes,std::memory_order_relaxed);
        if(!retained)free_staging(slab.data);
    }
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
    // Every operation owns an explicit OVERLAPPED event. The shared file-object
    // event is unused; suppress its redundant updates across PLE I/O workers.
    // Unsupported file systems retain their normal notification behavior.
    (void)SetFileCompletionNotificationModes(impl_->handle,FILE_SKIP_SET_EVENT_ON_HANDLE);
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
                    const auto slab=owner.borrow_staging(count);
                    data=slab.data;bytes=slab.bytes;
                }
                ~Buffer() {
                    if (!data) return;
                    owner.return_staging({data,bytes});
                }
            } buffer(*impl_, transfer_bytes, bounce);
            const auto actual = impl_->transfer(aligned_offset, bounce ? buffer.data : destination, transfer_bytes);
            impl_->physical.fetch_add(actual, std::memory_order_relaxed);
            if (actual < span) throw std::runtime_error("model file was truncated during range read");
            if (bounce) copy_read_buffer(destination, static_cast<std::byte*>(buffer.data) + prefix, copy_bytes);
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
            p.staging.load(), p.peak.load(), p.nanos.load(),p.cached_staging->load(),p.staging_allocations.load()};
}

#if defined(__linux__)
namespace {
// Each SSD worker owns its kernel queue. A shared userspace dispatch mutex
// would serialize independent experts, even with several outer read workers.
struct LinuxAioContext {
    aio_context_t context=0;
    LinuxAioContext(){if(::syscall(SYS_io_setup,64,&context)<0)context=0;}
    void reset() noexcept {
        if(context){while(::syscall(SYS_io_destroy,context)<0 && errno==EINTR){}context=0;}
    }
    ~LinuxAioContext(){reset();}
    bool read(std::array<iocb*,64>& requests,std::size_t count,
              std::array<std::int64_t,64>& results) {
        if(!context)return false;
        std::size_t submitted=0,completed=0;int failure=0;
        while(submitted<count) {
            const auto n=::syscall(SYS_io_submit,context,count-submitted,requests.data()+submitted);
            if(n<0 && errno==EINTR)continue;
            if(n<=0){failure=n<0?errno:EIO;break;}
            submitted+=std::size_t(n);
        }
        // Drain partial submissions, including a later submission failure,
        // before any destination or iocb can be released or reused.
        while(completed<submitted) {
            std::array<io_event,64> events{};
            const auto n=::syscall(SYS_io_getevents,context,1,submitted-completed,events.data(),nullptr);
            if(n<0 && errno==EINTR)continue;
            if(n<=0) {
                const auto error=n<0?errno:EIO;reset();
                throw std::system_error(error,std::generic_category(),"complete Linux model-file batch");
            }
            for(long i=0;i<n;++i)results.at(events[i].data)=events[i].res;
            completed+=std::size_t(n);
        }
        // Filesystems without native AIO still use the same O_DIRECT fd.
        // They run on the caller's existing SSD worker, without a global pool.
        if(!submitted && (failure==EINVAL || failure==EOPNOTSUPP || failure==ENOSYS))return false;
        if(failure)throw std::system_error(failure,std::generic_category(),"submit Linux model-file batch");
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))
            std::fprintf(stderr,"model_linux_aio_batch enabled=1 max_requests=64\n");
        return true;
    }
};
}
#endif

void FileRangeReader::read_batch(const std::vector<ReadSpan>& spans) const {
    for (const auto& span : spans) {
        if (span.offset > impl_->size || span.size > impl_->size - span.offset)
            throw std::out_of_range("batched model-file range is out of bounds");
        if (span.size && !span.destination)
            throw std::invalid_argument("batched model-file destination is null");
    }
#if defined(_WIN32)
    if(impl_->mode==FileReadMode::Buffered) {
        // Buffered reads accept the caller's exact spans. Reuse a bounded
        // event ring per I/O worker: no per-range event/temporary allocation,
        // and all five NINT streams can be in flight before any wait.
        struct Events {
            std::array<HANDLE,64> handles{};
            ~Events(){for(auto h:handles)if(h)CloseHandle(h);}
            HANDLE at(std::size_t i) {
                if(!handles[i])handles[i]=CreateEventW(nullptr,TRUE,FALSE,nullptr);
                if(!handles[i])throw std::system_error(GetLastError(),std::system_category(),"create buffered batch event");
                return handles[i];
            }
        };
        static thread_local Events events;
        struct Requests {
            HANDLE file;
            std::array<OVERLAPPED,64> native{};
            std::array<DWORD,64> bytes{};
            std::array<bool,64> submitted{},finished{};
            std::size_t count=0;
            explicit Requests(HANDLE h):file(h){}
            ~Requests() {
                // Even submission/short-read failures must drain every read
                // before caller-owned destinations and OVERLAPPEDs go away.
                for(std::size_t i=0;i<count;++i)if(submitted[i] && !finished[i]) {
                    DWORD ignored=0;GetOverlappedResult(file,&native[i],&ignored,TRUE);
                }
            }
        };
        const auto started=std::chrono::steady_clock::now();
        try {
            std::size_t span_index=0,span_offset=0;
            while(span_index<spans.size()) {
                Requests requests(impl_->handle);
                while(requests.count<requests.native.size() && span_index<spans.size()) {
                    const auto& span=spans[span_index];
                    if(span_offset==span.size){++span_index;span_offset=0;continue;}
                    const auto i=requests.count++;
                    const auto offset=span.offset+span_offset;
                    auto& request=requests.native[i];
                    request.Offset=static_cast<DWORD>(offset);request.OffsetHigh=static_cast<DWORD>(offset>>32);
                    request.hEvent=events.at(i);
                    // ReadFile resets this event before starting the operation.
                    // The preceding batch was fully drained before slot reuse.
                    requests.bytes[i]=static_cast<DWORD>(std::min<std::size_t>(span.size-span_offset,std::numeric_limits<DWORD>::max()));
                    impl_->calls.fetch_add(1,std::memory_order_relaxed);
                    const auto ok=ReadFile(impl_->handle,span.destination+span_offset,requests.bytes[i],nullptr,&request);
                    const auto error=ok?ERROR_SUCCESS:GetLastError();
                    if(!ok && error!=ERROR_IO_PENDING)throw std::system_error(error,std::system_category(),"submit buffered batch read");
                    requests.submitted[i]=true;span_offset+=requests.bytes[i];
                }
                for(std::size_t i=0;i<requests.count;++i) {
                    DWORD actual=0;
                    const auto ok=GetOverlappedResult(impl_->handle,&requests.native[i],&actual,TRUE);
                    requests.finished[i]=true;
                    if(!ok)throw std::system_error(GetLastError(),std::system_category(),"complete buffered batch read");
                    impl_->physical.fetch_add(actual,std::memory_order_relaxed);
                    if(actual!=requests.bytes[i])throw std::runtime_error("model file truncated during buffered batch read");
                    impl_->logical.fetch_add(actual,std::memory_order_relaxed);
                }
            }
        }catch(...){impl_->errors.fetch_add(1,std::memory_order_relaxed);throw;}
        impl_->nanos.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count(),std::memory_order_relaxed);
        return;
    }
    struct DirectEvents {
        std::array<HANDLE,64> handles{};
        ~DirectEvents(){for(auto handle:handles)if(handle)CloseHandle(handle);}
        HANDLE at(std::size_t i) {
            if(!handles[i])handles[i]=CreateEventW(nullptr,TRUE,FALSE,nullptr);
            if(!handles[i])throw std::system_error(GetLastError(),std::system_category(),"create direct batch event");
            return handles[i];
        }
    };
    static thread_local DirectEvents events;
    struct BatchStorage {
        const Impl& owner;
        void* data=nullptr;
        std::size_t capacity=0;
        explicit BatchStorage(const Impl& value):owner(value) {}
        void release() {
            if(!data)return;
            owner.return_staging({data,capacity});data=nullptr;capacity=0;
        }
        void ensure(std::size_t bytes) {
            if(bytes<=capacity)return;
            // Every preceding wave was drained before the arena can grow.
            release();const auto slab=owner.borrow_staging(bytes);data=slab.data;capacity=slab.bytes;
        }
        ~BatchStorage(){release();}
    };
    struct Request {
        const Impl& owner;
        OVERLAPPED native{};
        void* storage = nullptr;
        std::byte* destination = nullptr;
        std::size_t prefix = 0, bytes = 0, allocation = 0, storage_offset = 0;
        bool submitted = false, finished = false;
        bool aligned_target = false;
        explicit Request(const Impl& value) : owner(value) {}
        ~Request() {
            if (submitted && !finished) {
                DWORD ignored = 0;
                GetOverlappedResult(owner.handle, &native, &ignored, TRUE);
            }
        }
    };
    const auto started = std::chrono::steady_clock::now();
    // Destruction drains requests before releasing their common I/O target.
    BatchStorage storage(*impl_);
    std::array<std::optional<Request>,64> requests;
    std::size_t count=0;
    try {
        const auto alignment = impl_->alignment;
        // Keep several native reads outstanding even when only one cold
        // projection is needed. Each bounded wave still owns one staging arena.
        const auto limit = std::min(std::size_t(std::numeric_limits<DWORD>::max()) / alignment * alignment,
            direct_chunk_limit(alignment));
        std::size_t span_index = 0, span_offset = 0;
        while (span_index < spans.size()) {
            std::size_t batch_bytes=0;
            // Large PLE gathers can contain thousands of row ranges. Bound
            // aligned staging and outstanding IRPs, reusing the same events
            // after each fully drained batch.
            while (count < requests.size() && span_index < spans.size()) {
                const auto& span = spans[span_index];
                if (span_offset == span.size) { ++span_index; span_offset = 0; continue; }
                const auto offset = span.offset + span_offset;
                const auto base = offset / alignment * alignment;
                report_direct_chunking(span.size-span_offset,limit);
                auto* request=&requests[count].emplace(*impl_);
                request->prefix = static_cast<std::size_t>(offset - base);
                request->bytes = std::min(span.size - span_offset, limit - request->prefix);
                const auto required = request->prefix + request->bytes;
                request->allocation = required + (alignment - required % alignment) % alignment;
                request->destination = span.destination + span_offset;
                request->aligned_target=!request->prefix && request->bytes==request->allocation &&
                    reinterpret_cast<std::uintptr_t>(request->destination)%alignment==0;
                if(!request->aligned_target) {
                    if(request->allocation>SIZE_MAX-batch_bytes)throw std::overflow_error("direct read staging size overflow");
                    request->storage_offset=batch_bytes;batch_bytes+=request->allocation;
                }
                request->native.Offset = static_cast<DWORD>(base);
                request->native.OffsetHigh = static_cast<DWORD>(base >> 32);
                request->native.hEvent = events.at(count);
                span_offset += request->bytes;
                ++count;
            }
            storage.ensure(batch_bytes);
            static std::atomic<bool> aligned_reported{false};
            if(!aligned_reported.load(std::memory_order_relaxed)) {
                std::size_t direct=0;for(std::size_t i=0;i<count;++i)direct+=requests[i]->aligned_target;
                if(direct && !aligned_reported.exchange(true,std::memory_order_relaxed))
                    std::fprintf(stderr,"model_direct_batch_target enabled=1 requests=%zu direct_requests=%zu staging_bytes=%zu\n",count,direct,batch_bytes);
            }
            static std::atomic<bool> reported{false};
            if(count>1 && batch_bytes && !reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
                std::fprintf(stderr,"model_direct_staging_batch enabled=1 requests=%zu bytes=%zu arena_allocations=1\n",count,batch_bytes);
            // Keep every IO request in this bounded batch outstanding before
            // waiting for completion. Ordinary expert field batches fit here.
            for (std::size_t i=0;i<count;++i) {
                auto& request=requests[i];
                request->storage=request->aligned_target?request->destination:
                    static_cast<std::byte*>(storage.data)+request->storage_offset;
                impl_->calls.fetch_add(1, std::memory_order_relaxed);
                const auto ok = ReadFile(impl_->handle, request->storage,
                    static_cast<DWORD>(request->allocation), nullptr, &request->native);
                const auto error = ok ? ERROR_SUCCESS : GetLastError();
                if (!ok && error != ERROR_IO_PENDING)
                    throw std::system_error(error, std::system_category(), "submit batch read");
                request->submitted = true;
            }
            for (std::size_t i=0;i<count;++i) {
                auto& request=requests[i];
                DWORD actual = 0;
                const auto ok = GetOverlappedResult(impl_->handle, &request->native, &actual, TRUE);
                request->finished = true;
                if (!ok) throw std::system_error(GetLastError(), std::system_category(), "complete batch read");
                impl_->physical.fetch_add(actual, std::memory_order_relaxed);
                if (actual < request->prefix + request->bytes)
                    throw std::runtime_error("model file truncated during batch read");
            }
            for (std::size_t i=0;i<count;++i) {
                auto& request=requests[i];
                if(!request->aligned_target)copy_read_buffer(request->destination,
                    static_cast<std::byte*>(request->storage) + request->prefix, request->bytes);
                impl_->logical.fetch_add(request->bytes, std::memory_order_relaxed);
            }
            for(std::size_t i=0;i<count;++i)requests[i].reset();
            count=0;
        }
    } catch (...) {
        impl_->errors.fetch_add(1, std::memory_order_relaxed);
        throw;
    }
    impl_->nanos.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count(), std::memory_order_relaxed);
#elif defined(__linux__)
    if(impl_->mode==FileReadMode::Buffered) {
        for(const auto& span:spans)read(span.offset,span.destination,span.size);
        return;
    }
    static thread_local LinuxAioContext queue;
    const auto started=std::chrono::steady_clock::now();
    struct Storage {
        const Impl& owner;Impl::Slab slab{};
        explicit Storage(const Impl& value):owner(value){}
        ~Storage(){if(slab.data)owner.return_staging(slab);}
        void ensure(std::size_t bytes) {
            if(bytes<=slab.bytes)return;
            if(slab.data){owner.return_staging(slab);slab={};}
            slab=owner.borrow_staging(bytes);
        }
    } storage(*impl_);
    struct Part {std::uint64_t offset;std::size_t prefix,bytes,allocation,staging;std::byte* destination;bool aligned_target;};
    std::size_t index=0,consumed=0;
    try {
        while(index<spans.size()) {
            std::array<Part,64> parts{};std::size_t count=0,bytes=0,staging_bytes=0;
            const auto alignment=impl_->alignment;
            const auto wave_limit=std::max<std::size_t>(8u<<20,alignment);
            const auto request_limit=direct_chunk_limit(alignment);
            while(count<parts.size() && index<spans.size()) {
                const auto& span=spans[index];
                if(consumed==span.size){++index;consumed=0;continue;}
                const auto offset=span.offset+consumed,base=offset/alignment*alignment;
                const auto prefix=std::size_t(offset-base);
                report_direct_chunking(span.size-consumed,request_limit);
                const auto copy=std::min(span.size-consumed,request_limit-prefix);
                const auto allocation=(prefix+copy+alignment-1)/alignment*alignment;
                if(count && allocation>wave_limit-bytes)break;
                auto* destination=span.destination+consumed;
                const bool direct=!prefix && copy==allocation &&
                    reinterpret_cast<std::uintptr_t>(destination)%alignment==0;
                parts[count++]={base,prefix,copy,allocation,staging_bytes,destination,direct};
                if(!direct)staging_bytes+=allocation;
                bytes+=allocation;consumed+=copy;
            }
            if(!count)continue;
            storage.ensure(staging_bytes);
            static std::atomic<bool> aligned_reported{false};
            if(!aligned_reported.load(std::memory_order_relaxed)) {
                std::size_t direct=0;for(std::size_t i=0;i<count;++i)direct+=parts[i].aligned_target;
                if(direct && !aligned_reported.exchange(true,std::memory_order_relaxed))
                    std::fprintf(stderr,"model_direct_batch_target enabled=1 requests=%zu direct_requests=%zu staging_bytes=%zu\n",count,direct,staging_bytes);
            }
            const auto target=[&](const Part& part) {
                return part.aligned_target?part.destination:static_cast<std::byte*>(storage.slab.data)+part.staging;
            };
            std::array<iocb,64> native{};std::array<iocb*,64> requests{};
            std::array<std::int64_t,64> results{};
            for(std::size_t i=0;i<count;++i) {
                const auto& part=parts[i];auto& request=native[i];requests[i]=&request;
                request.aio_data=i;request.aio_lio_opcode=IOCB_CMD_PREAD;request.aio_fildes=impl_->handle;
                request.aio_buf=reinterpret_cast<std::uintptr_t>(target(part));
                request.aio_nbytes=part.allocation;request.aio_offset=part.offset;
            }
            if(!queue.read(requests,count,results)) {
                for(std::size_t i=0;i<count;++i)results[i]=impl_->transfer(parts[i].offset,
                    target(parts[i]),parts[i].allocation);
            }
            impl_->calls.fetch_add(count,std::memory_order_relaxed);
            for(std::size_t i=0;i<count;++i) {
                if(results[i]<0)throw std::system_error(int(-results[i]),std::generic_category(),"Linux model-file range read");
                impl_->physical.fetch_add(results[i],std::memory_order_relaxed);
                if(std::size_t(results[i])<parts[i].prefix+parts[i].bytes)
                    throw std::runtime_error("model file was truncated during Linux batch read");
            }
            for(std::size_t i=0;i<count;++i) {
                const auto& part=parts[i];
                if(!part.aligned_target)copy_read_buffer(part.destination,target(part)+part.prefix,part.bytes);
                impl_->logical.fetch_add(part.bytes,std::memory_order_relaxed);
            }
        }
    }catch(...){impl_->errors.fetch_add(1,std::memory_order_relaxed);throw;}
    impl_->nanos.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-started).count(),std::memory_order_relaxed);
#else
    for(const auto& span:spans)read(span.offset,span.destination,span.size);
#endif
}

void FileRangeReader::read_coalesced(const std::vector<ReadSpan>& spans) const {
    struct Interval {std::uint64_t first,last;};
    const auto alignment=impl_->alignment;
    std::vector<Interval> intervals;intervals.reserve(spans.size());
    std::size_t logical_bytes=0;
    for(const auto& span:spans) {
        if(span.offset>impl_->size || span.size>impl_->size-span.offset)
            throw std::out_of_range("batched range exceeds model shard");
        if(!span.size)continue;
        if(!span.destination)throw std::invalid_argument("batched range destination is null");
        if(span.size>SIZE_MAX-logical_bytes)throw std::overflow_error("batched logical byte count overflow");
        logical_bytes+=span.size;
        const auto last=span.offset+span.size;
        const auto remainder=(alignment-last%alignment)%alignment;
        intervals.push_back({span.offset/alignment*alignment,
            last+std::min<std::uint64_t>(remainder,impl_->size-last)});
    }
    if(intervals.empty())return;
    std::sort(intervals.begin(),intervals.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    std::vector<Interval> merged;merged.reserve(intervals.size());
    for(const auto interval:intervals) {
        if(!merged.empty() && interval.first<=merged.back().last)
            merged.back().last=std::max(merged.back().last,interval.last);
        else merged.push_back(interval);
    }
    if(alignment>1 && merged.size()>64) {
        std::vector<Interval> coalesced;coalesced.reserve(merged.size());
        std::uint64_t useful=0;
        for(const auto interval:merged) {
            const auto count=interval.last-interval.first;
            if(!coalesced.empty()) {
                auto& previous=coalesced.back();const auto cover=interval.last-previous.first;
                if(cover<=65536 && cover<=2*(useful+count)) {
                    previous.last=interval.last;useful+=count;continue;
                }
            }
            coalesced.push_back(interval);useful=count;
        }
        merged=std::move(coalesced);
    }
    if(merged.size()==intervals.size()) {
        if(alignment>1 && spans.size()>1) {
            auto ordered=spans;
            std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
            read_batch(ordered);
        }else read_batch(spans);
        return;
    }
    std::size_t bytes=0,physical_bytes=0;
    const auto allocated=[alignment](std::size_t count) {
        const auto padding=(alignment-count%alignment)%alignment;
        if(padding>SIZE_MAX-count)throw std::overflow_error("coalesced range alignment overflow");
        return count+padding;
    };
    for(const auto interval:merged) {
        const auto count=interval.last-interval.first;
        if(count>SIZE_MAX-physical_bytes)throw std::overflow_error("coalesced range size overflow");
        physical_bytes+=std::size_t(count);
        const auto capacity=allocated(std::size_t(count));
        if(capacity>SIZE_MAX-bytes)throw std::overflow_error("coalesced storage size overflow");
        bytes+=capacity;
    }
    struct Buffer {
        const Impl& owner;Impl::Slab slab;
        ~Buffer(){owner.return_staging(slab);}
    } storage{*impl_,impl_->borrow_staging(bytes)};
    std::vector<ReadSpan> reads;reads.reserve(merged.size());
    std::vector<ReadSpan> operations;operations.reserve(merged.size()+1);
    std::size_t offset=0;
    for(const auto interval:merged) {
        const auto count=std::size_t(interval.last-interval.first);
        auto* destination=static_cast<std::byte*>(storage.slab.data)+offset;
        reads.push_back({interval.first,destination,count});
        // A shard's unaligned EOF needs a bounce for its final sector only,
        // not another full coalesced payload. The prefix can be filled in place.
        const auto prefix=count/alignment*alignment;
        if(prefix)operations.push_back({interval.first,destination,prefix});
        if(prefix<count)operations.push_back({interval.first+prefix,destination+prefix,count-prefix});
        offset+=allocated(count);
    }
    read_batch(operations);
    for(const auto& span:spans)if(span.size) {
        const auto found=std::upper_bound(merged.begin(),merged.end(),span.offset,
            [](std::uint64_t at,const Interval& interval){return at<interval.first;});
        const auto index=std::size_t(found-merged.begin()-1);
        copy_read_buffer(span.destination,reads[index].destination+span.offset-merged[index].first,span.size);
    }
    // read_batch counted the merged ranges; report the original requested
    // bytes, including repeated/overlapping reads, without resetting peers.
    if(logical_bytes>=physical_bytes)impl_->logical.fetch_add(logical_bytes-physical_bytes,std::memory_order_relaxed);
    else impl_->logical.fetch_sub(physical_bytes-logical_bytes,std::memory_order_relaxed);
    static std::atomic<bool> reported{false};
    if(bytes>=(1u<<20) && !reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
        std::fprintf(stderr,"model_coalesced_read scope=first_large_read enabled=1 ranges=%zu groups=%zu aligned_storage_bytes=%zu\n",
            spans.size(),merged.size(),storage.slab.bytes);
}

} // namespace mfq
