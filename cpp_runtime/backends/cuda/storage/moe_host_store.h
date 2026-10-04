#pragma once

#include "mfq_tensor_backend.h"

#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
#endif

// Spill the existing packed runtime layout, without changing quantized values.
// The unlinked file lives until its last tensor view dies. Only cache misses
// touch its pages; the OS can reclaim them instead of retaining all experts in RAM.
inline std::size_t mmap_moe_host_fields(
        std::span<mfq_tensor_backend::Tensor*> fields,
        const std::string& directory) {
#ifdef _WIN32
    throw std::runtime_error("MFQ_MOE_SSD_CACHE_DIR currently requires POSIX mmap");
#else
    std::vector<std::size_t> offsets;
    std::size_t size = 0;
    for (const auto* field : fields) {
        MFQ_RUNTIME_CHECK(field->defined() && field->is_cpu() && field->is_contiguous(),
            "SSD MoE fields must be contiguous CPU tensors");
        const auto bytes = static_cast<std::size_t>(field->numel()) * field->element_size();
        MFQ_RUNTIME_CHECK(size <= std::numeric_limits<std::size_t>::max() - 63,
            "SSD MoE field alignment overflow");
        size = (size + 63) & ~std::size_t{63};
        offsets.push_back(size);
        MFQ_RUNTIME_CHECK(bytes <= std::numeric_limits<std::size_t>::max() - size,
            "SSD MoE field size overflow");
        size += bytes;
    }
    if (size == 0) return 0;
    MFQ_RUNTIME_CHECK(size <= static_cast<std::size_t>(std::numeric_limits<off_t>::max()),
        "SSD MoE file exceeds the file offset range");
    std::filesystem::create_directories(directory);
    auto path = (std::filesystem::path(directory) / "mfq-experts-XXXXXX").string();
    const int fd = mkstemp(path.data());
    MFQ_RUNTIME_CHECK(fd >= 0, "cannot create SSD MoE staging file in ", directory);
    if (unlink(path.c_str()) != 0) {
        close(fd);
        throw std::runtime_error("cannot unlink SSD MoE staging file");
    }
    std::unique_ptr<FILE, int (*)(FILE*)> file(fdopen(fd, "w+b"), std::fclose);
    if (!file) {
        close(fd);
        throw std::runtime_error("cannot open SSD MoE staging stream");
    }
#ifdef __linux__
    struct statfs filesystem{};
    MFQ_RUNTIME_CHECK(fstatfs(fd, &filesystem) == 0 &&
            filesystem.f_type != TMPFS_MAGIC && filesystem.f_type != RAMFS_MAGIC,
        "MFQ_MOE_SSD_CACHE_DIR must use disk storage, not tmpfs/ramfs");
#endif
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& field = *fields[i];
        const auto bytes = static_cast<std::size_t>(field.numel()) * field.element_size();
        MFQ_RUNTIME_CHECK(fseeko(file.get(), static_cast<off_t>(offsets[i]), SEEK_SET) == 0 &&
                std::fwrite(field.data_ptr(), 1, bytes, file.get()) == bytes,
            "cannot write SSD MoE staging file (check free disk space)");
    }
    MFQ_RUNTIME_CHECK(std::fflush(file.get()) == 0 &&
            ftruncate(fd, static_cast<off_t>(size)) == 0 && fsync(fd) == 0,
        "cannot flush SSD MoE staging file");
    void* base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    MFQ_RUNTIME_CHECK(base != MAP_FAILED, "cannot map SSD MoE staging file");
    auto owner = std::shared_ptr<void>(base, [size](void* data) { munmap(data, size); });
    // Keep normal read-ahead for each expert's contiguous packed fields.
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    for (std::size_t i = 0; i < fields.size(); ++i) {
        auto& field = *fields[i];
        auto* data = static_cast<std::byte*>(base) + offsets[i];
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        auto storage = std::make_shared<mfq::cuda::TensorStorage>();
        storage->owner = owner;
        storage->base = base;
        storage->bytes = size;
        auto view = mfq::cuda::make_contiguous_view(
            data, field.sizes().vec(), field.scalar_type(), field.device());
        field = mfq::cuda::Tensor(std::move(storage), view, offsets[i]);
#else
        field = mfq_tensor_backend::from_blob(data, field.sizes(),
            [owner](void*) {}, field.options());
#endif
    }
    return size;
#endif
}
