#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mfq::cache {

class DiskBudgetLock {
public:
    explicit DiskBudgetLock(const std::filesystem::path& path, bool wait = true) {
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        OVERLAPPED position{};
        locked_ = handle_ != INVALID_HANDLE_VALUE && LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK |
            (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY), 0, 1, 0, &position);
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        locked_ = fd_ >= 0 && ::flock(fd_, LOCK_EX | (wait ? 0 : LOCK_NB)) == 0;
#endif
    }
    ~DiskBudgetLock() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    DiskBudgetLock(const DiskBudgetLock&) = delete;
    DiskBudgetLock& operator=(const DiskBudgetLock&) = delete;
    bool locked() const { return locked_; }
private:
    bool locked_ = false;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

inline std::filesystem::path disk_budget_directory() {
    const char* root = std::getenv("MFQ_DISK_BUDGET_DIR");
    if (root && *root) return root;
#ifdef _WIN32
    const char* user = std::getenv("USERNAME");
    return std::filesystem::temp_directory_path() / (std::string("mfq-disk-budget-") + (user ? user : "user"));
#else
    return std::filesystem::temp_directory_path() / ("mfq-disk-budget-" + std::to_string(::getuid()));
#endif
}

inline std::uint64_t disk_volume(const std::filesystem::path& path) {
#ifdef _WIN32
    HANDLE handle = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    BY_HANDLE_FILE_INFORMATION info{};
    const bool ok = handle != INVALID_HANDLE_VALUE && GetFileInformationByHandle(handle, &info);
    if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    if (ok) return info.dwVolumeSerialNumber;
#else
    struct stat info{};
    if (::stat(path.c_str(), &info) == 0) return static_cast<std::uint64_t>(info.st_dev);
#endif
    throw std::runtime_error("cannot identify SSD volume");
}

inline std::uint64_t disk_reserved_locked(const std::filesystem::path& root, std::uint64_t volume) {
    std::uint64_t reserved = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (entry.path().extension() != ".lease") continue;
        DiskBudgetLock lease(entry.path(), false);
        if (lease.locked()) continue;
        std::ifstream input(entry.path(), std::ios::binary);
        input.seekg(1);
        std::uint64_t device = 0, bytes = 0;
        if (!(input >> device >> bytes)) throw std::runtime_error("invalid SSD reservation");
        if (device == volume) {
            if (bytes > std::numeric_limits<std::uint64_t>::max() - reserved)
                throw std::runtime_error("SSD reservation overflow");
            reserved += bytes;
        }
    }
    return reserved;
}

class DiskWriteGuard {
public:
    DiskWriteGuard(const std::filesystem::path& directory, std::uint64_t bytes, std::uint64_t margin = 0)
        : root_(disk_budget_directory()), lock_(prepare(root_) / "lock") {
        if (!lock_.locked()) throw std::runtime_error("cannot lock SSD budget");
        const auto free = std::filesystem::space(directory).available;
        const auto reserved = disk_reserved_locked(root_, disk_volume(directory));
        const auto available = free > reserved ? free - reserved : 0;
        allowed_ = margin <= available && bytes <= available - margin;
    }
    bool allowed() const { return allowed_; }
private:
    static std::filesystem::path prepare(const std::filesystem::path& root) {
        std::filesystem::create_directories(root);
        return root;
    }
    std::filesystem::path root_;
    DiskBudgetLock lock_;
    bool allowed_ = false;
};

inline std::uint64_t unreserved_disk_space(const std::filesystem::path& directory, std::uint64_t managed = 0) {
    const auto root = disk_budget_directory();
    std::filesystem::create_directories(root);
    DiskBudgetLock lock(root / "lock");
    if (!lock.locked()) throw std::runtime_error("cannot lock SSD budget");
    const auto available = std::filesystem::space(directory).available;
    const auto free = managed > std::numeric_limits<std::uint64_t>::max() - available
        ? std::numeric_limits<std::uint64_t>::max() : managed + available;
    const auto reserved = disk_reserved_locked(root, disk_volume(directory));
    return free > reserved ? free - reserved : 0;
}

}
