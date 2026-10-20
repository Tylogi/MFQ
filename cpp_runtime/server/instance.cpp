#include "instance.h"

#include "mfq_paged_prefix_cache.h"

#include <cerrno>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <share.h>
#include <sys/locking.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mfq::server {
namespace {

std::filesystem::path home_directory() {
#ifdef _WIN32
    if (const auto* home = _wgetenv(L"USERPROFILE")) return home;
#else
    if (const auto* home = std::getenv("HOME")) return std::filesystem::u8path(home);
#endif
    throw std::runtime_error("cannot locate the user home directory");
}

} // namespace

std::string normalize_uuid(std::string value) {
    if (value.empty()) {
        std::random_device random;
        std::uniform_int_distribution<int> byte(0, 255);
        constexpr char digits[] = "0123456789abcdef";
        for (int i = 0; i < 16; ++i) {
            if (i == 4 || i == 6 || i == 8 || i == 10) value += '-';
            int current = byte(random);
            if (i == 6) current = (current & 15) | 64;
            if (i == 8) current = (current & 63) | 128;
            value += digits[current >> 4];
            value += digits[current & 15];
        }
    }
    if (value.size() != 36) throw std::invalid_argument("invalid instance UUID");
    for (std::size_t i = 0; i < value.size(); ++i) {
        auto& c = value[i];
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (c != '-') throw std::invalid_argument("invalid instance UUID");
        } else {
            if (c >= 'A' && c <= 'F') c += 'a' - 'A';
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                throw std::invalid_argument("invalid instance UUID");
        }
    }
    return value;
}

namespace {

[[noreturn]] void file_error(const char* message) {
    throw std::system_error(errno, std::generic_category(), message);
}

void close_descriptor(int descriptor) {
#ifdef _WIN32
    _close(descriptor);
#else
    ::close(descriptor);
#endif
}

} // namespace

std::filesystem::path resolved_path(const std::filesystem::path& path) {
    const auto value = path.u8string();
    auto expanded = path;
    if (value == "~") expanded = home_directory();
    else if (value.rfind("~/", 0) == 0
#ifdef _WIN32
        || value.rfind("~\\", 0) == 0
#endif
    )
        expanded = home_directory() / std::filesystem::u8path(value.substr(2));
    return std::filesystem::weakly_canonical(std::filesystem::absolute(expanded));
}

InstanceLock::InstanceLock(const std::filesystem::path& data_dir, std::string instance_id, bool studio_owned) {
    auto workspace = resolved_path(data_dir);
    if (std::filesystem::exists(workspace) && !std::filesystem::is_directory(workspace))
        throw std::invalid_argument("--data-dir must be a directory");
#ifdef _WIN32
    auto lower = workspace.native();
    if (!LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, workspace.c_str(),
            static_cast<int>(lower.size()), lower.data(), static_cast<int>(lower.size()), nullptr, nullptr, 0))
        throw std::runtime_error("cannot normalize the workspace path");
    workspace = lower;
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    identity_ = {{"service", "mfq-server"}, {"protocol_version", "1.0"},
        {"studio_protocol_version", "1.0"}, {"version", MFQ_SERVER_VERSION},
        {"instance_id", normalize_uuid(std::move(instance_id))},
        {"workspace_id", mfq::cache::block_hash_hex(mfq::cache::sha256(workspace.u8string()))},
        {"pid", pid}, {"launch_mode", studio_owned ? "studio" : "cli"}};
    const auto filename = identity_["workspace_id"].get<std::string>() + ".lock";
    try {
#ifdef _WIN32
        const auto* local = _wgetenv(L"LOCALAPPDATA");
        const auto root = (local ? std::filesystem::path(local) : home_directory() / "AppData" / "Local") / "MFQ" / "run";
        std::filesystem::create_directories(root);
        const auto path = root / filename;
        const auto error = _wsopen_s(&descriptor_, path.c_str(), _O_RDWR | _O_CREAT | _O_BINARY | _O_NOINHERIT,
            _SH_DENYNO, _S_IREAD | _S_IWRITE);
        if (error) { errno = error; file_error("cannot open server instance lock"); }
        if (_filelengthi64(descriptor_) == 0 && _write(descriptor_, "\0", 1) != 1)
            file_error("cannot initialize server instance lock");
        if (_lseek(descriptor_, 0, SEEK_SET) < 0) file_error("cannot seek server instance lock");
        if (_locking(descriptor_, _LK_NBLCK, 1)) {
#else
        const auto root = std::filesystem::path("/tmp") / ("mfq-server-locks-" + std::to_string(getuid()));
        if (::mkdir(root.c_str(), 0700) && errno != EEXIST) file_error("cannot create server lock directory");
        const int directory = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (directory < 0) file_error("cannot open server lock directory");
        struct stat status{};
        if (::fstat(directory, &status) || status.st_uid != getuid() || (status.st_mode & 0022)) {
            ::close(directory);
            throw std::runtime_error("unsafe server lock directory ownership or permissions");
        }
        descriptor_ = ::openat(directory, filename.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
        const int open_error = errno;
        ::close(directory);
        if (descriptor_ < 0) { errno = open_error; file_error("cannot open server instance lock"); }
        if (::fstat(descriptor_, &status)) file_error("cannot inspect server instance lock");
        if (!S_ISREG(status.st_mode) || status.st_uid != getuid() || (status.st_mode & 0022))
            throw std::runtime_error("unsafe server instance lock ownership or permissions");
        if (::flock(descriptor_, LOCK_EX | LOCK_NB)) {
#endif
            if (errno == EAGAIN || errno == EACCES || errno == EDEADLK)
                throw std::runtime_error("This MFQ data directory is already managed by another server.");
            file_error("cannot acquire server instance lock");
        }
        const auto record = std::string(1, '\0') + identity_.dump();
#ifdef _WIN32
        if (_lseek(descriptor_, 0, SEEK_SET) < 0) file_error("cannot seek server instance lock");
#else
        if (::lseek(descriptor_, 0, SEEK_SET) < 0) file_error("cannot seek server instance lock");
#endif
        std::size_t written = 0;
        while (written < record.size()) {
#ifdef _WIN32
            const auto count = _write(descriptor_, record.data() + written, static_cast<unsigned>(record.size() - written));
#else
            const auto count = ::write(descriptor_, record.data() + written, record.size() - written);
#endif
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) file_error("cannot write server instance identity");
            written += static_cast<std::size_t>(count);
        }
#ifdef _WIN32
        if (_chsize_s(descriptor_, record.size())) file_error("cannot truncate server instance lock");
#else
        if (::ftruncate(descriptor_, static_cast<off_t>(record.size()))) file_error("cannot truncate server instance lock");
#endif
    } catch (...) {
        if (descriptor_ >= 0) close_descriptor(descriptor_);
        throw;
    }
}

InstanceLock::~InstanceLock() { if (descriptor_ >= 0) close_descriptor(descriptor_); }

} // namespace mfq::server
