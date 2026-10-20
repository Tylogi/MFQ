#include "files.h"
#include "instance.h"

#include <cerrno>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mfq::server {

void write_json(const std::filesystem::path& path, const nlohmann::json& value) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.parent_path() / (".mfq-" + normalize_uuid({}) + ".tmp");
    int descriptor = -1;
    bool created = false;
    try {
#ifdef _WIN32
        descriptor = _wopen(temporary.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY | _O_NOINHERIT,
            _S_IREAD | _S_IWRITE);
#else
        descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
#endif
        if (descriptor < 0) throw std::system_error(errno, std::generic_category());
        created = true;
        const auto content = value.dump();
        std::size_t written = 0;
        while (written < content.size()) {
#ifdef _WIN32
            const auto count = _write(descriptor, content.data() + written, static_cast<unsigned>(content.size() - written));
#else
            const auto count = ::write(descriptor, content.data() + written, content.size() - written);
#endif
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::system_error(errno, std::generic_category());
            written += static_cast<std::size_t>(count);
        }
#ifdef _WIN32
        if (_commit(descriptor)) throw std::system_error(errno, std::generic_category());
        _close(descriptor);
        descriptor = -1;
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("cannot replace JSON file");
#else
        if (::fsync(descriptor)) throw std::system_error(errno, std::generic_category());
        ::close(descriptor);
        descriptor = -1;
        std::filesystem::rename(temporary, path);
#endif
    } catch (...) {
#ifdef _WIN32
        if (descriptor >= 0) _close(descriptor);
#else
        if (descriptor >= 0) ::close(descriptor);
#endif
        std::error_code ignored;
        if (created) std::filesystem::remove(temporary, ignored);
        throw;
    }
}

} // namespace mfq::server
