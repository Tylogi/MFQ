#include "mfq_container.h"

#include "mlx_legacy_tensor_compat.h"

#include "mfq/mfq_model_source.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace mfq::metal {
namespace {

constexpr std::uint32_t kMaxRecordEntries =
    std::uint32_t{1} << 20;
std::uint64_t checked_file_size(
    const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (
        error
        || size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::uint64_t>::max())
        || size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamoff>::max())
    ) {
        throw std::runtime_error(
            "cannot determine usable MFQ file size: " + path.string());
    }
    return static_cast<std::uint64_t>(size);
}

} // namespace

struct MfqContainer::RandomAccessFiles {
    struct File {
        explicit File(std::filesystem::path source)
            : path(std::move(source)), size(checked_file_size(path)) {}

        ~File() {
            if (descriptor >= 0) {
                ::close(descriptor);
            }
        }

        int open() const {
            std::scoped_lock lock(mutex);
            if (descriptor >= 0) {
                return descriptor;
            }
            descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (descriptor < 0) {
                throw std::runtime_error(
                    "cannot open MFQ record source: " + path.string() +
                    ": " + std::strerror(errno));
            }
#if defined(__APPLE__) && defined(F_NOCACHE)
            if (::fcntl(descriptor, F_NOCACHE, 1) != 0) {
                const auto error = errno;
                ::close(descriptor);
                descriptor = -1;
                throw std::runtime_error(
                    "cannot enable direct MFQ record reads: " + path.string() +
                    ": " + std::strerror(error));
            }
#endif
            return descriptor;
        }

        std::filesystem::path path;
        std::uint64_t size = 0;
        mutable std::mutex mutex;
        mutable int descriptor = -1;
    };

    void read(
        const std::filesystem::path& path,
        std::uint64_t offset,
        std::span<std::byte> destination) {
        std::shared_ptr<File> source;
        {
            std::scoped_lock lock(mutex);
            const auto found = files.find(path);
            if (found != files.end()) {
                source = found->second;
            } else {
                source = std::make_shared<File>(path);
                files.emplace(path, source);
            }
        }
        if (offset > source->size ||
            destination.size() > source->size - offset) {
            throw std::runtime_error(
                "MFQ record source was truncated: " + path.string());
        }
        std::size_t done = 0;
        while (done < destination.size()) {
            constexpr std::size_t maximum_read = std::size_t{32} << 20;
            const auto requested = std::min(
                destination.size() - done, maximum_read);
            const auto count = ::pread(
                source->open(),
                destination.data() + done,
                requested,
                static_cast<off_t>(offset + done));
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(
                    "failed reading MFQ record source: " + path.string() +
                    ": " + std::strerror(errno));
            }
            if (count == 0) {
                throw std::runtime_error(
                    "MFQ record source was truncated: " + path.string());
            }
            done += static_cast<std::size_t>(count);
        }
    }

    std::mutex mutex;
    std::map<std::filesystem::path, std::shared_ptr<File>> files;
};

void MfqContainer::load_hf_directory(
    const std::filesystem::path& requested_path) {
    hf_source_ = std::make_shared<mfq::HfModelSource>(requested_path);
    header_.version = 2;
    header_.architecture = std::string(hf_source_->architecture());
    header_.extra_json = hf_source_->metadata();
    header_.record_count = static_cast<std::uint32_t>(
        hf_source_->tensors().size() + hf_source_->assets().size());
    source_paths_ = hf_source_->source_paths();

    for (const auto& tensor : hf_source_->tensors()) {
        MfqRecord record;
        record.name = tensor.name;
        record.dtype = tensor.dtype;
        record.stored_dtype = tensor.stored_dtype;
        record.source_path = requested_path;
        record.nbytes = tensor.nbytes;
        if (!records_.emplace(record.name, std::move(record)).second) {
            throw std::runtime_error(
                "duplicate canonical HF tensor: " + tensor.name);
        }
    }
    for (const auto& name : hf_source_->assets()) {
        MfqRecord record;
        record.name = name;
        record.dtype = "BLOB";
        record.source_path = requested_path;
        record.nbytes = hf_source_->read_asset(name).size();
        if (!records_.emplace(record.name, std::move(record)).second) {
            throw std::runtime_error("duplicate native HF asset: " + name);
        }
    }
    if (records_.size() > kMaxRecordEntries) {
        throw std::runtime_error("HF checkpoint has too many tensor records");
    }
}

MfqContainer::MfqContainer(std::filesystem::path path)
    : random_access_files_(std::make_shared<RandomAccessFiles>()) {
    std::error_code directory_error;
    if (std::filesystem::is_directory(path, directory_error) &&
        !directory_error) {
        load_hf_directory(path);
        return;
    }
    mfq::MfqModelSource source(std::move(path));
    header_.version = source.header().version;
    header_.architecture = source.header().architecture;
    header_.extra_json = source.header().metadata;
    header_.record_count = source.header().record_count;
    source_paths_ = source.source_paths();
    for (const auto& stored : source.records()) {
        MfqRecord record;
        record.name = stored.tensor.name;
        record.dtype = stored.tensor.dtype;
        record.stored_dtype = stored.tensor.stored_dtype;
        record.source_path = stored.source_path;
        record.offset = stored.offset;
        record.nbytes = stored.tensor.nbytes;
        records_.emplace(record.name, std::move(record));
    }
    install_legacy_tensor_compatibility(*this);
}

bool MfqContainer::contains(const std::string& name) const {
    if (records_.find(name) != records_.end()) {
        return true;
    }
    const auto alias = legacy_aliases_.find(name);
    return alias != legacy_aliases_.end() &&
        records_.find(alias->second) != records_.end();
}

const MfqRecord& MfqContainer::record(const std::string& name) const {
    auto found = records_.find(name);
    if (found == records_.end()) {
        const auto alias = legacy_aliases_.find(name);
        if (alias != legacy_aliases_.end()) {
            found = records_.find(alias->second);
        }
    }
    if (found == records_.end()) {
        throw std::runtime_error("missing MFQ record: " + name);
    }
    return found->second;
}

std::vector<std::uint8_t> MfqContainer::read(
    const std::string& name) const {
    const auto& value = record(name);
    return read_range(name, 0, value.nbytes);
}

MfqMappedBytes MfqContainer::map_record(
    const std::string& name) const {
    const auto& value = record(name);
    if (value.nbytes == 0) {
        return {};
    }
    if (hf_source_) {
        auto bytes = std::make_shared<std::vector<std::uint8_t>>(read(name));
        const auto* data = bytes->data();
        const auto size = bytes->size();
        std::shared_ptr<void> owner = bytes;
        return MfqMappedBytes(std::move(owner), data, size);
    }
    if (
        value.nbytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
        || value.offset > static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())
    ) {
        throw std::runtime_error(
            "MFQ record is too large to map: " + name);
    }
    const auto page_size = static_cast<std::uint64_t>(
        ::getpagesize());
    const auto mapped_offset =
        value.offset - value.offset % page_size;
    const auto delta = value.offset - mapped_offset;
    if (
        delta > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
            - value.nbytes
        || mapped_offset > static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())
    ) {
        throw std::runtime_error(
            "MFQ mapped record range overflows: " + name);
    }
    const auto mapped_size = static_cast<std::size_t>(
        delta + value.nbytes);
    const int descriptor = ::open(
        value.source_path.c_str(),
        O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        throw std::runtime_error(
            "cannot open MFQ record for mmap: " + name
            + ": " + std::strerror(errno));
    }
    void* mapping = ::mmap(
        nullptr,
        mapped_size,
        PROT_READ,
        MAP_PRIVATE,
        descriptor,
        static_cast<off_t>(mapped_offset));
    const int map_error = errno;
    ::close(descriptor);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(
            "cannot mmap MFQ record: " + name
            + ": " + std::strerror(map_error));
    }
    auto owner = std::shared_ptr<void>(
        mapping,
        [mapped_size](void* address) {
            ::munmap(address, mapped_size);
        });
    return MfqMappedBytes(
        std::move(owner),
        static_cast<const std::uint8_t*>(mapping)
            + static_cast<std::size_t>(delta),
        static_cast<std::size_t>(value.nbytes));
}

std::vector<std::uint8_t> MfqContainer::read_range(
    const std::string& name,
    std::uint64_t relative_offset,
    std::uint64_t nbytes) const {
    if (nbytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "MFQ record byte range is too large: " + name);
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(nbytes));
    read_range_into(
        name,
        relative_offset,
        std::as_writable_bytes(std::span<std::uint8_t>(result)));
    return result;
}

void MfqContainer::read_range_into(
    const std::string& name,
    std::uint64_t relative_offset,
    std::span<std::byte> destination) const {
    const auto& value = record(name);
    const auto nbytes = static_cast<std::uint64_t>(destination.size());
    if (
        relative_offset > value.nbytes
        || nbytes > value.nbytes - relative_offset
    ) {
        throw std::out_of_range(
            "MFQ record byte range is out of bounds: " + name);
    }
    if (nbytes >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())
        || value.offset >
            std::numeric_limits<std::uint64_t>::max()
                - relative_offset
    ) {
        throw std::runtime_error(
            "MFQ record byte range is too large: " + name);
    }
    if (nbytes == 0) {
        return;
    }
    if (hf_source_) {
        if (hf_source_->has_asset(value.name)) {
            const auto bytes = hf_source_->read_asset(value.name);
            std::copy_n(
                bytes.data() + relative_offset,
                destination.size(),
                destination.data());
        } else {
            hf_source_->read_range_into(
                value.name,
                relative_offset,
                destination.data(),
                destination.size());
        }
        return;
    }
    const auto absolute_offset =
        value.offset + relative_offset;
    if (
        absolute_offset >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())
    ) {
        throw std::runtime_error(
            "MFQ record byte offset is too large: " + name);
    }
    random_access_files_->read(
        value.source_path, absolute_offset, destination);
}


std::string MfqContainer::read_text(const std::string& name) const {
    const auto bytes = read(name);
    return std::string(bytes.begin(), bytes.end());
}

void MfqContainer::drop_source_file_cache() const noexcept {
    if (hf_source_) hf_source_->drop_file_cache();
}

std::optional<mfq::MfqModelGraph> MfqContainer::model_graph() const {
    const std::string asset(mfq::kMfqModelGraphAsset);
    if (!contains(asset)) return std::nullopt;
    return mfq::MfqModelGraph::from_json(read_text(asset));
}

void MfqContainer::install_legacy_aliases(
        std::unordered_map<std::string, std::string> canonical_to_stored,
        mfq::MfqLegacyTensorLayout layout) {
    if (model_graph()) {
        throw std::invalid_argument(
            "canonical MFQ model graphs cannot install legacy tensor aliases");
    }
    for (const auto& [canonical, stored] : canonical_to_stored) {
        if (canonical.empty() || stored.empty() || canonical == stored) {
            throw std::invalid_argument("invalid legacy MFQ tensor alias");
        }
        if (records_.find(canonical) != records_.end()) {
            throw std::invalid_argument(
                "legacy MFQ alias shadows a stored canonical tensor: " + canonical);
        }
        if (records_.find(stored) == records_.end()) {
            throw std::invalid_argument(
                "legacy MFQ alias refers to a missing tensor: " + stored);
        }
        if (canonical_to_stored.find(stored) != canonical_to_stored.end()) {
            throw std::invalid_argument(
                "legacy MFQ tensor alias chains are forbidden: " + canonical);
        }
        const auto existing = legacy_aliases_.find(canonical);
        if (existing != legacy_aliases_.end() && existing->second != stored) {
            throw std::invalid_argument(
                "legacy MFQ tensor alias is ambiguous: " + canonical);
        }
    }
    legacy_aliases_.insert(
        std::make_move_iterator(canonical_to_stored.begin()),
        std::make_move_iterator(canonical_to_stored.end()));
    legacy_tensor_layout_ = layout;
}


} // namespace mfq::metal
