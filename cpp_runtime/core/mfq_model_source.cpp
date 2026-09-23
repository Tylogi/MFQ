#include "mfq/mfq_model_source.h"

#include "mfq_format_compat.h"
#include "mfq_legacy_tensor_names.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mfq {
namespace {

constexpr std::uint64_t kMaxStringBytes = std::uint64_t{64} << 20;
constexpr std::uint32_t kMaxMetadataEntries = std::uint32_t{1} << 16;
constexpr std::uint32_t kMaxRecordEntries = std::uint32_t{1} << 20;
constexpr std::string_view kAssetPrefix = "__mfq_asset__/";

std::filesystem::path stable_path(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error("cannot open MFQ file: " + path.string());
    }
    auto result = std::filesystem::weakly_canonical(path, error);
    if (error) {
        error.clear();
        result = std::filesystem::absolute(path, error).lexically_normal();
    }
    if (error || !result.is_absolute()) {
        throw std::runtime_error("cannot resolve MFQ file path: " + path.string());
    }
    return result;
}

std::uint64_t usable_file_size(const std::filesystem::path& path) {
    std::error_code error;
    const auto value = std::filesystem::file_size(path, error);
    if (error || value > std::numeric_limits<std::uint64_t>::max() ||
        value > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("cannot determine usable MFQ file size: " +
                                 path.string());
    }
    return static_cast<std::uint64_t>(value);
}

class Input {
public:
    Input(std::istream& stream, std::uint64_t size, const std::filesystem::path& path)
        : stream_(stream), size_(size), path_(path) {}

    std::uint64_t position() const noexcept { return position_; }
    std::uint64_t remaining() const noexcept { return size_ - position_; }

    template <typename T>
    T scalar(const char* what) {
        T value{};
        read(reinterpret_cast<char*>(&value), sizeof(value), what);
        return value;
    }

    std::string string(const char* what) {
        const auto size = scalar<std::uint32_t>("string length");
        if (size > kMaxStringBytes) {
            throw std::runtime_error(std::string("MFQ ") + what +
                                     " exceeds the supported string length: " +
                                     path_.string());
        }
        std::string result(size, '\0');
        if (size != 0) read(result.data(), size, what);
        return result;
    }

private:
    void read(char* destination, std::uint64_t size, const char* what) {
        if (size > remaining() ||
            size > static_cast<std::uint64_t>(
                       std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error(std::string("unexpected EOF reading ") +
                                     what + ": " + path_.string());
        }
        stream_.read(destination, static_cast<std::streamsize>(size));
        if (!stream_) {
            throw std::runtime_error(std::string("failed reading MFQ ") + what +
                                     ": " + path_.string());
        }
        position_ += size;
    }

    std::istream& stream_;
    std::uint64_t size_;
    const std::filesystem::path& path_;
    std::uint64_t position_ = 0;
};

void validate_count(
    std::uint32_t count,
    std::uint32_t limit,
    std::uint64_t minimum_bytes,
    std::uint64_t remaining,
    const char* what,
    const std::filesystem::path& path) {
    if (count > limit || count > remaining / minimum_bytes) {
        throw std::runtime_error(std::string("invalid MFQ ") + what + ": " +
                                 path.string());
    }
}

std::uint64_t metadata_uint(
    const MfqSourceHeader& header,
    const std::string& key,
    std::uint64_t fallback) {
    const auto found = header.metadata.find(key);
    if (found == header.metadata.end()) return fallback;
    std::size_t parsed = 0;
    std::uint64_t result = 0;
    try {
        result = std::stoull(found->second, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " +
                                 found->second);
    }
    while (parsed < found->second.size() &&
           std::isspace(static_cast<unsigned char>(found->second[parsed]))) {
        ++parsed;
    }
    if (parsed != found->second.size()) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " +
                                 found->second);
    }
    return result;
}

std::vector<std::filesystem::path> shard_paths(
    const std::filesystem::path& path,
    std::uint64_t split_no,
    std::uint64_t split_count) {
    static const std::regex pattern(
        R"(^(.*)-([0-9]{5})-of-([0-9]{5})\.mfq$)");
    std::smatch match;
    const auto filename = path.filename().string();
    if (!std::regex_match(filename, match, pattern)) {
        throw std::runtime_error(
            "sharded MFQ path lacks -00001-of-00000 suffix: " + path.string());
    }
    if (std::stoull(match[2].str()) != split_no + 1 ||
        std::stoull(match[3].str()) != split_count) {
        throw std::runtime_error("MFQ shard filename/metadata mismatch: " +
                                 path.string());
    }
    std::vector<std::filesystem::path> result;
    result.reserve(static_cast<std::size_t>(split_count));
    for (std::uint64_t index = 1; index <= split_count; ++index) {
        std::ostringstream name;
        name << match[1].str() << '-' << std::setfill('0') << std::setw(5)
             << index << "-of-" << std::setw(5) << split_count << ".mfq";
        result.push_back(stable_path(path.parent_path() / name.str()));
    }
    return result;
}

struct ParsedFile {
    MfqSourceHeader header;
    std::vector<MfqStoredRecord> records;
};

ParsedFile parse_file(const std::filesystem::path& requested) {
    const auto path = stable_path(requested);
    const auto size = usable_file_size(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open MFQ file: " + path.string());
    Input input(stream, size, path);

    char magic[4]{};
    for (char& value : magic) value = input.scalar<char>("MFQ magic");
    if (std::memcmp(magic, "MFQ1", sizeof(magic)) != 0) {
        throw std::runtime_error("bad MFQ magic: " + path.string());
    }

    ParsedFile result;
    result.header.version = input.scalar<std::uint32_t>("MFQ version");
    if (result.header.version == 0 || result.header.version > 2) {
        throw std::runtime_error("unsupported MFQ version " +
                                 std::to_string(result.header.version) + ": " +
                                 path.string());
    }
    result.header.architecture = input.string("architecture");
    if (result.header.version >= 2) {
        const auto count = input.scalar<std::uint32_t>("metadata count");
        validate_count(count, kMaxMetadataEntries, 2 * sizeof(std::uint32_t),
                       input.remaining(), "metadata count", path);
        for (std::uint32_t index = 0; index < count; ++index) {
            auto key = input.string("metadata key");
            auto value = input.string("metadata value");
            if (!result.header.metadata.emplace(std::move(key), std::move(value)).second) {
                throw std::runtime_error("duplicate MFQ metadata key: " + path.string());
            }
        }
    }

    result.header.record_count = input.scalar<std::uint32_t>("record count");
    validate_count(result.header.record_count, kMaxRecordEntries,
                   2 * sizeof(std::uint32_t) + sizeof(std::uint64_t),
                   input.remaining(), "record count", path);
    result.records.reserve(result.header.record_count);
    for (std::uint32_t index = 0; index < result.header.record_count; ++index) {
        MfqStoredRecord record;
        record.tensor.name = input.string("record name");
        record.tensor.stored_dtype = input.string("record dtype");
        record.tensor.dtype = std::string(
            canonical_format_dtype(record.tensor.stored_dtype));
        record.tensor.nbytes = input.scalar<std::uint64_t>("record length");
        record.asset = record.tensor.name.compare(
                           0, kAssetPrefix.size(), kAssetPrefix) == 0;
        record.source_path = path;
        result.records.push_back(std::move(record));
    }
    auto offset = input.position();
    for (auto& record : result.records) {
        record.offset = offset;
        if (record.tensor.nbytes > size - offset) {
            throw std::runtime_error(
                "MFQ file length does not match record table: " + path.string());
        }
        offset += record.tensor.nbytes;
    }
    if (offset != size) {
        throw std::runtime_error(
            "MFQ file length does not match record table: " + path.string());
    }
    return result;
}

void read_exact(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::byte* destination,
    std::size_t size) {
    if (size == 0) return;
    if (offset > static_cast<std::uint64_t>(
                     std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<std::size_t>(
                   std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("model source byte range is too large");
    }
    std::ifstream stream(path, std::ios::binary);
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(reinterpret_cast<char*>(destination),
                static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("MFQ record source was truncated: " + path.string());
    }
}

} // namespace

struct MfqModelSource::Impl {
    MfqSourceHeader header;
    std::vector<std::filesystem::path> source_paths;
    std::vector<MfqStoredRecord> records;
    std::vector<TensorMetadata> tensors;
    std::vector<std::string> assets;
    MfqLegacyTensorAliases legacy_tensor_compatibility;
    std::unordered_map<std::string, std::size_t> records_by_name;
    std::unordered_map<std::string, std::size_t> tensors_by_name;
};

MfqModelSource::MfqModelSource(std::filesystem::path requested)
    : impl_(std::make_unique<Impl>()) {
    const auto initial_path = stable_path(requested);
    auto initial = parse_file(initial_path);
    const auto split_no = metadata_uint(initial.header, "split.no", 0);
    const auto split_count = metadata_uint(initial.header, "split.count", 1);
    if (split_count == 0 || split_count > 99999 || split_no >= split_count) {
        throw std::runtime_error("invalid MFQ split metadata: " +
                                 initial_path.string());
    }

    auto append = [&](ParsedFile parsed) {
        for (auto& record : parsed.records) {
            if (!impl_->records_by_name.emplace(
                    record.tensor.name, impl_->records.size()).second) {
                throw std::runtime_error("duplicate MFQ tensor record: " +
                                         record.source_path.string());
            }
            impl_->records.push_back(std::move(record));
        }
    };

    if (split_count == 1) {
        impl_->header = initial.header;
        impl_->source_paths.push_back(initial_path);
        append(std::move(initial));
    } else {
        const auto paths = shard_paths(initial_path, split_no, split_count);
        std::uint64_t actual_records = 0;
        std::uint64_t actual_tensors = 0;
        auto expected_records = std::numeric_limits<std::uint64_t>::max();
        auto expected_tensors = std::numeric_limits<std::uint64_t>::max();
        for (std::uint64_t index = 0; index < split_count; ++index) {
            auto current = parse_file(paths[static_cast<std::size_t>(index)]);
            if (current.header.version != initial.header.version ||
                current.header.architecture != initial.header.architecture ||
                metadata_uint(current.header, "split.no", split_count) != index ||
                metadata_uint(current.header, "split.count", 0) != split_count) {
                throw std::runtime_error("MFQ shard metadata mismatch: " +
                                         paths[static_cast<std::size_t>(index)].string());
            }
            if (index == 0) impl_->header = current.header;
            const auto records = metadata_uint(
                current.header, "split.records.count",
                std::numeric_limits<std::uint64_t>::max());
            const auto tensors = metadata_uint(
                current.header, "split.tensors.count",
                std::numeric_limits<std::uint64_t>::max());
            if (expected_records == std::numeric_limits<std::uint64_t>::max()) {
                expected_records = records;
            } else if (records != std::numeric_limits<std::uint64_t>::max() &&
                       records != expected_records) {
                throw std::runtime_error("MFQ shard record count mismatch");
            }
            if (expected_tensors == std::numeric_limits<std::uint64_t>::max()) {
                expected_tensors = tensors;
            } else if (tensors != std::numeric_limits<std::uint64_t>::max() &&
                       tensors != expected_tensors) {
                throw std::runtime_error("MFQ shard tensor count mismatch");
            }
            actual_records += current.header.record_count;
            for (const auto& record : current.records) {
                if (!record.asset) ++actual_tensors;
            }
            append(std::move(current));
        }
        if (expected_records != std::numeric_limits<std::uint64_t>::max() &&
            actual_records != expected_records) {
            throw std::runtime_error("MFQ shard record total mismatch");
        }
        if (expected_tensors != std::numeric_limits<std::uint64_t>::max() &&
            actual_tensors != expected_tensors) {
            throw std::runtime_error("MFQ shard tensor total mismatch");
        }
        impl_->source_paths = paths;
    }

    std::vector<std::string> stored_names;
    for (const auto& record : impl_->records) {
        if (record.asset) {
            impl_->assets.push_back(record.tensor.name);
        } else {
            stored_names.push_back(record.tensor.name);
        }
    }
    std::sort(impl_->assets.begin(), impl_->assets.end());

    std::unordered_map<std::string, std::string> stored_to_canonical;
    if (!has_asset(kModelGraphAsset) && has_asset(kModelConfigAsset)) {
        const auto config = read_asset(kModelConfigAsset);
        impl_->legacy_tensor_compatibility = make_legacy_tensor_aliases(
            architecture(),
            std::string_view(
                reinterpret_cast<const char*>(config.data()), config.size()),
            stored_names);
        for (const auto& [canonical, stored] :
             impl_->legacy_tensor_compatibility.canonical_to_stored) {
            const auto record = impl_->records_by_name.find(stored);
            if (record == impl_->records_by_name.end() ||
                    impl_->records[record->second].asset ||
                    impl_->records_by_name.find(canonical) !=
                        impl_->records_by_name.end()) {
                throw std::runtime_error(
                    "invalid legacy MFQ tensor alias: " + canonical);
            }
            const auto [found, inserted] =
                stored_to_canonical.emplace(stored, canonical);
            if (!inserted && found->second != canonical) {
                throw std::runtime_error(
                    "legacy MFQ tensor has multiple canonical names: " +
                    stored);
            }
        }
    }

    for (const auto& record : impl_->records) {
        if (record.asset) continue;
        auto tensor = record.tensor;
        if (const auto found = stored_to_canonical.find(tensor.name);
            found != stored_to_canonical.end()) {
            tensor.name = found->second;
        }
        if (!impl_->tensors_by_name.emplace(
                tensor.name, impl_->tensors.size()).second) {
            throw std::runtime_error(
                "legacy MFQ tensor aliases collide at canonical name: " +
                tensor.name);
        }
        impl_->tensors.push_back(std::move(tensor));
    }
}

MfqModelSource::~MfqModelSource() = default;
MfqModelSource::MfqModelSource(MfqModelSource&&) noexcept = default;
MfqModelSource& MfqModelSource::operator=(MfqModelSource&&) noexcept = default;

const MfqSourceHeader& MfqModelSource::header() const noexcept { return impl_->header; }
const std::vector<std::filesystem::path>& MfqModelSource::source_paths() const noexcept {
    return impl_->source_paths;
}
const std::vector<MfqStoredRecord>& MfqModelSource::records() const noexcept {
    return impl_->records;
}
std::string_view MfqModelSource::architecture() const noexcept {
    return impl_->header.architecture;
}
const std::unordered_map<std::string, std::string>&
MfqModelSource::metadata() const noexcept {
    return impl_->header.metadata;
}
const std::vector<TensorMetadata>& MfqModelSource::tensors() const noexcept {
    return impl_->tensors;
}

const TensorMetadata* MfqModelSource::find_tensor(std::string_view name) const noexcept {
    const auto found = impl_->tensors_by_name.find(std::string(name));
    return found == impl_->tensors_by_name.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

const MfqLegacyTensorAliases&
MfqModelSource::legacy_tensor_compatibility() const noexcept {
    return impl_->legacy_tensor_compatibility;
}

void MfqModelSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end()) {
        const auto alias = impl_->legacy_tensor_compatibility
            .canonical_to_stored.find(std::string(name));
        if (alias != impl_->legacy_tensor_compatibility
                .canonical_to_stored.end()) {
            found = impl_->records_by_name.find(alias->second);
        }
    }
    if (found == impl_->records_by_name.end() ||
            impl_->records[found->second].asset) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    const auto& record = impl_->records[found->second];
    if (relative_offset > record.tensor.nbytes ||
        size > record.tensor.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                std::string(name));
    }
    if (relative_offset > std::numeric_limits<std::uint64_t>::max() - record.offset) {
        throw std::overflow_error("model tensor byte offset overflow");
    }
    read_exact(record.source_path, record.offset + relative_offset, destination, size);
}

void MfqModelSource::drop_file_cache() const noexcept {
#if !defined(_WIN32) && defined(POSIX_FADV_DONTNEED)
    for (const auto& path : impl_->source_paths) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) continue;
        static_cast<void>(::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED));
        static_cast<void>(::close(fd));
    }
#endif
}

const std::vector<std::string>& MfqModelSource::assets() const noexcept {
    return impl_->assets;
}

bool MfqModelSource::has_asset(std::string_view name) const noexcept {
    const auto found = impl_->records_by_name.find(std::string(name));
    return found != impl_->records_by_name.end() &&
           impl_->records[found->second].asset;
}

std::vector<std::byte> MfqModelSource::read_asset(std::string_view name) const {
    const auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end() ||
        !impl_->records[found->second].asset) {
        throw std::runtime_error("model asset not found: " + std::string(name));
    }
    const auto& record = impl_->records[found->second];
    if (record.tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model asset is too large to read");
    }
    std::vector<std::byte> result(static_cast<std::size_t>(record.tensor.nbytes));
    read_exact(record.source_path, record.offset, result.data(), result.size());
    return result;
}

std::optional<ModelGraph> MfqModelSource::model_graph() const {
    if (!has_asset(kModelGraphAsset)) return std::nullopt;
    const auto bytes = read_asset(kModelGraphAsset);
    return ModelGraph::from_json(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace mfq
