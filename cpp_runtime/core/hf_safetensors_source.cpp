#include "mfq/hf_safetensors_source.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <utility>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mfq {
namespace {

using json = nlohmann::json;

constexpr std::string_view kAssetPrefix = "__mfq_asset__/";
constexpr std::uint64_t kMaxSafetensorsHeaderBytes = 100'000'000;
constexpr std::uint64_t kMaxJsonSidecarBytes = std::uint64_t{256} << 20;

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) >
                       std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("cannot size " + path.string());
    }
    std::vector<std::byte> result(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(result.size()));
    if (!stream && !result.empty()) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return result;
}

std::string read_text(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > kMaxJsonSidecarBytes) {
        throw std::runtime_error("JSON sidecar is too large: " + path.string());
    }
    const auto bytes = read_file(path);
    std::string result(bytes.size(), '\0');
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

json parse_json(std::string_view payload, const std::filesystem::path& path) {
    try {
        return json::parse(payload.begin(), payload.end());
    } catch (const json::exception& error) {
        throw std::runtime_error("invalid JSON in " + path.string() + ": " +
                                 error.what());
    }
}

std::uint64_t little_u64(const std::array<unsigned char, 8>& bytes) {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        result |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return result;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("Safetensors byte offset overflow");
    }
    return left + right;
}

std::vector<std::int64_t> parse_shape(
    const json& value,
    const std::string& name) {
    if (!value.is_array()) {
        throw std::runtime_error("Safetensors shape is not an array: " + name);
    }
    std::vector<std::int64_t> result;
    result.reserve(value.size());
    for (const auto& dimension : value) {
        if (!dimension.is_number_integer()) {
            throw std::runtime_error(
                "Safetensors shape has a non-integer dimension: " + name);
        }
        const auto size = dimension.get<std::int64_t>();
        if (size < 0) {
            throw std::runtime_error(
                "Safetensors shape has a negative dimension: " + name);
        }
        result.push_back(size);
    }
    return result;
}

bool inside(const std::filesystem::path& root, const std::filesystem::path& path) {
    auto parent = root.begin();
    auto child = path.begin();
    for (; parent != root.end(); ++parent, ++child) {
        if (child == path.end() || *child != *parent) return false;
    }
    return true;
}

std::filesystem::path shard_path(
    const std::filesystem::path& root,
    const std::string& name) {
    std::error_code error;
    const auto path = std::filesystem::weakly_canonical(root / name, error);
    if (error || !inside(root, path) ||
        !std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error("invalid Safetensors shard path: " + name);
    }
    return path;
}

} // namespace

struct HfSafetensorsSource::Impl {
    struct ShardReader {
        ShardReader(std::filesystem::path source, std::uint64_t source_size)
            : path(std::move(source)), size(source_size) {}

        ~ShardReader() {
#if !defined(_WIN32)
            if (descriptor >= 0) ::close(descriptor);
#endif
        }

        void read(
            std::uint64_t offset,
            std::byte* destination,
            std::size_t count) const {
            if (count == 0) return;
            if (offset > size || count > size - offset) {
                throw std::out_of_range(
                    "Safetensors byte range is outside shard " + path.string());
            }
#if defined(_WIN32)
            if (offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max()) ||
                count > static_cast<std::size_t>(
                            std::numeric_limits<std::streamsize>::max())) {
                throw std::runtime_error("Safetensors byte range is too large");
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (!stream.is_open()) stream.open(path, std::ios::binary);
            stream.clear();
            stream.seekg(static_cast<std::streamoff>(offset));
            stream.read(reinterpret_cast<char*>(destination),
                        static_cast<std::streamsize>(count));
            if (!stream) {
                throw std::runtime_error(
                    "unexpected EOF while reading " + path.string());
            }
#else
            const auto max_offset = static_cast<std::uint64_t>(
                std::numeric_limits<off_t>::max());
            if (offset > max_offset || count > max_offset - offset) {
                throw std::runtime_error("Safetensors byte range is too large");
            }
            const auto file = open();
            std::size_t done = 0;
            while (done < count) {
                constexpr std::size_t kMaxReadBytes = std::size_t{32} << 20;
                const auto request = std::min(count - done, kMaxReadBytes);
                const auto result = ::pread(
                    file,
                    destination + done,
                    request,
                    static_cast<off_t>(offset + done));
                if (result < 0) {
                    if (errno == EINTR) continue;
                    throw std::system_error(
                        errno, std::generic_category(),
                        "pread " + path.string());
                }
                if (result == 0) {
                    throw std::runtime_error(
                        "unexpected EOF while reading " + path.string());
                }
                done += static_cast<std::size_t>(result);
            }
#endif
        }

#if !defined(_WIN32)
        int open() const {
            std::lock_guard<std::mutex> lock(mutex);
            if (descriptor < 0) {
                descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (descriptor < 0) {
                    throw std::system_error(
                        errno, std::generic_category(),
                        "open " + path.string());
                }
#if defined(__APPLE__) && defined(F_NOCACHE)
                // Raw-HF tensors are copied into final runtime buffers or
                // consumed through bounded row/expert stores. Retaining the
                // same payload in the macOS file cache creates a transient
                // second model copy and can compress live GPU allocations.
                if (::fcntl(descriptor, F_NOCACHE, 1) != 0) {
                    const auto error = errno;
                    ::close(descriptor);
                    descriptor = -1;
                    throw std::system_error(
                        error,
                        std::generic_category(),
                        "F_NOCACHE " + path.string());
                }
#endif
            }
            return descriptor;
        }
#endif

        std::filesystem::path path;
        std::uint64_t size;
        mutable std::mutex mutex;
#if defined(_WIN32)
        mutable std::ifstream stream;
#else
        mutable int descriptor = -1;
#endif
    };

    struct PhysicalTensor {
        TensorMetadata metadata;
        HfTensorLocation location;
    };

    std::filesystem::path root;
    std::string architecture;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<std::filesystem::path> source_paths;
    std::vector<std::unique_ptr<ShardReader>> readers;
    std::unordered_map<std::string, PhysicalTensor> physical;
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, std::size_t> tensor_indices;
    std::unordered_map<std::string, std::string> exposed_to_source;
    std::unordered_map<std::string, std::filesystem::path> asset_paths;
    std::unordered_map<std::string, std::vector<std::byte>> inline_assets;
    std::vector<std::string> assets;
};

HfSafetensorsSource::HfSafetensorsSource(
    std::filesystem::path requested_root,
    std::unordered_map<std::string, std::string> canonical_to_source)
    : impl_(std::make_unique<Impl>()) {
    std::error_code error;
    impl_->root = std::filesystem::canonical(std::move(requested_root), error);
    if (error || !std::filesystem::is_directory(impl_->root, error) || error) {
        throw std::runtime_error("cannot open HF model directory: " +
                                 impl_->root.string());
    }

    const auto index_path = impl_->root / "model.safetensors.index.json";
    std::unordered_map<std::string, std::vector<std::string>> names_by_shard;
    std::vector<std::string> shard_names;
    if (std::filesystem::is_regular_file(index_path)) {
        const auto index = parse_json(read_text(index_path), index_path);
        const auto weight_map = index.find("weight_map");
        if (weight_map == index.end() || !weight_map->is_object() ||
            weight_map->empty()) {
            throw std::runtime_error(
                "Safetensors index requires a non-empty object weight_map: " +
                index_path.string());
        }
        for (const auto& [name, shard] : weight_map->items()) {
            if (!shard.is_string()) {
                throw std::runtime_error(
                    "Safetensors weight_map value is not a string: " + name);
            }
            names_by_shard[shard.get<std::string>()].push_back(name);
        }
        for (const auto& [name, unused] : names_by_shard) {
            static_cast<void>(unused);
            shard_names.push_back(name);
        }
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(impl_->root)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".safetensors") {
                shard_names.push_back(entry.path().filename().string());
            }
        }
        if (shard_names.empty()) {
            throw std::runtime_error("no Safetensors weights found under " +
                                     impl_->root.string());
        }
    }
    std::sort(shard_names.begin(), shard_names.end());

    for (const auto& shard_name : shard_names) {
        const auto path = shard_path(impl_->root, shard_name);
        const auto size = std::filesystem::file_size(path);
        if (size < 8) {
            throw std::runtime_error(
                "cannot read Safetensors header size: " + path.string());
        }
        std::ifstream stream(path, std::ios::binary);
        std::array<unsigned char, 8> raw_header_size{};
        stream.read(reinterpret_cast<char*>(raw_header_size.data()), 8);
        if (!stream) {
            throw std::runtime_error(
                "cannot read Safetensors header size: " + path.string());
        }
        const auto header_size = little_u64(raw_header_size);
        if (header_size > size - 8 ||
            header_size > kMaxSafetensorsHeaderBytes ||
            header_size > static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("invalid Safetensors header size: " +
                                     path.string());
        }
        std::string header(static_cast<std::size_t>(header_size), '\0');
        stream.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (!stream && !header.empty()) {
            throw std::runtime_error("cannot read Safetensors header: " +
                                     path.string());
        }
        const auto metadata = parse_json(header, path);
        const auto shard_index = impl_->source_paths.size();
        impl_->source_paths.push_back(path);
        impl_->readers.push_back(
            std::make_unique<Impl::ShardReader>(path, size));
        const auto payload_base = checked_add(8, header_size);

        std::vector<std::string> names;
        const auto indexed = names_by_shard.find(shard_name);
        if (indexed != names_by_shard.end()) {
            names = indexed->second;
        } else {
            for (const auto& [name, unused] : metadata.items()) {
                static_cast<void>(unused);
                if (name != "__metadata__") names.push_back(name);
            }
        }
        for (const auto& name : names) {
            const auto found = metadata.find(name);
            if (found == metadata.end() || !found->is_object()) {
                throw std::runtime_error(
                    "Safetensors index references a missing tensor: " + name);
            }
            const auto dtype = found->find("dtype");
            const auto shape = found->find("shape");
            const auto offsets = found->find("data_offsets");
            if (dtype == found->end() || !dtype->is_string() ||
                shape == found->end() || offsets == found->end() ||
                !offsets->is_array() || offsets->size() != 2 ||
                !(*offsets)[0].is_number_unsigned() ||
                !(*offsets)[1].is_number_unsigned()) {
                throw std::runtime_error(
                    "invalid Safetensors tensor metadata: " + name);
            }
            const auto begin = (*offsets)[0].get<std::uint64_t>();
            const auto end = (*offsets)[1].get<std::uint64_t>();
            if (end < begin || checked_add(payload_base, end) > size) {
                throw std::runtime_error(
                    "Safetensors tensor range is outside its shard: " + name);
            }
            Impl::PhysicalTensor tensor;
            tensor.metadata.name = name;
            tensor.metadata.dtype = dtype->get<std::string>();
            tensor.metadata.stored_dtype = tensor.metadata.dtype;
            tensor.metadata.shape = parse_shape(*shape, name);
            tensor.metadata.nbytes = end - begin;
            tensor.location = {shard_index, checked_add(payload_base, begin)};
            if (!impl_->physical.emplace(name, std::move(tensor)).second) {
                throw std::runtime_error("duplicate Safetensors tensor: " + name);
            }
        }
    }

    if (canonical_to_source.empty()) {
        for (const auto& [name, unused] : impl_->physical) {
            static_cast<void>(unused);
            canonical_to_source.emplace(name, name);
        }
    }
    std::vector<std::pair<std::string, std::string>> aliases(
        canonical_to_source.begin(), canonical_to_source.end());
    std::sort(aliases.begin(), aliases.end());
    std::unordered_set<std::string> claimed_sources;
    for (const auto& [canonical, source] : aliases) {
        const auto found = impl_->physical.find(source);
        if (canonical.empty() || source.empty() || found == impl_->physical.end()) {
            throw std::runtime_error("invalid HF canonical tensor mapping: " +
                                     canonical);
        }
        if (!claimed_sources.insert(source).second ||
            !impl_->exposed_to_source.emplace(canonical, source).second) {
            throw std::runtime_error("ambiguous HF canonical tensor mapping: " +
                                     canonical);
        }
        auto metadata = found->second.metadata;
        metadata.name = canonical;
        impl_->tensor_indices.emplace(canonical, impl_->tensors.size());
        impl_->tensors.push_back(std::move(metadata));
    }

    const auto add_asset = [&](std::string name, const std::filesystem::path& path) {
        std::error_code asset_error;
        const auto resolved = std::filesystem::canonical(path, asset_error);
        if (asset_error || !std::filesystem::is_regular_file(resolved, asset_error) ||
            asset_error || std::filesystem::file_size(resolved, asset_error) == 0 ||
            asset_error) {
            throw std::runtime_error("invalid native HF runtime asset: " +
                                     path.string());
        }
        if (impl_->inline_assets.find(name) != impl_->inline_assets.end() ||
            !impl_->asset_paths.emplace(name, resolved).second) {
            throw std::runtime_error("duplicate native HF runtime asset: " +
                                     path.string());
        }
        impl_->assets.push_back(std::move(name));
    };
    const auto add_inline_asset = [&](std::string name, std::string payload) {
        std::vector<std::byte> bytes(payload.size());
        if (!payload.empty()) {
            std::memcpy(bytes.data(), payload.data(), payload.size());
        }
        if (impl_->asset_paths.find(name) != impl_->asset_paths.end() ||
            !impl_->inline_assets.emplace(name, std::move(bytes)).second) {
            throw std::runtime_error("duplicate native HF inline asset: " + name);
        }
        impl_->assets.push_back(std::move(name));
    };

    const auto config = impl_->root / "config.json";
    if (std::filesystem::is_regular_file(config)) {
        auto parsed = parse_json(read_text(config), config);
        const auto inference_config = impl_->root / "inference" / "config.json";
        if (std::filesystem::is_regular_file(inference_config)) {
            const auto supplemental = parse_json(
                read_text(inference_config), inference_config);
            if (!parsed.is_object() || !supplemental.is_object()) {
                throw std::runtime_error(
                    "HF model configuration files must contain JSON objects");
            }
            auto* target = &parsed;
            if (const auto text = parsed.find("text_config");
                text != parsed.end() && text->is_object()) {
                target = &*text;
            }
            const auto* source = &supplemental;
            if (const auto text = supplemental.find("text_config");
                text != supplemental.end() && text->is_object()) {
                source = &*text;
            }
            for (const auto& [key, value] : source->items()) {
                if (!target->contains(key)) (*target)[key] = value;
            }
        }
        const auto model_type = parsed.find("model_type");
        if (model_type == parsed.end() || !model_type->is_string() ||
            model_type->get_ref<const std::string&>().empty()) {
            throw std::runtime_error("HF config.json has no model_type");
        }
        impl_->architecture = model_type->get<std::string>() + "-hf-full-mfq";
        impl_->metadata.emplace("source.format", "hf-safetensors");
        impl_->metadata.emplace("source.precision", "native");
        add_inline_asset(std::string(kModelConfigAsset), parsed.dump());
    }
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 4>
        sidecars{{
            {"tokenizer.json", "hf/tokenizer.json"},
            {"tokenizer_config.json", "hf/tokenizer_config.json"},
            {"chat_template.jinja", "hf/chat_template.jinja"},
            {"generation_config.json", "hf/generation_config.json"},
        }};
    for (const auto& [filename, asset_name] : sidecars) {
        const auto path = impl_->root / filename;
        if (std::filesystem::is_regular_file(path)) {
            add_asset(std::string(kAssetPrefix) + std::string(asset_name), path);
        }
    }
    const auto generation_path = impl_->root / "generation_config.json";
    if (std::filesystem::is_regular_file(generation_path)) {
        const auto generation = parse_json(
            read_text(generation_path), generation_path);
        if (!generation.is_object()) {
            throw std::runtime_error(
                "HF generation_config.json must be an object");
        }
        static constexpr std::array<
            std::pair<std::string_view, std::string_view>, 7> aliases{{
            {"max_new_tokens", "max_tokens"},
            {"temperature", "temperature"},
            {"top_k", "top_k"},
            {"top_p", "top_p"},
            {"presence_penalty", "presence_penalty"},
            {"frequency_penalty", "frequency_penalty"},
            {"repetition_penalty", "repetition_penalty"},
        }};
        json chat = json::object();
        for (const auto& [source, target] : aliases) {
            const auto found = generation.find(std::string(source));
            if (found != generation.end() && !found->is_null()) {
                chat[std::string(target)] = *found;
            }
        }
        if (!chat.empty()) {
            impl_->metadata.emplace(
                "runtime.sampling.v1",
                json({
                    {"schema", "mfq.runtime.sampling"},
                    {"version", 1},
                    {"chat", std::move(chat)},
                    {"provenance", {{"source", "hf:generation_config.json"}}},
                }).dump());
        }
    }
    const auto add_directory = [&](const std::filesystem::path& directory) {
        std::error_code directory_error;
        if (!std::filesystem::is_directory(directory, directory_error) ||
            directory_error) return;
        const auto resolved = std::filesystem::canonical(directory, directory_error);
        if (directory_error) {
            throw std::runtime_error("cannot resolve native HF runtime asset directory: " +
                                     directory.string());
        }
        for (std::filesystem::recursive_directory_iterator iterator(
                 resolved, directory_error), end;
             !directory_error && iterator != end;
             iterator.increment(directory_error)) {
            if (!iterator->is_regular_file(directory_error) || directory_error) continue;
            const auto relative = std::filesystem::relative(
                iterator->path(), resolved, directory_error);
            if (directory_error || relative.empty() || relative.is_absolute() ||
                *relative.begin() == "..") {
                throw std::runtime_error("invalid native HF runtime asset path: " +
                                         iterator->path().string());
            }
            add_asset(std::string(kAssetPrefix) + relative.generic_string(),
                      iterator->path());
        }
        if (directory_error) {
            throw std::runtime_error("cannot enumerate native HF runtime assets: " +
                                     resolved.string());
        }
    };
    add_directory(impl_->root / ".mfq-assets");
    if (const auto* configured = std::getenv("MFQ_RUNTIME_ASSET_DIRECTORY");
        configured != nullptr && *configured != '\0') {
        add_directory(configured);
    }
    std::sort(impl_->assets.begin(), impl_->assets.end());
}

HfSafetensorsSource::~HfSafetensorsSource() = default;
HfSafetensorsSource::HfSafetensorsSource(HfSafetensorsSource&&) noexcept = default;
HfSafetensorsSource& HfSafetensorsSource::operator=(
    HfSafetensorsSource&&) noexcept = default;

const std::filesystem::path& HfSafetensorsSource::root() const noexcept {
    return impl_->root;
}
const std::vector<std::filesystem::path>&
HfSafetensorsSource::source_paths() const noexcept {
    return impl_->source_paths;
}
void HfSafetensorsSource::drop_file_cache() const noexcept {
#if !defined(_WIN32) && defined(POSIX_FADV_DONTNEED)
    for (const auto& reader : impl_->readers) {
        try {
            const auto descriptor = reader->open();
            static_cast<void>(::posix_fadvise(
                descriptor,
                0,
                static_cast<off_t>(reader->size),
                POSIX_FADV_DONTNEED));
        } catch (...) {
        }
    }
#endif
}
std::string_view HfSafetensorsSource::architecture() const noexcept {
    return impl_->architecture;
}
const std::unordered_map<std::string, std::string>&
HfSafetensorsSource::metadata() const noexcept {
    return impl_->metadata;
}
const std::vector<TensorMetadata>& HfSafetensorsSource::tensors() const noexcept {
    return impl_->tensors;
}

const TensorMetadata* HfSafetensorsSource::find_tensor(
    std::string_view name) const noexcept {
    const auto found = impl_->tensor_indices.find(std::string(name));
    return found == impl_->tensor_indices.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

const HfTensorLocation& HfSafetensorsSource::location(
    std::string_view name) const {
    const auto alias = impl_->exposed_to_source.find(std::string(name));
    if (alias == impl_->exposed_to_source.end()) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    return impl_->physical.at(alias->second).location;
}

void HfSafetensorsSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    const auto alias = impl_->exposed_to_source.find(std::string(name));
    if (alias == impl_->exposed_to_source.end()) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    const auto& source = impl_->physical.at(alias->second);
    if (relative_offset > source.metadata.nbytes ||
        size > source.metadata.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                std::string(name));
    }
    read_shard_range_into(source.location.shard,
                          source.location.offset + relative_offset,
                          destination, size);
}

void HfSafetensorsSource::read_shard_range_into(
    std::size_t shard,
    std::uint64_t offset,
    std::byte* destination,
    std::size_t size) const {
    if (shard >= impl_->source_paths.size()) {
        throw std::out_of_range("Safetensors shard index out of range");
    }
    impl_->readers[shard]->read(offset, destination, size);
}

const std::vector<std::string>& HfSafetensorsSource::assets() const noexcept {
    return impl_->assets;
}

bool HfSafetensorsSource::has_asset(std::string_view name) const noexcept {
    const auto key = std::string(name);
    return impl_->inline_assets.find(key) != impl_->inline_assets.end() ||
        impl_->asset_paths.find(key) != impl_->asset_paths.end();
}

std::vector<std::byte> HfSafetensorsSource::read_asset(
    std::string_view name) const {
    const auto key = std::string(name);
    if (const auto inline_asset = impl_->inline_assets.find(key);
        inline_asset != impl_->inline_assets.end()) {
        return inline_asset->second;
    }
    const auto found = impl_->asset_paths.find(key);
    if (found == impl_->asset_paths.end()) {
        throw std::runtime_error("model asset not found: " + std::string(name));
    }
    return read_file(found->second);
}

std::optional<ModelGraph> HfSafetensorsSource::model_graph() const {
    if (!has_asset(kModelGraphAsset)) return std::nullopt;
    const auto bytes = read_asset(kModelGraphAsset);
    return ModelGraph::from_json(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace mfq
