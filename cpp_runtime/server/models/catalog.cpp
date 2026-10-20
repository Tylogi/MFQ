#include "catalog.h"
#include "api/errors.h"
#include "instance.h"
#include "storage/files.h"
#include "mfq/hf_safetensors_source.h"
#include "mfq/mfq_model_source.h"
#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

namespace mfq::server {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
constexpr std::size_t max_entries = 100000;
constexpr std::size_t max_index_bytes = 4 * 1024 * 1024;

std::string hash(const std::string& value) {
    return mfq::cache::block_hash_hex(mfq::cache::sha256(value)).substr(0, 32);
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool is_mfq(const fs::path& path) { return lowercase(path.extension().u8string()) == ".mfq"; }

bool is_hf(const fs::path& path) {
    std::error_code error;
    if (!fs::is_regular_file(path / "config.json", error)) return false;
    if (fs::is_regular_file(path / "model.safetensors.index.json", error)) return true;
    for (const auto& entry : fs::directory_iterator(path, fs::directory_options::skip_permission_denied, error))
        if (entry.is_regular_file(error) && entry.path().extension() == ".safetensors") return true;
    return false;
}

fs::path model_key(const fs::path& path) {
    if (!is_mfq(path)) return path;
    static const std::regex shard(R"(^(.*)-([0-9]{5})-of-([0-9]{5})$)");
    std::smatch match;
    const auto stem = path.stem().u8string();
    return std::regex_match(stem, match, shard) ? path.parent_path() / fs::u8path(match[1].str() + ".mfq") : path;
}

std::set<fs::path> discover(const fs::path& root) {
    std::set<fs::path> paths;
    std::error_code error;
    if (!fs::is_directory(root, error)) return paths;
    if (is_hf(root)) paths.insert(root);
    std::size_t visited = 0;
    for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, error);
            it != fs::recursive_directory_iterator(); it.increment(error)) {
        if (error) throw ApiError(500, "model_scan_failed", "could not scan model directory");
        if (++visited > max_entries || it.depth() > 32)
            throw ApiError(413, "model_catalog_too_large", "model directory exceeds scan limits");
        if (!it->is_regular_file(error)) continue;
        const auto& path = it->path();
        if (is_mfq(path)) paths.insert(fs::canonical(path));
        else if (path.filename() == "config.json" && is_hf(path.parent_path())) paths.insert(fs::canonical(path.parent_path()));
    }
    return paths;
}

std::string modified_at(fs::file_time_type value) {
    const auto system = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        value - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    const auto seconds = std::chrono::system_clock::to_time_t(system);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    std::ostringstream result;
    result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return result.str();
}

json context_capacity(const json& config) {
    for (const auto* source : {&config, config.contains("text_config") ? &config["text_config"] : &config,
            config.contains("language_config") ? &config["language_config"] : &config}) {
        if (!source->is_object()) continue;
        for (const auto* key : {"max_position_embeddings", "max_seq_len", "max_seq_length", "seq_length", "n_positions"}) {
            const auto found = source->find(key);
            if (found != source->end() && found->is_number_integer() && *found > 0 && *found <= 2147483647) return *found;
        }
    }
    return nullptr;
}

json inspect(const fs::path& path, const std::vector<fs::path>& group) {
    const bool hf = fs::is_directory(path);
    const auto name = hf ? path.filename().u8string() : model_key(path).stem().u8string();
    json result{{"id", hash(path.u8string())}, {"name", name}, {"architecture", "unknown"}, {"format", hf ? "hf" : "mfq"},
        {"shard_count", group.size()}, {"missing_shards", 0}, {"total_bytes", 0}, {"tensor_count", 0}, {"record_count", 0},
        {"dtypes", json::array()}, {"context_capacity", nullptr}, {"complete", false}, {"loadable", false}, {"error", nullptr},
        {"modified_at", modified_at(fs::last_write_time(path))}};
    if (!hf) {
        for (const auto& file : group) result["total_bytes"] = result["total_bytes"].get<std::uint64_t>() + fs::file_size(file);
        static const std::regex shard(R"(^.*-([0-9]{5})-of-([0-9]{5})$)");
        std::smatch match;
        const auto stem = path.stem().u8string();
        if (std::regex_match(stem, match, shard)) {
            const auto expected = std::stoul(match[2]);
            result["shard_count"] = expected;
            result["missing_shards"] = expected > group.size() ? expected - group.size() : 0;
        }
    }
    try {
        std::shared_ptr<const mfq::ModelSource> source = hf
            ? std::shared_ptr<const mfq::ModelSource>(std::make_shared<mfq::HfSafetensorsSource>(path))
            : std::make_shared<mfq::MfqModelSource>(path, mfq::FileReadMode::Buffered);
        auto architecture = std::string(source->architecture());
        result["architecture"] = architecture.empty() ? "unknown" : architecture;
        constexpr std::string_view suffix = "-hf-full-mfq";
        if (hf && architecture.size() >= suffix.size()) architecture.resize(architecture.size() - suffix.size());
        std::string fingerprint = architecture + '\0' + name;
        std::uint64_t bytes = 0;
        auto latest = fs::file_time_type::min();
        for (const auto& file : source->source_paths()) {
            const auto size = fs::file_size(file);
            bytes += size;
            latest = std::max(latest, fs::last_write_time(file));
            fingerprint += '\0' + file.filename().u8string() + ':' + std::to_string(size);
        }
        std::set<std::string> dtypes;
        for (const auto& tensor : source->tensors()) dtypes.insert(tensor.stored_dtype.empty() ? tensor.dtype : tensor.stored_dtype);
        if (!hf && !source->assets().empty()) dtypes.insert("BLOB");
        const auto config = source->has_asset(mfq::kModelConfigAsset) ? json::parse(source->model_config_json()) : json::object();
        if (!config.is_object()) throw std::runtime_error("model configuration must be an object");
        result.update({{"id", hash(fingerprint)}, {"total_bytes", bytes}, {"shard_count", source->source_paths().size()},
            {"missing_shards", 0}, {"tensor_count", source->tensors().size()}, {"record_count", source->tensors().size() + (hf ? 1 : source->assets().size())},
            {"dtypes", dtypes}, {"context_capacity", context_capacity(config)}, {"modified_at", modified_at(latest)}, {"complete", true},
            {"error", "native model execution is not connected yet"}});
    } catch (const std::exception& error) {
        result["error"] = "model metadata is invalid or incomplete";
        std::cerr << json{{"event", "model_inspection_failed"}, {"model", name}, {"detail", error.what()}}.dump() << '\n';
    }
    return result;
}

bool valid_id(const std::string& id) {
    return id.size() == 32 && std::all_of(id.begin(), id.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

} // namespace

ModelCatalog::ModelCatalog(std::vector<fs::path> roots) : secret_(normalize_uuid({})) {
    for (const auto& root : roots) {
        const auto path = resolved_path(root);
        if (std::find(roots_.begin(), roots_.end(), path) == roots_.end()) roots_.push_back(path);
        if (fs::exists(path) && !fs::is_directory(path)) throw std::invalid_argument("--model-dir must be a directory");
    }
    fs::create_directories(roots_.at(0));
}

std::set<fs::path> ModelCatalog::registered_paths() const {
    std::set<fs::path> result;
    for (const auto& root : roots_) {
        const auto index = root / ".mfq-files.json";
        if (!fs::exists(index)) continue;
        try {
            if (fs::file_size(index) > max_index_bytes) throw std::runtime_error("index too large");
            std::ifstream stream(index);
            const auto document = json::parse(stream);
            if (!document.is_object() || document.value("version", 0) != 1 || !document.at("files").is_array() || document["files"].size() > 10000)
                throw std::runtime_error("invalid index");
            for (const auto& item : document["files"]) {
                if (!item.is_string()) throw std::runtime_error("invalid index path");
                const auto value = item.get<std::string>();
                if (value.empty() || value.size() > 4096 || value.find('\0') != std::string::npos)
                    throw std::runtime_error("invalid index path");
                const auto path = fs::u8path(value);
                if (!path.is_absolute()) throw std::runtime_error("index paths must be absolute");
                result.insert(path);
            }
        } catch (...) { throw ApiError(500, "model_catalog_index_invalid", "model catalog index is invalid; existing file was preserved"); }
    }
    return result;
}

std::vector<ModelCatalog::Artifact> ModelCatalog::scan(const std::set<fs::path>& registered) const {
    auto paths = registered;
    for (const auto& root : roots_) {
        const auto found = discover(root);
        paths.insert(found.begin(), found.end());
    }
    std::map<fs::path, std::vector<fs::path>> groups;
    for (const auto& path : paths) {
        std::error_code error;
        if ((fs::is_directory(path, error) && is_hf(path)) || (is_mfq(path) && fs::is_regular_file(path, error))) {
            const auto canonical = fs::canonical(path);
            auto& group = groups[model_key(canonical)];
            if (std::find(group.begin(), group.end(), canonical) == group.end()) group.push_back(canonical);
        }
    }
    std::map<std::string, Artifact> names;
    std::set<std::pair<std::string, std::string>> representations;
    for (auto& [key, group] : groups) {
        std::sort(group.begin(), group.end());
        Artifact artifact{group.front(), inspect(group.front(), group)};
        const auto name = artifact.resource.at("name").get<std::string>();
        if (!representations.emplace(name, artifact.resource.at("format").get<std::string>()).second)
            throw ApiError(409, "duplicate_model_name", "multiple model artifacts have the same name");
        const auto previous = names.find(name);
        if (previous != names.end()) {
            if (artifact.resource["format"] == "hf") continue;
        }
        names.insert_or_assign(name, std::move(artifact));
    }
    std::vector<Artifact> result;
    for (auto& [name, artifact] : names) result.push_back(std::move(artifact));
    return result;
}

void ModelCatalog::refresh() {
    auto next = scan(registered_paths());
    artifacts_ = std::move(next);
    scanned_ = true;
    scanned_at_ = std::chrono::steady_clock::now();
}

json ModelCatalog::list(bool requested) {
    std::lock_guard<std::mutex> guard(mutex_);
    // shortcut: refresh scans on HTTP workers; move scans off workers when slow volumes require it.
    if (requested || !scanned_ || std::chrono::steady_clock::now() - scanned_at_ >= std::chrono::seconds(5)) refresh();
    auto data = json::array();
    for (const auto& item : artifacts_) data.push_back(item.resource);
    return {{"data", std::move(data)}};
}

std::string ModelCatalog::directory_id(const fs::path& path) {
    const auto id = hash(secret_ + path.u8string());
    if (directories_.size() >= max_entries && !directories_.count(id))
        throw ApiError(413, "model_catalog_too_large", "too many browsed model directories");
    directories_[id] = path;
    return id;
}

fs::path ModelCatalog::directory(const std::string& id, const std::string& path) const {
    if (!id.empty() && !valid_id(id)) throw ApiError(400, "invalid_model_directory_source", "invalid directory identifier");
    if (!id.empty() && !path.empty()) throw ApiError(400, "invalid_model_directory_source", "directory_id and path cannot be used together");
    if (path.size() > 4096 || path.find('\0') != std::string::npos)
        throw ApiError(400, "invalid_model_directory_source", "invalid directory path");
    fs::path selected;
    if (!path.empty()) selected = resolved_path(fs::u8path(path));
    else {
        const auto found = directories_.find(id);
        if (found != directories_.end()) selected = found->second;
    }
    std::error_code error;
    if (selected.empty() || !fs::is_directory(selected, error))
        throw ApiError(404, "model_directory_not_found", "model directory is unavailable");
    return fs::canonical(selected);
}

json ModelCatalog::directory_entry(const fs::path& path) {
    std::set<fs::path> models;
    if (is_hf(path)) models.insert(path);
    std::size_t visited = 0;
    for (const auto& item : fs::directory_iterator(path, fs::directory_options::skip_permission_denied)) {
        if (++visited > max_entries) throw ApiError(413, "model_catalog_too_large", "directory exceeds scan limits");
        std::error_code error;
        if (item.is_regular_file(error) && is_mfq(item.path())) models.insert(model_key(item.path()));
        else if (item.is_directory(error) && is_hf(item.path())) models.insert(item.path());
    }
    auto name = path.filename().u8string();
    if (name.empty()) name = path.root_path().u8string();
    return {{"id", directory_id(path)}, {"name", name}, {"model_file_count", models.size()}};
}

json ModelCatalog::browse(const fs::path& path) {
    const auto current = directory_entry(path);
    json result{{"current_id", current["id"]}, {"current_name", current["name"]}, {"current_path", path.u8string()},
        {"parent_id", nullptr}, {"model_file_count", current["model_file_count"]}, {"files", json::array()},
        {"can_open_directory", false}, {"can_open_in_finder", false}, {"data", json::array()}};
    if (path.parent_path() != path) result["parent_id"] = directory_id(path.parent_path());
    std::map<std::string, fs::path> children;
    for (const auto& entry : fs::directory_iterator(path, fs::directory_options::skip_permission_denied)) {
        std::error_code error;
        if (entry.is_directory(error)) children.emplace(entry.path().filename().u8string(), fs::canonical(entry.path()));
        else if (entry.is_regular_file(error) && is_mfq(entry.path()))
            result["files"].push_back({{"name", entry.path().filename().u8string()}, {"byte_size", entry.file_size()}});
    }
    for (const auto& [name, child] : children) result["data"].push_back(directory_entry(child));
    std::sort(result["files"].begin(), result["files"].end(), [](const json& a, const json& b) { return a["name"] < b["name"]; });
    return result;
}

json ModelCatalog::browse(const std::string& id, const std::string& path) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!id.empty() || !path.empty()) return browse(directory(id, path));
    auto data = json::array();
    for (const auto& root : roots_) if (fs::is_directory(root)) data.push_back(directory_entry(root));
    return {{"current_id", nullptr}, {"current_name", nullptr}, {"current_path", nullptr}, {"parent_id", nullptr},
        {"model_file_count", 0}, {"files", json::array()}, {"can_open_directory", false}, {"can_open_in_finder", false}, {"data", std::move(data)}};
}

json ModelCatalog::register_directory(const json& request) {
    if (!request.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    for (const auto& item : request.items())
        if ((item.key() != "path" && item.key() != "directory_id") || (!item.value().is_null() &&
                (!item.value().is_string() || item.value().get_ref<const std::string&>().empty())))
            throw ApiError(400, "invalid_request", "invalid model directory source");
    const auto id = request.contains("directory_id") && !request["directory_id"].is_null() ? request["directory_id"].get<std::string>() : "";
    const auto path = request.contains("path") && !request["path"].is_null() ? request["path"].get<std::string>() : "";
    if (id.empty() == path.empty()) throw ApiError(400, "invalid_model_directory_source", "exactly one directory source is required");
    std::lock_guard<std::mutex> guard(mutex_);
    const auto selected = directory(id, path);
    const auto added = discover(selected);
    if (added.empty()) throw ApiError(400, "model_registration_failed", "selected directory contains no supported models");
    auto paths = registered_paths();
    paths.insert(added.begin(), added.end());
    if (paths.size() > 10000) throw ApiError(413, "model_catalog_too_large", "too many registered model files");
    auto files = json::array();
    for (const auto& file : paths) files.push_back(file.u8string());
    const json document{{"version", 1}, {"files", std::move(files)}};
    if (document.dump().size() > max_index_bytes)
        throw ApiError(413, "model_catalog_too_large", "model catalog index exceeds size limit");
    auto next = scan(paths);
    try { write_json(roots_.front() / ".mfq-files.json", document); }
    catch (...) { throw ApiError(500, "model_registration_save_failed", "could not persist model registration"); }
    artifacts_ = std::move(next);
    scanned_ = true;
    scanned_at_ = std::chrono::steady_clock::now();
    auto data = json::array();
    for (const auto& artifact : artifacts_)
        if (std::mismatch(selected.begin(), selected.end(), artifact.path.begin(), artifact.path.end()).first == selected.end())
            data.push_back(artifact.resource);
    return {{"data", std::move(data)}};
}

json ModelCatalog::model_directory(const std::string& model_id) {
    if (!valid_id(model_id)) throw ApiError(400, "invalid_model_id", "invalid model identifier");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!scanned_) refresh();
    auto find = [&] { return std::find_if(artifacts_.begin(), artifacts_.end(), [&](const Artifact& item) { return item.resource["id"] == model_id; }); };
    auto found = find();
    if (found == artifacts_.end()) { refresh(); found = find(); }
    if (found == artifacts_.end()) throw ApiError(404, "model_artifact_not_found", "model artifact is unavailable");
    const auto path = found->resource["format"] == "hf" ? found->path : found->path.parent_path();
    return browse(directory({}, path.u8string()));
}

} // namespace mfq::server
