#include "auth.h"
#include "api/errors.h"
#include "instance.h"
#include "mfq_paged_prefix_cache.h"
#include "storage/files.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <regex>

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif

namespace mfq::server {
namespace {

using json = nlohmann::json;

std::string hash_key(const std::string& key) {
    return mfq::cache::block_hash_hex(mfq::cache::sha256(key));
}

bool valid_key(const std::string& key) {
    return !key.empty() && key.size() <= 512 &&
        std::all_of(key.begin(), key.end(), [](unsigned char c) { return c >= 33 && c <= 126; });
}

bool equal_hash(const std::string& left, const std::string& right) {
    if (left.size() != right.size()) return false;
    unsigned char difference = 0;
    for (std::size_t i = 0; i < left.size(); ++i) difference |= left[i] ^ right[i];
    return difference == 0;
}

std::string new_token() {
    std::array<unsigned char, 32> bytes{};
#ifdef _WIN32
    if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
        throw std::runtime_error("system random generator failed");
#else
    std::ifstream random("/dev/urandom", std::ios::binary);
    if (!random.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("system random generator failed");
#endif
    constexpr char digits[] = "0123456789abcdef";
    std::string token = "mfq_";
    for (const auto byte : bytes) {
        token += digits[byte >> 4];
        token += digits[byte & 15];
    }
    return token;
}

void validate_fields(const json& request, std::initializer_list<const char*> fields) {
    if (!request.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    for (const auto& field : request.items())
        if (std::find(fields.begin(), fields.end(), field.key()) == fields.end())
            throw ApiError(400, "invalid_request", "unknown request field");
}

bool valid_expiry(const json& expiry) {
    if (expiry.is_null()) return true;
    if (!expiry.is_string()) return false;
    static const std::regex format(
        R"(^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(\.[0-9]{1,6})?(Z|[+-]([0-9]{2}):([0-9]{2}))$)");
    std::smatch match;
    const auto value = expiry.get<std::string>();
    if (value.size() > 32 || !std::regex_match(value, match, format)) return false;
    const int year = std::stoi(match[1]), month = std::stoi(match[2]), day = std::stoi(match[3]);
    constexpr std::array<int, 12> days{31,28,31,30,31,30,31,31,30,31,30,31};
    if (!year || month < 1 || month > 12 || day < 1) return false;
    const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    if (day > days[month - 1] + (month == 2 && leap)) return false;
    return std::stoi(match[4]) < 24 && std::stoi(match[5]) < 60 && std::stoi(match[6]) < 60 &&
        (match[8] == "Z" || (std::stoi(match[9]) < 24 && std::stoi(match[10]) < 60));
}

void save_policy(const std::filesystem::path& path, const json& configuration) {
    try {
        write_json(path, configuration);
    } catch (...) {
        throw ApiError(500, "authentication_save_failed", "could not persist authentication settings");
    }
}

std::string required_scope(const std::string& method, const std::string& path) {
    const auto starts = [&](const char* prefix) { return path.rfind(prefix, 0) == 0; };
    const bool read = method == "GET" || method == "HEAD";
    if (path == "/api/v1/hub/download-directory") return "models";
    if (starts("/api/v1/runtime/cache/entries") || path == "/api/v1/runtime/cache/purge") return "admin";
    if (!read && (path == "/api/v1/runtime/listener" || path == "/api/v1/runtime/memory-policy" ||
        path == "/api/v1/runtime/model-aliases" || path == "/api/v1/runtime/context-policy")) return "admin";
    if (starts("/api/v1/auth/") || starts("/api/v1/cluster/")) return "admin";
    if (starts("/api/v1/models/directories")) return "models";
    if (starts("/api/v1/models/") && path.size() >= 10 && path.compare(path.size() - 10, 10, "/directory") == 0) return "models";
    if (starts("/api/v1/models") || starts("/api/v1/runtime") || starts("/api/v1/hub"))
        return read ? "inference" : "models";
    for (const auto* prefix : {"/api/v1/jobs", "/api/v1/artifacts", "/api/v1/datasets", "/api/v1/evaluations", "/api/v1/quantization"})
        if (starts(prefix)) return "jobs";
    return "inference";
}

} // namespace

Authentication::Authentication(std::filesystem::path policy_path, const std::string& root_key, bool network)
    : path_(std::move(policy_path)), network_(network) {
    if (!root_key.empty() && !valid_key(root_key)) throw std::invalid_argument("invalid root API key");
    enabled_ = !root_key.empty();
    if (enabled_) key_hash_ = hash_key(root_key);
    if (std::filesystem::exists(path_)) {
        if (std::filesystem::file_size(path_) > 4096) throw std::runtime_error("invalid saved authentication policy");
        std::ifstream stream(path_);
        const auto saved = json::parse(stream);
        if (!saved.is_object() || !saved.contains("enabled") || !saved["enabled"].is_boolean() ||
            !saved.contains("key_hash") || !saved["key_hash"].is_string())
            throw std::runtime_error("invalid saved authentication policy");
        const auto hash = saved["key_hash"].get<std::string>();
        if ((!hash.empty() && (hash.size() != 64 || !std::all_of(hash.begin(), hash.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            }))) || (saved["enabled"].get<bool>() && hash.empty()))
            throw std::runtime_error("invalid saved authentication policy");
        enabled_ = saved["enabled"].get<bool>();
        key_hash_ = hash;
    }
    if (network_ && !enabled_) throw std::invalid_argument("authentication is required for a non-loopback listener");
}

json Authentication::status() {
    std::lock_guard<std::mutex> guard(mutex_);
    return {{"enabled", enabled_}, {"has_key", !key_hash_.empty()}};
}

json Authentication::configure(const json& request) {
    validate_fields(request, {"enabled", "api_key"});
    if (!request.contains("enabled") || !request["enabled"].is_boolean())
        throw ApiError(400, "invalid_authentication_policy", "enabled must be a boolean");
    const bool enabled = request["enabled"].get<bool>();
    std::lock_guard<std::mutex> guard(mutex_);
    if (network_ && !enabled)
        throw ApiError(400, "authentication_required", "authentication cannot be disabled for a network listener");
    auto hash = key_hash_;
    if (request.contains("api_key") && !request["api_key"].is_null()) {
        if (!request["api_key"].is_string() || !valid_key(request["api_key"].get<std::string>()))
            throw ApiError(400, "invalid_authentication_policy", "API key must contain 1-512 printable ASCII characters without spaces");
        hash = hash_key(request["api_key"].get<std::string>());
    }
    if (enabled && hash.empty()) throw ApiError(400, "invalid_authentication_policy", "an API key is required to enable authentication");
    save_policy(path_, {{"enabled", enabled}, {"key_hash", hash}});
    enabled_ = enabled;
    key_hash_ = std::move(hash);
    return {{"enabled", enabled_}, {"has_key", !key_hash_.empty()}};
}

void Authentication::authorize(Storage& storage, const std::string& token, const std::string& method, const std::string& path) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!enabled_) return;
    if (!valid_key(token)) throw ApiError(401, "unauthorized", "invalid API credential");
    const auto hash = hash_key(token);
    if (equal_hash(hash, key_hash_)) return;
    const auto key = storage.authenticate_key(hash);
    if (!key) throw ApiError(401, "unauthorized", "invalid API credential");
    const auto required = required_scope(method, path);
    const auto& scopes = key->at("scopes");
    if (!scopes.is_array()) throw std::runtime_error("invalid stored API key scopes");
    if (std::find(scopes.begin(), scopes.end(), "admin") != scopes.end() ||
        std::find(scopes.begin(), scopes.end(), required) != scopes.end()) return;
    const auto& role = key->at("role");
    if (role == "administrator" || (role == "viewer" && required == "inference") ||
        (role == "operator" && (required == "inference" || required == "models" || required == "jobs"))) return;
    throw ApiError(403, "insufficient_scope", "API credential requires the " + required + " scope");
}

json Authentication::create_key(Storage& storage, json request) {
    validate_fields(request, {"name", "scopes", "role", "expires_at"});
    if (!request.contains("name") || !request["name"].is_string())
        throw ApiError(400, "invalid_request", "name is required");
    const auto name = request["name"].get<std::string>();
    const auto length = std::count_if(name.begin(), name.end(), [](unsigned char c) { return (c & 192) != 128; });
    if (length < 1 || length > 64) throw ApiError(400, "invalid_request", "name must contain 1-64 characters");
    if (!request.contains("scopes")) request["scopes"] = json::array();
    const auto& scopes = request["scopes"];
    if (!scopes.is_array() || scopes.size() > 4) throw ApiError(400, "invalid_request", "invalid API key scopes");
    for (auto it = scopes.begin(); it != scopes.end(); ++it) {
        if ((*it != "inference" && *it != "models" && *it != "jobs" && *it != "admin") ||
            std::find(scopes.begin(), it, *it) != it)
            throw ApiError(400, "invalid_request", "invalid or duplicate API key scope");
    }
    if (!request.contains("role")) request["role"] = nullptr;
    const auto& role = request["role"];
    if ((!role.is_null() && role != "viewer" && role != "operator" && role != "administrator") ||
        (role.is_null() && scopes.empty())) throw ApiError(400, "invalid_request", "API key requires scopes or a valid role");
    if (!request.contains("expires_at")) request["expires_at"] = nullptr;
    if (!valid_expiry(request["expires_at"])) throw ApiError(400, "invalid_request", "expires_at must be a valid ISO timestamp with a timezone");
    const auto token = new_token();
    return {{"key", storage.create_key(request, normalize_uuid({}), hash_key(token), token.substr(0, 12))}, {"token", token}};
}

json Authentication::rotate_key(Storage& storage, const std::string& id) {
    const auto token = new_token();
    return {{"key", storage.rotate_key(normalize_uuid(id), hash_key(token), token.substr(0, 12))}, {"token", token}};
}

} // namespace mfq::server
