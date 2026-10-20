#pragma once

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <nlohmann/json.hpp>

struct sqlite3;

namespace mfq::server {

class Storage {
public:
    explicit Storage(const std::filesystem::path& path);
    ~Storage();
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    nlohmann::json create_key(const nlohmann::json& request, const std::string& id,
        const std::string& hash, const std::string& prefix);
    nlohmann::json list_keys();
    std::optional<nlohmann::json> authenticate_key(const std::string& hash);
    nlohmann::json revoke_key(const std::string& id);
    nlohmann::json rotate_key(const std::string& id, const std::string& hash, const std::string& prefix);

private:
    sqlite3* database_ = nullptr;
    std::mutex mutex_;
};

} // namespace mfq::server
