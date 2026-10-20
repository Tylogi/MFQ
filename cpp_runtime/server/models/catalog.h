#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace mfq::server {

class ModelCatalog {
public:
    explicit ModelCatalog(std::vector<std::filesystem::path> roots);
    nlohmann::json list(bool refresh);
    nlohmann::json browse(const std::string& directory_id, const std::string& path);
    nlohmann::json register_directory(const nlohmann::json& request);
    nlohmann::json model_directory(const std::string& model_id);

private:
    struct Artifact { std::filesystem::path path; nlohmann::json resource; };
    std::set<std::filesystem::path> registered_paths() const;
    std::vector<Artifact> scan(const std::set<std::filesystem::path>& registered) const;
    void refresh();
    std::string directory_id(const std::filesystem::path& path);
    std::filesystem::path directory(const std::string& id, const std::string& path) const;
    nlohmann::json directory_entry(const std::filesystem::path& path);
    nlohmann::json browse(const std::filesystem::path& path);

    std::mutex mutex_;
    std::vector<std::filesystem::path> roots_;
    std::string secret_;
    std::map<std::string, std::filesystem::path> directories_;
    std::vector<Artifact> artifacts_;
    bool scanned_ = false;
    std::chrono::steady_clock::time_point scanned_at_;
};

} // namespace mfq::server
