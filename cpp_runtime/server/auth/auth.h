#pragma once

#include "storage/storage.h"

namespace mfq::server {

class Authentication {
public:
    Authentication(std::filesystem::path policy_path, const std::string& root_key, bool network);
    nlohmann::json status();
    nlohmann::json configure(const nlohmann::json& request);
    void authorize(Storage& storage, const std::string& token, const std::string& method, const std::string& path);
    nlohmann::json create_key(Storage& storage, nlohmann::json request);
    nlohmann::json rotate_key(Storage& storage, const std::string& id);

private:
    std::filesystem::path path_;
    std::mutex mutex_;
    bool enabled_ = false;
    bool network_;
    std::string key_hash_;
};

} // namespace mfq::server
