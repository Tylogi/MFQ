#pragma once

#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>

namespace mfq::server {

std::filesystem::path resolved_path(const std::filesystem::path& path);
std::string normalize_uuid(std::string value);

class InstanceLock {
public:
    InstanceLock(const std::filesystem::path& data_dir, std::string instance_id, bool studio_owned);
    ~InstanceLock();
    InstanceLock(const InstanceLock&) = delete;
    InstanceLock& operator=(const InstanceLock&) = delete;
    const nlohmann::json& identity() const { return identity_; }

private:
    nlohmann::json identity_;
    int descriptor_ = -1;
};

} // namespace mfq::server
