#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

namespace mfq::server {

void write_json(const std::filesystem::path& path, const nlohmann::json& value);

} // namespace mfq::server
