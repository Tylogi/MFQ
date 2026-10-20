#pragma once
#include <filesystem>
#include <string>

namespace mfq {
// A profile belongs to one exact checkpoint entry, not every model in a folder.
// Explicit CLI selection wins; absent sidecars preserve ordinary placement.
inline std::string resolve_moe_cache_profile_path(const std::string& explicit_path,
        const std::string& model_path,bool automatic=true) {
    if(!explicit_path.empty())return explicit_path;
    if(!automatic || model_path.empty())return {};
    auto model=std::filesystem::path(model_path);
    std::error_code error;
    auto sidecar=std::filesystem::is_directory(model,error)
        ?model/"moe-cache-profile.json":std::filesystem::path(model.string()+".moe-cache-profile.json");
    error.clear();
    return std::filesystem::is_regular_file(sidecar,error) && !error?sidecar.string():std::string{};
}
}
