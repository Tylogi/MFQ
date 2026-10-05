#include "transport.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

void write_text(const std::filesystem::path & path, const std::string & value) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create test profile");
    output << value;
}

} // namespace

int main() {
    require(
        static_cast<bool>(make_mfq_http_transport({})),
        "HTTP transport factory failed");
    require(
        static_cast<bool>(make_mfq_stdio_transport({})),
        "stdio transport factory failed");

    const auto defaults = resolve_mfq_runtime_profile("");
    require(defaults.source == "generic-defaults",
            "generic profile source mismatch");
    require(!defaults.chat.temperature,
            "generic profile must not guess model defaults");

    const auto root = std::filesystem::temp_directory_path() /
        "mfq-runtime-profile-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto model = root / "model-00002-of-00003.mfq";
    const auto family_sidecar = root / "model.runtime.json";
    const auto exact_sidecar = std::filesystem::path(
        model.string() + ".runtime.json");
    const auto explicit_profile = root / "server.json";
    write_text(family_sidecar,
        R"({"version":1,"chat":{"temperature":0.4,"top_p":0.6}})");
    write_text(exact_sidecar,
        R"({"version":1,"chat":{"temperature":0.3}})");
    write_text(explicit_profile,
        R"({"version":1,"chat":{"temperature":0.2}})");

    const auto resolved = resolve_mfq_runtime_profile(
        model.string(),
        R"({"version":1,"chat":{"temperature":0.5,"top_k":77}})",
        explicit_profile.string());
    require(std::abs(resolved.chat.temperature.value_or(-1.0) - 0.2) < 1e-12,
            "server profile did not win");
    require(std::abs(resolved.chat.top_p.value_or(-1.0) - 0.6) < 1e-12,
            "family sidecar field was lost");
    require(resolved.chat.top_k.value_or(-1) == 77,
            "embedded field was lost");
    require(resolved.source.find("runtime-explicit:") == 0,
            "profile source mismatch");

    std::filesystem::remove_all(root);
    return 0;
}
