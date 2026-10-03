#include "cli.h"

#include "cuda_execution.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::cuda::internal {
namespace {

std::vector<std::string> split_csv_values(
        const std::string& value,
        const char* option) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error(
                std::string(option) + " contains an empty item");
        }
        result.push_back(item.substr(first, last - first + 1));
    }
    if (result.empty()) {
        throw std::runtime_error(
            std::string(option) + " requires at least one value");
    }
    if (value.ends_with(',')) {
        throw std::runtime_error(
            std::string(option) + " contains an empty item");
    }
    return result;
}

} // namespace

std::vector<int64_t> parse_ids(const std::string& value) {
    std::vector<int64_t> ids;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            throw std::runtime_error("token ID lists cannot contain empty items");
        }
        item = item.substr(first, last - first + 1);
        int64_t id = 0;
        const auto [end, error] = std::from_chars(
            item.data(), item.data() + item.size(), id);
        if (error != std::errc{} || end != item.data() + item.size()) {
            throw std::runtime_error(
                "token ID lists must contain comma-separated integers");
        }
        ids.push_back(id);
    }
    if (ids.empty() || value.ends_with(',')) {
        throw std::runtime_error("token ID lists require at least one integer");
    }
    return ids;
}

KlMmqMode parse_kl_mmq_mode(const std::string& value) {
    if (value == "default") return KlMmqMode::Default;
    if (value == "nint8_1") return KlMmqMode::Nint8One;
    if (value == "fp16") return KlMmqMode::Fp16;
    throw std::runtime_error(
        "--kl-mmq must be default, nint8_1, or fp16");
}

std::vector<KlMmqMode> parse_kl_mmq_sequence(
        const std::string& value) {
    std::vector<KlMmqMode> modes;
    for (const auto& item :
         split_csv_values(value, "--kl-mmq-sequence")) {
        const KlMmqMode mode = parse_kl_mmq_mode(item);
        if (mode == KlMmqMode::Default) {
            throw std::runtime_error(
                "--kl-mmq-sequence accepts only nint8_1 and fp16");
        }
        if (std::find(modes.begin(), modes.end(), mode) != modes.end()) {
            throw std::runtime_error(
                "--kl-mmq-sequence contains a duplicate mode");
        }
        modes.push_back(mode);
    }
    return modes;
}

std::vector<int64_t> load_ids_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open --ids-file: " + path);
    const std::streamsize bytes = input.tellg();
    if (bytes <= 0 ||
            bytes % static_cast<std::streamsize>(sizeof(int32_t)) != 0) {
        throw std::runtime_error("--ids-file must contain raw int32 token ids");
    }
    input.seekg(0);
    std::vector<int32_t> stored(
        static_cast<size_t>(bytes /
            static_cast<std::streamsize>(sizeof(int32_t))));
    input.read(reinterpret_cast<char*>(stored.data()), bytes);
    if (!input) throw std::runtime_error("truncated --ids-file: " + path);
    return std::vector<int64_t>(stored.begin(), stored.end());
}

int with_command_errors(const std::function<int()>& fn) {
    try {
        return fn();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

} // namespace mfq::cuda::internal
