#include "cli.h"

#include "ops/execution/cuda_execution.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
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

namespace mfq::cuda::commands {
namespace {

double number(const std::string& text, std::string_view option) {
    double result = 0.0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result,
        std::chars_format::general);
    if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(result)) {
        usage_error(std::string(option) + " requires a finite number");
    }
    return result;
}

std::vector<std::string> comma_values(
        const std::string& text,
        std::string_view option) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        std::string item = text.substr(
            begin,
            comma == std::string::npos ? std::string::npos : comma - begin);
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            usage_error(std::string(option) + " contains an empty item");
        }
        result.push_back(item.substr(first, last - first + 1));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return result;
}

std::vector<int> parallel_devices(
        const std::string& text,
        std::string_view option) {
    const auto values = comma_values(text, option);
    if (values.size() == 1 && text.find(',') == std::string::npos) {
        const int count = integer<int>(values.front(), option);
        if (count < 2) {
            usage_error(std::string(option) +
                        " device count must be at least two");
        }
        std::vector<int> result(static_cast<std::size_t>(count));
        std::iota(result.begin(), result.end(), 0);
        return result;
    }
    std::vector<int> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        result.push_back(integer<int>(value, option));
    }
    if (result.size() < 2) {
        usage_error(std::string(option) + " requires at least two devices");
    }
    return result;
}

std::vector<double> parallel_split(
        const std::string& text,
        std::string_view option) {
    std::vector<double> result;
    for (const auto& value : comma_values(text, option)) {
        const double weight = number(value, option);
        if (weight <= 0.0) {
            usage_error(std::string(option) +
                        " values must be finite and positive");
        }
        result.push_back(weight);
    }
    return result;
}

std::vector<int> layer_ranges(
        const std::string& text,
        std::string_view option) {
    std::vector<int> result;
    for (const auto& value : comma_values(text, option)) {
        const std::size_t dash = value.find('-');
        if (dash == 0 || dash + 1 == value.size() ||
                (dash != std::string::npos &&
                 value.find('-', dash + 1) != std::string::npos)) {
            usage_error("invalid " + std::string(option) + " range: " + value);
        }
        const int first = integer<int>(value.substr(0, dash), option);
        const int last = dash == std::string::npos
            ? first
            : integer<int>(value.substr(dash + 1), option);
        if (first < 0 || last < first) {
            usage_error("invalid " + std::string(option) + " range: " + value);
        }
        for (int layer = first;; ++layer) {
            result.push_back(layer);
            if (layer == last) break;
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

bool parse_parallel_option(
        std::string_view option,
        ArgCursor& args,
        CudaLoadOptions& result) {
    if (option == "--tensor-parallel") {
        result.tensor_parallel.devices = parallel_devices(args.value(option), option);
    } else if (option == "--tensor-split") {
        result.tensor_parallel.split = parallel_split(args.value(option), option);
    } else if (option == "--expert-parallel") {
        result.expert_parallel.devices = parallel_devices(args.value(option), option);
    } else if (option == "--expert-split") {
        result.expert_parallel.split = parallel_split(args.value(option), option);
    } else if (option == "--layer-parallel") {
        result.layer_parallel.devices = parallel_devices(args.value(option), option);
    } else if (option == "--layer-split") {
        result.layer_parallel.split = parallel_split(args.value(option), option);
    } else if (option == "--parallel-test-duplicates" ||
               option == "--tensor-parallel-test-duplicates") {
        result.parallel_test_duplicates = true;
    } else {
        return false;
    }
    return true;
}

bool parse_resource_option(
        std::string_view option,
        ArgCursor& args,
        CudaLoadOptions& result) {
    if (option == "--threads" || option == "-t") {
        result.cpu_threads = integer<int>(args.value(option), option);
        if (*result.cpu_threads <= 0) usage_error("--threads must be positive");
    } else if (option == "--ctx-size") {
        result.context_size = integer<int64_t>(args.value(option), option);
        if (result.context_size < 0) usage_error("--ctx-size must be non-negative");
    } else if (option == "--cpu-offload-layers") {
        result.cpu_offload_layers = layer_ranges(args.value(option), option);
    } else if (option == "--n-gpu-layers" || option == "-ngl") {
        result.n_gpu_layers = integer<int>(args.value(option), option);
        if (*result.n_gpu_layers < 0) usage_error("--n-gpu-layers must be non-negative");
    } else if (option == "--moe-gpu-cache-gb") {
        result.moe_gpu_cache_gb = number(args.value(option), option);
        if (result.moe_gpu_cache_gb < 0.0) {
            usage_error("--moe-gpu-cache-gb must be non-negative");
        }
    } else if (option == "--moe-cache-profile") {
        result.moe_cache_profile_path = args.value(option);
    } else {
        return false;
    }
    return true;
}

} // namespace

void validate_integer_list(
        const std::string& text,
        std::string_view option) {
    for (const auto& value : comma_values(text, option)) {
        (void)integer<int64_t>(value, option);
    }
}

bool parse_cuda_load_option(
        std::string_view option,
        ArgCursor& args,
        CudaLoadOptions& result) {
    if (option == "--model") result.model_path = args.value(option);
    else if (option == "--yarn-context-extension") result.yarn_context_extension = true;
    else if (option == "--config") result.config_path = args.value(option);
    else if (option == "--tokenizer") result.tokenizer_model = args.value(option);
    else return parse_resource_option(option, args, result) ||
                parse_parallel_option(option, args, result);
    return true;
}

bool parse_token_input_option(
        std::string_view option,
        ArgCursor& args,
        TokenInputOptions& result) {
    if (option == "--ids") {
        result.ids_arg = args.value(option);
        validate_integer_list(result.ids_arg, option);
    } else if (option == "--ids-file") {
        result.ids_file = args.value(option);
    } else if (option == "--gen") {
        result.gen = integer<int>(args.value(option), option);
        if (result.gen < 0) usage_error("--gen must be non-negative");
    } else {
        return false;
    }
    return true;
}

} // namespace mfq::cuda::commands
