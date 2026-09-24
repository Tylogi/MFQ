#pragma once

#include "runtime/options.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace mfq::cuda::commands {

struct HelpRequested {};

class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void usage_error(const std::string& message) {
    throw UsageError(message);
}

class ArgCursor {
public:
    ArgCursor(int argc, char** argv, int first = 1)
        : argc_(argc), argv_(argv), index_(first) {}

    bool empty() const { return index_ >= argc_; }

    std::string_view next() {
        return argv_[index_++];
    }

    std::string value(std::string_view option) {
        if (empty() || looks_like_option(argv_[index_])) {
            usage_error("missing value for " + std::string(option));
        }
        return argv_[index_++];
    }

private:
    static bool looks_like_option(std::string_view value) {
        return value.starts_with("--") || value == "-h" ||
            value == "-t" || value == "-ngl";
    }

    int argc_;
    char** argv_;
    int index_;
};

template <typename T>
T integer(const std::string& text, std::string_view option) {
    static_assert(std::is_integral_v<T>);
    T result{};
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string(option) + " requires an integer");
    }
    return result;
}

inline void validate_integer_list(
        const std::string& text,
        std::string_view option) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        std::string item = text.substr(
            begin,
            comma == std::string::npos ? std::string::npos : comma - begin);
        const auto first = item.find_first_not_of(" \t\r\n");
        const auto last = item.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            usage_error(std::string(option) + " contains an empty item");
        }
        item = item.substr(first, last - first + 1);
        (void)integer<int64_t>(item, option);
        if (comma == std::string::npos) return;
        begin = comma + 1;
    }
    usage_error(std::string(option) + " requires comma-separated integers");
}

inline double number(const std::string& text, std::string_view option) {
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

inline bool parse_cuda_load_option(
        std::string_view option,
        ArgCursor& args,
        CudaLoadOptions& result) {
    if (option == "--model") result.model_path = args.value(option);
    else if (option == "--config") result.config_path = args.value(option);
    else if (option == "--threads" || option == "-t") {
        result.cpu_threads = integer<int>(args.value(option), option);
        if (result.cpu_threads <= 0) usage_error("--threads must be positive");
        result.cpu_threads_set = true;
    }
    else if (option == "--ctx-size") {
        result.context_size = integer<int64_t>(args.value(option), option);
        if (result.context_size < 0) {
            usage_error("--ctx-size must be non-negative");
        }
    }
    else if (option == "--cpu-offload-layers") {
        result.cpu_offload_layers_arg = args.value(option);
    }
    else if (option == "--n-gpu-layers" || option == "-ngl") {
        result.n_gpu_layers = integer<int>(args.value(option), option);
        if (result.n_gpu_layers < 0) {
            usage_error("--n-gpu-layers must be non-negative");
        }
        result.n_gpu_layers_set = true;
    }
    else if (option == "--moe-gpu-cache-gb") {
        result.moe_gpu_cache_gb = number(args.value(option), option);
        if (result.moe_gpu_cache_gb < 0.0) {
            usage_error("--moe-gpu-cache-gb must be non-negative");
        }
    }
    else if (option == "--moe-cache-profile") {
        result.moe_cache_profile_path = args.value(option);
    }
    else if (option == "--tensor-parallel") {
        result.tensor_parallel_arg = args.value(option);
    }
    else if (option == "--tensor-split") {
        result.tensor_split_arg = args.value(option);
    }
    else if (option == "--expert-parallel") {
        result.expert_parallel_arg = args.value(option);
    }
    else if (option == "--expert-split") {
        result.expert_split_arg = args.value(option);
    }
    else if (option == "--layer-parallel") {
        result.layer_parallel_arg = args.value(option);
    }
    else if (option == "--layer-split") {
        result.layer_split_arg = args.value(option);
    }
    else if (option == "--parallel-test-duplicates" ||
            option == "--tensor-parallel-test-duplicates") {
        result.parallel_test_duplicates = true;
    }
    else if (option == "--tokenizer") {
        result.tokenizer_model = args.value(option);
    }
    else return false;
    return true;
}

inline bool parse_token_input_option(
        std::string_view option,
        ArgCursor& args,
        TokenInputOptions& result) {
    if (option == "--ids") {
        result.ids_arg = args.value(option);
        validate_integer_list(result.ids_arg, option);
    }
    else if (option == "--ids-file") result.ids_file = args.value(option);
    else if (option == "--gen") {
        result.gen = integer<int>(args.value(option), option);
        if (result.gen < 0) usage_error("--gen must be non-negative");
    }
    else return false;
    return true;
}

inline int print_usage_error(const UsageError& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
}

} // namespace mfq::cuda::commands
