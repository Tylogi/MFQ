#pragma once

#include "cuda_engine.h"

#include <charconv>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

enum class KlMmqMode;

namespace mfq::cuda {

struct TokenInputOptions {
    std::string ids_arg, ids_file;
    int gen = 16;
};

} // namespace mfq::cuda

namespace mfq::cuda::internal {

std::vector<int64_t> parse_ids(const std::string& value);
KlMmqMode parse_kl_mmq_mode(const std::string& value);
std::vector<KlMmqMode> parse_kl_mmq_sequence(const std::string& value);
std::vector<int64_t> load_ids_file(const std::string& path);

int with_command_errors(const std::function<int()>& fn);

} // namespace mfq::cuda::internal

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

void validate_integer_list(
    const std::string& text,
    std::string_view option);

bool parse_cuda_load_option(
    std::string_view option,
    ArgCursor& args,
    CudaLoadOptions& result);

bool parse_token_input_option(
    std::string_view option,
    ArgCursor& args,
    TokenInputOptions& result);

inline int print_usage_error(const UsageError& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
}

} // namespace mfq::cuda::commands
