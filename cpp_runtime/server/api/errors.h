#pragma once

#include <stdexcept>
#include <string>

namespace mfq::server {

struct ApiError : std::runtime_error {
    int status;
    std::string code;
    ApiError(int status, const char* code, const std::string& message)
        : std::runtime_error(message), status(status), code(code) {}
};

} // namespace mfq::server
