#pragma once

#include "options.h"

#include <exception>
#include <iostream>
#include <utility>

namespace mfq::cuda::internal {

void setup_cuda_load(const CudaLoadOptions& options);
void reset_cuda_load() noexcept;

template <typename F>
int with_command_errors(F&& fn) {
    struct Cleanup {
        ~Cleanup() { reset_cuda_load(); }
    } cleanup;
    try {
        return std::forward<F>(fn)();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

} // namespace mfq::cuda::internal
