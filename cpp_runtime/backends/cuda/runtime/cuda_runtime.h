#pragma once

#include "options.h"

namespace mfq::cuda {
int run_cuda_inference(RuntimeOptions options);
int run_cuda_minicpmo_composite(const RuntimeOptions& options);
int run_cuda_minicpmo_duplex(const RuntimeOptions& options);
} // namespace mfq::cuda
