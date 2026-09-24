#pragma once

#include "options.h"

namespace mfq::cuda::internal {

int run_cuda_token_generation(
    mfq::cuda::CudaLoadOptions& load_options,
    const mfq::cuda::TokenInputOptions& token_options);

} // namespace mfq::cuda::internal
