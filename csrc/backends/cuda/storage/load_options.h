#pragma once

#include "mfq/cuda/engine.h"

#include <functional>

struct CudaExecutionContext;

namespace mfq::cuda::internal {

void setup_cuda_load(
    const CudaLoadOptions& options,
    CudaExecutionContext& execution);
int with_cuda_load(
    const CudaLoadOptions& options,
    const std::function<int(CudaExecutionContext&)>& run);

} // namespace mfq::cuda::internal
