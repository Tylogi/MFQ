#pragma once

#include "mlx_sampling.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace mfq::metal {

using MlxWarmupGenerate = std::function<void(const std::vector<std::int64_t>&,
    const MlxSamplingParams&, int)>;

void warm_mlx_inference(const std::vector<std::int64_t>& prompt,
    int context, int chunk_size, const MlxSamplingParams& defaults, bool mtp,
    const MlxWarmupGenerate& generate, const std::function<void()>& reset);

}
