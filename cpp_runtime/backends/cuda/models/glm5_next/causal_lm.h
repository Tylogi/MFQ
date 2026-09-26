#pragma once

#include "../../engine/cuda_transformer.h"
#include "models/include/glm5_next.h"
#include "layers.h"

#include <memory>

namespace mfq::cuda::glm5_next {

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const mfq::models::glm5_next::Config& config,
    int layer);

} // namespace mfq::cuda::glm5_next
