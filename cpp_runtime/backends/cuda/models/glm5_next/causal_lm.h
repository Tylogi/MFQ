#pragma once

#include "../../runtime/cuda_transformer.h"
#include "models/flash_next.h"
#include "layers.h"

#include <memory>

namespace mfq::cuda::flash_next {

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const mfq::models::flash_next::GlmConfig& config,
    int layer);

} // namespace mfq::cuda::flash_next
