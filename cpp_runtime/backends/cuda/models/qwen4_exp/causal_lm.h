#pragma once

#include "../../runtime/cuda_transformer.h"
#include "models/flash_next.h"
#include "layers.h"

#include <memory>

namespace mfq::cuda::flash_next {

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const mfq::models::flash_next::QwenConfig& config,
    int layer);
std::unique_ptr<::flash_runtime::Gr> load_final_mixer(
    const mfq::ModelSource& source,
    const mfq::models::flash_next::QwenConfig& config);

} // namespace mfq::cuda::flash_next
