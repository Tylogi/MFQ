#pragma once

#include "models/include/gemma4.h"

#include <memory>
#include <string>

namespace mfq { class ModelSource; }
struct Block;

namespace mfq::cuda::gemma4 {

using Config = mfq::models::gemma4::Config;

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const Config& config,
    int layer,
    const std::string& type);

} // namespace mfq::cuda::gemma4
