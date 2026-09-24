#pragma once

#include "models/include/model_config.h"

#include <memory>
#include <string>
#include <string_view>

namespace mfq { class ModelSource; }
struct Block;

std::unique_ptr<Block> load_transformer_block(
    const mfq::ModelSource& source,
    const mfq::models::ModelConfig& config,
    int layer,
    const std::string& type,
    bool minicpmo45 = false,
    std::string_view tensor_root = "model");
