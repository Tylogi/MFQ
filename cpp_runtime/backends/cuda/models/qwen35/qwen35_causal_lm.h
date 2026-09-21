#pragma once

#include "models/qwen35.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mfq { class ModelSource; }
struct Block;
struct TextSessionState;

namespace mfq::cuda::qwen35 {

using Config = mfq::models::qwen35::Config;

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const Config& config,
    int layer,
    const std::string& type,
    std::string_view tensor_root = "model");

bool supports_text_session_state(
    const std::vector<std::unique_ptr<::Block>>& blocks);

TextSessionState capture_text_session_state(
    const std::vector<std::unique_ptr<::Block>>& blocks,
    const std::vector<std::int64_t>& tokens,
    std::int64_t cache_position);

void restore_text_session_state(
    std::vector<std::unique_ptr<::Block>>& blocks,
    const TextSessionState& state);

} // namespace mfq::cuda::qwen35
