#include "models/include/flash_next.h"

namespace mfq::models::flash_next {

QwenConfig QwenConfig::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

GlmConfig GlmConfig::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

} // namespace mfq::models::flash_next
