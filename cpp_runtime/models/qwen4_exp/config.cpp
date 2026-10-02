#include "models/qwen4_exp/config.h"

namespace mfq::models::qwen4_exp {

Config Config::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

} // namespace mfq::models::qwen4_exp
