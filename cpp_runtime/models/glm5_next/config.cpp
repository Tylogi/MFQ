#include "models/glm5_next/config.h"

namespace mfq::models::glm5_next {

Config Config::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

} // namespace mfq::models::glm5_next
