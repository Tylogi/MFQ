#pragma once

#include "mtp_policy.h"

#include <string>
#include <utility>
#include <vector>

namespace mfq::engine::mtp {

void append_generation_metrics(
    std::vector<std::pair<std::string, double>>& metrics,
    const GenerationStats& stats);

} // namespace mfq::engine::mtp
