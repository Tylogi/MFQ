#pragma once

#include "model_graph/model_graph.h"

#include <functional>
#include <string_view>

namespace mfq {

// The only C++ compatibility boundary for artifacts written before
// model_graph.json. Backends must consume the returned semantic graph and must
// not infer components from header aliases or checkpoint tensor spellings.
ModelGraph synthesize_legacy_model_graph(
    std::string_view artifact_architecture,
    std::string_view model_config_json,
    const std::function<bool(std::string_view)>& contains_tensor);

} // namespace mfq
