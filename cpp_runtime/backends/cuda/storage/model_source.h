#pragma once

#include "mfq/model_source.h"

#include <string>

namespace mfq::cuda {

std::string load_model_config_json(
    const mfq::ModelSource& source,
    const std::string& external_path = {});

void validate_model_source(const mfq::ModelSource& source);

} // namespace mfq::cuda
