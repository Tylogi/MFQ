#pragma once

#include "models/common/model_config.h"

#include <string>
#include <string_view>

namespace mfq::models::minicpmo45 {

struct Config : ModelConfig {
    std::string version;
    std::string hidden_act;
    bool attention_bias = true;
    bool use_sliding_window = true;

    static Config from_json(std::string_view payload);

    template <class Source>
    static Config from_source(const Source& source) {
        return from_json(source.model_config_json());
    }
};

} // namespace mfq::models::minicpmo45
