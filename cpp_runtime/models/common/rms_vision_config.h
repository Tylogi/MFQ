#pragma once

#include "grid_vision.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <string>

namespace mfq::models {

// RMS-normalized axial ViT with a channel-first spatial downsampler and a
// bounded SwiGLU merger. Text positions remain sequential (NoPE MLA/KDA).
struct RmsVisionConfig {
    GridVisionConfig grid;
    int64_t projection_intermediate = 0;
    double epsilon = 1e-5;
    double limit = 10.0;

    static RmsVisionConfig parse(const nlohmann::json& value) {
        RmsVisionConfig c;
        auto& g = c.grid;
        g.hidden_size = value.at("hidden_size").get<int64_t>();
        g.intermediate_size = value.at("intermediate_size").get<int64_t>();
        g.depth = value.at("depth").get<int64_t>();
        g.num_heads = value.at("num_heads").get<int64_t>();
        g.in_channels = value.value("in_channels", int64_t{3});
        g.patch_size = value.at("patch_size").get<int64_t>();
        g.temporal_patch_size = value.at("temporal_patch_size").get<int64_t>();
        g.spatial_merge_size = value.value("spatial_merge_size", int64_t{2});
        g.out_hidden_size = value.at("out_hidden_size").get<int64_t>();
        g.rope_theta = value.value("rope_theta", 10000.0);
        if (value.contains("rope_parameters"))
            g.rope_theta = value.at("rope_parameters").value("rope_theta", g.rope_theta);
        c.projection_intermediate = value.at("projection_intermediate_size").get<int64_t>();
        c.epsilon = value.value("rms_norm_eps", 1e-5);
        c.limit = value.value("swiglu_limit", 10.0);
        g.validate(false);
        if (c.projection_intermediate <= 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0 ||
            !std::isfinite(c.limit) || c.limit <= 0 ||
            value.value("hidden_act", std::string("silu")) != "silu")
            throw std::invalid_argument("invalid RMS-SwiGLU vision config");
        return c;
    }
};

} // namespace mfq::models
