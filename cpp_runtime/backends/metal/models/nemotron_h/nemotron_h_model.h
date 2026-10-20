#pragma once

#include "mfq_container.h"
#include "mfq_model_graph.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"
#include "models/nemotron_h/config.h"
#include "models/nemotron_h/vision.h"

#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Nemotron-H reuses the backend-neutral text contract and the RADIO geometry
// from models/nemotron_h, so the Metal adapter parses exactly the same model
// config as the CUDA runtime.
using NemotronHConfig = mfq::models::nemotron_h::Config;
using NemotronHRadioConfig = mfq::models::nemotron_h::RadioVisionConfig;

inline constexpr std::string_view kNemotronHConfigAsset =
    "__mfq_asset__/model_config.json";

// Optional graph components resolved by semantic ID, never by model name.
struct NemotronHComponents {
    bool has_vision = false;
    bool has_predictor = false;
    NemotronHRadioConfig vision;
    std::string vision_input_contract;
    std::string vision_position_policy;
};

// Parse model_config.json and reject a graph whose components this backend
// cannot feed (a vision tower without the RADIO CPE contract, an unknown
// predictor implementation, or a foreign backbone).
NemotronHConfig load_nemotron_h_config(const MfqContainer& model);
NemotronHComponents resolve_nemotron_h_components(const MfqContainer& model);

mlx::core::array load_nemotron_h_dense(
    const MfqContainer& model,
    const std::string& name);
mlx::core::array load_nemotron_h_dense_vector(
    const MfqContainer& model,
    const std::string& name);
MlxRmsNorm load_nemotron_h_rms_norm(
    const MfqContainer& model,
    const std::string& name,
    double eps);

// Layer kinds the Metal block schedule dispatches to.
using NemotronHMixer = mfq::models::nemotron_h::MixerKind;

} // namespace mfq::metal
