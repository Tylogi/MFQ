#include "nemotron_h_model.h"

#include "mlx_legacy_tensor_compat.h"

#include <stdexcept>
#include <string>

namespace mfq::metal {
namespace {

using mlx::core::array;

bool dense_dtype_supported(const std::string& dtype) {
    return dtype == "BF16" || dtype == "F16" || dtype == "F32";
}

const mfq::MfqGraphComponent& require_component(
    const mfq::MfqModelGraph& graph,
    const char* kind) {
    const auto* component = graph.component(kind);
    if (component == nullptr) {
        throw std::runtime_error(
            std::string("Nemotron-H Metal graph has no ") + kind + " component");
    }
    return *component;
}

} // namespace

NemotronHConfig load_nemotron_h_config(const MfqContainer& model) {
    const auto graph = effective_model_graph(model);
    if (graph.backbone != "nemotron_h") {
        throw std::runtime_error(
            "unsupported native Metal backbone implementation: " + graph.backbone);
    }
    if (!model.contains(std::string(kNemotronHConfigAsset))) {
        throw std::runtime_error("MFQ has no embedded model config asset");
    }
    return NemotronHConfig::from_json(
        model.read_text(std::string(kNemotronHConfigAsset)), graph);
}

NemotronHComponents resolve_nemotron_h_components(const MfqContainer& model) {
    const auto graph = effective_model_graph(model);
    NemotronHComponents result;
    if (graph.component("vision") != nullptr) {
        const auto& vision = require_component(graph, "vision");
        // RADIO is selected by implementation and input contract only: the
        // tower takes normalized row-major patches and has no position
        // policy, so a mismatched declaration cannot be fed.
        if (vision.implementation != "radio" ||
            vision.input_contract != mfq::kMfqRadioCpeInputContract) {
            throw std::runtime_error(
                "Nemotron-H Metal vision requires the radio/radio_cpe.v1 vision component");
        }
        if (!vision.policy.empty() && vision.policy != "optional") {
            throw std::runtime_error(
                "Nemotron-H Metal RADIO vision must be declared optional, got: " +
                vision.policy);
        }
        if (!vision.position_policy.empty()) {
            throw std::runtime_error(
                "Nemotron-H Metal RADIO vision declares no position policy, got: " +
                vision.position_policy);
        }
        result.has_vision = true;
        result.vision_input_contract = vision.input_contract;
        result.vision_position_policy = vision.position_policy;
        result.vision = NemotronHRadioConfig::from_json(
            model.read_text(std::string(kNemotronHConfigAsset)));
    }
    if (graph.component("predictor") != nullptr) {
        const auto& predictor = require_component(graph, "predictor");
        if (predictor.implementation != "nemotron_h_mtp") {
            throw std::runtime_error(
                "Nemotron-H Metal multi-token prediction requires the "
                "nemotron_h_mtp predictor implementation");
        }
        result.has_predictor = true;
    }
    return result;
}

array load_nemotron_h_dense(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (!dense_dtype_supported(record.dtype)) {
        throw std::runtime_error(
            "Nemotron-H tensor must be dense BF16/F16/F32: " + name);
    }
    return mlx::core::contiguous(
        load_dense_array(record.dtype, model.read(name)));
}

array load_nemotron_h_dense_vector(
    const MfqContainer& model,
    const std::string& name) {
    auto value = load_nemotron_h_dense(model, name);
    if (value.ndim() != 1) {
        throw std::runtime_error(
            "Nemotron-H tensor must be a vector: " + name);
    }
    return value.dtype() == mlx::core::float32
        ? value
        : mlx::core::contiguous(
              mlx::core::astype(value, mlx::core::float32));
}

MlxRmsNorm load_nemotron_h_rms_norm(
    const MfqContainer& model,
    const std::string& name,
    double eps) {
    // Released Nemotron-H checkpoints use plain RMSNorm weights (no +1 shift).
    return MlxRmsNorm(
        load_nemotron_h_dense_vector(model, name),
        static_cast<float>(eps),
        0.0f);
}

} // namespace mfq::metal
