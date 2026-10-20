#include "models/nemotron_h/vision.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mfq::models::nemotron_h {
namespace {

using json = nlohmann::json;

const json& object(const json& parent, const char* name) {
    const auto found = parent.find(name);
    if (found == parent.end() || !found->is_object())
        throw std::runtime_error(std::string("Nemotron-H vision config requires an object ") +
                                 name);
    return *found;
}

std::int64_t integer(const json& parent, const char* name, std::int64_t fallback) {
    const auto found = parent.find(name);
    if (found == parent.end() || found->is_null())
        return fallback;
    if (!found->is_number_integer())
        throw std::runtime_error(std::string("Nemotron-H vision config ") + name +
                                 " must be an integer");
    return found->get<std::int64_t>();
}

double number(const json& parent, const char* name, double fallback) {
    const auto found = parent.find(name);
    if (found == parent.end() || found->is_null())
        return fallback;
    if (!found->is_number())
        throw std::runtime_error(std::string("Nemotron-H vision config ") + name +
                                 " must be a number");
    return found->get<double>();
}

bool flag(const json& parent, const char* name, bool fallback) {
    const auto found = parent.find(name);
    if (found == parent.end() || found->is_null())
        return fallback;
    if (!found->is_boolean())
        throw std::runtime_error(std::string("Nemotron-H vision config ") + name +
                                 " must be a boolean");
    return found->get<bool>();
}

std::string text(const json& parent, const char* name) {
    const auto found = parent.find(name);
    if (found == parent.end() || !found->is_string())
        throw std::runtime_error(std::string("Nemotron-H vision config ") + name +
                                 " must be a string");
    return found->get<std::string>();
}

const json& text_section(const json& root) {
    for (const char* key : {"llm_config", "text_config"}) {
        const auto found = root.find(key);
        if (found != root.end() && found->is_object())
            return *found;
    }
    return root;
}

// One axis of a bilinear resample with align_corners=False: the four-tap
// indices and weights for every output position.
struct AxisTap {
    std::int32_t first = 0;
    std::int32_t second = 0;
    float first_weight = 1.0f;
    float second_weight = 0.0f;
};

std::vector<AxisTap> axis_taps(std::int32_t input_size, std::int32_t output_size) {
    if (input_size <= 0 || output_size <= 0)
        throw std::runtime_error("Nemotron-H position interpolation needs positive extents");
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    std::vector<AxisTap> taps(static_cast<std::size_t>(output_size));
    for (std::int32_t index = 0; index < output_size; ++index) {
        // ATen clamps a negative source coordinate to zero and the high tap to
        // the last row, which is how the reference handles its border pixels.
        const double source =
            std::max(0.0, (index + 0.5) * scale - 0.5);
        const auto low = static_cast<std::int32_t>(std::floor(source));
        const auto fraction = static_cast<float>(source - std::floor(source));
        taps[static_cast<std::size_t>(index)] = AxisTap{
            low,
            std::min(low + 1, input_size - 1),
            1.0f - fraction,
            fraction,
        };
    }
    return taps;
}

} // namespace

void RadioVisionConfig::validate() const {
    if (hidden_size <= 0 || depth <= 0 || num_heads <= 0 || hidden_size % num_heads)
        throw std::runtime_error("Nemotron-H RADIO geometry is invalid");
    if (intermediate_size <= 0 || num_channels <= 0 || patch_size <= 0)
        throw std::runtime_error("Nemotron-H RADIO MLP or patch geometry is invalid");
    if (max_img_size < patch_size || max_img_size % patch_size)
        throw std::runtime_error(
            "Nemotron-H RADIO max_img_size must be a whole number of patches");
    if (num_cls_tokens <= 0 || num_registers < 0)
        throw std::runtime_error("Nemotron-H RADIO cls/register geometry is invalid");
    if (downsample_factor != 2)
        throw std::runtime_error("Nemotron-H RADIO requires a 2x2 pixel shuffle");
    if (projector_intermediate_size <= 0 || text_hidden_size <= 0)
        throw std::runtime_error("Nemotron-H RADIO projector geometry is invalid");
    if (layer_norm_eps <= 0.0 || projector_norm_eps <= 0.0)
        throw std::runtime_error("Nemotron-H RADIO normalization epsilon is invalid");
}

RadioVisionConfig RadioVisionConfig::from_json(std::string_view payload) {
    const auto document = json::parse(payload.begin(), payload.end(), nullptr, false);
    if (!document.is_object())
        throw std::runtime_error("Nemotron-H vision config must be an object");
    const auto& vision = object(document, "vision_config");
    if (text(vision, "model_type") != "radio")
        throw std::runtime_error("Nemotron-H vision tower must be a RADIO model");
    // The block graph has no LayerScale parameters in released checkpoints and
    // no SwiGLU variant: both would change the math this encoder implements.
    if (text(vision, "hidden_act") != "gelu" || !flag(vision, "qkv_bias", false) ||
        flag(vision, "use_swiglu_ffn", false) || number(vision, "layerscale_value", 1.0) != 1.0)
        throw std::runtime_error(
            "Nemotron-H RADIO expects gelu, biased qkv, no SwiGLU and layerscale 1");
    const double ratio = number(vision, "mlp_ratio", 4.0);
    const auto hidden = integer(vision, "hidden_size", 0);
    const auto intermediate = static_cast<std::int64_t>(static_cast<double>(hidden) * ratio);
    if (hidden <= 0 || static_cast<double>(intermediate) != static_cast<double>(hidden) * ratio)
        throw std::runtime_error(
            "Nemotron-H RADIO mlp_ratio must describe a whole intermediate width");
    // The reference `pixel_shuffle` regroups 2x2 patch blocks (its view calls
    // hardcode 0.5), so only the released 2x2 shuffle is executable.
    const double downsample = number(document, "downsample_ratio", 0.0);
    const auto factor = static_cast<std::int64_t>(std::lround(1.0 / downsample));
    if (downsample != 0.5 || factor != 2)
        throw std::runtime_error(
            "Nemotron-H RADIO requires downsample_ratio 0.5 (2x2 pixel shuffle)");
    if (text(document, "ps_version") != "v2")
        throw std::runtime_error("Nemotron-H RADIO requires ps_version v2");
    if (integer(vision, "hidden_size", 0) != integer(document, "vit_hidden_size", hidden))
        throw std::runtime_error("Nemotron-H RADIO vit_hidden_size disagrees with vision_config");
    RadioVisionConfig config;
    config.hidden_size = hidden;
    config.intermediate_size = intermediate;
    config.depth = integer(vision, "num_hidden_layers", 0);
    config.num_heads = integer(vision, "num_attention_heads", 0);
    config.num_channels = integer(vision, "num_channels", 3);
    config.patch_size = integer(vision, "patch_size", 0);
    config.max_img_size = integer(vision, "max_img_size", 0);
    config.num_cls_tokens = integer(vision, "num_cls_tokens", 0);
    config.num_registers = integer(vision, "num_registers", 0);
    config.layer_norm_eps = number(vision, "layer_norm_eps", 1e-6);
    config.downsample_factor = factor;
    config.projector_intermediate_size = integer(document, "projector_hidden_size", 0);
    config.text_hidden_size = integer(text_section(document), "hidden_size", 0);
    config.image_token_id = integer(document, "img_context_token_id", -1);
    config.validate();
    return config;
}

RadioPositionInterpolation make_radio_position_interpolation(
        std::int32_t rows, std::int32_t columns, std::int32_t position_side) {
    if (rows <= 0 || columns <= 0 || position_side <= 0)
        throw std::runtime_error("Nemotron-H RADIO position grid is invalid");
    const auto square =
        std::max(rows, columns);
    // The reference upsamples the square table to max(rows, columns) before
    // cropping, so both axes share one scale.
    const auto row_taps = axis_taps(position_side, square);
    const auto column_taps = axis_taps(position_side, square);
    RadioPositionInterpolation result;
    result.patch_count = static_cast<std::int64_t>(rows) * columns;
    result.indices.resize(static_cast<std::size_t>(result.patch_count) * 4);
    result.weights.resize(result.indices.size());
    std::size_t offset = 0;
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto& vertical = row_taps[static_cast<std::size_t>(row)];
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto& horizontal = column_taps[static_cast<std::size_t>(column)];
            const std::int32_t rows_of_taps[2] = {vertical.first, vertical.second};
            const std::int32_t columns_of_taps[2] = {horizontal.first, horizontal.second};
            for (int vertical_index = 0; vertical_index < 2; ++vertical_index) {
                for (int horizontal_index = 0; horizontal_index < 2; ++horizontal_index) {
                    const auto weight =
                        (vertical_index == 0 ? vertical.first_weight : vertical.second_weight) *
                        (horizontal_index == 0 ? horizontal.first_weight
                                               : horizontal.second_weight);
                    result.indices[offset] =
                        rows_of_taps[vertical_index] * position_side +
                        columns_of_taps[horizontal_index];
                    result.weights[offset] = weight;
                    ++offset;
                }
            }
        }
    }
    return result;
}

} // namespace mfq::models::nemotron_h
