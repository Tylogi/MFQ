// RADIO vision config contract: parses the released omni geometry and rejects
// the variants the encoder cannot execute. No GPU or weights required.
#include "models/nemotron_h/vision.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using mfq::models::nemotron_h::RadioVisionConfig;

// The keys of the released checkpoint's config.json that describe the vision
// tower and its projection (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B).
std::string released_config() {
    return R"({
  "downsample_ratio": 0.5,
  "img_context_token_id": 18,
  "projector_hidden_size": 20480,
  "ps_version": "v2",
  "vit_hidden_size": 1280,
  "llm_config": {"hidden_size": 4096, "model_type": "nemotron_h"},
  "vision_config": {
    "model_type": "radio",
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "num_channels": 3,
    "patch_size": 16,
    "max_img_size": 2048,
    "num_cls_tokens": 3,
    "num_registers": 7,
    "layer_norm_eps": 1e-06,
    "hidden_act": "gelu",
    "qkv_bias": true,
    "use_swiglu_ffn": false,
    "layerscale_value": 1.0
  }
})";
}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Replace one substring, so every rejection case only changes one fact.
std::string with(std::string payload, const std::string& from, const std::string& to) {
    const auto position = payload.find(from);
    require(position != std::string::npos, "test patch target is missing from the config");
    return payload.replace(position, from.size(), to);
}

void require_rejected(const std::string& payload, const char* message) {
    bool rejected = false;
    try {
        RadioVisionConfig::from_json(payload);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

} // namespace

int main() {
    try {
        const auto config = RadioVisionConfig::from_json(released_config());
        require(config.hidden_size == 1280 && config.depth == 32, "hidden size/depth mismatch");
        require(config.num_heads == 16 && config.head_dim() == 80, "head geometry mismatch");
        require(config.intermediate_size == 5120, "MLP ratio was not applied");
        require(config.patch_size == 16 && config.patch_width() == 768, "patch geometry mismatch");
        require(config.position_side() == 128, "cropped-position side mismatch");
        require(config.num_cls_tokens == 3 && config.num_registers == 7 &&
                    config.summary_tokens() == 10,
                "cls/register geometry mismatch");
        require(config.downsample_factor == 2 && config.shuffle_width() == 5120,
                "pixel shuffle geometry mismatch");
        require(config.projector_intermediate_size == 20480 && config.text_hidden_size == 4096,
                "projector geometry mismatch");
        require(config.image_token_id == 18, "image placeholder id mismatch");
        require(config.layer_norm_eps == 1e-6 && config.projector_norm_eps == 1e-5,
                "normalization epsilon mismatch");

        // Every variant that would silently change the numbers is refused.
        require_rejected(with(released_config(), "\"ps_version\": \"v2\"",
                              "\"ps_version\": \"v1\""),
                         "ps_version v1 was accepted");
        require_rejected(with(released_config(), "\"hidden_act\": \"gelu\"",
                              "\"hidden_act\": \"gelu_pytorch_tanh\""),
                         "a tanh GELU tower was accepted");
        require_rejected(with(released_config(), "\"qkv_bias\": true",
                              "\"qkv_bias\": false"),
                         "an unbiased qkv tower was accepted");
        require_rejected(with(released_config(), "\"use_swiglu_ffn\": false",
                              "\"use_swiglu_ffn\": true"),
                         "a SwiGLU tower was accepted");
        require_rejected(with(released_config(), "\"layerscale_value\": 1.0",
                              "\"layerscale_value\": 0.1"),
                         "a LayerScale tower was accepted without its weights");
        require_rejected(with(released_config(), "\"downsample_ratio\": 0.5",
                              "\"downsample_ratio\": 0.25"),
                         "a non-2x pixel shuffle was accepted");
        require_rejected(with(released_config(), "\"vit_hidden_size\": 1280",
                              "\"vit_hidden_size\": 1152"),
                         "a disagreeing vit_hidden_size was accepted");
        require_rejected(with(released_config(), "\"model_type\": \"radio\"",
                              "\"model_type\": \"vit\""),
                         "a non-RADIO tower was accepted");
        require_rejected(with(released_config(), "\"max_img_size\": 2048",
                              "\"max_img_size\": 2050"),
                         "a partial cropped-position row was accepted");
        require_rejected("{not json", "malformed config was accepted");

        std::cout << "RADIO vision config tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RADIO vision config test failed: " << error.what() << '\n';
        return 1;
    }
}
