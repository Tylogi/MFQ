#include "models/qwen4_exp/config.h"

namespace mfq::models::qwen4_exp {

Config Config::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

VisionConfig VisionConfig::from_json(std::string_view payload, const Config& text) {
    const auto document = nlohmann::json::parse(payload);
    const auto& vision = document.at("vision_config");
    if (vision.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu_pytorch_tanh" ||
            !vision.value("deepstack_visual_indexes", std::vector<int>{}).empty())
        throw std::runtime_error("unsupported Qwen4 grid-ViT activation or deepstack");
    VisionConfig result;
    auto& config = result.grid;
    config.hidden_size = vision.at("hidden_size").get<int64_t>();
    config.intermediate_size = vision.at("intermediate_size").get<int64_t>();
    config.depth = vision.at("depth").get<int64_t>();
    config.num_heads = vision.at("num_heads").get<int64_t>();
    config.in_channels = vision.value("in_channels", int64_t{3});
    config.patch_size = vision.at("patch_size").get<int64_t>();
    config.temporal_patch_size = vision.at("temporal_patch_size").get<int64_t>();
    config.spatial_merge_size = vision.value("spatial_merge_size", int64_t{2});
    config.out_hidden_size = vision.at("out_hidden_size").get<int64_t>();
    config.num_position_embeddings = vision.at("num_position_embeddings").get<int64_t>();
    config.rope_theta = vision.value("rope_theta", 10000.0);
    if (vision.contains("rope_parameters"))
        config.rope_theta = vision.at("rope_parameters").value("rope_theta", config.rope_theta);
    config.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
    config.validate();
    result.image_token_id = document.at("image_token_id").get<int64_t>();
    result.video_token_id = document.at("video_token_id").get<int64_t>();
    const auto image = result.image_token_id, video = result.video_token_id;
    if (config.out_hidden_size != text.hidden || text.sections.size() != 3 ||
            image < 0 || video < 0 || image >= text.vocab || video >= text.vocab || image == video)
        throw std::runtime_error("invalid Qwen4 grid-ViT text geometry or placeholder IDs");
    return result;
}

} // namespace mfq::models::qwen4_exp
