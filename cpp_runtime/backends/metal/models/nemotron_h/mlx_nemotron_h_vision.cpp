#include "mlx_nemotron_h_vision.h"

#include "mlx_nemotron_h_ops.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int I(std::int64_t value) {
    return static_cast<int>(value);
}

mfq::models::nemotron_h::RadioPositionInterpolation position_table(
    std::int64_t rows,
    std::int64_t columns,
    const NemotronHRadioConfig& config) {
    return mfq::models::nemotron_h::make_radio_position_interpolation(
        static_cast<std::int32_t>(rows),
        static_cast<std::int32_t>(columns),
        static_cast<std::int32_t>(config.position_side()));
}

array dense_vector(const MfqContainer& model, const std::string& name) {
    return load_nemotron_h_dense_vector(model, name);
}

array optional_dense(
    const MfqContainer& model,
    const std::string& name) {
    return load_nemotron_h_dense(model, name);
}

} // namespace

MlxRadioEncoder MlxRadioEncoder::load(
    const MfqContainer& model,
    const NemotronHRadioConfig& config) {
    config.validate();
    std::vector<MlxPreLnVisionBlock> layers;
    layers.reserve(static_cast<std::size_t>(config.depth));
    for (std::int64_t index = 0; index < config.depth; ++index) {
        layers.push_back(MlxPreLnVisionBlock::load(
            model,
            "vision.block." + std::to_string(index),
            static_cast<int>(config.num_heads),
            static_cast<float>(config.layer_norm_eps),
            MlxVisionActivation::gelu_erf));
    }
    return MlxRadioEncoder(
        config,
        optional_dense(model, "vision.patch_embedding.weight"),
        optional_dense(model, "vision.position_embedding.weight"),
        optional_dense(model, "vision.cls_register_token.weight"),
        std::move(layers));
}

MlxRadioEncoder::MlxRadioEncoder(
    NemotronHRadioConfig config,
    array patch_projection,
    array position_embedding,
    array cls_register_token,
    std::vector<MlxPreLnVisionBlock> layers)
    : config_(std::move(config)),
      patch_projection_(mlx::core::astype(
          std::move(patch_projection), mlx::core::float32)),
      position_embedding_(mlx::core::astype(
          std::move(position_embedding), mlx::core::float32)),
      cls_register_token_(mlx::core::astype(
          std::move(cls_register_token), mlx::core::float32)),
      layers_(std::move(layers)) {
    validate_components();
}

void MlxRadioEncoder::validate_components() const {
    const auto side = config_.position_side();
    if (patch_projection_.shape() !=
            Shape{I(config_.hidden_size), I(config_.patch_width())} ||
        position_embedding_.shape() !=
            Shape{1, I(side * side), I(config_.hidden_size)} ||
        cls_register_token_.shape() !=
            Shape{I(config_.summary_tokens()), I(config_.hidden_size)} ||
        layers_.size() != static_cast<std::size_t>(config_.depth)) {
        throw std::runtime_error(
            "Nemotron-H RADIO encoder tensor shapes disagree with config");
    }
}

array MlxRadioEncoder::interpolated_positions(
    std::int64_t rows,
    std::int64_t columns) const {
    const auto interpolation = position_table(rows, columns, config_);
    const auto table = mlx::core::reshape(
        position_embedding_,
        Shape{I(config_.position_side() * config_.position_side()),
              I(config_.hidden_size)});
    const auto indices = mlx::core::array(
        interpolation.indices.begin(),
        Shape{I(interpolation.patch_count * 4)},
        mlx::core::int32);
    const auto weights = mlx::core::reshape(
        mlx::core::array(
            interpolation.weights.begin(),
            Shape{I(interpolation.patch_count * 4)},
            mlx::core::float32),
        Shape{I(interpolation.patch_count), 4, 1});
    return mlx::core::sum(
        mlx::core::reshape(
            mlx::core::take(table, indices, 0),
            Shape{I(interpolation.patch_count), 4, I(config_.hidden_size)}) *
            weights,
        1);
}

array MlxRadioEncoder::embedding_input(
    const array& patches,
    std::int64_t rows,
    std::int64_t columns) const {
    if (rows <= 0 || columns <= 0 ||
        patches.ndim() != 2 || patches.shape(0) != rows * columns ||
        patches.shape(1) != config_.patch_width()) {
        throw std::runtime_error(
            "Nemotron-H RADIO pixel patches disagree with the image grid");
    }
    const auto count = rows * columns;
    auto hidden = mlx::core::matmul(
        mlx::core::astype(patches, mlx::core::float32),
        mlx::core::transpose(patch_projection_));
    hidden = hidden + interpolated_positions(rows, columns);
    auto prefix = mlx::core::reshape(
        cls_register_token_,
        Shape{1, I(config_.summary_tokens()), I(config_.hidden_size)});
    return mlx::core::concatenate(
        {prefix, mlx::core::reshape(
                     hidden, Shape{1, I(count), I(config_.hidden_size)})},
        1);
}

array MlxRadioEncoder::encode(
    const array& patches,
    std::int64_t rows,
    std::int64_t columns) const {
    if (rows <= 0 || columns <= 0 || rows > config_.position_side() ||
        columns > config_.position_side()) {
        throw std::runtime_error("Nemotron-H RADIO image patch grid is invalid");
    }
    const auto count = rows * columns;
    if (patches.ndim() != 2 || patches.shape(0) != count ||
        patches.shape(1) != config_.patch_width()) {
        throw std::runtime_error(
            "Nemotron-H RADIO pixel patches disagree with the image grid");
    }
    auto hidden = mlx::core::matmul(
        mlx::core::astype(patches, mlx::core::float32),
        mlx::core::transpose(patch_projection_));
    hidden = hidden + interpolated_positions(rows, columns);
    auto prefix = mlx::core::reshape(
        cls_register_token_,
        Shape{1, I(config_.summary_tokens()), I(config_.hidden_size)});
    hidden = mlx::core::concatenate(
        {prefix, mlx::core::reshape(
                     hidden, Shape{1, I(count), I(config_.hidden_size)})},
        1);
    for (const auto& layer : layers_) {
        hidden = layer.forward(hidden);
    }
    return mlx::core::reshape(
        mlx::core::slice(
            hidden,
            Shape{0, I(config_.summary_tokens()), 0},
            Shape{1, I(config_.summary_tokens() + count), I(config_.hidden_size)}),
        Shape{I(count), I(config_.hidden_size)});
}

MlxRadioProjector MlxRadioProjector::load(
    const MfqContainer& model,
    const NemotronHRadioConfig& config) {
    return MlxRadioProjector(
        config,
        MlxVisionLayerNorm(
            dense_vector(model, "vision.projector.norm.weight"),
            dense_vector(model, "vision.projector.norm.bias"),
            static_cast<float>(config.layer_norm_eps)),
        MlxRmsNorm(
            dense_vector(model, "vision.projector.mlp.norm.weight"),
            static_cast<float>(config.projector_norm_eps),
            0.0f),
        MlxLinear::load(model, "vision.projector.mlp.up.weight"),
        MlxLinear::load(model, "vision.projector.mlp.down.weight"));
}

MlxRadioProjector::MlxRadioProjector(
    NemotronHRadioConfig config,
    MlxVisionLayerNorm norm,
    MlxRmsNorm mlp_norm,
    MlxLinear up,
    MlxLinear down)
    : config_(std::move(config)),
      norm_(std::move(norm)),
      mlp_norm_(std::move(mlp_norm)),
      up_(std::move(up)),
      down_(std::move(down)) {
    validate_components();
}

void MlxRadioProjector::validate_components() const {
    if (norm_.width() != config_.hidden_size ||
        mlp_norm_.width() != config_.shuffle_width() ||
        up_.input_size() != config_.shuffle_width() ||
        up_.output_size() != config_.projector_intermediate_size ||
        down_.input_size() != config_.projector_intermediate_size ||
        down_.output_size() != config_.text_hidden_size) {
        throw std::runtime_error(
            "Nemotron-H RADIO projector geometry disagrees with its config");
    }
}

array MlxRadioProjector::project(
    const array& features,
    std::int64_t rows,
    std::int64_t columns) const {
    const auto stride = config_.downsample_factor;
    if (rows <= 0 || columns <= 0 || rows % stride || columns % stride) {
        throw std::runtime_error(
            "Nemotron-H RADIO patch grid does not divide the pixel shuffle factor");
    }
    const auto patches = rows * columns;
    if (features.ndim() != 3 || features.shape(0) != 1 ||
        features.shape(1) != patches ||
        features.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H RADIO tower features have an invalid shape");
    }
    auto hidden = norm_(features);
    // Pixel shuffle v2: row-pair / column-pair order. The trailing transpose
    // is not redundant; v1 keeps the transposed grid.
    hidden = mlx::core::reshape(
        hidden, Shape{1, I(rows), I(columns / stride), I(stride * config_.hidden_size)});
    hidden = mlx::core::transpose(hidden, {0, 2, 1, 3});
    hidden = mlx::core::reshape(
        hidden, Shape{1, I(columns / stride), I(rows / stride),
                      I(stride * stride * config_.hidden_size)});
    hidden = mlx::core::transpose(hidden, {0, 2, 1, 3});
    hidden = mlx::core::reshape(
        hidden, Shape{1, I(patches / (stride * stride)), I(config_.shuffle_width())});
    auto normalized = mlp_norm_(hidden);
    auto projected = up_(normalized);
    projected = relu_squared(projected);
    return mlx::core::reshape(
        down_(projected),
        Shape{I(patches / (stride * stride)), I(config_.text_hidden_size)});
}

MlxRadioVisionTower MlxRadioVisionTower::load(
    const MfqContainer& model,
    const NemotronHRadioConfig& config) {
    return MlxRadioVisionTower(
        MlxRadioEncoder::load(model, config),
        MlxRadioProjector::load(model, config));
}

array MlxRadioVisionTower::embedding_input(
    const array& patches,
    std::int64_t rows,
    std::int64_t columns) const {
    return encoder_.embedding_input(patches, rows, columns);
}

array MlxRadioVisionTower::features(
    const array& patches,
    std::int64_t rows,
    std::int64_t columns) const {
    return encoder_.encode(patches, rows, columns);
}

array MlxRadioVisionTower::embed(
    const array& patches,
    std::int64_t rows,
    std::int64_t columns) const {
    return projector_.project(
        mlx::core::reshape(
            encoder_.encode(patches, rows, columns),
            Shape{1, I(rows * columns), I(encoder_.config().hidden_size)}),
        rows,
        columns);
}

std::int64_t MlxRadioVisionTower::tokens(
    std::int64_t rows,
    std::int64_t columns) const noexcept {
    const auto stride = encoder_.config().downsample_factor;
    return rows * columns / (stride * stride);
}

MlxNemotronHRadioPromptComponent MlxNemotronHRadioPromptComponent::load(
    const MfqContainer& model,
    const NemotronHRadioConfig& config,
    std::string input_contract,
    std::string position_policy) {
    if (input_contract != mfq::kMfqRadioCpeInputContract) {
        throw std::invalid_argument(
            "unsupported RADIO vision input contract: " + input_contract);
    }
    if (!position_policy.empty()) {
        throw std::invalid_argument(
            "RADIO vision declares no position policy, got: " + position_policy);
    }
    if (config.image_token_id < 0) {
        throw std::invalid_argument(
            "Nemotron-H RADIO needs the `<image>` placeholder id");
    }
    return MlxNemotronHRadioPromptComponent(
        MlxRadioVisionTower::load(model, config),
        config,
        std::move(input_contract));
}

std::vector<std::pair<std::int64_t, std::int64_t>>
MlxNemotronHRadioPromptComponent::validated_grids(
    const MlxGridMediaInput& media) const {
    const auto stride = config_.downsample_factor;
    if (media.processor != input_contract_ ||
        media.pixel_values.ndim() != 2 ||
        media.pixel_values.shape(1) != config_.patch_width() ||
        media.image_grids.size() != media.media_types.size() ||
        media.image_grids.empty()) {
        throw std::invalid_argument("Nemotron-H RADIO media geometry is invalid");
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> grids;
    grids.reserve(media.image_grids.size());
    std::int64_t patches = 0;
    for (std::size_t image = 0; image < media.image_grids.size(); ++image) {
        if (media.media_types[image] != 1) {
            throw std::invalid_argument(
                "Nemotron-H RADIO accepts images only; video is not supported");
        }
        const auto& grid = media.image_grids[image];
        if (grid.temporal != 1 || grid.height <= 0 || grid.width <= 0 ||
            grid.height % stride || grid.width % stride) {
            throw std::invalid_argument(
                "Nemotron-H RADIO patch grid must be a positive multiple of the "
                "pixel shuffle factor");
        }
        patches += static_cast<std::int64_t>(grid.height) * grid.width;
        grids.emplace_back(grid.height, grid.width);
    }
    if (patches != media.pixel_values.shape(0)) {
        throw std::invalid_argument(
            "Nemotron-H RADIO patch count disagrees with pixel_values");
    }
    return grids;
}

std::vector<std::vector<std::int64_t>>
MlxNemotronHRadioPromptComponent::placeholder_runs(
    const std::vector<std::int64_t>& token_ids,
    const std::vector<std::pair<std::int64_t, std::int64_t>>& grids) const {
    std::vector<std::int64_t> positions;
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (token_ids[index] == config_.image_token_id) {
            positions.push_back(static_cast<std::int64_t>(index));
        }
    }
    std::vector<std::vector<std::int64_t>> runs;
    runs.reserve(grids.size());
    std::size_t offset = 0;
    for (const auto& [rows, columns] : grids) {
        const auto count =
            static_cast<std::size_t>(tower_.tokens(rows, columns));
        if (offset + count > positions.size()) {
            throw std::invalid_argument(
                "Nemotron-H prompt holds fewer `<image>` placeholders than the "
                "images require");
        }
        std::vector<std::int64_t> run(
            positions.begin() + static_cast<std::ptrdiff_t>(offset),
            positions.begin() + static_cast<std::ptrdiff_t>(offset + count));
        for (std::size_t index = 1; index < run.size(); ++index) {
            if (run[index] != run[0] + static_cast<std::int64_t>(index)) {
                throw std::invalid_argument(
                    "Nemotron-H `<image>` placeholders of one image must be "
                    "contiguous");
            }
        }
        runs.push_back(std::move(run));
        offset += count;
    }
    if (offset != positions.size()) {
        throw std::invalid_argument(
            "Nemotron-H prompt holds more `<image>` placeholders than the "
            "images require");
    }
    return runs;
}

MlxPreparedPrompt MlxNemotronHRadioPromptComponent::prepare(
    const std::vector<std::int64_t>& token_ids,
    const array& text_embeddings,
    const MlxGridMediaInput& media) const {
    const auto grids = validated_grids(media);
    (void)placeholder_runs(token_ids, grids);
    if (text_embeddings.ndim() != 3 || text_embeddings.shape(0) != 1 ||
        text_embeddings.shape(1) != static_cast<int>(token_ids.size()) ||
        text_embeddings.shape(2) != config_.text_hidden_size) {
        throw std::invalid_argument(
            "Nemotron-H RADIO text embeddings disagree with the prompt");
    }
    std::vector<array> projected;
    projected.reserve(grids.size());
    std::int64_t patch_offset = 0;
    for (const auto& [rows, columns] : grids) {
        const auto count = rows * columns;
        auto image = mlx::core::slice(
            media.pixel_values,
            Shape{I(patch_offset), 0},
            Shape{I(patch_offset + count), I(config_.patch_width())});
        projected.push_back(tower_.embed(image, rows, columns));
        patch_offset += count;
    }
    const auto replacement = projected.size() == 1
        ? projected.front()
        : mlx::core::concatenate(std::move(projected), 0);
    auto embeddings = replace_multimodal_token_embeddings(
        mlx::core::astype(text_embeddings, mlx::core::float32),
        replacement,
        token_ids,
        {config_.image_token_id});
    const auto tokens = static_cast<std::int64_t>(token_ids.size());
    auto positions = mlx::core::reshape(
        mlx::core::arange(tokens, mlx::core::int32),
        Shape{1, I(tokens)});
    return MlxPreparedPrompt{
        token_ids, std::move(embeddings), std::move(positions), 0};
}

} // namespace mfq::metal
