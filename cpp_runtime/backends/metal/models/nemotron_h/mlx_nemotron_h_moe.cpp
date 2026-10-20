#include "mlx_nemotron_h_moe.h"

#include "mlx_moe_ops.h"
#include "mlx_nemotron_h_ops.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("invalid Nemotron-H MoE ") + name);
    }
    return static_cast<int>(value);
}

std::string layer_prefix(
    const std::string& tensor_root,
    std::size_t layer_index) {
    return tensor_root + ".block." + std::to_string(layer_index) + ".";
}

std::optional<array> load_optional_vector(
    const MfqContainer& model,
    const std::string& name) {
    if (!model.contains(name)) return std::nullopt;
    return load_nemotron_h_dense_vector(model, name);
}

bool is_packed_expert_bank(const MfqContainer& model, const std::string& name) {
    return model.contains(name) && model.record(name).dtype == "MFE";
}

MlxRoutedLinear load_routed(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "MFE") {
        throw std::runtime_error(
            "Nemotron-H routed expert tensor must use MFE: " + name);
    }
    const auto mapped = model.map_record(name);
    return MlxRoutedLinear(MlxMoeWeight::from_blob(mapped.view()));
}

std::optional<array> to_float32(std::optional<array> value) {
    if (!value.has_value() || value->dtype() == mlx::core::float32) {
        return value;
    }
    return mlx::core::astype(std::move(*value), mlx::core::float32);
}

} // namespace

MlxNemotronHMoeBlock MlxNemotronHMoeBlock::load(
    const MfqContainer& model,
    const NemotronHConfig& config,
    NemotronHMixer kind,
    const std::string& tensor_root,
    std::size_t layer_index) {
    const auto prefix = layer_prefix(tensor_root, layer_index);
    const auto mlp = prefix + "mlp.";
    auto norm = load_nemotron_h_rms_norm(
        model, prefix + "attention.norm.weight", config.rms_norm_eps);
    if (kind == NemotronHMixer::mlp) {
        return MlxNemotronHMoeBlock(
            config,
            std::move(norm),
            MlxLinear::load(model, mlp + "up.weight"),
            MlxLinear::load(model, mlp + "down.weight"));
    }
    if (kind != NemotronHMixer::moe) {
        throw std::runtime_error(
            "Nemotron-H MLP block requires a moe or mlp layer");
    }
    const std::string up_name = mlp + "experts.up.weight";
    const std::string down_name = mlp + "experts.down.weight";
    std::optional<MlxRoutedLinear> routed_up;
    std::optional<MlxRoutedLinear> routed_down;
    std::optional<array> dense_up;
    std::optional<array> dense_down;
    if (is_packed_expert_bank(model, up_name) ||
        is_packed_expert_bank(model, down_name)) {
        routed_up.emplace(load_routed(model, up_name));
        routed_down.emplace(load_routed(model, down_name));
    } else {
        dense_up = load_nemotron_h_dense(model, up_name);
        dense_down = load_nemotron_h_dense(model, down_name);
    }
    return MlxNemotronHMoeBlock(
        config,
        std::move(norm),
        MlxLinear::load(model, mlp + "router.weight"),
        load_optional_vector(model, mlp + "router.bias"),
        MlxLinear::load(model, mlp + "latent.down.weight"),
        MlxLinear::load(model, mlp + "latent.up.weight"),
        std::move(routed_up),
        std::move(routed_down),
        std::move(dense_up),
        std::move(dense_down),
        MlxLinear::load(model, mlp + "shared_expert.up.weight"),
        MlxLinear::load(model, mlp + "shared_expert.down.weight"));
}

MlxNemotronHMoeBlock::MlxNemotronHMoeBlock(
    NemotronHConfig config,
    MlxRmsNorm ffn_norm,
    MlxLinear up,
    MlxLinear down)
    : config_(std::move(config)),
      ffn_norm_(std::move(ffn_norm)),
      up_(std::move(up)),
      down_(std::move(down)) {
    validate_components();
}

MlxNemotronHMoeBlock::MlxNemotronHMoeBlock(
    NemotronHConfig config,
    MlxRmsNorm ffn_norm,
    MlxLinear router,
    std::optional<array> router_bias,
    MlxLinear latent_in,
    MlxLinear latent_out,
    std::optional<MlxRoutedLinear> routed_up,
    std::optional<MlxRoutedLinear> routed_down,
    std::optional<array> dense_up,
    std::optional<array> dense_down,
    MlxLinear shared_up,
    MlxLinear shared_down)
    : config_(std::move(config)),
      is_moe_(true),
      ffn_norm_(std::move(ffn_norm)),
      router_(std::move(router)),
      router_bias_(std::move(router_bias)),
      latent_in_(std::move(latent_in)),
      latent_out_(std::move(latent_out)),
      routed_up_(std::move(routed_up)),
      routed_down_(std::move(routed_down)),
      dense_up_(to_float32(std::move(dense_up))),
      dense_down_(to_float32(std::move(dense_down))),
      shared_up_(std::move(shared_up)),
      shared_down_(std::move(shared_down)) {
    validate_components();
}

void MlxNemotronHMoeBlock::validate_components() const {
    const int hidden = checked_int(config_.hidden_size, "hidden_size");
    if (ffn_norm_.width() != hidden) {
        throw std::runtime_error(
            "Nemotron-H MLP norm width disagrees with hidden_size");
    }
    if (!is_moe_) {
        const int intermediate =
            checked_int(config_.intermediate_size, "intermediate_size");
        if (!up_ || !down_ || up_->input_size() != hidden ||
            up_->output_size() != intermediate ||
            down_->input_size() != intermediate ||
            down_->output_size() != hidden) {
            throw std::runtime_error(
                "Nemotron-H dense MLP component dimensions are inconsistent");
        }
        return;
    }
    const int experts =
        checked_int(config_.num_routed_experts, "routed expert count");
    const int latent = checked_int(config_.moe_latent_size, "latent size");
    const int intermediate =
        checked_int(config_.moe_intermediate_size, "expert intermediate size");
    const int shared =
        checked_int(config_.moe_shared_intermediate_size, "shared intermediate size");
    const bool routed_present = routed_up_.has_value() && routed_down_.has_value();
    const bool dense_present = dense_up_.has_value() && dense_down_.has_value();
    if (routed_present == dense_present) {
        throw std::runtime_error(
            "Nemotron-H MoE requires exactly one routed expert bank");
    }
    if (
        !router_ || router_->input_size() != hidden ||
        router_->output_size() != experts ||
        (router_bias_.has_value() &&
         router_bias_->shape() != Shape{experts}) ||
        !latent_in_ || latent_in_->input_size() != hidden ||
        latent_in_->output_size() != latent || !latent_out_ ||
        latent_out_->input_size() != latent || latent_out_->output_size() != hidden ||
        !shared_up_ || shared_up_->input_size() != hidden ||
        shared_up_->output_size() != shared || !shared_down_ ||
        shared_down_->input_size() != shared ||
        shared_down_->output_size() != hidden) {
        throw std::runtime_error(
            "Nemotron-H latent MoE component dimensions are inconsistent");
    }
    if (routed_present) {
        if (routed_up_->weight().neuron_len() != latent ||
            routed_up_->weight().out_per_expert() != intermediate ||
            routed_down_->weight().neuron_len() != intermediate ||
            routed_down_->weight().out_per_expert() != latent) {
            throw std::runtime_error(
                "Nemotron-H routed expert bank dimensions are inconsistent");
        }
    } else if (dense_up_->shape() != Shape{experts, intermediate, latent} ||
               dense_down_->shape() != Shape{experts, latent, intermediate}) {
        throw std::runtime_error(
            "Nemotron-H dense routed expert bank dimensions are inconsistent");
    }
}

array MlxNemotronHMoeBlock::forward(const array& input) const {
    if (input.ndim() != 3 || input.shape(0) <= 0 || input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H MLP input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    const auto normalized = ffn_norm_(input);
    return is_moe_ ? forward_moe(normalized) : forward_dense(normalized);
}

array MlxNemotronHMoeBlock::forward_dense(const array& input) const {
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    const int hidden = static_cast<int>(config_.hidden_size);
    const auto flat = mlx::core::reshape(
        round_to_float16(input), Shape{batch * tokens, hidden});
    const auto hidden_values = relu_squared((*up_)(flat));
    const auto output = (*down_)(hidden_values);
    return mlx::core::reshape(output, Shape{batch, tokens, hidden});
}

array MlxNemotronHMoeBlock::forward_moe(const array& input) const {
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    const int hidden = static_cast<int>(config_.hidden_size);
    const int latent = static_cast<int>(config_.moe_latent_size);
    const int top_k = static_cast<int>(config_.num_experts_per_tok);
    const int rows = batch * tokens;
    const auto flat = mlx::core::reshape(
        round_to_float16(input), Shape{rows, hidden});
    const auto logits = mlx::core::astype(
        (*router_)(flat), mlx::core::float32);
    const auto routing = moe_topk(
        logits,
        top_k,
        /*use_sigmoid=*/true,
        /*use_sqrt_softplus=*/false,
        config_.norm_topk_prob,
        /*delayed_softmax=*/false,
        router_bias_,
        std::nullopt,
        1e-20f,
        static_cast<float>(config_.routed_scaling_factor));
    const auto ids = mlx::core::contiguous(routing.ids);
    const auto weights = mlx::core::reshape(
        routing.weights, Shape{rows, top_k, 1});
    const auto latent_values = (*latent_in_)(flat);
    const auto experts = [&]() -> array {
        if (routed_up_) {
            return routed_up_->forward(latent_values, ids);
        }
        const auto selected = mlx::core::take(*dense_up_, ids, 0);
        return mlx::core::reshape(
            mlx::core::matmul(
                selected,
                mlx::core::reshape(
                    latent_values, Shape{rows, 1, latent, 1})),
            Shape{rows, top_k, static_cast<int>(config_.moe_intermediate_size)});
    }();
    const auto activated = relu_squared(experts);
    const auto down = [&]() -> array {
        if (routed_down_) {
            return routed_down_->forward(activated, ids);
        }
        const auto selected = mlx::core::take(*dense_down_, ids, 0);
        return mlx::core::reshape(
            mlx::core::matmul(selected, mlx::core::expand_dims(activated, -1)),
            Shape{rows, top_k, latent});
    }();
    auto routed = mlx::core::sum(down * weights, 1);
    routed = (*latent_out_)(routed);
    const auto shared = round_to_float16(
        (*shared_down_)(relu_squared((*shared_up_)(flat))));
    const auto combined = mlx::core::astype(
        routed + shared, input.dtype());
    return mlx::core::reshape(combined, Shape{batch, tokens, hidden});
}

} // namespace mfq::metal
