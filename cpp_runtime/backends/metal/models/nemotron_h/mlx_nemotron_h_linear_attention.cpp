#include "mlx_nemotron_h_linear_attention.h"

#include "mlx_detached_copy.h"
#include "mlx_linear_attention.h"
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
            std::string("invalid Nemotron-H mamba ") + name);
    }
    return static_cast<int>(value);
}

std::string layer_prefix(
    const std::string& tensor_root,
    std::size_t layer_index) {
    return tensor_root + ".block." + std::to_string(layer_index) + ".";
}

array float32_contiguous(array value) {
    if (value.dtype() != mlx::core::float32) {
        value = mlx::core::astype(value, mlx::core::float32);
    }
    return mlx::core::contiguous(std::move(value));
}

// Causal depthwise convolution over the cached window, delegated to the
// shared primitive so the single-token fast kernel and its generic reference
// stay in one place.
struct MlxCausalConvolution {
    array output;
    array state;
};

MlxCausalConvolution causal_convolution_silu(
    const array& input,
    const array& state,
    const array& weight,
    const std::optional<array>& bias) {
    // The shared primitive keeps the single-token fast kernel and the generic
    // causal reference in one place; the state is [batch, kernel-1, channels].
    auto result = cached_depthwise_conv_silu(input, weight, state, 1, bias);
    return {std::move(result.output), std::move(result.state)};
}

} // namespace

MlxNemotronHLinearAttentionBlock MlxNemotronHLinearAttentionBlock::load(
    const MfqContainer& model,
    const NemotronHConfig& config,
    const std::string& tensor_root,
    std::size_t layer_index) {
    const auto prefix = layer_prefix(tensor_root, layer_index);
    const auto linear = prefix + "linear_attention.";
    std::optional<array> convolution_bias;
    if (model.contains(linear + "conv.bias")) {
        convolution_bias = load_nemotron_h_dense_vector(model, linear + "conv.bias");
    }
    auto a_log = load_nemotron_h_dense_vector(model, linear + "a");
    return MlxNemotronHLinearAttentionBlock(
        config,
        load_nemotron_h_rms_norm(
            model, prefix + "attention.norm.weight", config.rms_norm_eps),
        MlxLinear::load(model, linear + "in_projection.weight"),
        float32_contiguous(load_nemotron_h_dense(model, linear + "conv.weight")),
        std::move(convolution_bias),
        float32_contiguous(-mlx::core::exp(a_log)),
        load_nemotron_h_dense_vector(model, linear + "d"),
        load_nemotron_h_dense_vector(model, linear + "dt_bias"),
        load_nemotron_h_dense_vector(model, linear + "norm.weight"),
        MlxLinear::load(model, linear + "output.weight"));
}

MlxNemotronHLinearAttentionBlock::MlxNemotronHLinearAttentionBlock(
    NemotronHConfig config,
    MlxRmsNorm mixer_norm,
    MlxLinear in_projection,
    array convolution_weight,
    std::optional<array> convolution_bias,
    array decay,
    array skip,
    array dt_bias,
    array scan_weight,
    MlxLinear out_projection)
    : config_(std::move(config)),
      mixer_norm_(std::move(mixer_norm)),
      in_projection_(std::move(in_projection)),
      convolution_weight_(std::move(convolution_weight)),
      convolution_bias_(std::move(convolution_bias)),
      decay_(float32_contiguous(std::move(decay))),
      skip_(float32_contiguous(std::move(skip))),
      dt_bias_(float32_contiguous(std::move(dt_bias))),
      scan_weight_(float32_contiguous(std::move(scan_weight))),
      out_projection_(std::move(out_projection)) {
    validate_components();
}

void MlxNemotronHLinearAttentionBlock::validate_components() const {
    const int hidden = checked_int(config_.hidden_size, "hidden_size");
    const int heads = checked_int(config_.mamba_heads, "head count");
    const int head_dim = checked_int(config_.mamba_head_dim, "head dimension");
    const int groups = checked_int(config_.mamba_groups, "group count");
    const int state = checked_int(config_.ssm_state_size, "state size");
    const int kernel = checked_int(config_.conv_kernel, "convolution kernel");
    const int inner = heads * head_dim;
    const int conv_width = inner + 2 * groups * state;
    const int projection_width = 2 * inner + 2 * groups * state + heads;
    const bool conv_shape =
        convolution_weight_.ndim() == 3 &&
        convolution_weight_.shape(0) == conv_width &&
        convolution_weight_.shape(1) == 1 &&
        convolution_weight_.shape(2) == kernel;
    if (kernel < 2 || heads % groups != 0 || state % 32 != 0 ||
        mixer_norm_.width() != hidden ||
        in_projection_.input_size() != hidden ||
        in_projection_.output_size() != projection_width ||
        !conv_shape ||
        (convolution_bias_.has_value() &&
         convolution_bias_->shape() != Shape{conv_width}) ||
        decay_.shape() != Shape{heads} || skip_.shape() != Shape{heads} ||
        dt_bias_.shape() != Shape{heads} ||
        scan_weight_.shape() != Shape{inner} ||
        out_projection_.input_size() != inner ||
        out_projection_.output_size() != hidden) {
        throw std::runtime_error(
            "Nemotron-H mamba component dimensions are inconsistent");
    }
}

void MlxNemotronHLinearAttentionBlock::reset_cache(int batch) {
    if (batch <= 0) {
        throw std::runtime_error(
            "Nemotron-H mamba cache batch must be positive");
    }
    const int kernel = checked_int(config_.conv_kernel, "convolution kernel");
    const int conv_width = static_cast<int>(config_.mamba_conv_width());
    const int heads = static_cast<int>(config_.mamba_heads);
    const int head_dim = static_cast<int>(config_.mamba_head_dim);
    const int state = static_cast<int>(config_.ssm_state_size);
    if (!zero_convolution_state_ || !zero_recurrent_state_ ||
        zero_cache_batch_ != batch) {
        zero_convolution_state_ = mlx::core::zeros(
            Shape{batch, kernel - 1, conv_width}, mlx::core::float32);
        zero_recurrent_state_ = mlx::core::zeros(
            Shape{batch, heads, head_dim, state}, mlx::core::float32);
        zero_cache_batch_ = batch;
    }
    convolution_state_ = *zero_convolution_state_;
    recurrent_state_ = *zero_recurrent_state_;
    cache_position_ = 0;
    cache_batch_ = batch;
    checkpoint_convolution_.reset();
    checkpoint_recurrent_.reset();
    speculative_xbc_ = array(0.0f);
    speculative_dt_ = array(0.0f);
    speculative_start_ = -1;
    speculative_confirmed_ = 0;
    speculative_tokens_ = 0;
    speculative_pending_ = false;
}

void MlxNemotronHLinearAttentionBlock::materialize_cache() {
    if (convolution_state_ && recurrent_state_) {
        mlx::core::eval(*convolution_state_, *recurrent_state_);
    }
}

void MlxNemotronHLinearAttentionBlock::clear_cache() noexcept {
    convolution_state_.reset();
    recurrent_state_.reset();
    zero_convolution_state_.reset();
    zero_recurrent_state_.reset();
    zero_cache_batch_ = 0;
    cache_position_ = 0;
    cache_batch_ = 0;
    checkpoint_convolution_.reset();
    checkpoint_recurrent_.reset();
    speculative_xbc_ = array(0.0f);
    speculative_dt_ = array(0.0f);
    speculative_start_ = -1;
    speculative_confirmed_ = 0;
    speculative_tokens_ = 0;
    speculative_pending_ = false;
}

MlxNemotronHRecurrentSnapshot MlxNemotronHLinearAttentionBlock::snapshot_cache() const {
    if (!convolution_state_ || !recurrent_state_ || cache_position_ <= 0 ||
        cache_batch_ <= 0 || speculative_pending_) {
        throw std::runtime_error(
            "Nemotron-H mamba cache is unavailable");
    }
    auto convolution = detached_copy(*convolution_state_);
    auto recurrent = detached_copy(*recurrent_state_);
    mlx::core::eval(convolution, recurrent);
    return {std::move(convolution), std::move(recurrent), cache_position_, cache_batch_};
}

void MlxNemotronHLinearAttentionBlock::restore_cache(
    const MlxNemotronHRecurrentSnapshot& snapshot) {
    const int kernel = static_cast<int>(config_.conv_kernel);
    const int conv_width = static_cast<int>(config_.mamba_conv_width());
    const int heads = static_cast<int>(config_.mamba_heads);
    const int head_dim = static_cast<int>(config_.mamba_head_dim);
    const int state = static_cast<int>(config_.ssm_state_size);
    if (snapshot.batch <= 0 || snapshot.position <= 0 ||
        snapshot.convolution_state.shape() !=
            Shape{snapshot.batch, kernel - 1, conv_width} ||
        snapshot.recurrent_state.shape() !=
            Shape{snapshot.batch, heads, head_dim, state} ||
        snapshot.convolution_state.dtype() != mlx::core::float32 ||
        snapshot.recurrent_state.dtype() != mlx::core::float32) {
        throw std::runtime_error(
            "Nemotron-H mamba cache snapshot topology mismatch");
    }
    convolution_state_ = detached_copy(snapshot.convolution_state);
    recurrent_state_ = detached_copy(snapshot.recurrent_state);
    mlx::core::eval(*convolution_state_, *recurrent_state_);
    cache_position_ = snapshot.position;
    cache_batch_ = snapshot.batch;
    checkpoint_convolution_.reset();
    checkpoint_recurrent_.reset();
    speculative_pending_ = false;
}

array MlxNemotronHLinearAttentionBlock::forward(
    const array& input,
    bool use_cache) {
    if (input.ndim() != 3 || input.shape(0) <= 0 || input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H mamba input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    if (use_cache && !convolution_state_) {
        reset_cache(input.shape(0));
    }
    return forward(input, use_cache ? cache_position_ : 0, use_cache);
}

array MlxNemotronHLinearAttentionBlock::forward(
    const array& input,
    int position_offset,
    bool use_cache) {
    if (input.ndim() != 3 || input.shape(0) <= 0 || input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H mamba input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    const int maximum = checked_int(
        config_.max_position_embeddings, "maximum position count");
    if (position_offset < 0 || position_offset > maximum - tokens) {
        throw std::runtime_error(
            "Nemotron-H mamba position range is invalid");
    }
    if (use_cache) {
        if (!convolution_state_ || !recurrent_state_) {
            if (position_offset != 0) {
                throw std::runtime_error(
                    "Nemotron-H mamba cache must start at position zero");
            }
            reset_cache(batch);
        }
        if (cache_batch_ != batch) {
            throw std::runtime_error(
                "Nemotron-H mamba cache batch mismatch");
        }
        if (cache_position_ != position_offset) {
            throw std::runtime_error(
                "Nemotron-H mamba cache append must be contiguous");
        }
    }

    const int heads = static_cast<int>(config_.mamba_heads);
    const int head_dim = static_cast<int>(config_.mamba_head_dim);
    const int groups = static_cast<int>(config_.mamba_groups);
    const int state_width = static_cast<int>(config_.ssm_state_size);
    const int inner = heads * head_dim;
    const int conv_width = inner + 2 * groups * state_width;
    const auto activation_dtype = input.dtype();

    const auto normalized = mixer_norm_(input);
    auto projected = float32_contiguous(in_projection_(normalized));
    if (projected.shape(2) != inner + conv_width + heads) {
        throw std::runtime_error(
            "Nemotron-H mamba fused projection width disagrees with config");
    }
    auto gate = mlx::core::slice(projected, Shape{0, 0, 0},
                                 Shape{batch, tokens, inner});
    auto xbc = mlx::core::slice(projected, Shape{0, 0, inner},
                                Shape{batch, tokens, inner + conv_width});
    auto dt = mlx::core::slice(projected, Shape{0, 0, inner + conv_width},
                               Shape{batch, tokens, inner + conv_width + heads});
    if (speculative_pending_) {
        speculative_xbc_ = xbc;
        speculative_dt_ = dt;
    }

    const auto zero_convolution = mlx::core::zeros(
        Shape{batch, static_cast<int>(config_.conv_kernel) - 1, conv_width},
        mlx::core::float32);
    auto convolution = causal_convolution_silu(
        xbc,
        use_cache ? *convolution_state_ : zero_convolution,
        convolution_weight_,
        convolution_bias_);
    const auto convolved = std::move(convolution.output);
    if (use_cache) {
        convolution_state_ = std::move(convolution.state);
    }
    auto x = mlx::core::reshape(
        mlx::core::slice(convolved, Shape{0, 0, 0},
                         Shape{batch, tokens, inner}),
        Shape{batch, tokens, heads, head_dim});
    auto b_signal = mlx::core::reshape(
        mlx::core::slice(convolved, Shape{0, 0, inner},
                         Shape{batch, tokens, inner + groups * state_width}),
        Shape{batch, tokens, groups, state_width});
    auto c_signal = mlx::core::reshape(
        mlx::core::slice(
            convolved, Shape{0, 0, inner + groups * state_width},
            Shape{batch, tokens, inner + 2 * groups * state_width}),
        Shape{batch, tokens, groups, state_width});

    auto scan = selective_state_scan(
        x,
        dt,
        dt_bias_,
        decay_,
        b_signal,
        c_signal,
        skip_,
        use_cache ? recurrent_state_ : std::nullopt,
        config_.time_step_limit_lo,
        config_.time_step_limit_hi);
    const auto scanned = std::move(scan.output);
    if (use_cache) {
        recurrent_state_ = std::move(scan.state);
    }
    const auto normalized_scan = grouped_gated_rms_norm(
        mlx::core::reshape(scanned, Shape{batch * tokens, inner}),
        mlx::core::reshape(gate, Shape{batch * tokens, inner}),
        scan_weight_,
        inner / groups,
        static_cast<float>(config_.rms_norm_eps));
    auto mixed = out_projection_(
        mlx::core::reshape(normalized_scan, Shape{batch, tokens, inner}));
    if (use_cache) {
        cache_position_ += tokens;
    }
    return mlx::core::astype(std::move(mixed), activation_dtype);
}

array MlxNemotronHLinearAttentionBlock::forward_speculative(
    const array& input,
    int position_offset,
    int confirmed_tokens) {
    if (input.ndim() != 3 || input.shape(0) <= 0 ||
        input.shape(1) <= confirmed_tokens || confirmed_tokens <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H speculative mamba input is invalid");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    if (!convolution_state_ || !recurrent_state_) {
        if (position_offset != 0) {
            throw std::runtime_error(
                "Nemotron-H speculative cache must start at position zero");
        }
        reset_cache(batch);
    }
    if (cache_batch_ != batch || cache_position_ != position_offset) {
        throw std::runtime_error(
            "Nemotron-H speculative cache append must be contiguous");
    }
    checkpoint_convolution_ = *convolution_state_;
    checkpoint_recurrent_ = *recurrent_state_;
    speculative_start_ = position_offset;
    speculative_confirmed_ = confirmed_tokens;
    speculative_tokens_ = tokens;
    speculative_pending_ = true;
    try {
        return forward(input, position_offset, true);
    } catch (...) {
        convolution_state_ = checkpoint_convolution_;
        recurrent_state_ = checkpoint_recurrent_;
        cache_position_ = position_offset;
        commit_speculative();
        throw;
    }
}

void MlxNemotronHLinearAttentionBlock::commit_speculative() noexcept {
    checkpoint_convolution_.reset();
    checkpoint_recurrent_.reset();
    speculative_xbc_ = array(0.0f);
    speculative_dt_ = array(0.0f);
    speculative_start_ = -1;
    speculative_confirmed_ = 0;
    speculative_tokens_ = 0;
    speculative_pending_ = false;
}

void MlxNemotronHLinearAttentionBlock::rollback_speculative(int accepted_tokens) {
    if (!speculative_pending_ || !checkpoint_convolution_ ||
        !checkpoint_recurrent_ || speculative_xbc_.ndim() != 3 ||
        speculative_dt_.ndim() != 3 || accepted_tokens < 0 ||
        speculative_confirmed_ + accepted_tokens > speculative_tokens_) {
        throw std::runtime_error(
            "Nemotron-H speculative mamba rollback is invalid");
    }
    const int retained = speculative_confirmed_ + accepted_tokens;
    convolution_state_ = checkpoint_convolution_;
    recurrent_state_ = checkpoint_recurrent_;
    try {
        replay_recurrent(retained);
    } catch (...) {
        commit_speculative();
        throw;
    }
    commit_speculative();
}

void MlxNemotronHLinearAttentionBlock::replay_recurrent(int retained_tokens) {
    if (retained_tokens <= 0 || retained_tokens > speculative_tokens_) {
        throw std::runtime_error(
            "Nemotron-H mamba recurrent replay range is invalid");
    }
    const int batch = speculative_xbc_.shape(0);
    const int heads = static_cast<int>(config_.mamba_heads);
    const int head_dim = static_cast<int>(config_.mamba_head_dim);
    const int groups = static_cast<int>(config_.mamba_groups);
    const int state_width = static_cast<int>(config_.ssm_state_size);
    const int inner = heads * head_dim;
    const auto xbc = mlx::core::slice(
        speculative_xbc_, Shape{0, 0, 0},
        Shape{batch, retained_tokens, speculative_xbc_.shape(2)});
    const auto dt = mlx::core::slice(
        speculative_dt_, Shape{0, 0, 0},
        Shape{batch, retained_tokens, heads});
    auto convolution = causal_convolution_silu(
        xbc,
        *convolution_state_,
        convolution_weight_,
        convolution_bias_);
    const auto convolved = std::move(convolution.output);
    convolution_state_ = std::move(convolution.state);
    auto x = mlx::core::reshape(
        mlx::core::slice(convolved, Shape{0, 0, 0},
                         Shape{batch, retained_tokens, inner}),
        Shape{batch, retained_tokens, heads, head_dim});
    auto b_signal = mlx::core::reshape(
        mlx::core::slice(convolved, Shape{0, 0, inner},
                         Shape{batch, retained_tokens, inner + groups * state_width}),
        Shape{batch, retained_tokens, groups, state_width});
    auto c_signal = mlx::core::reshape(
        mlx::core::slice(
            convolved, Shape{0, 0, inner + groups * state_width},
            Shape{batch, retained_tokens, inner + 2 * groups * state_width}),
        Shape{batch, retained_tokens, groups, state_width});
    const auto scanned = selective_state_scan(
        x,
        dt,
        dt_bias_,
        decay_,
        b_signal,
        c_signal,
        skip_,
        *recurrent_state_,
        config_.time_step_limit_lo,
        config_.time_step_limit_hi);
    recurrent_state_ = scanned.state;
    cache_position_ = speculative_start_ + retained_tokens;
}

} // namespace mfq::metal
