#include "mlx_nemotron_h_attention.h"

#include "mlx_nemotron_h_ops.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("invalid Nemotron-H attention ") + name);
    }
    return static_cast<int>(value);
}

std::string layer_prefix(
    const std::string& tensor_root,
    std::size_t layer_index) {
    return tensor_root + ".block." + std::to_string(layer_index) + ".";
}

array attention_operand(const array& value) {
    // The CUDA runtime stores and multiplies attention operands in float16.
    return round_to_float16(value);
}

} // namespace

MlxNemotronHAttentionBlock MlxNemotronHAttentionBlock::load(
    const MfqContainer& model,
    const NemotronHConfig& config,
    const std::string& tensor_root,
    std::size_t layer_index) {
    const auto prefix = layer_prefix(tensor_root, layer_index);
    const auto attention = prefix + "attention.";
    return MlxNemotronHAttentionBlock(
        config,
        load_nemotron_h_rms_norm(model, attention + "norm.weight", config.rms_norm_eps),
        MlxLinear::load(model, attention + "query.weight"),
        MlxLinear::load(model, attention + "key.weight"),
        MlxLinear::load(model, attention + "value.weight"),
        MlxLinear::load(model, attention + "output.weight"));
}

MlxNemotronHAttentionBlock::MlxNemotronHAttentionBlock(
    NemotronHConfig config,
    MlxRmsNorm attention_norm,
    MlxLinear query,
    MlxLinear key,
    MlxLinear value,
    MlxLinear output)
    : config_(std::move(config)),
      attention_norm_(std::move(attention_norm)),
      query_(std::move(query)),
      key_(std::move(key)),
      value_(std::move(value)),
      output_(std::move(output)) {
    std::vector<const MlxLinear*> projections{&query_, &key_, &value_};
    qkv_.emplace(std::move(projections));
    validate_components();
}

void MlxNemotronHAttentionBlock::validate_components() const {
    const int hidden = checked_int(config_.hidden_size, "hidden_size");
    const int heads = checked_int(config_.num_attention_heads, "head count");
    const int kv_heads =
        checked_int(config_.num_key_value_heads, "KV head count");
    const int head_dim = checked_int(config_.head_dim, "head dimension");
    if (heads % kv_heads != 0 ||
        attention_norm_.width() != hidden ||
        query_.input_size() != hidden ||
        query_.output_size() != heads * head_dim ||
        key_.input_size() != hidden ||
        key_.output_size() != kv_heads * head_dim ||
        value_.input_size() != hidden ||
        value_.output_size() != kv_heads * head_dim ||
        output_.input_size() != heads * head_dim ||
        output_.output_size() != hidden) {
        throw std::runtime_error(
            "Nemotron-H attention component dimensions are inconsistent");
    }
}

void MlxNemotronHAttentionBlock::reset_cache(int batch, int initial_capacity) {
    if (batch <= 0) {
        throw std::runtime_error(
            "Nemotron-H attention cache batch must be positive");
    }
    cache_ = std::make_unique<MlxKvCache>(
        batch,
        checked_int(config_.num_key_value_heads, "KV head count"),
        checked_int(config_.max_position_embeddings, "maximum position count"),
        checked_int(config_.head_dim, "head dimension"),
        initial_capacity);
    cache_batch_ = batch;
}

void MlxNemotronHAttentionBlock::materialize_cache() {
    if (cache_) cache_->materialize();
}

void MlxNemotronHAttentionBlock::clear_cache() noexcept {
    cache_.reset();
    cache_batch_ = 0;
}

MlxKvCacheSnapshot MlxNemotronHAttentionBlock::snapshot_cache(bool detached) const {
    if (!cache_ || cache_->position() <= 0) {
        throw std::runtime_error("Nemotron-H attention cache is unavailable");
    }
    return cache_->snapshot(detached);
}

void MlxNemotronHAttentionBlock::restore_cache(const MlxKvCacheSnapshot& snapshot) {
    reset_cache(snapshot.batch, snapshot.capacity);
    cache_->restore_snapshot(snapshot);
}

void MlxNemotronHAttentionBlock::trim_cache(int tokens) {
    if (!cache_) {
        throw std::runtime_error("Nemotron-H attention cache is unavailable");
    }
    cache_->trim(tokens);
}

int MlxNemotronHAttentionBlock::cache_position() const noexcept {
    return cache_ ? cache_->position() : 0;
}

array MlxNemotronHAttentionBlock::forward(const array& input, bool use_cache) {
    if (use_cache && !cache_) {
        reset_cache(input.shape(0));
    }
    return forward(input, use_cache ? cache_position() : 0, use_cache);
}

array MlxNemotronHAttentionBlock::forward(
    const array& input,
    int position_offset,
    bool use_cache) {
    return forward_impl(input, position_offset, use_cache);
}

array MlxNemotronHAttentionBlock::forward_impl(
    const array& input,
    int position_offset,
    bool use_cache) {
    if (input.ndim() != 3 || input.shape(0) <= 0 || input.shape(1) <= 0 ||
        input.shape(2) != config_.hidden_size) {
        throw std::runtime_error(
            "Nemotron-H attention input must have a non-empty "
            "[batch,tokens,hidden] shape");
    }
    const int batch = input.shape(0);
    const int tokens = input.shape(1);
    const int maximum = checked_int(
        config_.max_position_embeddings, "maximum position count");
    if (position_offset < 0 || position_offset > maximum - tokens) {
        throw std::runtime_error(
            "Nemotron-H attention position range is invalid");
    }
    if (use_cache) {
        if (!cache_) {
            if (position_offset != 0) {
                throw std::runtime_error(
                    "Nemotron-H attention cache must start at position zero");
            }
            reset_cache(batch);
        }
        if (cache_batch_ != batch) {
            throw std::runtime_error(
                "Nemotron-H attention cache batch mismatch");
        }
        if (cache_position() != position_offset) {
            throw std::runtime_error(
                "Nemotron-H attention cache append must be contiguous");
        }
    }
    const int heads = static_cast<int>(config_.num_attention_heads);
    const int kv_heads = static_cast<int>(config_.num_key_value_heads);
    const int head_dim = static_cast<int>(config_.head_dim);

    const auto normalized = attention_norm_(input);
    auto projected = (*qkv_)(normalized);
    const auto query = mlx::core::transpose(
        mlx::core::reshape(
            attention_operand(std::move(projected.at(0))),
            Shape{batch, tokens, heads, head_dim}),
        {0, 2, 1, 3});
    const auto key = mlx::core::transpose(
        mlx::core::reshape(
            attention_operand(std::move(projected.at(1))),
            Shape{batch, tokens, kv_heads, head_dim}),
        {0, 2, 1, 3});
    const auto value = mlx::core::transpose(
        mlx::core::reshape(
            attention_operand(std::move(projected.at(2))),
            Shape{batch, tokens, kv_heads, head_dim}),
        {0, 2, 1, 3});

    auto attended = [&]() -> array {
        if (!use_cache) {
            return scaled_dot_product_attention(query, key, value, true);
        }
        auto cached = cache_->append(key, value);
        return cache_->quantized()
            ? cache_->attention(query)
            : scaled_dot_product_attention(
                  query, cached.first, cached.second, true);
    }();
    auto flat = mlx::core::reshape(
        mlx::core::transpose(attended, {0, 2, 1, 3}),
        Shape{batch, tokens, heads * head_dim});
    return output_(attention_operand(std::move(flat)));
}

} // namespace mfq::metal
