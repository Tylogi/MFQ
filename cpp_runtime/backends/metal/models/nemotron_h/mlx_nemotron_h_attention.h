#pragma once

#include "nemotron_h_model.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// NoPE grouped-query attention mixer. Nemotron-H never rotates queries or
// keys; positions only address the append-only KV history.
class MlxNemotronHAttentionBlock {
public:
    static MlxNemotronHAttentionBlock load(
        const MfqContainer& model,
        const NemotronHConfig& config,
        const std::string& tensor_root,
        std::size_t layer_index);

    MlxNemotronHAttentionBlock(
        NemotronHConfig config,
        MlxRmsNorm attention_norm,
        MlxLinear query,
        MlxLinear key,
        MlxLinear value,
        MlxLinear output);

    // Returns the mixer branch, not the residual sum.
    mlx::core::array forward(
        const mlx::core::array& input,
        bool use_cache);
    mlx::core::array forward(
        const mlx::core::array& input,
        int position_offset,
        bool use_cache);

    void reset_cache(int batch, int initial_capacity = 16);
    void materialize_cache();
    void clear_cache() noexcept;
    MlxKvCacheSnapshot snapshot_cache(bool detached = true) const;
    void restore_cache(const MlxKvCacheSnapshot& snapshot);
    void trim_cache(int tokens);

    int cache_position() const noexcept;
    int cache_batch() const noexcept {
        return cache_batch_;
    }

private:
    void validate_components() const;
    mlx::core::array forward_impl(
        const mlx::core::array& input,
        int position_offset,
        bool use_cache);

    NemotronHConfig config_;
    MlxRmsNorm attention_norm_;
    MlxLinear query_;
    MlxLinear key_;
    MlxLinear value_;
    MlxLinear output_;
    std::optional<MlxProjectionBatch> qkv_;
    std::unique_ptr<MlxKvCache> cache_;
    int cache_batch_ = 0;
};

} // namespace mfq::metal
