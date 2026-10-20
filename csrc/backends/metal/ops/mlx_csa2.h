#pragma once

#include <optional>
#include <mlx/mlx.h>

namespace mfq::metal {

mlx::core::array mlx_csa2_pack(const mlx::core::array& value,
    int bits, int group_size, bool e4m3_scale = false);
mlx::core::array mlx_csa2_unpack(const mlx::core::array& value,
    int bits, int group_size, bool e4m3_scale = false,
    mlx::core::Dtype dtype = mlx::core::float16);
mlx::core::array mlx_csa2_index_scores(const mlx::core::array& query,
    const mlx::core::array& keys, const mlx::core::array& weights,
    int position, int ratio,
    const std::optional<mlx::core::array>& candidates = std::nullopt);
mlx::core::array mlx_csa2_topk(const mlx::core::array& scores, int count,
    const std::optional<mlx::core::array>& ids = std::nullopt);
mlx::core::array mlx_csa2_attention(const mlx::core::array& query,
    const mlx::core::array& local, const std::optional<mlx::core::array>& pooled,
    int pool_length, const mlx::core::array& topk,
    const mlx::core::array& sinks, int position, int ratio, int window);

}
