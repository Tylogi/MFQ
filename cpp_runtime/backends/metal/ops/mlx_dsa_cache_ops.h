#pragma once

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDsaCompressedWindow {
    mlx::core::array latent;
    mlx::core::array partial_kv;
    mlx::core::array partial_score;
};

MlxDsaCompressedWindow mlx_dsa_compress_window(
    const mlx::core::array& kv, const mlx::core::array& score,
    const mlx::core::array& partial_kv, const mlx::core::array& partial_score,
    const mlx::core::array& norm, int partial_length,
    mlx::core::Dtype dtype, float eps);

struct MlxDsaPackedCacheUpdate {
    mlx::core::array cache;
    mlx::core::array normalized;
};

MlxDsaPackedCacheUpdate mlx_dsa_norm_rope_pack_store(
    const mlx::core::array& cache, const mlx::core::array& values,
    const mlx::core::array& norm, const mlx::core::array& cosine,
    const mlx::core::array& sine, int begin, int rotary,
    int group_size, bool e4m3_scale, float eps,
    bool normalize, bool return_normalized = false);

}
