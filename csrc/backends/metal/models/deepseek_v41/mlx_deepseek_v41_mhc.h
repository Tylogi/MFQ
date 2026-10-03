#pragma once

#include "deepseek_v41_model.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV41HcMetadataResult {
    mlx::core::array post;
    mlx::core::array combination;
    mlx::core::array pre;
};

// Exact official-geometry decode kernels live with the V4.1 Mega-mHC
// adapter. They are deliberately not branches in the healthy V4 HC path.
mlx::core::array deepseek_v41_hc_collapse_norm(
    const mlx::core::array& residual,
    const mlx::core::array& pre,
    const mlx::core::array& norm,
    float norm_eps = 1e-6f);

MlxDeepseekV41HcMetadataResult deepseek_v41_hc_metadata_exact(
    const mlx::core::array& normalized_mixes,
    const mlx::core::array& scale,
    const mlx::core::array& base,
    int sinkhorn_iterations = 20,
    float eps = 1e-6f);

mlx::core::array deepseek_v41_hc_post(
    const mlx::core::array& branch,
    const mlx::core::array& residual,
    const mlx::core::array& post,
    const mlx::core::array& combination);

mlx::core::array deepseek_v41_hc_post_sum(
    const mlx::core::array& routed,
    const mlx::core::array& shared,
    const mlx::core::array& residual,
    const mlx::core::array& post,
    const mlx::core::array& combination);

struct MlxDeepseekV41MhcResult {
    mlx::core::array branch;
    mlx::core::array next_pre;
    MlxDeepseekV41HcMetadataResult expansion;
};

// Single-pass Mega-mHC adapter. V4.1 deliberately carries the pre-mix from
// the preceding sublayer; this differs from the older V4 layer lifecycle.
class MlxDeepseekV41Mhc {
public:
    static MlxDeepseekV41Mhc load(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        const std::string& prefix,
        const std::string& norm_name);

    MlxDeepseekV41MhcResult collapse(
        const mlx::core::array& residual,
        const mlx::core::array& previous_pre) const;

    mlx::core::array expand(
        const mlx::core::array& branch,
        const mlx::core::array& residual,
        const MlxDeepseekV41HcMetadataResult& expansion) const;
    mlx::core::array expand_sum(
        const mlx::core::array& routed,
        const mlx::core::array& shared,
        const mlx::core::array& residual,
        const MlxDeepseekV41HcMetadataResult& expansion) const;

    static mlx::core::array identity_pre(
        int batch,
        int tokens);

private:
    MlxDeepseekV41Mhc(
        DeepseekV41Config config,
        MlxLinear function,
        mlx::core::array base,
        mlx::core::array scale,
        mlx::core::array norm);

    DeepseekV41Config config_;
    MlxLinear function_;
    mlx::core::array base_;
    mlx::core::array scale_;
    MlxRmsNorm norm_;
};

} // namespace mfq::metal
