#pragma once

#include "mfq_tensor_backend.h"

#include <optional>
#include <vector>

namespace mfq_qwen4_exp {
using mfq_tensor_backend::Tensor;

Tensor grouped_rms_norm(const Tensor&, const Tensor&, int64_t group_size, double eps);
std::vector<Tensor> gated_residual_pre(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const std::optional<Tensor>& injection, int64_t hidden_size,
    int64_t hc_count, double eps);
Tensor gated_residual_post(
    const Tensor&, const Tensor&, const Tensor&, int64_t hc_count);
Tensor block_scores(const Tensor&, const Tensor&);
std::vector<Tensor> ple_dilated_conv_silu(
    const Tensor&, const Tensor&, const std::optional<Tensor>& state,
    int64_t dilation);
Tensor dense_gqa_attention(
    const Tensor&, const Tensor&, const Tensor&, int64_t offset);
Tensor sparse_gqa_attention(
    const Tensor&, const Tensor&, const Tensor&, const Tensor& indices);

} // namespace mfq_qwen4_exp
