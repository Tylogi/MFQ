#pragma once

#include "mfq_tensor_backend.h"

#include <optional>
#include <vector>

namespace mfq_glm5_next {
using mfq_tensor_backend::Tensor;

std::vector<Tensor> mhc_pre(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int64_t sinkhorn_iterations, double hc_eps, double rms_eps);
Tensor mhc_post(const Tensor&, const Tensor&, const Tensor&, const Tensor&);
Tensor kda_forget_gate(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int64_t num_heads, int64_t head_dim, double lower_bound);
Tensor kpool_scores(const Tensor&, const Tensor&, const Tensor&);
Tensor kpool_states(
    const Tensor&, const Tensor&, const Tensor&, int64_t pool);
Tensor dense_mla_attention(
    const Tensor&, const Tensor&, int64_t offset,
    std::optional<double> scale = std::nullopt);
Tensor sparse_mla_attention(
    const Tensor&, const Tensor&, const Tensor& indices,
    std::optional<double> scale = std::nullopt);

} // namespace mfq_glm5_next
