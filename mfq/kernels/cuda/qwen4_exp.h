#pragma once

#include "mfq_tensor_backend.h"

#include <optional>
#include <vector>

namespace mfq_qwen4_exp {
using mfq_tensor_backend::Tensor;

Tensor grouped_rms_norm(const Tensor&, const Tensor&, int64_t group_size, double eps);
Tensor gated_residual_bottleneck(const Tensor&, int64_t streams);
Tensor gated_residual_injection(const Tensor&, int64_t streams);
Tensor gated_residual_mix(const Tensor& projection, const Tensor& normalized, int64_t streams);
std::vector<Tensor> gated_residual_pre(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const std::optional<Tensor>& injection, int64_t hidden_size,
    int64_t hc_count, double eps);
Tensor gated_residual_post(
    const Tensor&, const Tensor&, const Tensor&, int64_t hc_count);
std::vector<Tensor> gated_residual_post_norm(
    const Tensor& branch, const Tensor& residual, const Tensor& injection,
    const Tensor& norm, int64_t hc_count, double eps);
Tensor block_scores(const Tensor&, const Tensor&);
std::vector<Tensor> ple_dilated_conv_silu(
    const Tensor&, const Tensor&, const std::optional<Tensor>& state,
    int64_t dilation);
Tensor dense_gqa_attention(
    const Tensor&, const Tensor&, const Tensor&, int64_t offset);
Tensor sparse_gqa_attention(
    const Tensor&, const Tensor&, const Tensor&, const Tensor& indices);
Tensor sparse_gqa_attention_gate(const Tensor&, const Tensor&, const Tensor&,
    const Tensor& indices, const Tensor& gate, bool half_output);
Tensor qsa_selected_tokens(const Tensor& blocks, const Tensor& positions,
    int64_t pool, int64_t budget);
Tensor attention_gate(const Tensor& attended, const Tensor& gate, bool half_output);
Tensor causal_gqa_attention_gate(const Tensor&, const Tensor&, const Tensor&,
    const Tensor& positions, const Tensor& gate, int64_t columns, bool half_output);

} // namespace mfq_qwen4_exp
