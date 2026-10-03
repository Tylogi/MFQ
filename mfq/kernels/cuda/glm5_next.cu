#include "glm5_next.h"

#include "selected_attention.h"

#include <cmath>

namespace mfq_glm5_next {
namespace tb = mfq_tensor_backend;
using tb::Tensor;

std::vector<Tensor> mhc_pre(const Tensor& input, const Tensor& function,
    const Tensor& base, const Tensor& scale, int64_t iterations, double hc_eps, double rms_eps) {
    mfq_selected_attention::values({&input, &function, &base, &scale});
    MFQ_RUNTIME_CHECK(input.dim() == 4 && input.size(2) > 0 && input.size(3) > 0,
        "GLM mHC requires [B,T,C,H] input");
    const auto streams = input.size(2), hidden = input.size(3);
    const auto mix = (2 + streams) * streams;
    MFQ_RUNTIME_CHECK(function.dim() == 2 && function.size(0) == mix &&
        function.size(1) == streams * hidden && base.dim() == 1 && base.size(0) == mix &&
        scale.dim() == 1 && scale.size(0) == 3 && iterations > 0,
        "GLM mHC parameters disagree");
    MfqCudaGuard guard(input.device());
    auto flat = input.to(tb::kFloat32).reshape({input.size(0), input.size(1), streams * hidden});
    flat = flat * tb::rsqrt((flat * flat).mean(-1, true) + rms_eps);
    auto logits = tb::matmul(flat, function.to(tb::kFloat32).transpose(-1, -2));
    auto pre = tb::sigmoid(logits.narrow(-1, 0, streams) * scale.select(0, 0) +
        base.narrow(0, 0, streams)) + hc_eps;
    auto post = 2.0 * tb::sigmoid(logits.narrow(-1, streams, streams) * scale.select(0, 1) +
        base.narrow(0, streams, streams));
    auto combination_logits = logits.narrow(-1, 2 * streams, streams * streams)
        .reshape({input.size(0), input.size(1), streams, streams}) * scale.select(0, 2) +
        base.narrow(0, 2 * streams, streams * streams).reshape({streams, streams});
    auto combination = tb::softmax(combination_logits, -1) + hc_eps;
    combination = combination / (combination.sum(-2, true) + hc_eps);
    for (int64_t i = 1; i < iterations; ++i) {
        combination = combination / (combination.sum(-1, true) + hc_eps);
        combination = combination / (combination.sum(-2, true) + hc_eps);
    }
    auto collapsed = (pre.unsqueeze(-1) * input).sum(-2).to(input.scalar_type());
    return {post, combination, collapsed};
}

Tensor mhc_post(const Tensor& branch, const Tensor& residual,
                const Tensor& post, const Tensor& combination) {
    mfq_selected_attention::values({&branch, &residual, &post, &combination});
    MFQ_RUNTIME_CHECK(residual.dim() == 4 && branch.dim() == 3 &&
        branch.size(0) == residual.size(0) && branch.size(1) == residual.size(1) &&
        branch.size(2) == residual.size(3), "GLM mHC branch/residual dimensions disagree");
    const auto b = branch.size(0), t = branch.size(1), c = residual.size(2);
    MFQ_RUNTIME_CHECK(post.sizes().vec() == std::vector<int64_t>({b,t,c}) &&
        combination.sizes().vec() == std::vector<int64_t>({b,t,c,c}),
        "GLM mHC post metadata dimensions disagree");
    MfqCudaGuard guard(branch.device());
    auto mixed = mfq_selected_attention::promoted_matmul(
        combination.transpose(-1, -2), residual);
    return post.to(residual.scalar_type()).unsqueeze(-1) * branch.unsqueeze(-2) + mixed;
}

Tensor kda_forget_gate(const Tensor& input, const Tensor& fa, const Tensor& fb,
    const Tensor& bias, const Tensor& a_log, int64_t heads, int64_t width, double lower_bound) {
    mfq_selected_attention::values({&input, &fa, &fb, &bias, &a_log});
    MFQ_RUNTIME_CHECK(input.dim() >= 1 && heads > 0 && width > 0 && fa.dim() == 2 &&
        fa.size(0) == width && fa.size(1) == input.size(-1) && fb.dim() == 2 &&
        fb.size(0) == heads * width && fb.size(1) == width && bias.dim() == 1 &&
        bias.size(0) == heads * width && a_log.dim() == 1 && a_log.size(0) == heads &&
        std::isfinite(lower_bound), "GLM KDA forget-gate dimensions disagree");
    MfqCudaGuard guard(input.device());
    auto reduced = mfq_selected_attention::promoted_matmul(input, fa.transpose(-1, -2));
    auto gate = mfq_selected_attention::promoted_matmul(
        reduced, fb.transpose(-1, -2)).to(tb::kFloat32) + bias.to(tb::kFloat32);
    auto shape = input.sizes().vec();
    shape.back() = heads;
    shape.push_back(width);
    gate = gate.reshape(shape);
    std::vector<int64_t> rate_shape(shape.size(), 1);
    rate_shape[shape.size() - 2] = heads;
    auto rate = a_log.to(tb::kFloat32).exp().reshape(rate_shape);
    return lower_bound * tb::sigmoid(rate * gate);
}

Tensor kpool_scores(const Tensor& query, const Tensor& pooled, const Tensor& weights) {
    mfq_selected_attention::values({&query, &pooled, &weights});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && weights.dim() == 3 &&
        weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
        weights.size(2) == query.size(2), "GLM k-pool head weights disagree");
    MfqCudaGuard guard(query.device());
    auto scores = mfq_selected_attention::dots(query, pooled) /
        std::sqrt(double(query.size(3)));
    auto head_weights = weights.to(tb::kFloat32) / std::sqrt(double(query.size(2)));
    return (scores * head_weights.unsqueeze(-1)).sum(-2);
}

Tensor kpool_states(const Tensor& keys, const Tensor& gates,
                    const Tensor& ape, int64_t pool) {
    mfq_selected_attention::values({&keys, &gates, &ape});
    MFQ_RUNTIME_CHECK(keys.dim() == 3 && gates.sizes() == keys.sizes() && pool > 0 &&
        ape.dim() == 2 && ape.size(0) == pool && ape.size(1) == keys.size(2),
        "GLM k-pool parameter dimensions disagree");
    MfqCudaGuard guard(keys.device());
    const auto complete = keys.size(1) / pool;
    if (complete == 0) return tb::zeros({keys.size(0), 0, keys.size(2)}, keys.options());
    const std::vector<int64_t> shape{keys.size(0), complete, pool, keys.size(2)};
    auto grouped_keys = keys.narrow(1, 0, complete * pool).reshape(shape);
    auto grouped_gates = gates.narrow(1, 0, complete * pool).reshape(shape);
    auto probabilities = tb::softmax(grouped_gates.to(tb::kFloat32) +
        ape.unsqueeze(0).unsqueeze(0), -2);
    return (probabilities.to(keys.scalar_type()) * grouped_keys).sum(-2);
}

Tensor dense_mla_attention(const Tensor& query, const Tensor& cache,
                           int64_t offset, std::optional<double> scale) {
    mfq_selected_attention::values({&query, &cache});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && cache.dim() == 3,
        "GLM MLA requires [B,H,T,D] query and [B,K,D] cache");
    return mfq_selected_attention::dense(query, cache.unsqueeze(1),
        cache.unsqueeze(1), offset,
        scale.value_or(1.0 / std::sqrt(double(query.size(3)))));
}

Tensor sparse_mla_attention(const Tensor& query, const Tensor& cache,
    const Tensor& indices, std::optional<double> scale) {
    mfq_selected_attention::values({&query, &cache});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && cache.dim() == 3,
        "GLM MLA requires [B,H,T,D] query and [B,K,D] cache");
    return mfq_selected_attention::sparse(query, cache.unsqueeze(1),
        cache.unsqueeze(1), indices,
        scale.value_or(1.0 / std::sqrt(double(query.size(3)))), true);
}

} // namespace mfq_glm5_next
