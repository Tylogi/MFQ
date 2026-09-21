#pragma once

// Shared stateful Flash-Next building blocks. Keep checkpoint ownership here,
// outside quantized projections and the stateless CUDA compute primitives.
#include "mfq/kernels/cuda/flash_next.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

std::vector<mfq_tensor_backend::Tensor> gdn_cuda(
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor, MfqOptional<mfq_tensor_backend::Tensor>);
mfq_tensor_backend::Tensor ssm_conv_silu_cuda(
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor, int64_t);

namespace mfq::flash_next {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;
using Linear = std::function<Tensor(const Tensor&)>;

inline Tensor rms_norm(const Tensor& value, const Tensor& weight, double eps) {
    auto f = value.to(tb::kFloat32);
    return (f * tb::rsqrt((f * f).mean(-1, true) + eps) * weight.to(tb::kFloat32))
        .to(value.scalar_type());
}

class SequenceCache {
public:
    SequenceCache(int64_t maximum, int64_t width)
        : maximum_(maximum), width_(width) {
        MFQ_RUNTIME_CHECK(maximum > 0 && width > 0, "invalid Flash-Next cache dimensions");
    }
    void reset() { values_ = {}; position_ = 0; }
    int64_t position() const { return position_; }
    const Tensor& storage() const { return values_; }
    void truncate(int64_t keep) {
        MFQ_RUNTIME_CHECK(keep >= 0 && keep <= position_, "invalid Flash-Next cache truncation");
        position_ = keep;
    }
    Tensor append(const Tensor& value) {
        MFQ_RUNTIME_CHECK(value.is_cuda() && value.dim() == 3 && value.size(0) > 0 &&
            value.size(1) > 0 && value.size(2) == width_, "invalid Flash-Next cache append");
        MFQ_RUNTIME_CHECK(!values_.defined() || (value.size(0) == values_.size(0) &&
            value.device() == values_.device()), "reset Flash-Next cache before changing batch/device");
        MFQ_RUNTIME_CHECK(value.size(1) <= maximum_ - position_, "Flash-Next cache exceeds max_context");
        const auto required = position_ + value.size(1);
        if (!values_.defined() || required > values_.size(1)) {
            int64_t capacity = values_.defined() ? values_.size(1) : std::min<int64_t>(16, maximum_);
            while (capacity < required) capacity = std::min(maximum_, capacity > maximum_ / 2 ? maximum_ : capacity * 2);
            auto expanded = tb::zeros({value.size(0), capacity, width_}, value.options().dtype(tb::kFloat16));
            if (position_) expanded.narrow(1, 0, position_).copy_(values_.narrow(1, 0, position_));
            values_ = std::move(expanded);
        }
        values_.narrow(1, position_, value.size(1)).copy_(value.to(tb::kFloat16));
        position_ = required;
        return values_.narrow(1, 0, position_);
    }
private:
    int64_t maximum_, width_, position_ = 0;
    Tensor values_;
};

// Both public graphs select complete visible pools, expand their token IDs,
// pad the fixed budget with -1, and optionally append the incomplete tail.
// argpartition's order/tie order is unspecified in MLX; CUDA top-k selects the
// same set for distinct scores. Attention receives this explicit selected order.
inline Tensor select_pooled_blocks(const Tensor& scores, int64_t query_offset,
    int64_t logical_length, int64_t pool, int64_t budget, bool include_tail) {
    MFQ_RUNTIME_CHECK(scores.is_cuda() && scores.dim() == 3 && pool > 0 && budget > 0 &&
        budget % pool == 0 && query_offset >= 0 && logical_length >= query_offset &&
        scores.size(1) <= logical_length - query_offset && scores.size(2) == logical_length / pool,
        "Flash-Next pool selection geometry mismatch");
    const auto b = scores.size(0), t = scores.size(1), pools = scores.size(2);
    const auto options = scores.options().dtype(tb::kInt64);
    auto absolute = tb::arange(t, options) + query_offset;
    auto selected = tb::full({b,t,budget}, -1, options);
    const auto count = std::min(budget / pool, pools);
    if (count) {
        auto ends = tb::arange(pools, options) * pool + (pool - 1);
        auto visible = (ends.reshape({1,1,pools}) <= absolute.reshape({1,t,1})).expand({b,t,pools});
        auto ranked = tb::where(visible, scores.to(tb::kFloat32), tb::full_like(scores, -1e30).to(tb::kFloat32));
        auto indices = std::get<1>(tb::topk(ranked, count, -1, true, false));
        auto valid = visible.gather(-1, indices).unsqueeze(-1).expand({b,t,count,pool});
        auto expanded = indices.unsqueeze(-1) * pool + tb::arange(pool, options);
        expanded = tb::where(valid, expanded, tb::full_like(expanded, -1));
        selected.narrow(-1, 0, count * pool).copy_(expanded.reshape({b,t,count * pool}));
    }
    if (include_tail && pool > 1) {
        auto count = (absolute + 1).remainder(pool);
        auto offsets = tb::arange(pool - 1, options).unsqueeze(0);
        auto tail = (absolute + 1 - count).unsqueeze(1) + offsets;
        tail = tb::where(offsets < count.unsqueeze(1), tail, tb::full_like(tail, -1));
        selected = tb::cat({selected, tail.unsqueeze(0).expand({b,t,pool - 1})}, -1);
    }
    return selected.to(tb::kInt32).contiguous();
}
} // namespace mfq::flash_next
