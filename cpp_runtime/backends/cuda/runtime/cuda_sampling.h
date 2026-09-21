#pragma once

#include "mfq/sampling.h"
#include "mfq/runtime.h"
#include "mfq_cuda_ops.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace mfq::cuda {

class SamplingOps {
public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Params = ::MfqSamplingParams;

    SamplingOps(Tensor random_host, Tensor random_cuda)
        : random_host_(std::move(random_host)),
          random_cuda_(std::move(random_cuda)) {}

    Tensor sample_greedy(Tensor logits) {
        return ::sample_greedy_cuda(std::move(logits));
    }

    Tensor sample_stochastic(
            Tensor logits, float random, const Params& params) {
        *random_host_.data_ptr<float>() = random;
        random_cuda_.copy_(random_host_, true);
        return sample_stochastic(
            std::move(logits), random_cuda_, params);
    }

    Tensor sample_stochastic(
            Tensor logits, Tensor random_cuda, const Params& params) {
        if (params.top_k > 0) {
            return ::sample_top_k_top_p_cuda(
                std::move(logits), std::move(random_cuda),
                params.temperature, params.top_k, params.top_p);
        }
        return ::sample_softmax_cuda(
            std::move(logits), std::move(random_cuda), params.temperature);
    }

    Tensor apply_penalties(
            Tensor logits, const Tensor& counts, const Params& params) {
        return ::sample_apply_penalties_cuda(
            std::move(logits), counts, params.presence_penalty,
            params.frequency_penalty, params.repetition_penalty);
    }

private:
    Tensor random_host_;
    Tensor random_cuda_;
};

using Sampler = mfq::Sampler<SamplingOps>;

inline SamplingOps::Tensor sample_logits(
        Sampler& sampler,
        SamplingOps::Tensor logits,
        const SamplingOps::Tensor& counts,
        const MfqTokenConstraintPtr& token_constraint = {}) {
    logits = logits.contiguous().view({1, -1});
    if (sampler.has_penalties()) {
        logits = sampler.apply_penalties(std::move(logits), counts);
    }

    auto next = sampler.sample(logits);
    if (token_constraint && token_constraint->allows &&
        !token_constraint->allows(next.item<std::int64_t>())) {
        auto masked = logits
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous();
        token_constraint->apply(
            masked.data_ptr<float>(),
            static_cast<std::size_t>(masked.numel()));
        next = sampler.sample(masked.to(logits.device()));
        if (!token_constraint->allows(next.item<std::int64_t>())) {
            throw std::runtime_error(
                "CUDA constrained sampler returned an invalid token");
        }
    }
    if (token_constraint && token_constraint->accept) {
        token_constraint->accept(next.item<std::int64_t>());
    }
    return next;
}

} // namespace mfq::cuda
