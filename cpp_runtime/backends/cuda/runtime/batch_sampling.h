#pragma once

#include "cuda_sampling.h"

#include <vector>

namespace mfq::cuda {

struct BatchSamplingInput {
    Sampler* sampler = nullptr;
    SamplingOps::Tensor counts;
    MfqTokenConstraintPtr constraint;
};

struct BatchSamplingResult {
    SamplingOps::Tensor next;
    SamplingOps::Tensor host;
};

// Reusable pinned metadata/mask and readback arenas. Reserve before capturing;
// prepare only copies into these stable addresses when the capacity is unchanged.
// Logits are consumed; clone them before sampling if the original values are needed.
// Each prepare/run/finish cycle must complete before the next prepare.
class BatchSampling {
public:
    using Tensor = SamplingOps::Tensor;
    void reserve(int rows, int vocabulary, mfq_tensor_backend::Device device);
    const SamplingPlan& prepare(const Tensor& logits, const std::vector<BatchSamplingInput>& inputs);
    Tensor run(Tensor logits) const;
    BatchSamplingResult finish(Tensor output);
    BatchSamplingResult sample(Tensor logits, const std::vector<BatchSamplingInput>& inputs);
    const SamplingRow* metadata() const;
    const SamplingPlan& plan() const noexcept { return plan_; }

private:
    int capacity_ = 0;
    int vocabulary_ = 0;
    Tensor metadata_host_, metadata_cuda_, result_host_;
    std::vector<BatchSamplingInput> inputs_;
    SamplingPlan plan_;
};

} // namespace mfq::cuda
