#pragma once

#include "mfq/mfe_quant_expert_store.h"
#include "moe.h"
#include <atomic>

// Shared source for GPU cache misses and original-FP32 CPU cold projections.
// Only one prototype per format cohort is read at registration. Its codebook
// is shared by all experts; fields are released after arena registration.
class MoeQuantRangeSource {
public:
    explicit MoeQuantRangeSource(std::shared_ptr<mfq::MfeQuantExpertStore> store);
    const std::shared_ptr<MixedMoeRuntime>& metadata() const { return metadata_; }
    const mfq::MfeQuantExpertStore& store() const { return *store_; }
    MixedMoePool read_expert(int expert) const;
    mfq_tensor_backend::Tensor forward_cpu(
        CudaExecutionContext& execution,mfq_tensor_backend::Tensor input,const MoeRoutePlan& route) const;
    std::uint64_t cpu_projections() const { return cpu_projections_.load(); }
private:
    std::shared_ptr<mfq::MfeQuantExpertStore> store_;
    std::shared_ptr<MixedMoeRuntime> metadata_;
    std::vector<std::int64_t> q_strides_;
    mutable std::atomic<std::uint64_t> cpu_projections_{0};
};
