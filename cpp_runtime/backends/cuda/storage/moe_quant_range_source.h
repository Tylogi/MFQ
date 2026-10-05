#pragma once

#include "mfq/mfe_quant_expert_store.h"
#include "moe.h"
#include "runtime/moe_host_expert_cache.h"
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
    void bind_host_cache(std::shared_ptr<MoeHostExpertCache> cache,int source_id);
    MoeHostExpertCache::Lease acquire_expert(int expert) const;
    bool demote_expert(int expert,const MoeHostExpertCache::Load& copy);
    void mark_gpu_resident(int expert,bool resident);
    bool gpu_resident(int expert) const;
    std::size_t expert_field_bytes(int expert) const;
    const std::shared_ptr<MoeHostExpertCache>& host_cache() const { return host_cache_; }
    mfq::MoeCacheKey host_key(int expert) const;
    mfq_tensor_backend::Tensor forward_cpu(
        CudaExecutionContext& execution,mfq_tensor_backend::Tensor input,const MoeRoutePlan& route) const;
    std::uint64_t cpu_projections() const { return cpu_projections_.load(); }
    std::uint64_t materializations() const { return materializations_.load(); }
private:
    std::shared_ptr<mfq::MfeQuantExpertStore> store_;
    std::shared_ptr<MixedMoeRuntime> metadata_;
    std::vector<std::int64_t> q_strides_;
    std::vector<std::size_t> field_bytes_;
    std::shared_ptr<MoeHostExpertCache> host_cache_;
    int source_id_=-1;
    std::unique_ptr<std::atomic_bool[]> gpu_resident_;
    mutable std::atomic<std::uint64_t> cpu_projections_{0};
    mutable std::atomic<std::uint64_t> materializations_{0};
};
