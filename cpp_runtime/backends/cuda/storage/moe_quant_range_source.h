#pragma once

#include "mfq/mfe_quant_expert_store.h"
#include "moe.h"
#include "runtime/moe_host_expert_cache.h"
#include <atomic>
#include <unordered_map>

// Shared source for GPU cache misses and original-FP32 CPU cold projections.
// Only one prototype per format cohort is read at registration. Its codebook
// is shared by all experts; fields are released after arena registration.
class MoeQuantRangeSource {
public:
    explicit MoeQuantRangeSource(std::shared_ptr<mfq::MfeQuantExpertStore> store);
    const std::shared_ptr<MixedMoeRuntime>& metadata() const { return metadata_; }
    const mfq::MfeQuantExpertStore& store() const { return *store_; }
    MixedMoePool read_expert(int expert) const;
    MixedMoePool decode_expert(int expert,const mfq::MfeQuantExpert& encoded) const;
    void mark_preload_complete(bool assert_resident) { disk_sealed_.store(assert_resident); preloaded_.store(true); }
    bool expert_disk_sealed() const { return disk_sealed_.load(); }
    void bind_host_cache(std::shared_ptr<MoeHostExpertCache> cache,int source_id);
    MoeHostExpertCache::Lease acquire_expert(int expert) const;
    bool demote_expert(int expert,const MoeHostExpertCache::Load& copy);
    void mark_gpu_resident(int expert,bool resident);
    // Cache transactions already publish/remove the corresponding RAM entry.
    void publish_gpu_resident(int expert,bool resident) noexcept { gpu_resident_[expert].store(resident); }
    bool gpu_resident(int expert) const;
    std::size_t expert_field_bytes(int expert) const;
    const std::shared_ptr<MoeHostExpertCache>& host_cache() const { return host_cache_; }
    mfq::MoeCacheKey host_key(int expert) const;
    mfq_tensor_backend::Tensor forward_cpu(
        CudaExecutionContext& execution,mfq_tensor_backend::Tensor input,const MoeRoutePlan& route,
        const std::unordered_map<int,MoeHostExpertCache::Lease>* prepared = nullptr) const;
    std::uint64_t cpu_projections() const { return cpu_projections_.load(); }
    void record_cpu_projections(std::size_t experts) const { cpu_projections_.fetch_add(experts); }
    std::uint64_t materializations() const { return materializations_.load(); }
    std::uint64_t dense_materializations() const { return dense_materializations_.load(); }
    std::uint64_t expert_disk_reads_after_preload() const { return disk_reads_after_preload_.load(); }
private:
    std::shared_ptr<mfq::MfeQuantExpertStore> store_;
    std::shared_ptr<MixedMoeRuntime> metadata_;
    std::vector<std::int64_t> q_strides_;
    std::vector<std::size_t> field_bytes_;
    bool dense_groups_=false;
    std::shared_ptr<MoeHostExpertCache> host_cache_;
    int source_id_=-1;
    std::unique_ptr<std::atomic_bool[]> gpu_resident_;
    mutable std::atomic<std::uint64_t> cpu_projections_{0};
    mutable std::atomic<std::uint64_t> materializations_{0};
    mutable std::atomic<std::uint64_t> dense_materializations_{0};
    std::atomic_bool disk_sealed_{false};
    mutable std::atomic<std::uint64_t> sealed_disk_attempts_{0};
    std::atomic_bool preloaded_{false};
    mutable std::atomic<std::uint64_t> disk_reads_after_preload_{0};
};
