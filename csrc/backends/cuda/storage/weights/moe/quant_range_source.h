#pragma once

#include "quant/mfe_expert_store.h"
#include "ops/moe/moe.h"
#include "host_expert_cache.h"
#include <atomic>
#include <array>
#include <unordered_map>

// Shared source for GPU cache misses and original-FP32 CPU cold projections.
// Only one prototype per format cohort is read at registration. Its codebook
// is shared by all experts; fields are released after arena registration.
class MoePinnedExpertArena;
class MoeQuantRangeSource {
public:
    explicit MoeQuantRangeSource(std::shared_ptr<mfq::MfeQuantExpertStore> store);
    const std::shared_ptr<MixedMoeRuntime>& metadata() const { return metadata_; }
    const mfq::MfeQuantExpertStore& store() const { return *store_; }
    MixedMoePool read_expert(int expert) const;
    MixedMoePool allocate_expert(int expert) const;
    void copy_expert_fields(int expert,const std::vector<mfq_tensor_backend::Tensor>& targets,bool touch_cache=true) const;
    bool copy_nvq_expert_fields(const std::vector<int>& experts,
        const std::array<mfq_tensor_backend::Tensor,4>& targets) const;
    MixedMoePool decode_expert(int expert,const mfq::MfeQuantExpert& encoded) const;
    void mark_preload_complete(bool assert_resident) { disk_sealed_.store(assert_resident); preloaded_.store(true); }
    bool expert_disk_sealed() const { return disk_sealed_.load(); }
    void bind_host_cache(std::shared_ptr<MoeHostExpertCache> cache,int source_id);
    void bind_pinned_arena(std::shared_ptr<MoePinnedExpertArena> arena) {pinned_arena_=std::move(arena);}
    MoeHostExpertCache::Lease acquire_expert(int expert,bool touch=true) const;
    bool demote_expert(int expert,const MoeHostExpertCache::Load& copy);
    void mark_gpu_resident(int expert,bool resident);
    // Cache transactions already publish/remove the corresponding RAM entry.
    void publish_gpu_resident(int expert,bool resident) noexcept { gpu_resident_[expert].store(resident); }
    bool gpu_resident(int expert) const;
    std::size_t expert_field_bytes(int expert) const;
    const std::shared_ptr<MoeHostExpertCache>& host_cache() const { return host_cache_; }
    mfq::engine::MoeCacheKey host_key(int expert) const;
    mfq_tensor_backend::Tensor forward_cpu(
        CudaExecutionContext& execution,mfq_tensor_backend::Tensor input,const MoeRoutePlan& route,
        const std::unordered_map<int,MoeHostExpertCache::Lease>* prepared = nullptr) const;
    std::uint64_t cpu_projections() const { return cpu_projections_.load(); }
    void record_cpu_projections(std::size_t experts) const { cpu_projections_.fetch_add(experts); }
    std::uint64_t materializations() const { return materializations_.load(); }
    std::uint64_t dense_materializations() const { return dense_materializations_.load(); }
    std::uint64_t expert_disk_reads_after_preload() const { return disk_reads_after_preload_.load(); }
    std::uint64_t expert_disk_bytes_after_preload() const { return disk_bytes_after_preload_.load(); }
    std::uint64_t expert_encoded_ns() const { return encoded_ns_.load(); }
    std::uint64_t expert_materialize_ns() const { return materialize_ns_.load(); }
    std::uint64_t expert_nint_materialize_ns() const { return nint_materialize_ns_.load(); }
    std::uint64_t expert_nvq_materialize_ns() const { return nvq_materialize_ns_.load(); }
    std::uint64_t direct_field_reads() const { return direct_field_reads_.load(); }
    std::uint64_t bulk_field_reads() const { return bulk_field_reads_.load(); }
    // Per-format totals: reads, source bytes, encoded read ns, materialization ns.
    using FormatIoStats=std::array<std::array<std::uint64_t,4>,18>;
    FormatIoStats format_io_stats() const;
private:
    void record_format_io(int format,std::uint64_t bytes,std::uint64_t encoded_ns,
                          std::uint64_t materialize_ns,std::uint64_t reads=1) const;
    MixedMoePool read_expert_canonical(int expert) const;
    struct FieldLayout {
        std::vector<std::int64_t> shape;
        mfq_tensor_backend::ScalarType dtype;
        std::size_t offset=0,bytes=0;
    };
    std::shared_ptr<mfq::MfeQuantExpertStore> store_;
    std::shared_ptr<MixedMoeRuntime> metadata_;
    std::vector<std::int64_t> q_strides_;
    std::vector<std::size_t> field_bytes_;
    std::vector<std::vector<FieldLayout>> field_layouts_;
    std::vector<std::size_t> arena_bytes_;
    std::shared_ptr<MoePinnedExpertArena> pinned_arena_;
    bool dense_groups_=false;
    bool profile_ssd_=false;
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
    mutable std::atomic<std::uint64_t> disk_bytes_after_preload_{0};
    mutable std::atomic<std::uint64_t> encoded_ns_{0},materialize_ns_{0};
    mutable std::atomic<std::uint64_t> nint_materialize_ns_{0},nvq_materialize_ns_{0};
    mutable std::atomic<std::uint64_t> direct_field_reads_{0};
    mutable std::atomic<std::uint64_t> bulk_field_reads_{0};
    mutable std::array<std::array<std::atomic<std::uint64_t>,4>,18> format_io_{};
};
