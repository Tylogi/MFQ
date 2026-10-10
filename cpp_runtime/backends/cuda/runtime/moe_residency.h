#pragma once
#include <cstdint>
#include <memory>
#include <functional>
#include <vector>
class MoeExpertCache;
class MoeResidencyManager {
public:
    struct MemorySample {std::uint64_t resident_bytes=0,available_bytes=0;};
    struct Stats {
        std::uint64_t observed_routes=0,scheduled_rounds=0,candidate_bundles=0;
        std::uint64_t planned_bundles=0,committed_bundles=0,memory_rejections=0;
        std::uint64_t host_pack_bytes=0,direct_upload_bytes=0,backup_peak_bytes=0,shared_backup_bytes=0;
        std::uint64_t candidate_projections=0,planned_projections=0,committed_projections=0,projection_memory_rejections=0;
        std::uint64_t prepare_ns=0,join_ns=0,dma_wait_ns=0,map_ns=0,host_exchange_ns=0,apply_ns=0;
        std::uint64_t decode_prepare_ns=0,decode_join_ns=0,decode_dma_wait_ns=0,decode_map_ns=0,decode_host_exchange_ns=0,decode_apply_ns=0;
        std::uint64_t backup_copies=0,upload_copies=0;
        std::uint64_t parallel_host=0,parallel_host_bytes=0,parallel_host_batches=0;
        std::uint64_t batched=0,batch_device_bytes=0,batch_descriptor_copies=0;
        std::uint64_t async_publish=0,async_publish_ns=0,decode_async_publish_ns=0;
        std::uint64_t window_prepares=0;
        std::uint64_t cached_copy_bytes=0,cached_copy_fields=0;
        std::uint64_t expert_fence_waits=0;
        std::uint64_t cached_copy_batches=0;
        std::uint64_t expert_fence_wait_ns=0,backup_submit_ns=0,refill_submit_ns=0;
        std::uint64_t cached_copy_dma_bytes=0,cached_copy_dma_fields=0;
        std::uint64_t direct_exchange_rounds=0,direct_exchange_bytes=0,direct_exchange_ns=0;
        std::uint64_t cached_copy_descriptors=0,cached_copy_blocks=0;
        std::uint64_t exchange_defaults=0;
    };
    using Observer=std::function<void(const char*,std::size_t)>;
    using MemoryProbe=std::function<MemorySample()>;
    explicit MoeResidencyManager(MoeExpertCache*,Observer observer={},MemoryProbe memory_probe={});
    ~MoeResidencyManager();
    void before_layer(int layer);
    void after_layer(int layer,const std::vector<int32_t>& ids,int tokens,bool window_inflight=false);
    bool prepare_during_window() const noexcept;
    void finish_window(int layer);
    void record_window_expert_fence(int layer,void* stream);
    void apply_pending();
    Stats stats() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void prepare();
    void prepare_batched(std::int64_t bytes);
    void prepare_direct_exchange();
    void publish_pending();
    void rollback();
};
