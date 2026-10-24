#pragma once
#include "mfe_ffn.h"
namespace mfq::cuda {
// Per-call device scratch. Pairs retain token-major order within each expert;
// unique_pairs contains one original pair per expert, in unspecified order.
// Pair arrays have tokens*routes entries; offsets/counts have experts entries,
// with -1/0 for absent experts. unique_count has one entry. The caller owns all
// six arrays through prepare completion and subsequent graph replays.
struct MfeExpertPlan {
    int32_t* sorted_pairs=nullptr;
    int32_t* pair_ranks=nullptr;
    int32_t* expert_offsets=nullptr;
    int32_t* expert_counts=nullptr;
    int32_t* unique_pairs=nullptr;
    int32_t* unique_count=nullptr;
};
constexpr int kMfeExpertPlanMaxPairs=1024;
// Original compressed field addresses, published only after their first reader
// has completed. Geometry and activation pointers stay in the GPU templates.
struct alignas(16) MfeCachedExpert {
    uint64_t fields[3][8]{};
    uint64_t ready_mask=0;
};
struct MfePrepareSharedGate {
    const int8_t* activation=nullptr;
    const float* activation_scales=nullptr;
    const uint8_t* packed=nullptr;
    const int64_t* offset=nullptr;
    const uint8_t* scales=nullptr;
    const uint8_t* minima=nullptr;
    const float* outer_scale=nullptr;
    const float* outer_minimum=nullptr;
    float* terms=nullptr;
    uint32_t* completed=nullptr;
    __half* output=nullptr;
    int groups=0,quantization_total=0,quantization_rotation=0;
};
struct MfeCachedDispatch {
    const MfeCachedExpert* mapped=nullptr;
    const int32_t* primary_cohorts=nullptr;
    MfePackedProjection* projections=nullptr;
    uint64_t* snapshot_tags=nullptr;
    uint32_t* snapshot_ready=nullptr;
    int primary_counts[3]{};
    bool primary_incomplete=false;
    MfePrepareSharedGate gate{};
};
void mfe_ffn_prepare_warp(const MfeFfnBatch&,const MfeInputQuantization*,int count,int groups,cudaStream_t,
    const MfeExpertPlan& plan={});
void mfe_ffn_prepare_publish(const MfeFfnBatch&,const MfeInputQuantization*,int count,int groups,
    int32_t* mapped_ids,uint32_t* ready,cudaStream_t,const MfeExpertPlan& plan={});
void mfe_ffn_prepare_publish_cached(const MfeFfnBatch&,const MfeInputQuantization*,int count,int groups,
    int32_t* mapped_ids,uint32_t* ready,const MfeCachedDispatch&,cudaStream_t,const MfeExpertPlan& plan={});
void mfe_ffn_wait_transfer_packet(const MfeFfnBatch&,const int32_t* mapped_plan,cudaStream_t);
// Optional M1 route output needs routes+2 integers: count, original slots,
// then the proven all-resident reduction flag.
void mfe_ffn_wait_transfer_packet_snapshot(const MfeFfnBatch&,const int32_t* mapped_plan,
    const void* mapped_table,void* snapshot,uint64_t capacity,cudaStream_t,
    const void* mapped_descriptors=nullptr,void* device_descriptors=nullptr,uint64_t descriptor_capacity_words=0,
    bool cached_resident_packet=false,int32_t* transfer_routes=nullptr);
}
