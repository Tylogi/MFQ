#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace mfq::cuda {
struct MoeMappedCopyDescriptor {
    uint64_t destination=0,source=0,offset=0,bytes=0;
};
static_assert(sizeof(MoeMappedCopyDescriptor)==32);
cudaError_t moe_mapped_copy_launch_geometry(int* blocks,int* threads);
void moe_snapshot_mapped(const void* host_table,const uint32_t* cancelled,void* device_table,
    uint64_t capacity,cudaStream_t);
void moe_copy_mapped(const void* table,const uint32_t* cancelled,int blocks,int threads,cudaStream_t);
void moe_publish_routes(const void* half_input,const int32_t* ids,float* mapped_input,
    int32_t* mapped_ids,uint32_t* sequence,int input_elements,int route_entries,cudaStream_t);
void wait_mapped_flag(const uint32_t* flag,cudaStream_t);
void wait_mapped_plan(const uint32_t* flag,const int32_t* host_kind,int32_t* device_kind,
    int experts,const uint32_t* host_abort,uint32_t* device_abort,cudaStream_t);
void signal_mapped_flag(uint32_t* flag,cudaStream_t);
void moe_update_dispatch_maps(const uint64_t* pointers,int maps,int experts,
    const int32_t* kind,const uint32_t* abort,cudaStream_t);
void moe_classify_resident(const uint64_t* dispatch_pointers,int gate_maps,int up_maps,int down_maps,
    int experts,int32_t* kind,uint32_t* abort,cudaStream_t);
void moe_update_nvq_maps(const uint64_t* maps,int pools,int experts,int32_t* pool,int32_t* local,cudaStream_t);
void moe_plan_nvq_routes(const uint64_t* maps,int pools,int experts,const int32_t* ids,
    int pairs,int32_t* plan,cudaStream_t);
cudaError_t moe_wire_launch_geometry(int* blocks,int* threads);
void moe_scatter_wire(const void* wire,const uint32_t* cancelled,int blocks,int threads,cudaStream_t);
void moe_scatter_wire_phase(const void* wire,const uint32_t* cancelled,int blocks,int threads,int phase,cudaStream_t);
void moe_merge_results(const void* hot,const void* transfer,const void* cpu,const int32_t* ids,
    const int32_t* kind,void* output,int entries,int width,cudaStream_t);
}
