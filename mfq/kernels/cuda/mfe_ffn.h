#pragma once
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace mfq::cuda {
struct MfePackedProjection {
    const void* fields[9]{};
    std::int64_t sizes[3]{};
    const std::int32_t* expert_local=nullptr;
    int family=0; // 1: canonical NINT; 2: packed NVQ.
    int local_experts=0,output_rows=0,input_width=0,groups=0,group_size=0;
    int format=0,sub_bits=0,sign_mode=0,nvec=0,nsign=0;
    int q_expert_stride=0,fallback=-1;
};
struct MfeInputQuantization {
    std::int8_t* values=nullptr;
    float* scales=nullptr;
    int groups=0,group_size=0;
};
struct MfeFfnBatch {
    const MfePackedProjection* projections=nullptr;
    const std::int32_t* cohort_by_expert=nullptr;
    int projection_offsets[3]{},projection_counts[3]{};
    const MfePackedProjection* shared=nullptr; // Gate, Up, Down.
    const __half* input=nullptr;
    const std::int32_t* ids=nullptr;
    const float* route_weights=nullptr;
    std::int32_t* kinds=nullptr; // 0: CPU; positive: GPU, stored in VRAM.
    const std::int32_t* host_kinds=nullptr;
    const std::int32_t* host_transfer_index=nullptr;
    const std::uint32_t* host_aborted=nullptr;
    const std::uint32_t* plan_ready=nullptr;
    const std::uint32_t* transfer_ready=nullptr;
    const MfePackedProjection* transferred=nullptr;
    std::int32_t* transfer_index=nullptr;
    const __half* cpu_pairs=nullptr;
    const std::uint32_t* cpu_ready=nullptr;
    std::uint32_t* aborted=nullptr;
    const void* shared_gate=nullptr; // Already evaluated in its original dtype.
    const __half* sigmoid_table=nullptr;
    __half* hidden=nullptr;
    std::int8_t* hidden_quantized=nullptr;
    float* hidden_scales=nullptr;
    std::uint32_t* completion=nullptr;
    __half* resident_pairs=nullptr;
    void* output=nullptr;
    int tokens=0,routes=0,experts=0,input_width=0,output_width=0;
    int intermediate=0,shared_intermediate=0;
    int hidden_stride=0,quantized_stride=0,scale_stride=0;
    bool shared_product_half=false,shared_gate_bfloat=false,output_float=false;
    bool output_pairs=false;
    bool resident_plan_overlap=false;
};
void mfe_ffn_prepare(const MfeFfnBatch&,const MfeInputQuantization*,int count,int groups,cudaStream_t);
void mfe_ffn_gate_up(const MfeFfnBatch&,cudaStream_t);
void mfe_ffn_resident(const MfeFfnBatch&,cudaStream_t);
void mfe_ffn_wait_transfer(const MfeFfnBatch&,cudaStream_t);
void mfe_ffn_transferred(const MfeFfnBatch&,cudaStream_t);
void mfe_ffn_wait_cpu(const MfeFfnBatch&,cudaStream_t);
void mfe_ffn_down_reduce_compute(const MfeFfnBatch&,cudaStream_t);
void mfe_ffn_down_reduce(const MfeFfnBatch&,cudaStream_t);
}
