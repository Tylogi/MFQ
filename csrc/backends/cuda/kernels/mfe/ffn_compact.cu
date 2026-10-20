#define MFQ_MFE_COMPACT_NVQ 1
#include "native/env.h"
#include "ffn_device.cuh"
#include "ffn_compact.h"
#include <atomic>
#include <cstdio>

namespace mfq::cuda {
void mfe_ffn_parity_read_gate_up(const MfeFfnBatch&,int,int,cudaStream_t);
void mfe_ffn_parity_group_gate_up(const MfeFfnBatch&,int,cudaStream_t,const int32_t*);
void mfe_ffn_parity_token_single_gate_up(const MfeFfnBatch&,int,int,cudaStream_t);
void mfe_ffn_parity_transfer_routes(const MfeFfnBatch&,const int32_t*,cudaStream_t);
void mfe_ffn_parity_read_down(const MfeFfnBatch&,int,cudaStream_t);
void mfe_ffn_parity_token_single_down(const MfeFfnBatch&,int,cudaStream_t);
void mfe_ffn_cached_down(const MfeFfnBatch&,const uint32_t*,cudaStream_t);
namespace {
bool enabled(const char* name) {
    const auto* setting=mfq::cuda::env_lookup(name);
    return !setting || setting[0]!='0';
}
bool launch_contract(const MfeFfnBatch& b,bool down) {
    if(!mfe_ffn_compact_requested(b,down) || !enabled("MFQ_MFE_GROUP_DOT") ||
            !enabled("MFQ_MFE_GPU_BUNDLE") || !mfe_ffn_format_specialization_enabled())return false;
    if(down) {
        if(const auto* quad=mfq::cuda::env_lookup("MFQ_MFE_CANONICAL_QUAD"))if(quad[0]=='1')return false;
    }else {
        if(!enabled("MFQ_MFE_PARALLEL_GU"))return false;
        if(const auto* warps=mfq::cuda::env_lookup("MFQ_MFE_GU_WARPS"))if(std::atoi(warps)!=8)return false;
    }
    return true;
}
template<uint32_t Formats>
void gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    partition|=16;
    if(b.plan_ready && b.resident_plan_overlap && enabled("MFQ_MFE_EARLY_GU"))partition|=8;
    if(b.shared && enabled("MFQ_MFE_SHARED_FIRST"))partition|=4;
    mfe_ffn_parallel_gate_up_kernel<2,true,Formats,true,false>
        <<<dim3((width+7)/8,b.tokens*(b.routes+int(b.shared!=nullptr))),dim3(32,8),0,stream>>>(b,partition);
}
bool group_gate_up(const MfeFfnBatch& b,int partition,cudaStream_t stream,const int32_t* routes) {
    if(b.tokens!=1 || b.token_group_counts || b.token_group_aligned_gu ||
            !enabled("MFQ_MFE_GU_GROUP_QUANT") || !enabled("MFQ_MFE_PARITY_READ") ||
            !launch_contract(b,false))return false;
    mfe_ffn_parity_group_gate_up(b,partition,stream,routes);return true;
}
template<uint32_t Formats>
void down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    partition|=16;
    if(b.shared) {
        const auto kernel=mfe_ffn_down_reduce_kernel_quad<true,true,Formats,true,false>;
        const int warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(kernel),b.routes+1,b.tokens);
        kernel<<<dim3((b.output_width+3)/4,b.tokens),dim3(32,warps),0,stream>>>(b,partition,nullptr);
    }else {
        const auto kernel=mfe_ffn_down_reduce_kernel<true,true,Formats,true,false>;
        const int warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(kernel),b.routes,b.tokens);
        kernel<<<dim3((b.output_width+1)/2,b.tokens),dim3(32,warps),0,stream>>>(b,partition,nullptr);
    }
}

// Unshared routes keep the same inlined compact decoder and register budget
// as ordinary decode. Only their row index comes from the token-group plan.
template<uint32_t Formats,int Rows>
__global__ void __launch_bounds__(128,8) token_single_down(MfeFfnBatch b,int partition) {
    const int group=token_group_index<1>(b,blockIdx.y);
    if(group<0 || b.token_group_counts[group]!=1)return;
    const int slot=b.token_group_slots[group*8];
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slot%all_routes;
    const bool shared=route==b.routes;
    if(shared && b.shared_intermediate==0)return;
    const int expert=shared?0:b.ids[(slot/all_routes)*b.routes+route];
    const int kind=shared?1:route_kind(b,expert);
    if(kind<=0 || (partition && kind!=partition))return;
    const int first=(blockIdx.x*blockDim.y+threadIdx.y)*Rows;
    if(first>=b.output_width)return;
    int local;const auto* d=projection(b,2,expert,shared,local,b.plan_ready && kind==2);
    __half values[Rows];
    if constexpr(Rows==4)project_quad<true,Formats,true>(d,local,first,
        b.hidden_quantized+int64_t(slot)*b.quantized_stride,b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum,
        false,b.hidden+int64_t(slot)*b.hidden_stride);
    else project_pair<true,Formats,true>(d,local,first,
        b.hidden_quantized+int64_t(slot)*b.quantized_stride,b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum,
        b.hidden+int64_t(slot)*b.hidden_stride);
#pragma unroll
    for(int r=0;r<Rows;++r)if(threadIdx.x==r && first+r<b.output_width)
        b.resident_pairs[int64_t(slot)*b.output_width+first+r]=values[r];
}
template<uint32_t Formats>
void single_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    partition|=16;
    if(b.plan_ready && b.resident_plan_overlap && enabled("MFQ_MFE_EARLY_GU"))partition|=8;
    mfe_ffn_parallel_gate_up_kernel<2,true,Formats,true,false,true>
        <<<dim3((width+7)/8,b.token_groups),dim3(32,8),0,stream>>>(b,partition);
}
template<uint32_t Formats>
void single_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(b.shared)token_single_down<Formats,4>
        <<<dim3((b.output_width+15)/16,b.token_groups),dim3(32,4),0,stream>>>(b,partition);
    else token_single_down<Formats,2>
        <<<dim3((b.output_width+7)/8,b.token_groups),dim3(32,4),0,stream>>>(b,partition);
}
}
bool mfe_ffn_compact_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!launch_contract(b,false))return false;
    const auto* parity_read=mfq::cuda::env_lookup("MFQ_MFE_PARITY_READ");
    if(parity_read?parity_read[0]!='0':b.tokens==1) {
        mfe_ffn_parity_read_gate_up(b,partition,width,stream);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_parity_read_dispatch scope=first_gu_launch enabled=1 tokens=%d partition=%d mask=%u\n",
            b.tokens,partition,b.nvq_format_mask);
        return true;
    }
    if(!(b.nvq_format_mask&~0x3f820u))gate_up<0x3f820u>(b,partition,width,stream);
    else gate_up<0x3f922u>(b,partition,width,stream);
    return true;
}
// Transfer-only work is usually a small fraction of the routed experts. Its
// existing completion path avoids a second mostly empty quantization node.
// Resident GU continues to use the independent quantization schedule.
bool mfe_ffn_compact_group_gate_up(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    return group_gate_up(b,partition,stream,nullptr);
}
bool mfe_ffn_compact_transferred_fused(const MfeFfnBatch& b,cudaStream_t stream,const int32_t* routes) {
    if(!enabled("MFQ_MFE_TRANSFER_HIDDEN_FUSED") || b.tokens!=1 || !b.plan_ready || !b.completion ||
            b.token_group_counts || b.token_group_aligned_gu || !launch_contract(b,false))return false;
    if(group_gate_up(b,2,stream,routes))return true;
    const int width=std::max(b.intermediate,b.shared?b.shared_intermediate:0);
    const auto* parity_read=mfq::cuda::env_lookup("MFQ_MFE_PARITY_READ");
    if(routes && (!parity_read || parity_read[0]!='0'))mfe_ffn_parity_transfer_routes(b,routes,stream);
    else if(!parity_read || parity_read[0]!='0')mfe_ffn_parity_read_gate_up(b,2,width,stream);
    else if(!(b.nvq_format_mask&~0x3f820u))gate_up<0x3f820u>(b,2,width,stream);
    else gate_up<0x3f922u>(b,2,width,stream);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_transfer_hidden_fused_dispatch scope=first_transfer_gu enabled=1 tokens=%d\n",b.tokens);
    return true;
}
bool mfe_ffn_compact_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!launch_contract(b,true))return false;
    if(enabled("MFQ_MFE_PARITY_DOWN") && enabled("MFQ_MFE_PARITY_READ")) {
        mfe_ffn_parity_read_down(b,partition,stream);
        return true;
    }
    if(!(b.nvq_format_mask&~0x3f820u))down<0x3f820u>(b,partition,stream);
    else down<0x3f922u>(b,partition,stream);
    return true;
}
bool mfe_ffn_compact_cached_down(const MfeFfnBatch& b,const uint32_t* cached,cudaStream_t stream) {
    if(!cached || b.tokens!=1 || !b.plan_ready || !b.resident_pairs || b.output_pairs ||
            !enabled("MFQ_MFE_CACHED_REDUCE") || !enabled("MFQ_MFE_PARITY_DOWN") ||
            !enabled("MFQ_MFE_PARITY_READ") || !launch_contract(b,true))return false;
    mfe_ffn_cached_down(b,cached,stream);return true;
}
bool mfe_ffn_compact_token_single_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!launch_contract(b,false))return false;
    if(!b.completion && enabled("MFQ_MFE_TOKEN_SINGLE_PARITY") &&
            enabled("MFQ_MFE_PARITY_READ") && enabled("MFQ_MFE_GU_PREQUANTIZED")) {
        mfe_ffn_parity_token_single_gate_up(b,partition,width,stream);
        return true;
    }
    if(!(b.nvq_format_mask&~0x3f820u))single_gate_up<0x3f820u>(b,partition,width,stream);
    else single_gate_up<0x3f922u>(b,partition,width,stream);
    return true;
}
bool mfe_ffn_compact_token_single_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!launch_contract(b,true))return false;
    if(enabled("MFQ_MFE_PARITY_DOWN") && enabled("MFQ_MFE_PARITY_READ")) {
        mfe_ffn_parity_token_single_down(b,partition,stream);
        return true;
    }
    if(!(b.nvq_format_mask&~0x3f820u))single_down<0x3f820u>(b,partition,stream);
    else single_down<0x3f922u>(b,partition,stream);
    return true;
}
}
