#define MFQ_MFE_COMPACT_NVQ 1
#include "mfe_ffn_device.cuh"
#include "mfe_ffn_compact.h"

namespace mfq::cuda {
namespace {
bool enabled(const char* name) {
    const auto* setting=std::getenv(name);
    return !setting || setting[0]!='0';
}
bool launch_contract(const MfeFfnBatch& b,bool down) {
    if(!mfe_ffn_compact_requested(b,down) || !enabled("MFQ_MFE_GROUP_DOT") ||
            !enabled("MFQ_MFE_GPU_BUNDLE") || !mfe_ffn_format_specialization_enabled())return false;
    if(down) {
        if(const auto* quad=std::getenv("MFQ_MFE_CANONICAL_QUAD"))if(quad[0]=='1')return false;
    }else {
        if(!enabled("MFQ_MFE_PARALLEL_GU"))return false;
        if(const auto* warps=std::getenv("MFQ_MFE_GU_WARPS"))if(std::atoi(warps)!=8)return false;
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
template<uint32_t Formats>
void down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    partition|=16;
    if(b.shared) {
        const auto kernel=mfe_ffn_down_reduce_kernel_quad<true,true,Formats,true,false>;
        const int warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(kernel),b.routes+1,b.tokens);
        kernel<<<dim3((b.output_width+3)/4,b.tokens),dim3(32,warps),0,stream>>>(b,partition);
    }else {
        const auto kernel=mfe_ffn_down_reduce_kernel<true,true,Formats,true,false>;
        const int warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(kernel),b.routes,b.tokens);
        kernel<<<dim3((b.output_width+1)/2,b.tokens),dim3(32,warps),0,stream>>>(b,partition);
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
    const int expert=shared?0:b.ids[(slot/all_routes)*b.routes+route];
    const int kind=shared?1:route_kind(b,expert);
    if(kind<=0 || (partition && kind!=partition))return;
    const int first=(blockIdx.x*blockDim.y+threadIdx.y)*Rows;
    if(first>=b.output_width)return;
    int local;const auto* d=projection(b,2,expert,shared,local,b.plan_ready && kind==2);
    __half values[Rows];
    if constexpr(Rows==4)project_quad<true,Formats,true>(d,local,first,
        b.hidden_quantized+int64_t(slot)*b.quantized_stride,b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum);
    else project_pair<true,Formats,true>(d,local,first,
        b.hidden_quantized+int64_t(slot)*b.quantized_stride,b.hidden_scales+int64_t(slot)*b.scale_stride,values,b.warp_multi_sum);
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
    if(!(b.nvq_format_mask&~0x3f820u))gate_up<0x3f820u>(b,partition,width,stream);
    else gate_up<0x3f922u>(b,partition,width,stream);
    return true;
}
bool mfe_ffn_compact_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!launch_contract(b,true))return false;
    if(!(b.nvq_format_mask&~0x3f820u))down<0x3f820u>(b,partition,stream);
    else down<0x3f922u>(b,partition,stream);
    return true;
}
bool mfe_ffn_compact_token_single_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!launch_contract(b,false))return false;
    if(!(b.nvq_format_mask&~0x3f820u))single_gate_up<0x3f820u>(b,partition,width,stream);
    else single_gate_up<0x3f922u>(b,partition,width,stream);
    return true;
}
bool mfe_ffn_compact_token_single_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!launch_contract(b,true))return false;
    if(!(b.nvq_format_mask&~0x3f820u))single_down<0x3f820u>(b,partition,stream);
    else single_down<0x3f922u>(b,partition,stream);
    return true;
}
}
