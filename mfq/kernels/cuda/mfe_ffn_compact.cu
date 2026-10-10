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
}
