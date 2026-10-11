#ifdef MFQ_GU_COMPACT_ONLY
#include "mfe_gu_compact_probe_body.inc"
#else
#include "mfe_gu_register_probe_body.inc"
#endif
#include <stdexcept>

namespace mfq::cuda {
template<uint32_t Formats,int MinimumBlocks,int MaximumThreads=512>
void probe_launch_gu(const MfeFfnBatch& b,cudaStream_t stream) {
    const int width=std::max(b.intermediate,b.shared?b.shared_intermediate:0);
    const int partition=16|(b.shared?4:0);
    const dim3 grid((width+7)/8,b.tokens*(b.routes+int(b.shared!=nullptr)));
    mfe_ffn_parallel_gate_up_kernel<MinimumBlocks,true,Formats,false,false,MaximumThreads>
        <<<grid,dim3(32,8),0,stream>>>(b,partition);
}
template<uint32_t Formats>
void probe_choose_gu(const MfeFfnBatch& b,cudaStream_t stream,int variant) {
    if(variant==0)probe_launch_gu<Formats,2>(b,stream);
    else if(variant==1)probe_launch_gu<Formats,5,256>(b,stream);
    else if(variant==2)probe_launch_gu<Formats,3>(b,stream);
    else throw std::runtime_error("invalid GU register variant");
}
void mfe_gu_register_probe(const MfeFfnBatch& b,cudaStream_t stream,int variant) {
    if(b.plan_ready || b.nvq_format_mask&~0x3f922u)
        throw std::runtime_error("GU register probe requires resident supported formats");
    if(!(b.nvq_format_mask&~0x3f820u))probe_choose_gu<0x3f820u>(b,stream,variant);
    else probe_choose_gu<0x3f922u>(b,stream,variant);
}
template<uint32_t Formats,int MinimumBlocks,int MaximumThreads=512>
cudaFuncAttributes probe_gu_attributes() {
    cudaFuncAttributes result{};
    const auto error=cudaFuncGetAttributes(&result,
        mfe_ffn_parallel_gate_up_kernel<MinimumBlocks,true,Formats,false,false,MaximumThreads>);
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    return result;
}
cudaFuncAttributes mfe_gu_register_probe_attributes(uint32_t mask,int variant) {
    if(mask&~0x3f922u)throw std::runtime_error("unsupported GU format mask");
    if(!(mask&~0x3f820u)) {
        if(variant==0)return probe_gu_attributes<0x3f820u,2>();
        if(variant==1)return probe_gu_attributes<0x3f820u,5,256>();
        if(variant==2)return probe_gu_attributes<0x3f820u,3>();
    }else {
        if(variant==0)return probe_gu_attributes<0x3f922u,2>();
        if(variant==1)return probe_gu_attributes<0x3f922u,5,256>();
        if(variant==2)return probe_gu_attributes<0x3f922u,3>();
    }
    throw std::runtime_error("invalid GU register variant");
}
}
