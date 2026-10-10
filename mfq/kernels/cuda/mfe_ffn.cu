#include "mfe_ffn_device.cuh"
#include "mfe_ffn_compact.h"
namespace mfq::cuda {
namespace {
bool mfe_group_dot_enabled() {
    const char* setting=std::getenv("MFQ_MFE_GROUP_DOT");
    return !setting || setting[0]!='0';
}
bool mfe_gpu_bundle_enabled() {
    const char* setting=std::getenv("MFQ_MFE_GPU_BUNDLE");
    return !setting || setting[0]!='0';
}
template<bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
void launch_mfe_down_path(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if constexpr(Grouped)partition|=16;
    const char* canonical_quad=std::getenv("MFQ_MFE_CANONICAL_QUAD");
    if(b.shared && (!b.shared_dense_math || (canonical_quad && canonical_quad[0]=='1'))) {
        if(mfe_gpu_bundle_enabled()) {
            const auto warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(mfe_ffn_down_reduce_kernel_quad<true,Grouped,Formats,DefaultMath,E8Narrow>),
                b.routes+(b.shared?1:0),b.tokens);
            mfe_ffn_down_reduce_kernel_quad<true,Grouped,Formats,DefaultMath,E8Narrow><<<dim3((b.output_width+3)/4,b.tokens),dim3(32,warps),0,stream>>>(b,partition);
        }else {
            const auto warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(mfe_ffn_down_reduce_kernel_quad<false,Grouped,Formats,DefaultMath,E8Narrow>),
                b.routes+(b.shared?1:0),b.tokens);
            mfe_ffn_down_reduce_kernel_quad<false,Grouped,Formats,DefaultMath,E8Narrow><<<dim3((b.output_width+3)/4,b.tokens),dim3(32,warps),0,stream>>>(b,partition);
        }
        return;
    }
    if(mfe_gpu_bundle_enabled()) {
        const auto warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(mfe_ffn_down_reduce_kernel<true,Grouped,Formats,DefaultMath,E8Narrow>),
            b.routes+(b.shared?1:0),b.tokens);
        mfe_ffn_down_reduce_kernel<true,Grouped,Formats,DefaultMath,E8Narrow><<<dim3((b.output_width+1)/2,b.tokens),dim3(32,warps),0,stream>>>(b,partition);
    }else {
        const auto warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(mfe_ffn_down_reduce_kernel<false,Grouped,Formats,DefaultMath,E8Narrow>),
            b.routes+(b.shared?1:0),b.tokens);
        mfe_ffn_down_reduce_kernel<false,Grouped,Formats,DefaultMath,E8Narrow><<<dim3((b.output_width+1)/2,b.tokens),dim3(32,warps),0,stream>>>(b,partition);
    }
}
template<bool Grouped,uint32_t Formats=0x3fffeu,bool DefaultMath=false,bool E8Narrow=false>
void launch_mfe_gate_up_path(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if constexpr(Grouped)partition|=16;
    const char* setting=std::getenv("MFQ_MFE_PARALLEL_GU");
    const bool parallel=!setting || setting[0]!='0';
    if(parallel) {
        const char* shared_first=std::getenv("MFQ_MFE_SHARED_FIRST");
        const char* early=std::getenv("MFQ_MFE_EARLY_GU");
        if(b.plan_ready && b.resident_plan_overlap && (!early || early[0]!='0'))partition|=8;
        // Bit 2 changes CTA order; the lower bits retain the transfer partition.
        if(b.shared && (!shared_first || shared_first[0]!='0'))partition|=4;
        const bool bundle=mfe_gpu_bundle_enabled();
        int warps=bundle?8:4;
        if(const char* requested=std::getenv("MFQ_MFE_GU_WARPS")) {
            const int value=std::atoi(requested);
            if(value==4 || value==8 || value==16)warps=value;
        }
        const dim3 grid((width+warps-1)/warps,b.tokens*(b.routes+(b.shared?1:0)));
        if(bundle)
            mfe_ffn_parallel_gate_up_kernel<2,Grouped,Formats,DefaultMath,E8Narrow><<<grid,dim3(32,warps),0,stream>>>(b,partition);
        else
            mfe_ffn_parallel_gate_up_kernel<1,Grouped,Formats,DefaultMath,E8Narrow><<<grid,dim3(32,warps),0,stream>>>(b,partition);
    } else {
        mfe_ffn_gate_up_kernel<Grouped,Formats,DefaultMath,E8Narrow><<<dim3((width+7)/8,b.tokens*(b.routes+(b.shared?1:0))),dim3(32,4),0,stream>>>(b,partition);
    }
}
template<bool DefaultMath,bool E8Narrow=false>
void launch_mfe_down_math(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    const bool specialize=mfe_ffn_format_specialization_enabled();
    if(mfe_group_dot_enabled() && specialize && !(b.nvq_format_mask&~0x3f922u)) {
        if(!(b.nvq_format_mask&~0x3f820u))launch_mfe_down_path<true,0x3f820u,DefaultMath,E8Narrow>(b,partition,stream);
        else launch_mfe_down_path<true,0x3f922u,DefaultMath,E8Narrow>(b,partition,stream);
    }else if(mfe_group_dot_enabled())launch_mfe_down_path<true,0x3fffeu,DefaultMath,E8Narrow>(b,partition,stream);
    else launch_mfe_down_path<false>(b,partition,stream);
}
template<bool DefaultMath,bool E8Narrow=false>
void launch_mfe_gate_up_math(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    const bool specialize=mfe_ffn_format_specialization_enabled();
    if(mfe_group_dot_enabled() && specialize && !(b.nvq_format_mask&~0x3f922u)) {
        if(!(b.nvq_format_mask&~0x3f820u))launch_mfe_gate_up_path<true,0x3f820u,DefaultMath,E8Narrow>(b,partition,width,stream);
        else launch_mfe_gate_up_path<true,0x3f922u,DefaultMath,E8Narrow>(b,partition,width,stream);
    }else if(mfe_group_dot_enabled())launch_mfe_gate_up_path<true,0x3fffeu,DefaultMath,E8Narrow>(b,partition,width,stream);
    else launch_mfe_gate_up_path<false>(b,partition,width,stream);
}
// 1: both stages; 2: Gate/Up only; 3: Down only. Opt-in during evaluation.
bool mfe_default_math_enabled(const MfeFfnBatch& b,bool down) {
    const char* setting=std::getenv("MFQ_MFE_DEFAULT_MATH");
    return b.default_math && setting &&
        (setting[0]=='1' || setting[0]==(down?'3':'2'));
}
// Reuse the proven default-arithmetic contract to keep the active kernel
// compact. Both contracts are required; raw and special-math callers stay wide.
bool mfe_e8_narrow_enabled(const MfeFfnBatch& b,bool down) {
    return b.e8_narrow && b.default_math && mfe_group_dot_enabled() &&
        mfe_ffn_e8_narrow_requested(b,down);
}
void launch_mfe_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(mfe_ffn_compact_down(b,partition,stream))return;
    if(mfe_e8_narrow_enabled(b,true))launch_mfe_down_math<true,true>(b,partition,stream);
    else if(mfe_default_math_enabled(b,true))launch_mfe_down_math<true>(b,partition,stream);
    else launch_mfe_down_math<false>(b,partition,stream);
}
void launch_mfe_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(mfe_ffn_compact_gate_up(b,partition,width,stream))return;
    if(mfe_e8_narrow_enabled(b,false))launch_mfe_gate_up_math<true,true>(b,partition,width,stream);
    else if(mfe_default_math_enabled(b,false))launch_mfe_gate_up_math<true>(b,partition,width,stream);
    else launch_mfe_gate_up_math<false>(b,partition,width,stream);
}

}
void mfe_ffn_prepare(const MfeFfnBatch& b,const MfeInputQuantization* inputs,int count,int groups,cudaStream_t stream) {
    int blocks=b.tokens*groups;
    blocks=std::max(blocks,(b.tokens*(b.routes+(b.shared?1:0))*b.scale_stride+63)/64);
    mfe_ffn_prepare_kernel<<<blocks,64,0,stream>>>(b,inputs,count);
}
void mfe_ffn_gate_up(const MfeFfnBatch& b,cudaStream_t stream) {
    if(b.plan_ready) {
        mfq::cuda::mfe_ffn_resident(b,stream);mfq::cuda::mfe_ffn_wait_transfer(b,stream);mfq::cuda::mfe_ffn_transferred(b,stream);return;
    }
    const int width=std::max(b.intermediate,b.shared?b.shared_intermediate:0);
    launch_mfe_gate_up(b,0,width,stream);
}
void mfe_ffn_resident(const MfeFfnBatch& b,cudaStream_t stream) {
    const int width=std::max(b.intermediate,b.shared?b.shared_intermediate:0);
    launch_mfe_gate_up(b,1,width,stream);
    launch_mfe_down(b,1,stream);
}
void mfe_ffn_wait_transfer(const MfeFfnBatch& b,cudaStream_t stream) {
    if(b.transfer_ready)mfe_ffn_transfer_ready_kernel<<<1,32,0,stream>>>(b);
}
void mfe_ffn_transferred(const MfeFfnBatch& b,cudaStream_t stream) {
    const int width=std::max(b.intermediate,b.shared?b.shared_intermediate:0);
    launch_mfe_gate_up(b,2,width,stream);
}
void mfe_ffn_wait_cpu(const MfeFfnBatch& b,cudaStream_t stream) {
    if(b.cpu_ready)mfe_ffn_cpu_ready_kernel<<<1,32,0,stream>>>(b);
}
void mfe_ffn_down_reduce_compute(const MfeFfnBatch& b,cudaStream_t stream) {
    launch_mfe_down(b,b.plan_ready?2:0,stream);
}
void mfe_ffn_down_reduce(const MfeFfnBatch& b,cudaStream_t stream) {
    mfq::cuda::mfe_ffn_wait_cpu(b,stream);
    mfq::cuda::mfe_ffn_down_reduce_compute(b,stream);
}
}
