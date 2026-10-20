#define MFQ_MFE_COMPACT_NVQ 1
#define MFQ_MFE_PARITY_READ 1
#include "ffn_device.cuh"
#include <atomic>
#include <cstdio>

namespace mfq::cuda { namespace {
template<uint32_t Formats>
void launch_cached(const MfeFfnBatch& b,const uint32_t* cached,cudaStream_t stream) {
    if(b.shared) {
        const auto kernel=mfe_ffn_down_reduce_kernel_quad<true,true,Formats,true,false,true>;
        const int warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(kernel),b.routes+1,b.tokens);
        kernel<<<dim3((b.output_width+3)/4,b.tokens),dim3(32,warps),0,stream>>>(b,2,cached);
    }else {
        const auto kernel=mfe_ffn_down_reduce_kernel<true,true,Formats,true,false,true>;
        const int warps=mfe_ffn_down_launch_warps(reinterpret_cast<const void*>(kernel),b.routes,b.tokens);
        kernel<<<dim3((b.output_width+1)/2,b.tokens),dim3(32,warps),0,stream>>>(b,2,cached);
    }
}
}
void mfe_ffn_cached_down(const MfeFfnBatch& b,const uint32_t* cached,cudaStream_t stream) {
    if(!(b.nvq_format_mask&~0x3f820u))launch_cached<0x3f820u>(b,cached,stream);
    else launch_cached<0x3f922u>(b,cached,stream);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_cached_reduce_dispatch scope=first_cached_down enabled=1 tokens=1 routes=%d shared=%d lanes=32\n",
        b.routes,int(b.shared!=nullptr));
}
}
