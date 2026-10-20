#include "native/env.h"
#define MFQ_MFE_COMPACT_NVQ 1
#define MFQ_MFE_PARITY_READ 1
#include "ffn_device.cuh"
#include "ffn_gu_prequantized.cuh"
#include "ffn_gu_group_quant.cuh"
#include "ffn_gu_transfer_routes.cuh"
#include "ffn_resident_down.cuh"
#include <atomic>
#include <cstdio>

namespace mfq::cuda {
namespace {
template<uint32_t Formats>
void parity_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    const auto* resident=mfq::cuda::env_lookup("MFQ_MFE_RESIDENT_DOWN");
    if((partition&3)==1 && b.resident_pairs && (!resident || resident[0]!='0')) {
        mfe_ffn_resident_down<Formats>(b,stream);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_resident_down_dispatch scope=first_resident_down enabled=1 tokens=%d routes=%d mask=%u warps=4 compiled_formats=%u\n",
            b.tokens,b.routes,b.nvq_format_mask,Formats);
        return;
    }
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

template<uint32_t Formats,int Rows>
__global__ void __launch_bounds__(128,8) parity_single_down(MfeFfnBatch b,int partition) {
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
void parity_token_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(b.shared)parity_single_down<Formats,4><<<
        dim3((b.output_width+15)/16,mfe_token_group_bound(b,1)),dim3(32,4),0,stream>>>(b,partition);
    else parity_single_down<Formats,2><<<
        dim3((b.output_width+7)/8,mfe_token_group_bound(b,1)),dim3(32,4),0,stream>>>(b,partition);
}

template<uint32_t Formats>
void parity_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    partition|=16;
    const auto enabled=[](const char* name){const auto* p=mfq::cuda::env_lookup(name);return !p || p[0]!='0';};
    if(b.plan_ready && b.resident_plan_overlap && enabled("MFQ_MFE_EARLY_GU"))partition|=8;
    if(b.shared && enabled("MFQ_MFE_SHARED_FIRST"))partition|=4;
    const auto grid=dim3((width+7)/8,b.tokens*(b.routes+int(b.shared!=nullptr)));
    if(!b.completion && enabled("MFQ_MFE_GU_PREQUANTIZED")) {
        const int tiles=b.routes*((b.intermediate+7)/8)+(b.shared?(b.shared_intermediate+7)/8:0);
        const bool simple_m1=b.tokens==1 && (!b.shared || (b.intermediate+7)/8==(b.shared_intermediate+7)/8);
        if(simple_m1)mfe_ffn_gu_prequantized_kernel<Formats,true><<<grid,dim3(32,8),0,stream>>>(b,partition);
        else mfe_ffn_gu_prequantized_kernel<Formats,false><<<dim3(tiles,b.tokens),dim3(32,8),0,stream>>>(b,partition);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_gu_prequantized_dispatch scope=first_gu_launch enabled=1 tokens=%d partition=%d mask=%u packed_grid=%d simple_m1=%d route_reads=1 launched_blocks=%d old_blocks=%d\n",
            b.tokens,partition,b.nvq_format_mask,int(!simple_m1),int(simple_m1),tiles*b.tokens,int(grid.x*grid.y));
    }else mfe_ffn_parallel_gate_up_kernel<2,true,Formats,true,false>
        <<<grid,dim3(32,8),0,stream>>>(b,partition);
}
template<uint32_t Formats>
void parity_token_single_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    partition|=16;
    const auto* early=mfq::cuda::env_lookup("MFQ_MFE_EARLY_GU");
    if(b.plan_ready && b.resident_plan_overlap && (!early || early[0]!='0'))partition|=8;
    // Token-group slots already contain their original shared/routed position.
    // The separate hidden quantizer owns Down metadata and completion.
    mfe_ffn_gu_prequantized_kernel<Formats,false,true><<<
        dim3((width+7)/8,mfe_token_group_bound(b,1)),dim3(32,8),0,stream>>>(b,partition);
}
}
// Called only after the ordinary compact launcher proves its full contract.
void mfe_ffn_parity_read_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!(b.nvq_format_mask&~0xf820u))parity_down<0xf820u>(b,partition,stream);
    else parity_down<0x3f922u>(b,partition,stream);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_parity_down_dispatch scope=first_down_launch enabled=1 tokens=%d partition=%d mask=%u\n",
        b.tokens,partition,b.nvq_format_mask);
}
void mfe_ffn_parity_token_single_down(const MfeFfnBatch& b,int partition,cudaStream_t stream) {
    if(!(b.nvq_format_mask&~0xf820u))parity_token_down<0xf820u>(b,partition,stream);
    else parity_token_down<0x3f922u>(b,partition,stream);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_token_single_parity_down_dispatch scope=first_single_group enabled=1 tokens=%d partition=%d mask=%u\n",
        b.tokens,partition,b.nvq_format_mask);
}
void mfe_ffn_parity_read_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!(b.nvq_format_mask&~0xf820u))parity_gate_up<0xf820u>(b,partition,width,stream);
    else parity_gate_up<0x3f922u>(b,partition,width,stream);
}
void mfe_ffn_parity_token_single_gate_up(const MfeFfnBatch& b,int partition,int width,cudaStream_t stream) {
    if(!(b.nvq_format_mask&~0xf820u))parity_token_single_gate_up<0xf820u>(b,partition,width,stream);
    else parity_token_single_gate_up<0x3f922u>(b,partition,width,stream);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_token_single_parity_dispatch scope=first_single_group enabled=1 tokens=%d partition=%d mask=%u\n",
        b.tokens,partition,b.nvq_format_mask);
}
void mfe_ffn_parity_transfer_routes(const MfeFfnBatch& b,const int32_t* routes,cudaStream_t stream) {
    const auto grid=dim3((b.intermediate+7)/8,std::min(b.routes,4));
    if(!(b.nvq_format_mask&~0xf820u))mfe_ffn_gu_transfer_routes_kernel<0xf820u>
        <<<grid,dim3(32,8),0,stream>>>(b,routes);
    else mfe_ffn_gu_transfer_routes_kernel<0x3f922u>
        <<<grid,dim3(32,8),0,stream>>>(b,routes);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_transfer_route_list_dispatch scope=first_transfer_gu enabled=1 tokens=1 workers=%u blocks=%u routes=%d\n",
        grid.y,grid.x*grid.y,b.routes);
}
}

namespace mfq::cuda {
void mfe_ffn_parity_group_gate_up(const MfeFfnBatch& b,int partition,cudaStream_t stream,const int32_t* routes) {
    const auto* early=mfq::cuda::env_lookup("MFQ_MFE_EARLY_GU");
    if(b.plan_ready && b.resident_plan_overlap && (!early || early[0]!='0'))partition|=8;
    // Four warps per projection cover a GS24 group in three balanced rounds.
    if(routes && (partition&3)==2) {
        const dim3 grid(b.scale_stride,b.routes),block(32,8);
        if(!(b.nvq_format_mask&~0xf820u))mfe_ffn_gu_group_quant_kernel<0xf820u,8,true><<<grid,block,0,stream>>>(b,partition,routes);
        else mfe_ffn_gu_group_quant_kernel<0x3f922u,8,true><<<grid,block,0,stream>>>(b,partition,routes);
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "mfe_gu_group_route_dispatch scope=first_transfer_group_gu enabled=1 tokens=1 warps=8 register_blocks=3 row_order=gate_then_up descriptor_copy=warp_shuffle\n");
        return;
    }
    // Larger groups use the same loop and retain its descriptor/row checks.
    const dim3 grid(b.scale_stride,b.tokens*(b.routes+int(b.shared!=nullptr))),block(32,8);
    if(!(b.nvq_format_mask&~0xf820u))mfe_ffn_gu_group_quant_kernel<0xf820u,8><<<grid,block,0,stream>>>(b,partition);
    else mfe_ffn_gu_group_quant_kernel<0x3f922u,8><<<grid,block,0,stream>>>(b,partition);
    static std::atomic<bool> reported{false};
    if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
        "mfe_gu_group_quant_dispatch scope=first_group_gu enabled=1 tokens=%d partition=%d mask=%u warps=8 direct_quant=1 integer_chains=2 row_order=gate_then_up descriptor_copy=warp_shuffle\n",
        b.tokens,partition,b.nvq_format_mask);
}
}
