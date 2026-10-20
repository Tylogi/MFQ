#pragma once

namespace mfq::cuda { namespace {

// Each CTA projects one original route slot. Resident projection needs no
// cross-route reduction, shared result array, or final-output branches.
// Inactive slots are cleared just as in the combined Down kernel; the shared
// expert's original slot remains untouched while its independent stream runs.
template<uint32_t Formats,int Rows>
__global__ void __launch_bounds__(128,8) mfe_ffn_resident_down_kernel(MfeFfnBatch b) {
    const int slot=blockIdx.y;
    const int all_routes=b.routes+int(b.shared!=nullptr),route=slot%all_routes;
    const bool shared=route==b.routes;
    if(shared && !b.shared_intermediate)return;
    const int first=(blockIdx.x*4+threadIdx.y)*Rows;
    if(first>=b.output_width)return;
    __half values[Rows]{};
    const int expert=shared?0:b.ids[(slot/all_routes)*b.routes+route];
    if(shared || (unsigned(expert)<unsigned(b.experts) && route_kind(b,expert)==1)) {
        int local;const auto* d=projection(b,2,expert,shared,local,false);
        const auto* q=b.hidden_quantized+int64_t(slot)*b.quantized_stride;
        const auto* scales=b.hidden_scales+int64_t(slot)*b.scale_stride;
        if constexpr(Rows==4)project_quad<true,Formats,true>(d,local,first,q,scales,values,b.warp_multi_sum);
        else project_pair<true,Formats,true>(d,local,first,q,scales,values,b.warp_multi_sum);
    }
#pragma unroll
    for(int r=0;r<Rows;++r)if(threadIdx.x==r && first+r<b.output_width)
        b.resident_pairs[int64_t(slot)*b.output_width+first+r]=values[r];
}

template<uint32_t Formats>
void mfe_ffn_resident_down(const MfeFfnBatch& b,cudaStream_t stream) {
    if(b.shared)mfe_ffn_resident_down_kernel<Formats,4><<<
        dim3((b.output_width+15)/16,b.tokens*(b.routes+1)),dim3(32,4),0,stream>>>(b);
    else mfe_ffn_resident_down_kernel<Formats,2><<<
        dim3((b.output_width+7)/8,b.tokens*b.routes),dim3(32,4),0,stream>>>(b);
}

} }
