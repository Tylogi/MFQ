#pragma once

// Original Uniform16 activation integers, stored as signed16. Two lanes
// consume each GS48 group; the two integer chains hide DP2A dependency latency.
template<bool HalfGroups>
__device__ __forceinline__ float gr_down_direct_s16(const MhcMatrix& w,GrTwoPlane q,
        int row,int first,int last,int lane) {
    const auto* weights=w.q+(w.offsets[row]>>3);
    const float outer_scale=w.row_scale[row],outer_minimum=w.row_minimum[row];
    const int half=HalfGroups?lane&1:0;
    float partial=0.f;
    for(int group=first+(HalfGroups?lane>>1:lane);group<last;group+=HalfGroups?16:32) {
        int even=0,odd=0;
        #pragma unroll
        for(int chunk=0;chunk<(HalfGroups?3:6);++chunk) {
            const int column=group*48+half*24+chunk*8;
            if(column<w.width) {
                const auto codes=*reinterpret_cast<const int2*>(weights+column);
                const int at=(group-first)*48+half*24+chunk*8;
                const auto values=*reinterpret_cast<const int4*>(q.high+2*at);
                even=gr_s16_u8_low(values.x,unsigned(codes.x),even);
                odd=gr_s16_u8_high(values.y,unsigned(codes.x),odd);
                even=gr_s16_u8_low(values.z,unsigned(codes.y),even);
                odd=gr_s16_u8_high(values.w,unsigned(codes.y),odd);
            }
        }
        const size_t metadata=size_t(row)*w.groups+group;
        const auto statistics=q.scales[group-first];
        const float scale=outer_scale*float(w.scales[metadata]);
        const float minimum=outer_minimum*float(w.minima[metadata]);
        partial=fmaf(statistics.y,fmaf(scale,float(even+odd),
            -minimum*(half==0?statistics.x:0.f)),partial);
    }
    #pragma unroll
    for(int shift=16;shift;shift>>=1)partial+=__shfl_down_sync(0xffffffffu,partial,shift);
    return partial;
}
