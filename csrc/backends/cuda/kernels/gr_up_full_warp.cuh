#pragma once

// One output per warp doubles the useful warp population. Four lanes read
// a GS48 group in 12-byte pieces, then combine integers into the original
// 24-byte halves before executing the original sixteen-term FP32 tree.
template<int Capacity>
__device__ __forceinline__ void gr_up_uniform16_full_warp(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    const int group=lane>>2,half=(lane>>1)&1,piece=lane&1,at=group*48+half*24+piece*12;
    if(group<7) {
        int high_values[Capacity][3],low_values[Capacity][3];
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
            #pragma unroll
            for(int word=0;word<3;++word)if(at+word*4<w.width) {
                const int offset=k*336+at+word*4;
                high_values[k][word]=*reinterpret_cast<const int*>(q.high+offset);
                low_values[k][word]=*reinterpret_cast<const int*>(q.low+offset);
            }
        }
        #pragma unroll
        for(int pair=0;pair<4;pair+=2) {
            int weights[2][3];float scales[2],minima[2];
            #pragma unroll
            for(int s=0;s<2;++s) {
                const int row=(pair+s)*hidden+column;
                const auto* source=w.q+(w.offsets[row]>>3)+at;
                const size_t meta=size_t(row)*7+group;
                scales[s]=w.row_scale[row]*float(w.scales[meta]);
                minima[s]=w.row_minimum[row]*float(w.minima[meta]);
                #pragma unroll
                for(int word=0;word<3;++word)if(at+word*4<w.width)
                    weights[s][word]=*reinterpret_cast<const int*>(source+word*4);
            }
            #pragma unroll
            for(int s=0;s<2;++s)for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                int high=0,low=0;
                #pragma unroll
                for(int word=0;word<3;++word)if(at+word*4<w.width) {
                    high=gr_unsigned_signed_dot(weights[s][word],high_values[k][word],high);
                    low=int(__dp4a(unsigned(weights[s][word]),unsigned(low_values[k][word]),unsigned(low)));
                }
                high+=__shfl_xor_sync(0x0fffffffu,high,1);
                low+=__shfl_xor_sync(0x0fffffffu,low,1);
                if(piece==0) {
                    int total=0;
                    if(half==0) {const auto sum=q.sums[k*7+group];total=sum.x*256+sum.z;}
                    const float dot=fmaf(256.f,float(high),float(low));
                    sums[k][pair+s]=q.scales[k*7+group].y*fmaf(scales[s],dot,-minima[s]*float(total));
                }
            }
        }
    }
    #pragma unroll
    for(int s=0;s<4;++s)for(int k=0;k<Capacity;++k) {
        #pragma unroll
        for(int shift=16;shift>=2;shift>>=1)
            sums[k][s]+=__shfl_down_sync(0xffffffffu,sums[k][s],shift);
    }
}
