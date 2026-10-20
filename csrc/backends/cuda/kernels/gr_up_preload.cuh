#pragma once

// Load the four streams before their dots to overlap independent weight
// loads, as in Strata's fast Up. M1's 80 blocks nearly fill the 3090 Ti once.
template<int Capacity>
__device__ __forceinline__ void gr_up_uniform16_preload(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4]) {
    const int group=(lane&15)>>1,half=lane&1,at=group*48+half*24;
    if(group<7) {
        int2 weights[4][3],high_values[Capacity][3],low_values[Capacity][3];
        float scales[4],minima[4];
        #pragma unroll
        for(int stream=0;stream<4;++stream) {
            const int row=stream*hidden+column;
            const auto* source=w.q+(w.offsets[row]>>3)+at;
            const size_t meta=size_t(row)*7+group;
            scales[stream]=w.row_scale[row]*float(w.scales[meta]);
            minima[stream]=w.row_minimum[row]*float(w.minima[meta]);
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                if((reinterpret_cast<uintptr_t>(source)&7u)==0)
                    weights[stream][chunk]=*reinterpret_cast<const int2*>(source+chunk*8);
                else {
                    const auto* words=reinterpret_cast<const int*>(source+chunk*8);
                    weights[stream][chunk]=make_int2(words[0],words[1]);
                }
            }
        }
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                const int offset=k*336+at+chunk*8;
                high_values[k][chunk]=*reinterpret_cast<const int2*>(q.high+offset);
                low_values[k][chunk]=*reinterpret_cast<const int2*>(q.low+offset);
            }
        }
        #pragma unroll
        for(int stream=0;stream<4;++stream) {
            #pragma unroll
            for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                int high=0,low=0;
                #pragma unroll
                for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                    const auto weight=weights[stream][chunk],h=high_values[k][chunk],l=low_values[k][chunk];
                    high=gr_unsigned_signed_dot(weight.x,h.x,high);
                    high=gr_unsigned_signed_dot(weight.y,h.y,high);
                    low=int(__dp4a(unsigned(weight.x),unsigned(l.x),unsigned(low)));
                    low=int(__dp4a(unsigned(weight.y),unsigned(l.y),unsigned(low)));
                }
                int total=0;
                if(half==0) {
                    const auto sum=q.sums[k*7+group];total=sum.x*256+sum.z;
                }
                const float dot=fmaf(256.f,float(high),float(low));
                sums[k][stream]=q.scales[k*7+group].y*fmaf(scales[stream],dot,-minima[stream]*float(total));
            }
        }
    }
    #pragma unroll
    for(int stream=0;stream<4;++stream) {
        #pragma unroll
        for(int k=0;k<Capacity;++k) {
            #pragma unroll
            for(int shift=8;shift;shift>>=1)sums[k][stream]+=__shfl_down_sync(0xffffffffu,sums[k][stream],shift,16);
        }
    }
}

// Prepared global activation planes have longer pointer/load lifetimes.
// Keep two independent weight streams in flight instead of four so their
// register footprint stays below the 64-register occupancy boundary.
template<int Capacity,int Metadata=0>
__device__ __forceinline__ void gr_up_uniform16_pair_preload(const MhcMatrix& w,GrTwoPlane q,
        int column,int hidden,int samples,int sample_begin,int lane,float (&sums)[Capacity][4],
        const float2* coefficients=nullptr,const int4* row_records=nullptr,const uint16_t* group_records=nullptr) {
    const int group=(lane&15)>>1,half=lane&1,at=group*48+half*24;
    if(group<7) {
        int2 high_values[Capacity][3],low_values[Capacity][3];
        #pragma unroll
        for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                const int offset=k*336+at+chunk*8;
                high_values[k][chunk]=*reinterpret_cast<const int2*>(q.high+offset);
                low_values[k][chunk]=*reinterpret_cast<const int2*>(q.low+offset);
            }
        }
        #pragma unroll
        for(int pair=0;pair<4;pair+=2) {
            int2 weights[2][3];float scales[2],minima[2];
            #pragma unroll
            for(int s=0;s<2;++s) {
                const int row=(pair+s)*hidden+column;
                const uint8_t* source;
                const size_t meta=size_t(row)*7+group;
                if constexpr(Metadata==2) {
                    const auto record=row_records[row];
                    const uint64_t offset=uint64_t(uint32_t(record.x))|(uint64_t(uint32_t(record.y))<<32);
                    source=w.q+offset+at;
                    const auto code=group_records[meta];
                    scales[s]=__int_as_float(record.z)*float(code&255u);
                    minima[s]=__int_as_float(record.w)*float(code>>8);
                }else if constexpr(Metadata==1) {
                    source=w.q+(w.offsets[row]>>3)+at;
                    const auto coefficient=coefficients[meta];
                    scales[s]=coefficient.x;minima[s]=coefficient.y;
                }else {
                    source=w.q+(w.offsets[row]>>3)+at;
                    scales[s]=w.row_scale[row]*float(w.scales[meta]);
                    minima[s]=w.row_minimum[row]*float(w.minima[meta]);
                }
                #pragma unroll
                for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                    if((reinterpret_cast<uintptr_t>(source)&7u)==0)
                        weights[s][chunk]=*reinterpret_cast<const int2*>(source+chunk*8);
                    else {
                        const auto* words=reinterpret_cast<const int*>(source+chunk*8);
                        weights[s][chunk]=make_int2(words[0],words[1]);
                    }
                }
            }
            #pragma unroll
            for(int s=0;s<2;++s)for(int k=0;k<Capacity;++k)if(sample_begin+k<samples) {
                int high=0,low=0;
                #pragma unroll
                for(int chunk=0;chunk<3;++chunk)if(at+chunk*8<w.width) {
                    const auto weight=weights[s][chunk],h=high_values[k][chunk],l=low_values[k][chunk];
                    high=gr_unsigned_signed_dot(weight.x,h.x,high);
                    high=gr_unsigned_signed_dot(weight.y,h.y,high);
                    low=int(__dp4a(unsigned(weight.x),unsigned(l.x),unsigned(low)));
                    low=int(__dp4a(unsigned(weight.y),unsigned(l.y),unsigned(low)));
                }
                int total=0;
                if(half==0) {const auto sum=q.sums[k*7+group];total=sum.x*256+sum.z;}
                const float dot=fmaf(256.f,float(high),float(low));
                sums[k][pair+s]=q.scales[k*7+group].y*fmaf(scales[s],dot,-minima[s]*float(total));
            }
        }
    }
    #pragma unroll
    for(int s=0;s<4;++s)for(int k=0;k<Capacity;++k) {
        #pragma unroll
        for(int shift=8;shift;shift>>=1)sums[k][s]+=__shfl_down_sync(0xffffffffu,sums[k][s],shift,16);
    }
}
