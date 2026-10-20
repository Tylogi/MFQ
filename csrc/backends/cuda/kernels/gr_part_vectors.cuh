#pragma once

template<int Splits,bool Published>
__device__ __forceinline__ float gr_reduce_split_major_fixed(const float* parts,int row,int rows) {
    float values[Splits];
    #pragma unroll
    for(int split=0;split<Splits;++split) {
        const auto* source=parts+split*rows+row;
        if constexpr(Published)
            asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(values[split]) : "l"(source));
        else values[split]=*source;
    }
    float sum=0.f;
    #pragma unroll
    for(int split=0;split<Splits;++split)sum=__fadd_rn(sum,values[split]);
    return sum;
}

template<bool Published>
__device__ __forceinline__ float gr_reduce_split_major_parts(const float* parts,size_t base,int row,int rows,int splits) {
    if(splits==7)return gr_reduce_split_major_fixed<7,Published>(parts+base,row,rows);
    if(splits==8)return gr_reduce_split_major_fixed<8,Published>(parts+base,row,rows);
    if(splits==13)return gr_reduce_split_major_fixed<13,Published>(parts+base,row,rows);
    if(splits==16)return gr_reduce_split_major_fixed<16,Published>(parts+base,row,rows);
    if(splits==1)return parts[base+row];
    float sum=0.f;
    for(int split=0;split<splits;++split) {
        if constexpr(Published)sum+=reinterpret_cast<const volatile float*>(parts)[base+split*rows+row];
        else sum+=parts[base+split*rows+row];
    }
    return sum;
}

template<bool Published>
__device__ __forceinline__ float4 gr_read_part_vector(const float* source) {
    if constexpr(Published) {
        float4 value;
        asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(value.x),"=f"(value.y),"=f"(value.z),"=f"(value.w) : "l"(source));
        return value;
    }else return *reinterpret_cast<const float4*>(source);
}

template<int Count,int Offset>
__device__ __forceinline__ float gr_sum_part_vectors(float4 a,float4 b,float4 c,float4 d) {
    static_assert(Count+Offset<=16);
    const float values[16]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,c.x,c.y,c.z,c.w,d.x,d.y,d.z,d.w};
    float sum=0.f;
    #pragma unroll
    for(int split=0;split<Count;++split)sum=__fadd_rn(sum,values[Offset+split]);
    return sum;
}

// GS48 groups have 48 rows (the last has 32), so their publication boundaries
// are four-float aligned. Four aligned loads cover a GS48 row's 13 splits
// without crossing into an unpublished neighboring group. Preserve every
// original FP32 addition, including the initial +0.
template<bool Published>
__device__ __forceinline__ float gr_reduce_down_parts(const float* parts,size_t base,int row,int splits) {
    if(splits==13 || (splits==16 && (base&3u)==0)) {
        const size_t start=base+size_t(row)*splits;
        const auto* aligned=parts+(start&~size_t(3));
        const auto a=gr_read_part_vector<Published>(aligned);
        const auto b=gr_read_part_vector<Published>(aligned+4);
        const auto c=gr_read_part_vector<Published>(aligned+8);
        const auto d=gr_read_part_vector<Published>(aligned+12);
        if(splits==16)return gr_sum_part_vectors<16,0>(a,b,c,d);
        switch(start&3u) {
        case 0:return gr_sum_part_vectors<13,0>(a,b,c,d);
        case 1:return gr_sum_part_vectors<13,1>(a,b,c,d);
        case 2:return gr_sum_part_vectors<13,2>(a,b,c,d);
        default:return gr_sum_part_vectors<13,3>(a,b,c,d);
        }
    }
    const auto read=[&](size_t at) {
        if constexpr(Published)return reinterpret_cast<const volatile float*>(parts)[at];
        else return parts[at];
    };
    if(splits==1)return read(base+row);
    float sum=0.f;
    for(int split=0;split<splits;++split)sum+=read(base+size_t(row)*splits+split);
    return sum;
}
