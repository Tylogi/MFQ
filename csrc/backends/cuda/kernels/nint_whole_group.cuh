#pragma once
#include "nint_whole_group_decode.cuh"

struct NintWholeArgs {
    const uint8_t *weights,*bits,*scale,*minimum;
    const int64_t* offsets;
    const float *outer_scale,*outer_minimum;
    const int8_t* activation;
    const float* activation_scale;
    __half* output;
    uint64_t bytes;
    int rows,groups;
    int activation_rows=1;
};

// Four warps own the same row as the original dense NINT kernel. Only the
// integer decoder changes: all accumulation and both XOR trees are retained.
template<int GS,int Bits,int Rows>
__device__ __forceinline__ void nint_whole_row(NintWholeArgs a) {
    const int row=blockIdx.x,lane=threadIdx.x,warp=threadIdx.y;
    const uint64_t row_bit=uint64_t(a.offsets[row]);
    const float outer_scale=a.outer_scale[row],outer_minimum=a.outer_minimum[row];
    float acc[Rows]={};
    for(int group=warp*32+lane;group<a.groups;group+=128) {
        const NintWholeGroup<GS,Bits> packed(a.weights,row_bit+uint64_t(group*GS)*Bits,a.bytes);
        int dots[Rows]={},sums[Rows]={};
        const auto consume=[&](auto chunk_constant) {
            constexpr int chunk=decltype(chunk_constant)::value;
            const int column=group*GS+chunk*4;
            const int qw=packed.template codes<chunk>();
            #pragma unroll
            for(int m=0;m<Rows;++m)if(m<a.activation_rows) {
                const int qx=*reinterpret_cast<const int*>(a.activation+size_t(m)*a.groups*GS+column);
                sums[m]=__dp4a(0x01010101,qx,sums[m]);
                dots[m]=__dp4a(Bits==8?qw^int(0x80808080u):qw,qx,dots[m]);
            }
        };
        consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
        consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
        consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
        if constexpr(GS==28)consume(std::integral_constant<int,6>{});
        const float scale=outer_scale*float(a.scale[size_t(row)*a.groups+group]);
        const float minimum=outer_minimum*float(a.minimum[size_t(row)*a.groups+group]);
        #pragma unroll
        for(int m=0;m<Rows;++m)if(m<a.activation_rows) {
            const int dot=dots[m]+(Bits==8?128*sums[m]:0);
            acc[m]+=a.activation_scale[size_t(m)*a.groups+group]*(scale*float(dot)-minimum*float(sums[m]));
        }
    }
#pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)acc[m]+=__shfl_xor_sync(0xffffffffu,acc[m],offset);
    }
    __shared__ float partial[Rows][4];
    #pragma unroll
    for(int m=0;m<Rows;++m)if(lane==0)partial[m][warp]=acc[m];
    __syncthreads();
    if(warp==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            float value=lane<4?partial[m][lane]:0.f;
            #pragma unroll
            for(int offset=16;offset>0;offset>>=1)value+=__shfl_xor_sync(0xffffffffu,value,offset);
            if(lane==0 && m<a.activation_rows)a.output[size_t(m)*a.rows+row]=__float2half_rn(value);
        }
    }
}

template<int GS,int Rows=1>
__global__ void __launch_bounds__(128) nint_whole_group_kernel(NintWholeArgs a) {
    if(int(blockIdx.x)>=a.rows)return;
    // Read current device metadata, including for graph replay. All four
    // warps see the same row bit width and enter the same specialization.
    switch(a.bits[blockIdx.x]) {
    case 1:nint_whole_row<GS,1,Rows>(a);break;
    case 2:nint_whole_row<GS,2,Rows>(a);break;
    case 3:nint_whole_row<GS,3,Rows>(a);break;
    case 4:nint_whole_row<GS,4,Rows>(a);break;
    case 5:nint_whole_row<GS,5,Rows>(a);break;
    case 6:nint_whole_row<GS,6,Rows>(a);break;
    case 7:nint_whole_row<GS,7,Rows>(a);break;
    case 8:nint_whole_row<GS,8,Rows>(a);break;
    }
}
