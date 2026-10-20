#pragma once
#include "nint_whole_group_decode.cuh"
#include "nint_short_decode.h"



// Strata's Q81Il reads all token columns of one packed chunk together.
// This copy contains only short-lived activation codes; weights stay packed.
template<int Rows>
__device__ __forceinline__ void nint_interleaved_codes(NintWholeArgs a,int group,int chunk,int (&codes)[Rows]) {
    constexpr int Columns=Rows<=2?2:Rows<=4?4:8;
    const size_t index=size_t(group)*7+chunk;
    const int* source=a.activation_interleaved+index*((Rows==5 || Rows==6)?4:Columns);
    if constexpr(Columns==2) {
        const int2 v=*reinterpret_cast<const int2*>(source);
        codes[0]=v.x;if constexpr(Rows>1)codes[1]=v.y;
    }else {
        const int4 v=*reinterpret_cast<const int4*>(source);
        codes[0]=v.x;if constexpr(Rows>1)codes[1]=v.y;
        if constexpr(Rows>2)codes[2]=v.z;
        if constexpr(Rows>3)codes[3]=v.w;
        if constexpr(Rows>4) {
            if constexpr(Rows==5 || Rows==6) {
                const int* tail=a.activation_interleaved+size_t(a.groups)*7*4+index*(Rows-4);
                if constexpr(Rows==5)codes[4]=*tail;
                else {
                    const int2 w=*reinterpret_cast<const int2*>(tail);
                    codes[4]=w.x;codes[5]=w.y;
                }
            }else {
            const int4 w=*reinterpret_cast<const int4*>(source+4);
            codes[4]=w.x;if constexpr(Rows>5)codes[5]=w.y;
            if constexpr(Rows>6)codes[6]=w.z;
            if constexpr(Rows>7)codes[7]=w.w;
            }
        }
    }
}

// Four warps own the same row as the original dense NINT kernel. Only the
// integer decoder changes: all accumulation and both XOR trees are retained.
template<int GS,int Bits,int Rows,int Warps=4,bool Staged=false,bool CachedSum=false,int RowsPerBlock=1,bool Interleaved=false,typename Packed=NintWholeGroup<GS,Bits,Staged>,bool ExactRows=false>
__device__ __forceinline__ void nint_whole_row(NintWholeArgs a,uint64_t row_bit) {
    const int local_row=RowsPerBlock==1?0:threadIdx.y/Warps;
    const int row=int(blockIdx.x)*RowsPerBlock+local_row,lane=threadIdx.x;
    const int warp=threadIdx.y-local_row*Warps;
    const float outer_scale=a.outer_scale[row],outer_minimum=a.outer_minimum[row];
    float acc[Rows]={};
    for(int group=warp*32+lane;group<a.groups;group+=128) {
        const Packed packed(a.weights,row_bit+uint64_t(group*GS)*Bits,a.bytes,a.group_load);
        int dots[Rows]={},sums[CachedSum?1:Rows]={};
        const auto consume=[&](auto chunk_constant) {
            constexpr int chunk=decltype(chunk_constant)::value;
            const int qw=packed.template codes<chunk>();
            if constexpr(Interleaved) {
                int qx[Rows];nint_interleaved_codes(a,group,chunk,qx);
                #pragma unroll
                for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                    if constexpr(!CachedSum)sums[m]=__dp4a(0x01010101,qx[m],sums[m]);
                    dots[m]=__dp4a(Bits==8?qw^int(0x80808080u):qw,qx[m],dots[m]);
                }
            }else {
            const int column=group*GS+chunk*4;
            #pragma unroll
            for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                const int qx=*reinterpret_cast<const int*>(a.activation+size_t(m)*a.groups*GS+column);
                if constexpr(!CachedSum)sums[m]=__dp4a(0x01010101,qx,sums[m]);
                dots[m]=__dp4a(Bits==8?qw^int(0x80808080u):qw,qx,dots[m]);
            }
            }
        };
        consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
        consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
        consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
        if constexpr(GS==28)consume(std::integral_constant<int,6>{});
        const float scale=outer_scale*float(a.scale[size_t(row)*a.groups+group]);
        const float minimum=outer_minimum*float(a.minimum[size_t(row)*a.groups+group]);
        #pragma unroll
        for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
            const int sum=CachedSum?a.activation_sum[size_t(m)*a.groups+group]:sums[m];
            const int dot=dots[m]+(Bits==8?128*sum:0);
            acc[m]+=a.activation_scale[size_t(m)*a.groups+group]*(scale*float(dot)-minimum*float(sum));
        }
    }
#pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)acc[m]+=__shfl_xor_sync(0xffffffffu,acc[m],offset);
    }
    __shared__ float partial[Rows][RowsPerBlock][4];
    #pragma unroll
    for(int m=0;m<Rows;++m)if(lane==0)partial[m][local_row][warp]=acc[m];
    __syncthreads();
    if(warp==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            float value=lane<Warps?partial[m][local_row][lane]:0.f;
            #pragma unroll
            for(int offset=16;offset>0;offset>>=1)value+=__shfl_xor_sync(0xffffffffu,value,offset);
            if(lane==0 && (ExactRows || m<a.activation_rows))a.output[size_t(m)*a.rows+row]=__float2half_rn(value);
        }
    }
}

template<int GS,int Rows=1,int Warps=4,bool Staged=false,bool CachedSum=false>
__global__ void __launch_bounds__(128) nint_whole_group_kernel(NintWholeArgs a) {
    if(int(blockIdx.x)>=a.rows)return;
    uint64_t row_bit=uint64_t(a.offsets[blockIdx.x]);
    const int row_bits=a.bits[blockIdx.x];
    __shared__ __align__(16) uint8_t packed_row[Staged?GS*96+32:1];
    if constexpr(Staged) {
        // Stage the original packed bytes with contiguous 16-byte transfers.
        // The short input projection has at most 96 groups; no dequantized
        // weight cache or extra persistent representation is created.
        const uint64_t start=(row_bit>>7)*16;
        const int count=(int(row_bit&127)+a.groups*GS*row_bits+7)/8;
        const int bytes=(count+15)&~15;
        const int tid=threadIdx.y*32+threadIdx.x;
        for(int at=tid*16;at<bytes;at+=Warps*32*16) {
            if(start+at+16<=a.bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
                const auto destination=unsigned(__cvta_generic_to_shared(packed_row+at));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                    "r"(destination),"l"(a.weights+start+at):"memory");
#else
                *reinterpret_cast<uint4*>(packed_row+at)=*reinterpret_cast<const uint4*>(a.weights+start+at);
#endif
            }else {
                #pragma unroll
                for(int b=0;b<16;++b)packed_row[at+b]=start+at+b<a.bytes?a.weights[start+at+b]:0;
            }
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
        __syncthreads();
        a.weights=packed_row;a.bytes=bytes;row_bit&=127;
    }
    // Read current device metadata, including for graph replay. All four
    // warps see the same row bit width and enter the same specialization.
    switch(row_bits) {
    case 1:nint_whole_row<GS,1,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    case 2:nint_whole_row<GS,2,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    case 3:nint_whole_row<GS,3,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    case 4:nint_whole_row<GS,4,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    case 5:nint_whole_row<GS,5,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    case 6:nint_whole_row<GS,6,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    case 7:nint_whole_row<GS,7,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    case 8:nint_whole_row<GS,8,Rows,Warps,Staged,CachedSum>(a,row_bit);break;
    }
}
