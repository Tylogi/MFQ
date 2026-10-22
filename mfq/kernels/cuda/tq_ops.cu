#define TRELLIS_FIXED_BOOK 1
#define TRELLIS_NORMALIZED_WINDOW 1
#define TRELLIS_PAIR_BOOK 1
#include "tq_decode.cuh"
#include "tq_ops.h"
#include <cuda_bf16.h>

template<bool BF> __device__ __forceinline__ float scalar_float(uint16_t value) {
    if constexpr(BF)return __bfloat162float(__ushort_as_bfloat16(value));
    return __half2float(__ushort_as_half(value));
}
template<bool BF> __device__ __forceinline__ uint16_t scalar_round(float value) {
    if constexpr(BF)return __bfloat16_as_ushort(__float2bfloat16_rn(value));
    return __half_as_ushort(__float2half_rn(value));
}

template<bool BF,int W,bool C,int Bits>
__device__ __forceinline__ float projection_dot(TrellisProjectionView m,uint64_t d,int row,const uint16_t* x) {
    float acc=0;int tid=threadIdx.x%(W*32);
    for(int col=tid*16;col<m.stored_columns;col+=W*512) {
        half weight[16];
        decode_vector<16,true,C,Bits>(m.words,d,m.scales,static_cast<const half*>(m.relative),nullptr,nullptr,
            row,col,m.stored_columns,m.gslog,weight);
        uint32_t packed_x[8]={};
        if(col+16<=m.input_columns && (reinterpret_cast<uintptr_t>(x)&15)==0) {
            uint4 a=__ldg(reinterpret_cast<const uint4*>(x+col));
            uint4 b=__ldg(reinterpret_cast<const uint4*>(x+col+8));
            packed_x[0]=a.x;packed_x[1]=a.y;packed_x[2]=a.z;packed_x[3]=a.w;
            packed_x[4]=b.x;packed_x[5]=b.y;packed_x[6]=b.z;packed_x[7]=b.w;
        } else {
            #pragma unroll
            for(int j=0;j<16;++j)if(col+j<m.input_columns)packed_x[j/2]|=uint32_t(__ldg(x+col+j))<<((j&1)*16);
        }
        #pragma unroll
        for(int j=0;j<16;++j)acc=__fmaf_rn(__half2float(weight[j]),scalar_float<BF>(uint16_t(packed_x[j/2]>>((j&1)*16))),acc);
    }
    return acc;
}
template<bool BF,int W,bool C>
__device__ __forceinline__ float width_dispatch(TrellisProjectionView m,int row,const uint16_t* x) {
    uint64_t d=__ldg(m.descriptors+row);
    #define GB(K) case K-1:return projection_dot<BF,W,C,K>(m,d,row,x)
    switch((d>>32)&7){GB(1);GB(2);GB(3);GB(4);GB(5);GB(6);GB(7);GB(8);}
    return 0.f;
}
template<bool BF,int W>
__device__ __forceinline__ float matrix_dot(TrellisProjectionView m,int row,const uint16_t* x) {
    if(m.columns)return width_dispatch<BF,W,true>(m,row,x);
    return width_dispatch<BF,W,false>(m,row,x);
}
template<int W,int R>
__device__ __forceinline__ float reduce(float value,float (&partial)[R][W]) {
    int lane=threadIdx.x&31,warp=threadIdx.x/32,slot=warp/W;
    #pragma unroll
    for(int j=16;j;j/=2)value+=__shfl_down_sync(0xffffffff,value,j);
    if constexpr(W>1) {
        if(lane==0)partial[slot][warp%W]=value;
        __syncthreads();
        value=lane<W?partial[slot][lane]:0.f;
        #pragma unroll
        for(int j=16;j;j/=2)value+=__shfl_down_sync(0xffffffff,value,j);
    }
    return value;
}
template<bool BF,int W,int R,bool Pair>
__global__ void projection_kernel(TrellisProjectionView g,TrellisProjectionView u,const uint16_t* x,uint16_t* y) {
    int row=blockIdx.x*R+threadIdx.x/(W*32);float ga=0,ua=0;
    if(row<g.rows) {
        ga=matrix_dot<BF,W>(g,row,x);
        if constexpr(Pair)ua=matrix_dot<BF,W>(u,row,x);
    }
    __shared__ float sg[R][W],su[R][W];
    ga=reduce<W,R>(ga,sg);if constexpr(Pair)ua=reduce<W,R>(ua,su);
    if(row<g.rows && threadIdx.x%(W*32)==0) {
        if(g.bias)ga+=__ldg(g.bias+row);
        if constexpr(Pair) {
            if(u.bias)ua+=__ldg(u.bias+row);
            ga=scalar_float<BF>(scalar_round<BF>(ga));ua=scalar_float<BF>(scalar_round<BF>(ua));
            y[row]=scalar_round<BF>(ua*(ga/(1.f+expf(-ga))));
        } else y[row]=scalar_round<BF>(ga);
    }
}
template<bool BF,int W,int R,bool Pair>
int launch_projection(TrellisProjectionView g,TrellisProjectionView u,const void* x,void* y,cudaStream_t s) {
    int e=launchable(projection_kernel<BF,W,R,Pair>,W*R*32);if(e)return e;
    projection_kernel<BF,W,R,Pair><<<(g.rows+R-1)/R,W*R*32,0,s>>>(g,u,static_cast<const uint16_t*>(x),static_cast<uint16_t*>(y));
    return int(cudaGetLastError());
}
template<bool BF,bool Pair>
int geometry(TrellisProjectionView g,TrellisProjectionView u,const void* x,void* y,int w,int r,cudaStream_t s) {
    #define G(W,R) if(w==W && r==R)return launch_projection<BF,W,R,Pair>(g,u,x,y,s)
    G(1,4);G(2,2);G(4,1);G(8,1);G(16,1);G(32,1);
    return -2;
}
extern "C" int mfq_tq_gemv(TrellisProjectionView g,const void* x,void* y,int dtype,int w,int r,cudaStream_t s) {
    if(g.input_columns<=0 || g.input_columns>g.stored_columns || g.rows<=0 || g.stored_columns%256 ||
       !g.words || !g.descriptors || !g.scales || !g.relative || !x || !y ||
       (!g.columns && (g.gslog<4 || g.gslog>20)))return -1;
    if(w==0){w=g.stored_columns<=2048?2:4;r=g.stored_columns<=2048?2:1;}
    if(dtype==0)return geometry<false,false>(g,g,x,y,w,r,s);
    if(dtype==1)return geometry<true,false>(g,g,x,y,w,r,s);
    return -1;
}
extern "C" int mfq_tq_swiglu(TrellisProjectionView g,TrellisProjectionView u,const void* x,void* h,int dtype,int w,int r,cudaStream_t s) {
    if(g.rows!=u.rows || g.input_columns!=u.input_columns || g.rows<=0 || g.input_columns<=0 ||
       g.stored_columns<g.input_columns || u.stored_columns<u.input_columns ||
       g.stored_columns%256 || u.stored_columns%256 || !g.words || !u.words ||
       !g.descriptors || !u.descriptors || !g.scales || !u.scales || !g.relative || !u.relative || !x || !h ||
       (!g.columns && (g.gslog<4 || g.gslog>20)) || (!u.columns && (u.gslog<4 || u.gslog>20)))return -1;
    if(w==0){w=4;r=1;}
    if(dtype==0)return geometry<false,true>(g,u,x,h,w,r,s);
    if(dtype==1)return geometry<true,true>(g,u,x,h,w,r,s);
    return -1;
}
extern "C" int mfq_tq_ffn(TrellisProjectionView g,TrellisProjectionView u,TrellisProjectionView d,const void* x,void* h,void* y,int dtype,cudaStream_t s) {
    if(d.input_columns!=g.rows)return -1;
    int e=mfq_tq_swiglu(g,u,x,h,dtype,0,0,s);if(e)return e;
    return mfq_tq_gemv(d,h,y,dtype,0,0,s);
}
