#pragma once

#include "gr_up_half_prepare.cuh"

template<bool OnePlane>
__device__ __forceinline__ void gr_prepare_up_half_group(const float* low,GrUpHalf q,
        int sample,int group,int lane) {
    const int column=group*48+lane;
    const float x0=column<320?low[column]:0.f;
    const float x1=lane<16 && column+32<320?low[column+32]:0.f;
    gr_prepare_up_half_values<OnePlane>(x0,x1,q,sample,group,lane);
}

template<bool OnePlane,bool RegisterPrepare=false,bool VectorParts=false>
__global__ void gated_residual_prepare_up_half_kernel(MhcMatrix down,const float* parts,
        GrUpHalf q,int down_splits,int injection_stride,bool half) {
    constexpr int Rank=320,Groups=7;
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sample=blockIdx.x;
    const int stride=down.outputs*down_splits+injection_stride;
    if constexpr(RegisterPrepare) {
        if(warp<Groups) {
            const auto activate=[&](int row) {
                if(row>=Rank)return 0.f;
                float sum=down_splits==1?parts[size_t(sample)*stride+row]:0.f;
                if(down_splits>1)for(int split=0;split<down_splits;++split)
                    sum+=parts[size_t(sample)*stride+row*down_splits+split];
                return mhc_activated(sum,half,4,false);
            };
            const int column=warp*48+lane;
            gr_prepare_up_half_values<OnePlane>(activate(column),lane<16?activate(column+32):0.f,q,sample,warp,lane);
        }
    }else {
        __shared__ float low[Rank];
        for(int row=tid;row<Rank;row+=256) {
            float sum;
            if constexpr(VectorParts)sum=gr_reduce_down_parts<false>(parts,size_t(sample)*stride,row,down_splits);
            else {
                sum=down_splits==1?parts[size_t(sample)*stride+row]:0.f;
                if(down_splits>1)for(int split=0;split<down_splits;++split)
                    sum+=parts[size_t(sample)*stride+row*down_splits+split];
            }
            low[row]=mhc_activated(sum,half,4,false);
        }
        __syncthreads();
        if(warp<Groups)gr_prepare_up_half_group<OnePlane>(low,q,sample,warp,lane);
    }
}

__device__ __forceinline__ int gr_up_half_weight_pair(const uint8_t* source) {
    const unsigned pair=*reinterpret_cast<const uint16_t*>(source);
    const unsigned encoded=__byte_perm(pair,0x64006400u,0x5150);
    int converted;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(converted) : "r"(encoded),"r"(0x64006400u));
    return converted;
}

__device__ __forceinline__ void gr_up_mma16816_half(float (&dot)[4],const int (&a)[4],const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(dot[0]),"+f"(dot[1]),"+f"(dot[2]),"+f"(dot[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

template<int Rows,bool OnePlane,bool LocalPrepare=false,int GroupUnroll=1>
__global__ void __launch_bounds__(128,6) gated_residual_up_half_mma_kernel(
        MhcMatrix down,MhcMatrix up,MhcMatrix inject,const void* normalized,const float* parts,
        void* mixed,void* injection,int down_splits,int injection_splits,
        const float* dense_projection,bool half,GrUpHalf q) {
    constexpr int Hidden=2560,Streams=4,Rank=320,Groups=7,Elements=336;
    const int tid=threadIdx.x,lane=tid&31,stream=tid>>5,first=int(blockIdx.x)*16;
    const int row0=stream*Hidden+first+(lane>>2),row1=row0+8;
    const int token0=(lane&3)*2,word=(lane&3)*2;
    const int stride=down.outputs*down_splits+inject.outputs*injection_splits;
    if(blockIdx.x==0) {
        if(inject.dense && !dense_projection && tid<32) {
            #pragma unroll
            for(int m=0;m<Rows;++m)for(int s=0;s<Streams;++s) {
                float sum=parts[size_t(m)*stride+down.outputs*down_splits+s*injection_splits+lane];
                #pragma unroll
                for(int shift=16;shift;shift>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,shift));
                if(lane==0)mhc_write(injection,m*Streams+s,mhc_activated(__fadd_rn(sum,0.f),false,Streams,true),false);
            }
        }else if(tid<Rows*Streams) {
            const int m=tid/Streams,s=tid%Streams;
            const size_t start=size_t(m)*stride+down.outputs*down_splits+s*injection_splits;
            float sum=dense_projection?dense_projection[m*Streams+s]:injection_splits==1?parts[start]:0.f;
            if(!dense_projection && injection_splits>1)for(int split=0;split<injection_splits;++split)sum+=parts[start+split];
            mhc_write(injection,tid,mhc_activated(sum,half && !inject.dense,Streams,true),half && !inject.dense);
        }
    }
    if constexpr(LocalPrepare) {
        __shared__ float local_low[Rank];
        __shared__ __align__(16) __half local_values[Elements*(OnePlane?1:2)];
        __shared__ float2 local_statistics[Groups];
        static_assert(Rows==1);
        for(int row=tid;row<Rank;row+=128) {
            float sum=down_splits==1?parts[row]:0.f;
            if(down_splits>1)for(int split=0;split<down_splits;++split)
                sum+=parts[row*down_splits+split];
            local_low[row]=mhc_activated(sum,half,Streams,false);
        }
        __syncthreads();
        q.high=local_values;
        q.low=OnePlane?nullptr:local_values+Elements;
        q.statistics=local_statistics;
        for(int group=stream;group<Groups;group+=Streams)
            gr_prepare_up_half_group<OnePlane>(local_low,q,0,group,lane);
        __syncthreads();
    }
    const auto* weight0=up.q+(up.offsets[row0]>>3);
    const auto* weight1=up.q+(up.offsets[row1]>>3);
    const float outer_scale0=up.row_scale[row0],outer_scale1=up.row_scale[row1];
    const float outer_min0=up.row_minimum[row0],outer_min1=up.row_minimum[row1];
    float result[4]{};
    #pragma unroll (GroupUnroll)
    for(int group=0;group<Groups;++group) {
        float dot[OnePlane?1:2][4]{};
        #pragma unroll
        for(int chunk=0;chunk<3;++chunk) {
            const int begin=chunk*16+word;
            int a[4]{};
            if(group*48+begin<Rank) {
                a[0]=gr_up_half_weight_pair(weight0+group*48+begin);
                a[1]=gr_up_half_weight_pair(weight1+group*48+begin);
            }
            if(group*48+begin+8<Rank) {
                a[2]=gr_up_half_weight_pair(weight0+group*48+begin+8);
                a[3]=gr_up_half_weight_pair(weight1+group*48+begin+8);
            }
            #pragma unroll
            for(int plane=0;plane<(OnePlane?1:2);++plane) {
                const auto* values=plane==0?q.high:q.low;
                int b[2]{};const int token=lane>>2;
                if(token<Rows) {
                    const auto* source=values+token*Elements+group*48;
                    b[0]=*reinterpret_cast<const int*>(source+begin);
                    b[1]=*reinterpret_cast<const int*>(source+begin+8);
                }
                gr_up_mma16816_half(dot[plane],a,b);
            }
        }
        const float scale0=outer_scale0*float(up.scales[size_t(row0)*Groups+group]);
        const float scale1=outer_scale1*float(up.scales[size_t(row1)*Groups+group]);
        const float min0=outer_min0*float(up.minima[size_t(row0)*Groups+group]);
        const float min1=outer_min1*float(up.minima[size_t(row1)*Groups+group]);
        #pragma unroll
        for(int fragment=0;fragment<4;++fragment) {
            const int token=token0+(fragment&1);
            if(token<Rows) {
                const auto statistics=q.statistics[token*Groups+group];
                float combined=dot[0][fragment];
                if constexpr(!OnePlane)combined+=dot[1][fragment];
                const float value=fmaf(fragment<2?scale0:scale1,combined,
                    -(fragment<2?min0:min1)*statistics.y);
                if constexpr(OnePlane)result[fragment]+=value;
                else result[fragment]+=statistics.x*value;
            }
        }
    }
    __shared__ float products[Rows][Streams][16];
    #pragma unroll
    for(int fragment=0;fragment<4;++fragment) {
        const int token=token0+(fragment&1),column=(lane>>2)+(fragment<2?0:8);
        if(token<Rows) {
            const float projected=mhc_round(result[fragment],half);
            const float gate=mhc_round(mhc_sigmoid(projected,up.fast_sigmoid),half);
            products[token][stream][column]=mhc_round(__fmul_rn(gate,
                mhc_read(normalized,(size_t(token)*Streams+stream)*Hidden+first+column,half)),half);
        }
    }
    __syncthreads();
    if(tid<Rows*16) {
        const int token=tid/16,column=tid%16;float sum=0.f;
        #pragma unroll
        for(int s=0;s<Streams;++s)sum=__fadd_rn(sum,products[token][s][column]);
        mhc_write(mixed,size_t(token)*Hidden+first+column,sum/float(Streams),half);
    }
}
