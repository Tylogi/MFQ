#pragma once

// Same signed-byte fragments as the existing NVQ m16n8k32 path. The A
// fragment is sixteen original NINT8 output rows, B is up to eight tokens.
__device__ __forceinline__ void gr_mma16832(int (&dot)[4],const int (&a)[4],const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ void gr_up_mma16832_unsigned(int (&dot)[4],const int (&a)[4],const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

template<int Rows,bool LocalPrepare,bool Uniform16=false>
__global__ void __launch_bounds__(128,6) gated_residual_up_mma_kernel(
        MhcMatrix down,MhcMatrix up,MhcMatrix inject,const void* normalized,const float* parts,
        void* mixed,void* injection,int down_splits,int injection_splits,
        const float* dense_projection,bool half,GrTwoPlane prepared) {
    constexpr int Hidden=2560,Streams=4,Rank=320,Groups=7,Elements=336;
    const int tid=threadIdx.x,lane=tid&31,stream=tid>>5,first=int(blockIdx.x)*16;
    const int row0=stream*Hidden+first+(lane>>2),row1=row0+8;
    const int token0=(lane&3)*2,word=(lane&3)*4;
    const int stride=down.outputs*down_splits+inject.outputs*injection_splits;
    // Injection is unchanged and is written by only the first output CTA.
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
            const bool injection_half=half && !inject.dense;
            mhc_write(injection,tid,mhc_activated(sum,injection_half,Streams,true),injection_half);
        }
    }
    GrTwoPlane q=prepared;
    if constexpr(LocalPrepare) {
        __shared__ float local_low[Rank];
        __shared__ __align__(16) int8_t local_planes[3*Elements];
        __shared__ float2 local_scales[Groups];
        __shared__ int4 local_sums[Groups];
        static_assert(Rows==1);
        for(int row=tid;row<Rank;row+=128) {
            float sum=down_splits==1?parts[row]:0.f;
            if(down_splits>1)for(int split=0;split<down_splits;++split)sum+=parts[row*down_splits+split];
            local_low[row]=mhc_activated(sum,half,Streams,false);
        }
        __syncthreads();
        q={local_planes,local_planes+Elements,local_planes+2*Elements,local_scales,local_sums};
        for(int group=stream;group<Groups;group+=Streams)
            gr_quantize_activation_group<48,true,false>(up,local_low,false,q,group,0,lane);
        __syncthreads();
    }
    const auto* weight0=up.q+(up.offsets[row0]>>3);
    const auto* weight1=up.q+(up.offsets[row1]>>3);
    const float outer_scale0=up.row_scale[row0],outer_scale1=up.row_scale[row1];
    const float outer_min0=up.row_minimum[row0],outer_min1=up.row_minimum[row1];
    float result[4]{};
    #pragma unroll 1
    for(int group=0;group<Groups;++group) {
        int dot[Uniform16?2:3][4]{};
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
            const int begin=chunk*32+word;
            int a[4]{},raw[4]{};
            if(begin<48 && group*48+begin<Rank) {
                raw[0]=*reinterpret_cast<const int*>(weight0+group*48+begin);
                raw[1]=*reinterpret_cast<const int*>(weight1+group*48+begin);
                a[0]=raw[0]^int(0x80808080u);a[1]=raw[1]^int(0x80808080u);
            }
            if(begin+16<48 && group*48+begin+16<Rank) {
                raw[2]=*reinterpret_cast<const int*>(weight0+group*48+begin+16);
                raw[3]=*reinterpret_cast<const int*>(weight1+group*48+begin+16);
                a[2]=raw[2]^int(0x80808080u);a[3]=raw[3]^int(0x80808080u);
            }
            #pragma unroll
            for(int plane=0;plane<(Uniform16?2:3);++plane) {
                const int8_t* values=plane==0?q.high:!Uniform16 && plane==1?q.middle:q.low;
                int b[2]{};const int token=lane>>2;
                if(token<Rows) {
                    const auto* source=values+token*Elements+group*48;
                    if(begin<48)b[0]=*reinterpret_cast<const int*>(source+begin);
                    if(begin+16<48)b[1]=*reinterpret_cast<const int*>(source+begin+16);
                }
                if constexpr(Uniform16) {
                    if(plane==1)gr_up_mma16832_unsigned(dot[plane],raw,b);
                    else gr_mma16832(dot[plane],a,b);
                }else gr_mma16832(dot[plane],a,b);
            }
        }
        const float scale0=outer_scale0*float(up.scales[size_t(row0)*Groups+group]);
        const float scale1=outer_scale1*float(up.scales[size_t(row1)*Groups+group]);
        const float minimum0=outer_min0*float(up.minima[size_t(row0)*Groups+group]);
        const float minimum1=outer_min1*float(up.minima[size_t(row1)*Groups+group]);
        const float correction0=Uniform16?-minimum0:fmaf(128.f,scale0,-minimum0);
        const float correction1=Uniform16?-minimum1:fmaf(128.f,scale1,-minimum1);
        #pragma unroll
        for(int fragment=0;fragment<4;++fragment) {
            const int token=token0+(fragment&1);
            if(token<Rows) {
                const auto sums=q.sums[token*Groups+group];
                const int total=Uniform16?sums.x*256+sums.z:sums.x*65536+sums.y*256+sums.z;
                float combined;
                if constexpr(Uniform16)
                    combined=fmaf(256.f,float(dot[0][fragment]+128*sums.x),float(dot[1][fragment]));
                else combined=fmaf(65536.f,float(dot[0][fragment]),
                    fmaf(256.f,float(dot[1][fragment]),float(dot[2][fragment])));
                const float scale=fragment<2?scale0:scale1;
                const float correction=fragment<2?correction0:correction1;
                result[fragment]=__fadd_rn(result[fragment],
                    q.scales[token*Groups+group].y*fmaf(scale,combined,correction*float(total)));
            }
        }
    }
    __shared__ float products[Rows][Streams][16];
    #pragma unroll
    for(int fragment=0;fragment<4;++fragment) {
        const int token=token0+(fragment&1),column=(lane>>2)+(fragment<2?0:8);
        if(token<Rows) {
            const float projected=mhc_round(result[fragment],half);
            const float gate=mhc_round(1.f/(1.f+expf(-projected)),half);
            products[token][stream][column]=mhc_round(__fmul_rn(gate,
                mhc_read(normalized,(size_t(token)*Streams+stream)*Hidden+first+column,half)),half);
        }
    }
    __syncthreads();
    if(tid<Rows*16) {
        const int token=tid/16,column=tid%16;
        float sum=0.f;
        #pragma unroll
        for(int s=0;s<Streams;++s)sum=__fadd_rn(sum,products[token][s][column]);
        mhc_write(mixed,size_t(token)*Hidden+first+column,sum/float(Streams),half);
    }
}
