#pragma once

__device__ __forceinline__ void gr_up_mma_u8_s8(int (&dot)[4],const int (&a)[4],const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ float gr_up_original_group_tree(const float (&v)[8]) {
    const float a=__fadd_rn(__fadd_rn(v[0],v[4]),__fadd_rn(v[2],v[6]));
    const float b=__fadd_rn(__fadd_rn(v[1],v[5]),__fadd_rn(v[3],v[7]));
    return __fadd_rn(a,b);
}

// A's sixteen rows are eight outputs, each split into two 24-value halves.
// B's six valid columns are the three exact signed digits of each half.
// The diagonal selects the correct half; every integer dot and the original
// sixteen-lane FP32 group tree are retained. Original unsigned NINT bytes.
template<int GroupUnroll=7,bool SharedPartials=false>
__global__ void __launch_bounds__(128,SharedPartials?6:4) gated_residual_up_int16_mma_kernel(
        MhcMatrix up,MhcMatrix inject,const void* normalized,const float* parts,
        void* mixed,void* injection,int down_splits,int injection_splits,bool half,GrTwoPlane q) {
    constexpr int Hidden=2560,Streams=4,Rank=320,Groups=7;
    const int tid=threadIdx.x,lane=tid&31,stream=tid>>5,first=int(blockIdx.x)*8;
    const int fragment_row=lane>>2,half_row=fragment_row&1;
    const int row0=stream*Hidden+first+fragment_row/2,row1=row0+4;
    const int word=(lane&3)*4;
    if(blockIdx.x==0 && tid<Streams) {
        const size_t start=Rank*down_splits+tid*injection_splits;
        float sum=injection_splits==1?parts[start]:0.f;
        if(injection_splits>1)for(int split=0;split<injection_splits;++split)sum+=parts[start+split];
        mhc_write(injection,tid,mhc_activated(sum,half,Streams,true),half);
    }
    const auto* weight0=up.q+(up.offsets[row0]>>3);
    const auto* weight1=up.q+(up.offsets[row1]>>3);
    const float scale0=up.row_scale[row0],scale1=up.row_scale[row1];
    const float min0=up.row_minimum[row0],min1=up.row_minimum[row1];
    float values[SharedPartials?1:2][SharedPartials?1:8]{};
    __shared__ float partials[SharedPartials?Streams*16*9:1];
    #pragma unroll (GroupUnroll)
    for(int group=0;group<Groups;++group) {
        int a[4]{};
        const int offset=group*48+half_row*24+word;
        if(word<24 && offset<Rank) {
            a[0]=*reinterpret_cast<const int*>(weight0+offset);
            a[1]=*reinterpret_cast<const int*>(weight1+offset);
        }
        if(word+16<24 && offset+16<Rank) {
            a[2]=*reinterpret_cast<const int*>(weight0+offset+16);
            a[3]=*reinterpret_cast<const int*>(weight1+offset+16);
        }
        int b[2]{};
        const int digit_column=lane>>2;
        if(digit_column<6) {
            const int digit_half=digit_column/3,plane=digit_column%3;
            const auto* source=q.high+group*192+plane*64+digit_half*32+word;
            b[0]=*reinterpret_cast<const int*>(source);
            b[1]=*reinterpret_cast<const int*>(source+16);
        }
        int dot[4]{};gr_up_mma_u8_s8(dot,a,b);
        const auto stats=q.scales[group];
        const float total=half_row==0?stats.x:0.f;
        const float scales[2]={scale0*float(up.scales[size_t(row0)*Groups+group]),
                               scale1*float(up.scales[size_t(row1)*Groups+group])};
        const float minima[2]={min0*float(up.minima[size_t(row0)*Groups+group]),
                               min1*float(up.minima[size_t(row1)*Groups+group])};
        #pragma unroll
        for(int r=0;r<2;++r) {
            int digits[3];
            #pragma unroll
            for(int plane=0;plane<3;++plane) {
                const int col=half_row*3+plane;
                const int source_lane=(lane&~3)+(col>>1);
                const int fragment=dot[2*r+(col&1)];
                digits[plane]=__shfl_sync(0xffffffffu,fragment,source_lane);
            }
            const int integer_dot=digits[0]*16384+digits[1]*128+digits[2];
            const float value=stats.y*fmaf(scales[r],float(integer_dot),-minima[r]*total);
            if constexpr(SharedPartials) {
                if((lane&3)==0)partials[(stream*16+fragment_row+8*r)*9+group]=value;
            }else values[r][group]=value;
        }
    }
    float results[2];
    if constexpr(SharedPartials)__syncwarp();
    #pragma unroll
    for(int r=0;r<2;++r) {
        float own;
        if constexpr(SharedPartials) {
            float groups[8]{};
            #pragma unroll
            for(int group=0;group<Groups;++group)groups[group]=partials[(stream*16+fragment_row+8*r)*9+group];
            own=gr_up_original_group_tree(groups);
        }else own=gr_up_original_group_tree(values[r]);
        const float other=__shfl_sync(0xffffffffu,own,(lane&~3)^4);
        results[r]=__fadd_rn(own,other);
    }
    __shared__ float products[Streams][8];
    if((lane&3)==0 && half_row==0) {
        #pragma unroll
        for(int r=0;r<2;++r) {
            const int column=fragment_row/2+4*r;
            const float projected=mhc_round(results[r],half);
            const float gate=mhc_round(mhc_sigmoid(projected,up.fast_sigmoid),half);
            products[stream][column]=mhc_round(__fmul_rn(gate,
                mhc_read(normalized,size_t(stream)*Hidden+first+column,half)),half);
        }
    }
    __syncthreads();
    if(tid<8) {
        float sum=0.f;
        #pragma unroll
        for(int s=0;s<Streams;++s)sum=__fadd_rn(sum,products[s][tid]);
        mhc_write(mixed,first+tid,__fdiv_rn(sum,float(Streams)),half);
    }
}
