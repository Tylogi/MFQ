#pragma once

__device__ __forceinline__ void gr_mma16832_unsigned(int (&dot)[4],const int (&a)[4],const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

// All short-batch columns share the original weight tile. Split K provides
// enough independent CTAs even for one token; no weight expansion is used.
template<int Rows,int Planes,bool PrepareUp=false,bool SplitMajor=false>
__global__ void __launch_bounds__(128,6) gated_residual_down_mma_kernel(
        MhcMatrix down,MhcMatrix inject,const void* normalized,bool half,
        float* parts,int splits,int injection_splits,bool vector_injection,
        const float* dense_input,GrTwoPlane q,GrTwoPlane prepared_up={},int counter_slot=-1) {
    constexpr int GS=48;
    down.width=10240;down.outputs=320;down.groups=214;down.gs=48;
    inject.width=10240;inject.outputs=4;
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int split=int(blockIdx.x)%splits,tile=int(blockIdx.x)/splits;
    const int row0=tile*64+warp*16+(lane>>2),row1=row0+8;
    const int first=down.groups*split/splits,last=down.groups*(split+1)/splits;
    const int word=(lane&3)*4,token0=(lane&3)*2;
    const int elements=down.groups*GS,stride=down.outputs*splits+inject.outputs*injection_splits;
    // Preserve the injection's original split arithmetic and FP32 order.
    if(inject.dense && dense_input && tid<32) {
        const int per_block=32/inject.outputs,virtual_lane=int(blockIdx.x)*per_block+tid/inject.outputs;
        const int row=tid%inject.outputs;
        if(virtual_lane<32) {
            #pragma unroll
            for(int token=0;token<Rows;++token) {
                float sum=0.f;
                for(int begin=virtual_lane*4;begin<inject.width;begin+=128) {
                    if(vector_injection) {
                        const auto x=*reinterpret_cast<const float4*>(dense_input+size_t(token)*inject.width+begin);
                        const auto w=*reinterpret_cast<const float4*>(inject.row_scale+size_t(begin)*inject.outputs+row*4);
                        sum=fmaf(x.x,w.x,sum);sum=fmaf(x.y,w.y,sum);
                        sum=fmaf(x.z,w.z,sum);sum=fmaf(x.w,w.w,sum);
                    }else {
                        #pragma unroll
                        for(int item=0;item<4;++item)sum=fmaf(dense_input[size_t(token)*inject.width+begin+item],
                            inject.row_scale[(size_t(begin)+item)*inject.outputs+row],sum);
                    }
                }
                parts[size_t(token)*stride+down.outputs*splits+row*injection_splits+virtual_lane]=sum;
            }
        }
    }
    if(!inject.dense) {
        const auto* packed=inject.q+(inject.offsets[warp]>>3);
        const float outer_scale=inject.row_scale[warp],outer_minimum=inject.row_minimum[warp];
        for(int part=int(blockIdx.x);part<injection_splits;part+=int(gridDim.x)) {
            const int begin=inject.groups*part/injection_splits,end=inject.groups*(part+1)/injection_splits;
            float sums[Rows]{};
            for(int group=begin;group<end;++group) {
                const size_t meta=size_t(warp)*inject.groups+group;
                const float scale=__fmul_rn(outer_scale,float(inject.scales[meta]));
                const float minimum=__fmul_rn(outer_minimum,float(inject.minima[meta]));
                #pragma unroll
                for(int chunk=0;chunk<2;++chunk) {
                    const int j=lane+chunk*32,column=group*inject.gs+j;
                    if(j<inject.gs && column<inject.width) {
                        const float weight=__fsub_rn(__fmul_rn(scale,float(packed[column])),minimum);
                        #pragma unroll
                        for(int token=0;token<Rows;++token)sums[token]=fmaf(weight,
                            mhc_read(normalized,size_t(token)*down.width+column,half),sums[token]);
                    }
                }
            }
            #pragma unroll
            for(int shift=16;shift;shift>>=1) {
                #pragma unroll
                for(int token=0;token<Rows;++token)sums[token]+=__shfl_down_sync(0xffffffffu,sums[token],shift);
            }
            #pragma unroll
            for(int token=0;token<Rows;++token) {
                if(lane==0)parts[size_t(token)*stride+down.outputs*splits+warp*injection_splits+part]=sums[token];
            }
        }
    }
    const auto* weight0=down.q+(down.offsets[row0]>>3);
    const auto* weight1=down.q+(down.offsets[row1]>>3);
    const float outer_scale0=down.row_scale[row0],outer_scale1=down.row_scale[row1];
    const float outer_min0=down.row_minimum[row0],outer_min1=down.row_minimum[row1];
    float result[4]{};
    for(int group=first;group<last;++group) {
        int dots[Planes==3?3:2][4]{};
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
            const int begin=chunk*32+word,column=group*GS+begin;
            int raw[4]{};
            if(begin<GS && column<down.width) {
                raw[0]=*reinterpret_cast<const int*>(weight0+column);
                raw[1]=*reinterpret_cast<const int*>(weight1+column);
            }
            if(begin+16<GS && column+16<down.width) {
                raw[2]=*reinterpret_cast<const int*>(weight0+column+16);
                raw[3]=*reinterpret_cast<const int*>(weight1+column+16);
            }
            int centered[4];
            #pragma unroll
            for(int item=0;item<4;++item)centered[item]=raw[item]^int(0x80808080u);
            // K padding must be zero in signed space too.
            if(begin>=GS || column>=down.width)centered[0]=centered[1]=0;
            if(begin+16>=GS || column+16>=down.width)centered[2]=centered[3]=0;
            #pragma unroll
            for(int plane=0;plane<(Planes==3?3:2);++plane) {
                const int8_t* values=plane==0?q.high:Planes==3 && plane==1?q.middle:q.low;
                int b[2]{};const int token=lane>>2;
                if(token<Rows) {
                    const auto* source=values+token*elements+group*GS;
                    if(begin<GS)b[0]=*reinterpret_cast<const int*>(source+begin);
                    if(begin+16<GS)b[1]=*reinterpret_cast<const int*>(source+begin+16);
                }
                if constexpr(Planes==4) {
                    if(plane==1)gr_mma16832_unsigned(dots[plane],raw,b);
                    else gr_mma16832(dots[plane],centered,b);
                }else gr_mma16832(dots[plane],centered,b);
            }
        }
        const float scale0=outer_scale0*float(down.scales[size_t(row0)*down.groups+group]);
        const float scale1=outer_scale1*float(down.scales[size_t(row1)*down.groups+group]);
        const float minimum0=outer_min0*float(down.minima[size_t(row0)*down.groups+group]);
        const float minimum1=outer_min1*float(down.minima[size_t(row1)*down.groups+group]);
        #pragma unroll
        for(int fragment=0;fragment<4;++fragment) {
            const int token=token0+(fragment&1);
            if(token<Rows) {
                const auto sums=q.sums[token*down.groups+group];
                const auto scales=q.scales[token*down.groups+group];
                const float scale=fragment<2?scale0:scale1,minimum=fragment<2?minimum0:minimum1;
                const float high=fmaf(scale,float(dots[0][fragment]+128*sums.x),-minimum*float(sums.x));
                result[fragment]=fmaf(scales.x,high,result[fragment]);
                if constexpr(Planes==3) {
                    const float middle=fmaf(scale,float(dots[1][fragment]+128*sums.y),-minimum*float(sums.y));
                    result[fragment]=fmaf(scales.x*(1.f/256.f),middle,result[fragment]);
                }
                const int low_dot=dots[Planes==3?2:1][fragment]+(Planes==3?128*sums.z:0);
                const float low=fmaf(scale,float(low_dot),-minimum*float(sums.z));
                result[fragment]=fmaf(scales.y,low,result[fragment]);
            }
        }
    }
    #pragma unroll
    for(int fragment=0;fragment<4;++fragment) {
        const int token=token0+(fragment&1),row=fragment<2?row0:row1;
        if(token<Rows)parts[size_t(token)*stride+(SplitMajor?split*down.outputs+row:row*splits+split)]=result[fragment];
    }
    if constexpr(PrepareUp) {
        static_assert(Rows==1 && Planes==4 && SplitMajor);
        gr_down_mma_finish_up_prepare(parts,splits,tile,half,prepared_up,counter_slot);
    }
}
