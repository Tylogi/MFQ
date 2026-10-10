#include "../ops/include/float_projection.h"
#include "qwen4_exp.h"
#include "../ops/include/gated_residual_fused.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <type_traits>


namespace {
struct MhcMatrix {
    const uint8_t *q,*bits,*scales,*minima;
    const int64_t* offsets;
    const float *row_scale,*row_minimum;
    const __half* zero_scale;
    int width,outputs,groups,gs;
    bool zero,aligned;
    const __nv_bfloat16* dense=nullptr;
};
struct MhcInput {
    const void *branch,*residual,*injection;
    const float* norm;
    const float* cached_gates;
    int hidden,streams;
    bool branch_half,residual_half,injection_half,output_half,after;
};
__device__ __forceinline__ float mhc_read(const void* p,size_t i,bool half) {
    return half?__half2float(static_cast<const __half*>(p)[i]):static_cast<const float*>(p)[i];
}
__device__ __forceinline__ float mhc_round(float v,bool half) {
    return half?__half2float(__float2half_rn(v)):v;
}
__device__ __forceinline__ void mhc_write(void* p,size_t i,float v,bool half) {
    if(half)static_cast<__half*>(p)[i]=__float2half_rn(v);else static_cast<float*>(p)[i]=v;
}
__device__ __forceinline__ float mhc_updated(MhcInput p,int sample,int stream,int column) {
    const size_t at=(size_t(sample)*p.streams+stream)*p.hidden+column;
    const float residual=mhc_read(p.residual,at,p.residual_half);
    if(!p.after)return residual;
    const float branch=mhc_read(p.branch,size_t(sample)*p.hidden+column,p.branch_half);
    const float gate=p.cached_gates[stream];
    const float product=mhc_round(__fmul_rn(branch,gate),p.branch_half && p.injection_half);
    return mhc_round(__fadd_rn(residual,product),p.output_half);
}
__device__ __forceinline__ void mhc_normalized_slice(MhcInput p,const float* inverse,int sample,int begin,int end,float* slice) {
    const int first=begin/p.hidden,last=(end-1)/p.hidden;
    for(int s=first;s<=last;++s) {
        const int from=max(0,begin-s*p.hidden),until=min(p.hidden,end-s*p.hidden);
        for(int column=from+int(threadIdx.x);column<until;column+=blockDim.x) {
            const float value=mhc_updated(p,sample,s,column);
            slice[s*p.hidden+column-begin]=mhc_round(__fmul_rn(__fmul_rn(value,inverse[s]),__fadd_rn(p.norm[s*p.hidden+column],1.f)),p.output_half);
        }
    }
}
__device__ __forceinline__ float mhc_activated(float value,bool half,int streams,bool injection) {
    const float projected=mhc_round(value,half);
    const float low=mhc_round(__fdiv_rn(projected,float(streams)),half);
    const float gate=mhc_round(1.f/(1.f+expf(-low)),half);
    return mhc_round(__fmul_rn(injection?2.f:low,gate),half);
}

template<int FixedGroup=0,bool StaticUnsignedQ8=false>
__device__ __forceinline__ float mhc_affine(MhcMatrix w,const float* normalized,int row,int first,int last,int lane) {
    if constexpr(!StaticUnsignedQ8)if(w.dense) {
        float sum=0.f;
        for(int group=first;group<last;++group) {
            const int column=group*w.gs+lane;
            if(column<w.width)sum=fmaf(__bfloat162float(w.dense[size_t(row)*w.width+column]),normalized[column-first*w.gs],sum);
        }
        for(int stride=16;stride;stride>>=1)sum+=__shfl_down_sync(0xffffffff,sum,stride);
        return sum;
    }
    if(StaticUnsignedQ8 || (w.aligned && !w.zero)) {
        const int gs=FixedGroup?FixedGroup:w.gs;
        const uint8_t* packed=w.q+(w.offsets[row]>>3);
        const float outer_scale=w.row_scale[row],outer_minimum=w.row_minimum[row];
        float sum=0.f;
        for(int group=first;group<last;++group) {
            const size_t meta=size_t(row)*w.groups+group;
            const float scale=__fmul_rn(outer_scale,float(w.scales[meta]));
            const float minimum=__fmul_rn(outer_minimum,float(w.minima[meta]));
            #pragma unroll
            for(int chunk=0;chunk<2;++chunk) {
                const int j=lane+chunk*32,column=group*gs+j;
                if(j<gs && column<w.width)
                    sum=fmaf(__fsub_rn(__fmul_rn(scale,float(packed[column])),minimum),normalized[column-first*gs],sum);
            }
        }
        for(int stride=16;stride;stride>>=1)sum+=__shfl_down_sync(0xffffffff,sum,stride);
        return sum;
    }
    if constexpr(!StaticUnsignedQ8) {
    const int bits=w.zero || w.aligned?8:w.bits[row];
    const uint64_t offset=w.zero?0:uint64_t(w.offsets[row]);
    const float outer_scale=w.zero?0.f:w.row_scale[row],outer_minimum=w.zero?0.f:w.row_minimum[row];
    float sum=0.f;
    for(int group=first;group<last;++group) {
        const size_t meta=size_t(row)*w.groups+group;
        const float scale=w.zero?__half2float(w.zero_scale[meta]):__fmul_rn(outer_scale,float(w.scales[meta]));
        const float minimum=w.zero?0.f:__fmul_rn(outer_minimum,float(w.minima[meta]));
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
            const int j=lane+chunk*32;if(j>=w.gs)continue;
            const int column=group*w.gs+j;if(column>=w.width)continue;
            float q;
            if(w.zero)q=float(static_cast<const int8_t*>(static_cast<const void*>(w.q))[size_t(row)*w.groups*w.gs+column]);
            else if(w.aligned)q=float(w.q[(offset>>3)+column]);
            else {
                const uint64_t bit=offset+uint64_t(column)*bits;const int shift=bit&7;
                unsigned value=w.q[bit>>3];if(shift+bits>8)value|=unsigned(w.q[(bit>>3)+1])<<8;
                q=float((value>>shift)&((1u<<bits)-1));
            }
            sum=fmaf(__fsub_rn(__fmul_rn(scale,q),minimum),normalized[column-first*w.gs],sum);
        }
    }
    for(int stride=16;stride;stride>>=1)sum+=__shfl_down_sync(0xffffffff,sum,stride);
    return sum;
    }else return 0.f; // The static aligned-q8 branch above always returns.
}

__global__ void gated_residual_norm_kernel(MhcInput input,void* updated,void* normalized,float* promoted_normalized,float eps,const float* prepared_right) {
    const int sample=blockIdx.y,tid=threadIdx.x;
    __shared__ float partial[256],inverse[4],prior_gates[4];
    if(tid<input.streams)prior_gates[tid]=input.after?mhc_read(input.injection,size_t(sample)*input.streams+tid,input.injection_half):0.f;
    input.cached_gates=prior_gates;
    __syncthreads();
    for(int s=int(blockIdx.x);s<=int(blockIdx.x);++s) {
        float sum=0.f;
        for(int column=tid;column<input.hidden;column+=256) {
            const float value=mhc_updated(input,sample,s,column);
            sum=__fadd_rn(sum,__fmul_rn(value,value));
        }
        partial[tid]=sum;
        __syncthreads();
        if(tid<32) {
            const float a=__fadd_rn(partial[tid],partial[tid+128]);
            const float b=__fadd_rn(partial[tid+64],partial[tid+192]);
            const float c=__fadd_rn(partial[tid+32],partial[tid+160]);
            const float d=__fadd_rn(partial[tid+96],partial[tid+224]);
            float value=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
            #pragma unroll
            for(int stride=16;stride>0;stride>>=1) {
                const float other=__shfl_down_sync(0xffffffff,value,stride);
                if(tid<stride)value=__fadd_rn(value,other);
            }
            if(tid==0)inverse[s]=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(value,float(input.hidden)),eps)));
        }
        __syncthreads();
    }
    for(int s=int(blockIdx.x);s<=int(blockIdx.x);++s)for(int column=tid;column<input.hidden;column+=256) {
        const size_t at=(size_t(sample)*input.streams+s)*input.hidden+column;
        const float value=mhc_updated(input,sample,s,column);
        mhc_write(updated,at,value,input.output_half);
        const float norm_value=mhc_round(__fmul_rn(__fmul_rn(value,inverse[s]),__fadd_rn(input.norm[s*input.hidden+column],1.f)),input.output_half);
        mhc_write(normalized,at,norm_value,input.output_half);
        if(promoted_normalized)promoted_normalized[at]=norm_value;
    }

    if(prepared_right && blockIdx.x==0 && blockIdx.y==0) {
        const char* address=reinterpret_cast<const char*>(prepared_right);
        const size_t bytes=size_t(input.hidden)*input.streams*input.streams*sizeof(float);
        for(size_t at=size_t(tid)*128;at<bytes;at+=size_t(blockDim.x)*128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(address+at));
    }
}
// Strata's GR staging uses cp.async for aligned float tiles. Keep the
// activation bytes and lane arithmetic unchanged; unaligned slices use scalar
// copies. Only Float normalization output uses the asynchronous path.
__device__ __forceinline__ void mhc_copy_normalized(float* destination,const void* source,
        int count,bool half,bool asynchronous) {
    const int tid=threadIdx.x;
    const bool aligned=((reinterpret_cast<std::uintptr_t>(source)|
        reinterpret_cast<std::uintptr_t>(destination))&15u)==0;
    if(asynchronous && !half && aligned) {
        const auto* input=static_cast<const float*>(source);
        for(int column=tid*4;column+4<=count;column+=blockDim.x*4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            const unsigned shared_address=unsigned(__cvta_generic_to_shared(destination+column));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                "r"(shared_address),"l"(input+column):"memory");
#else
            *reinterpret_cast<float4*>(destination+column)=*reinterpret_cast<const float4*>(input+column);
#endif
        }
        for(int column=(count&~3)+tid;column<count;column+=blockDim.x)
            destination[column]=input[column];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    }else for(int column=tid;column<count;column+=blockDim.x)
        destination[column]=mhc_read(source,column,half);
}

template<int FixedGroup=0,int ProjectionThreads=256,int InjectionKind=0,bool StaticFormat=false,bool VectorInjection=false>
__global__ void gated_residual_projection_kernel(MhcInput input,MhcMatrix down,MhcMatrix inject,
    const void* normalized,float* parts,int down_splits,int injection_splits,int row_tile,int down_capacity,
    const float* dense_input,bool asynchronous) {
    const int sample=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int down_blocks=(down.outputs+row_tile-1)/row_tile;
    const int split=blockIdx.x%down_splits,row_base=(blockIdx.x/down_splits)*row_tile;
    const int first=down.groups*split/down_splits,last=down.groups*(split+1)/down_splits;
    const int begin_column=first*down.gs,end_column=min(last*down.gs,down.width);
    extern __shared__ __align__(16) float normalized_shared[];
    const size_t base=size_t(sample)*(down.outputs*down_splits+inject.outputs*injection_splits);
    {
        const size_t byte_offset=(size_t(sample)*down.width+begin_column)*(input.output_half?sizeof(__half):sizeof(float));
        mhc_copy_normalized(normalized_shared,static_cast<const char*>(normalized)+byte_offset,
            end_column-begin_column,input.output_half,asynchronous);
        __syncthreads();
        for(int row=row_base+warp;row<row_base+row_tile && row<down.outputs;row+=ProjectionThreads/32) {
            const float sum=mhc_affine<FixedGroup,StaticFormat>(down,normalized_shared,row,first,last,lane);
            if(lane==0)parts[base+row*down_splits+split]=sum;
        }
    }
    if constexpr(InjectionKind!=2)if(dense_input && tid<32) {
        const int per_block=32/inject.outputs;
        const int virtual_lane=int(blockIdx.x)*per_block+tid/inject.outputs;
        const int row=tid%inject.outputs;
        if(tid<per_block*inject.outputs && virtual_lane<32) {
            float sum=0.f;
            for(int begin=virtual_lane*4;begin<inject.width;begin+=128) {
                if constexpr(VectorInjection) {
                    const auto x=*reinterpret_cast<const float4*>(dense_input+size_t(sample)*inject.width+begin);
                    const auto w=*reinterpret_cast<const float4*>(inject.row_scale+size_t(begin)*inject.outputs+row*4);
                    sum=fmaf(x.x,w.x,sum);sum=fmaf(x.y,w.y,sum);
                    sum=fmaf(x.z,w.z,sum);sum=fmaf(x.w,w.w,sum);
                } else {
                #pragma unroll
                for(int item=0;item<4;++item)if(begin+item<inject.width)
                    sum=fmaf(dense_input[size_t(sample)*inject.width+begin+item],
                        inject.row_scale[(size_t(begin)+item)*inject.outputs+row],sum);
                }
            }
            parts[base+down.outputs*down_splits+row*injection_splits+virtual_lane]=sum;
        }
    }
    if constexpr(InjectionKind!=1)if(InjectionKind==2 || !inject.dense) {
        const int first_inject=((begin_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups;
        const int last_inject=min(injection_splits,((end_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups);
        for(int part=first_inject+int(blockIdx.x)/down_splits;part<last_inject;part+=down_blocks) {
        const int begin=inject.groups*part/injection_splits,end=inject.groups*(part+1)/injection_splits;
        float* slice=normalized_shared+down_capacity;
        const int injection_end=end*inject.gs<inject.width?end*inject.gs:inject.width;
        const size_t byte_offset=(size_t(sample)*down.width+begin*inject.gs)*(input.output_half?sizeof(__half):sizeof(float));
        mhc_copy_normalized(slice,static_cast<const char*>(normalized)+byte_offset,
            injection_end-begin*inject.gs,input.output_half,asynchronous);
        __syncthreads();
        if(warp<inject.outputs) {
            const float sum=mhc_affine<0,InjectionKind==2>(inject,slice,warp,begin,end,lane);
            if(lane==0)parts[base+down.outputs*down_splits+warp*injection_splits+part]=sum;
        }
        __syncthreads();
        }
    }

}

template<int FixedGroup=0>
__global__ void gated_residual_down_kernel(MhcInput input,MhcMatrix down,MhcMatrix inject,
    void* updated,void* normalized,float* promoted_normalized,float* parts,int down_splits,int injection_splits,float eps,int row_tile,int down_capacity) {
    const int sample=blockIdx.y,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int down_blocks=(down.outputs+row_tile-1)/row_tile;
    const int split=blockIdx.x%down_splits,row_base=(blockIdx.x/down_splits)*row_tile;
    const int first=down.groups*split/down_splits,last=down.groups*(split+1)/down_splits;
    const int begin_column=first*down.gs,end_column=min(last*down.gs,down.width);
    const int first_inject=((begin_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups;
    const int last_inject=min(injection_splits,((end_column+inject.gs-1)/inject.gs*injection_splits+inject.groups-1)/inject.groups);
    int first_stream=begin_column/input.hidden,last_stream=(end_column-1)/input.hidden;
    if(first_inject+int(blockIdx.x)/down_splits<last_inject)
        last_stream=max(last_stream,(min(inject.groups*last_inject/injection_splits*inject.gs,inject.width)-1)/input.hidden);
    if(blockIdx.x==0) {first_stream=0;last_stream=input.streams-1;}
    __shared__ float partial[256],inverse[4],prior_gates[4];
    extern __shared__ float normalized_shared[];
    if(tid<input.streams)prior_gates[tid]=input.after?mhc_read(input.injection,size_t(sample)*input.streams+tid,input.injection_half):0.f;
    input.cached_gates=prior_gates;
    __syncthreads();
    for(int s=first_stream;s<=last_stream;++s) {
        float sum=0.f;
        for(int column=tid;column<input.hidden;column+=256) {
            const float value=mhc_updated(input,sample,s,column);
            sum=__fadd_rn(sum,__fmul_rn(value,value));
        }
        partial[tid]=sum;
        __syncthreads();
        for(int stride=128;stride>=32;stride>>=1) {
            if(tid<stride)partial[tid]=__fadd_rn(partial[tid],partial[tid+stride]);
            __syncthreads();
        }
        if(tid<32) {
            float value=partial[tid];
            #pragma unroll
            for(int stride=16;stride>0;stride>>=1) {
                const float other=__shfl_down_sync(0xffffffff,value,stride);
                if(tid<stride)value=__fadd_rn(value,other);
            }
            if(tid==0)inverse[s]=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(value,float(input.hidden)),eps)));
        }
        __syncthreads();
    }
    __syncthreads();
    if(blockIdx.x==0)for(int s=0;s<input.streams;++s)for(int column=tid;column<input.hidden;column+=256) {
        const size_t at=(size_t(sample)*input.streams+s)*input.hidden+column;
        const float value=mhc_updated(input,sample,s,column);
        mhc_write(updated,at,value,input.output_half);
        const float norm_value=mhc_round(__fmul_rn(__fmul_rn(value,inverse[s]),__fadd_rn(input.norm[s*input.hidden+column],1.f)),input.output_half);
        mhc_write(normalized,at,norm_value,input.output_half);
        if(promoted_normalized)promoted_normalized[at]=norm_value;
    }
    const size_t base=size_t(sample)*(down.outputs*down_splits+inject.outputs*injection_splits);
    {
        mhc_normalized_slice(input,inverse,sample,begin_column,end_column,normalized_shared);
        __syncthreads();
        for(int row=row_base+warp;row<row_base+row_tile && row<down.outputs;row+=8) {
            const float sum=mhc_affine<FixedGroup>(down,normalized_shared,row,first,last,lane);
            if(lane==0)parts[base+row*down_splits+split]=sum;
        }
        __syncthreads();
    }
    if(!inject.dense)for(int part=first_inject+int(blockIdx.x)/down_splits;part<last_inject;part+=down_blocks) {
        const int begin=inject.groups*part/injection_splits,end=inject.groups*(part+1)/injection_splits;
        float* slice=normalized_shared+down_capacity;
        const int injection_end=end*inject.gs<inject.width?end*inject.gs:inject.width;
        mhc_normalized_slice(input,inverse,sample,begin*inject.gs,injection_end,slice);
        __syncthreads();
        if(warp<inject.outputs) {
            const float sum=mhc_affine(inject,slice,warp,begin,end,lane);
            if(lane==0)parts[base+down.outputs*down_splits+warp*injection_splits+part]=sum;
        }
        __syncthreads();
    }

    // A single existing CTA warms only prepared FP32 injection cache lines.
    // No additional kernel, projection, or weight arithmetic is introduced.
    if(inject.dense && inject.row_scale && blockIdx.x==0 && blockIdx.y==0) {
        const char* address=reinterpret_cast<const char*>(inject.row_scale);
        const size_t bytes=size_t(inject.width)*inject.outputs*sizeof(float);
        for(size_t at=size_t(tid)*128;at<bytes;at+=size_t(blockDim.x)*128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(address+at));
    }
}

template<int FixedGroup=0,int FixedStreams=0,int FixedWidth=0>
__global__ void gated_residual_up_kernel(MhcMatrix down,MhcMatrix up,MhcMatrix inject,
    const void* normalized,const float* parts,void* mixed,void* injection,
    int hidden,int streams,bool half,int down_splits,int injection_splits,const float* dense_projection) {
    extern __shared__ float low[];
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,sample=blockIdx.y;
    if constexpr(FixedStreams>0)streams=FixedStreams;
    const size_t base=size_t(sample)*(down.outputs*down_splits+inject.outputs*injection_splits);
    for(int row=tid;row<down.outputs;row+=blockDim.x) {
        float sum=down_splits==1?parts[base+row]:0.f;
        if(down_splits>1)for(int split=0;split<down_splits;++split)sum+=parts[base+row*down_splits+split];
        low[row]=mhc_activated(sum,half,streams,false);
    }
    if(inject.dense && !dense_projection && blockIdx.x==0 && tid<32) {
        #pragma unroll
        for(int row=0;row<4;++row)if(row<inject.outputs) {
            float sum=parts[base+down.outputs*down_splits+row*injection_splits+lane];
            #pragma unroll
            for(int shift=16;shift>0;shift>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,shift));
            if(lane==0)mhc_write(injection,size_t(sample)*inject.outputs+row,
                mhc_activated(__fadd_rn(sum,0.f),false,streams,true),false);
        }
    }
    if(blockIdx.x==0 && tid<inject.outputs && (!inject.dense || dense_projection)) {
        const size_t start=base+down.outputs*down_splits+tid*injection_splits;
        float sum=inject.dense?dense_projection[size_t(sample)*inject.outputs+tid]:(injection_splits==1?parts[start]:0.f);
        if(!inject.dense && injection_splits>1)for(int split=0;split<injection_splits;++split)sum+=parts[start+split];
        const bool injection_half=half && !inject.dense;
        mhc_write(injection,size_t(sample)*inject.outputs+tid,mhc_activated(sum,injection_half,streams,true),injection_half);
    }
    __syncthreads();
    const int column=blockIdx.x*(blockDim.x/32)+warp;if(column>=hidden)return;
    float sums[4]{};
    if(up.aligned && !up.zero) {
        const int gs=FixedGroup?FixedGroup:up.gs;
        constexpr int fixed_groups=FixedWidth?(FixedWidth+FixedGroup-1)/FixedGroup:0;
        const int groups=fixed_groups?fixed_groups:up.groups;
        const int width=FixedWidth?FixedWidth:up.width;
        const uint8_t* packed[4];
        float outer_scale[4],outer_minimum[4];
        float cached_scale[4]{},cached_minimum[4]{};
        int rows[4];
        #pragma unroll
        for(int s=0;s<4;++s)if(s<streams) {
            rows[s]=s*hidden+column;
            packed[s]=up.q+(up.offsets[rows[s]]>>3);
            outer_scale[s]=up.row_scale[rows[s]];
            outer_minimum[s]=up.row_minimum[rows[s]];
            if constexpr(fixed_groups>0 && fixed_groups<=32) {
                if(lane<fixed_groups) {
                    const size_t meta=size_t(rows[s])*fixed_groups+lane;
                    cached_scale[s]=__fmul_rn(outer_scale[s],float(up.scales[meta]));
                    cached_minimum[s]=__fmul_rn(outer_minimum[s],float(up.minima[meta]));
                }
            }
        }
        #pragma unroll 1
        for(int group=0;group<groups;++group) {
            float scale[4],minimum[4];
            #pragma unroll
            for(int s=0;s<4;++s)if(s<streams) {
                if constexpr(fixed_groups>0 && fixed_groups<=32) {
                    scale[s]=__shfl_sync(0xffffffff,cached_scale[s],group);
                    minimum[s]=__shfl_sync(0xffffffff,cached_minimum[s],group);
                } else {
                    const size_t meta=size_t(rows[s])*up.groups+group;
                    scale[s]=__fmul_rn(outer_scale[s],float(up.scales[meta]));
                    minimum[s]=__fmul_rn(outer_minimum[s],float(up.minima[meta]));
                }
            }
            #pragma unroll
            for(int chunk=0;chunk<2;++chunk) {
                const int j=lane+chunk*32,at=group*gs+j;
                if(j<gs && at<width) {
                    const float activation=low[at];
                    #pragma unroll
                    for(int s=0;s<4;++s)if(s<streams) {
                        const float weight=__fsub_rn(__fmul_rn(scale[s],float(packed[s][at])),minimum[s]);
                        sums[s]=fmaf(weight,activation,sums[s]);
                    }
                }
            }
        }
        for(int stride=16;stride;stride>>=1) {
            #pragma unroll
            for(int s=0;s<4;++s)if(s<streams)sums[s]+=__shfl_down_sync(0xffffffff,sums[s],stride);
        }
    } else {
        #pragma unroll
        for(int s=0;s<4;++s)if(s<streams)sums[s]=mhc_affine(up,low,s*hidden+column,0,up.groups,lane);
    }
    if(lane==0) {
        float value=0.f;
        #pragma unroll
        for(int s=0;s<4;++s)if(s<streams) {
            const float projected=mhc_round(sums[s],half);
            const float gate=mhc_round(1.f/(1.f+expf(-projected)),half);
            const float product=mhc_round(__fmul_rn(gate,mhc_read(normalized,(size_t(sample)*streams+s)*hidden+column,half)),half);
            value=__fadd_rn(value,product);
        }
        mhc_write(mixed,size_t(sample)*hidden+column,__fdiv_rn(value,float(streams)),half);
    }
}

MhcMatrix mhc_matrix(const NintWeight& w) {
    return {w.q_packed.data_ptr<uint8_t>(),w.q8_zero?nullptr:w.row_q_bits.data_ptr<uint8_t>(),
        w.q8_zero?nullptr:w.sub_scale.data_ptr<uint8_t>(),w.q8_zero?nullptr:w.sub_min.data_ptr<uint8_t>(),
        w.q8_zero?nullptr:w.row_q_bit_offsets.data_ptr<int64_t>(),w.q8_zero?nullptr:w.neuron_scale.data_ptr<float>(),
        w.q8_zero?nullptr:w.neuron_min.data_ptr<float>(),w.q8_zero?w.q8_zero_scale.data_ptr<__half>():nullptr,
        int(w.neuron_len),int(w.out),int(w.ng),int(w.gs),w.q8_zero,w.aligned_q8};
}
}

mfq_tensor_backend::Tensor prepare_gr_dense_vector_right(const mfq_tensor_backend::Tensor& right) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(right.is_cuda() && right.dim()==2 && right.scalar_type()==tb::kFloat32 &&
        right.size(0)>0 && right.size(0)%4==0 && right.size(1)>0 && right.size(1)<=4,
        "vector residual injection requires F32 [4*k, 1..4] weights");
    return right.reshape({right.size(0)/4,4,right.size(1)}).transpose(1,2).contiguous();
}

std::vector<mfq_tensor_backend::Tensor> gated_residual_two_stage_cuda(
    const mfq_tensor_backend::Tensor& branch,const mfq_tensor_backend::Tensor& residual,
    const mfq_tensor_backend::Tensor& injection,const mfq_tensor_backend::Tensor& norm,
    const NintWeight& down,const NintWeight& up,const NintWeight* inject,int64_t streams,double eps,bool after,
    const mfq_tensor_backend::Tensor& dense_injection,const mfq_tensor_backend::Tensor& prepared_injection_right,
    const std::function<void(const char*)>& marker) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK((inject!=nullptr)!=dense_injection.defined(),"exactly one residual injection weight is required");
    const auto matrix_valid=[](const NintWeight& w) {return w.out>0 && w.neuron_len>0 && w.gs>0 && w.gs<=64 &&
        w.ng==(w.neuron_len+w.gs-1)/w.gs;};
    MFQ_RUNTIME_CHECK(matrix_valid(down) && matrix_valid(up) && (!inject || matrix_valid(*inject)),"fused residual quantization geometry invalid");
    const auto valid=[](const tb::Tensor& x) {return x.is_cuda() && x.is_contiguous() &&
        (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32);};
    MFQ_RUNTIME_CHECK(valid(residual) && streams>0 && streams<=4 && residual.dim()>0 && residual.size(-1)%streams==0,
        "fused residual residual geometry/dtype unsupported");
    const int hidden=residual.size(-1)/streams,rows=residual.numel()/residual.size(-1);
    MFQ_RUNTIME_CHECK(rows>0 && rows<=65535 && norm.is_cuda() && norm.is_contiguous() && norm.scalar_type()==tb::kFloat32 &&
        norm.numel()==residual.size(-1) && norm.device()==residual.device(),"fused residual norm unsupported");
    MFQ_RUNTIME_CHECK(!after || (valid(branch) && valid(injection) && branch.numel()==rows*hidden &&
        injection.numel()==rows*streams && branch.device()==residual.device() && injection.device()==residual.device()),
        "fused residual prior branch/injection disagree");
    MFQ_RUNTIME_CHECK(down.neuron_len==residual.size(-1) && up.neuron_len==down.out && up.out==residual.size(-1) &&
        down.q_packed.device()==residual.device() && up.q_packed.device()==residual.device() &&
        (dense_injection.defined()?(dense_injection.is_cuda() && dense_injection.is_contiguous() && dense_injection.scalar_type()==tb::kBFloat16 &&
        dense_injection.device()==residual.device() && dense_injection.dim()==2 && dense_injection.size(0)==streams && dense_injection.size(1)==down.neuron_len)
        :(inject->neuron_len==down.neuron_len && inject->out==streams && inject->q_packed.device()==residual.device())),"fused residual matrices disagree");
    MfqCudaGuard guard(residual.device());
    const int resident=nint_float_projection_resident_blocks(residual.get_device());
    const auto splits=[&](const NintWeight& w) {const int blocks=((w.out+3)/4)*rows;
        return int(std::min<int64_t>(w.ng,std::max(1,(resident+blocks-1)/blocks)));};
    MFQ_RUNTIME_CHECK(splits(up)==1,"fused residual requires original up projection unsplit");
    const int ds=splits(down);
    const char* norm_setting=std::getenv("MFQ_GR_SHARE_NORM");
    const bool share_norm=!norm_setting || norm_setting[0]!='0';
    const char* group_setting=std::getenv("MFQ_GR_FIXED_GROUPS");
    const bool fixed_groups=!group_setting || group_setting[0]=='1';
    const char* format_setting=std::getenv("MFQ_GR_FIXED_FORMATS");
    const bool fixed_formats=!format_setting || format_setting[0]=='1';
    const char* thread_setting=std::getenv("MFQ_GR_UP_THREADS");
    const int up_threads=thread_setting?std::atoi(thread_setting):512;
    MFQ_RUNTIME_CHECK(up_threads==128 || up_threads==256 || up_threads==512,
        "residual Up threads must be 128, 256 or 512");
    const char* projection_setting=std::getenv("MFQ_GR_PROJECTION_THREADS");
    int projection_threads=projection_setting?std::atoi(projection_setting):256;
    MFQ_RUNTIME_CHECK(projection_threads==128 || projection_threads==256 || projection_threads==512,
        "residual projection threads must be 128, 256 or 512");
    if(!share_norm)projection_threads=256;
    const int tile=projection_threads/32;
    const char* warp_setting=std::getenv("MFQ_GR_DENSE_FUSION");
    const bool dense_warp=(!warp_setting || warp_setting[0]!='0') && share_norm &&
        rows==1 && dense_injection.defined() && prepared_injection_right.defined() &&
        ((down.out+tile-1)/tile)*ds >= (32+(32/streams)-1)/(32/streams);
    const int is=dense_warp?32:(dense_injection.defined()?std::min<int64_t>((down.neuron_len+31)/32,resident):splits(*inject));
    const bool bh=after && branch.scalar_type()==tb::kFloat16,rh=residual.scalar_type()==tb::kFloat16,
        ih=after && injection.scalar_type()==tb::kFloat16,half=rh && (!after || (bh && ih));
    const auto options=residual.options().dtype(half?tb::kFloat16:tb::kFloat32);
    auto normalized=tb::empty(residual.sizes().vec(),options),updated=tb::empty(residual.sizes().vec(),options);
    auto shape=residual.sizes().vec();shape.back()=hidden;auto mixed=tb::empty(shape,options);
    shape.back()=streams;auto next_injection=tb::empty(shape,options.dtype(dense_injection.defined() || !half?tb::kFloat32:tb::kFloat16));
    const char* async_setting=std::getenv("MFQ_GR_ASYNC_NORMALIZED");
    const bool asynchronous=share_norm && !half && (!async_setting || async_setting[0]=='1');
    auto promoted_normalized=(dense_injection.defined() || asynchronous) && half?
        tb::empty(residual.sizes().vec(),options.dtype(tb::kFloat32)):tb::Tensor{};
    const bool vector_injection=prepared_injection_right.defined() && prepared_injection_right.dim()==3;
    if(prepared_injection_right.defined()) {
        const bool shape_matches=vector_injection?
            (down.neuron_len%4==0 && prepared_injection_right.size(0)==down.neuron_len/4 &&
             prepared_injection_right.size(1)==streams && prepared_injection_right.size(2)==4 &&
             (reinterpret_cast<std::uintptr_t>(prepared_injection_right.data_ptr())&15u)==0):
            (prepared_injection_right.dim()==2 && prepared_injection_right.size(0)==down.neuron_len &&
             prepared_injection_right.size(1)==streams);
        MFQ_RUNTIME_CHECK(dense_injection.defined() && prepared_injection_right.is_cuda() &&
            prepared_injection_right.device()==residual.device() && prepared_injection_right.is_contiguous() &&
            prepared_injection_right.scalar_type()==tb::kFloat32 && shape_matches,
            "prepared residual injection matrix shape/dtype/device disagrees");
    }
    auto parts=tb::empty({rows,down.out*ds+streams*is},residual.options().dtype(tb::kFloat32));
    MhcInput input{after?branch.data_ptr():nullptr,residual.data_ptr(),after?injection.data_ptr():nullptr,
        norm.data_ptr<float>(),nullptr,hidden,int(streams),bh,rh,ih,half,after};
    auto projection_input=input;
    if(asynchronous)projection_input.output_half=false;
    const void* projection_normalized=asynchronous && promoted_normalized.defined()?
        promoted_normalized.data_ptr():normalized.data_ptr();
    const auto d=mhc_matrix(down),u=mhc_matrix(up);
    MhcMatrix i{};
    if(dense_injection.defined()) {
        i.width=down.neuron_len;i.outputs=streams;i.gs=32;i.groups=(i.width+31)/32;
        i.dense=static_cast<const __nv_bfloat16*>(dense_injection.data_ptr());
        if(prepared_injection_right.defined())i.row_scale=prepared_injection_right.data_ptr<float>();
    } else i=mhc_matrix(*inject);
    const int down_capacity=((down.ng+ds-1)/ds)*down.gs,inject_capacity=dense_warp?0:((i.groups+is-1)/is)*i.gs;
    const size_t shared=(down_capacity+inject_capacity)*sizeof(float);
    MFQ_RUNTIME_CHECK(shared+1056<=48*1024,"fused residual projection exceeds shared memory capacity");
    if(marker)marker("begin");
    if(share_norm) {
        gated_residual_norm_kernel<<<dim3(unsigned(streams),unsigned(rows)),256,0,mfq_current_cuda_stream()>>>(
            input,updated.data_ptr(),normalized.data_ptr(),promoted_normalized.defined()?promoted_normalized.data_ptr<float>():nullptr,float(eps),dense_injection.defined()&&prepared_injection_right.defined()?prepared_injection_right.data_ptr<float>():nullptr);
        if(marker)marker("norm");
    }
    const auto launch_down=[&](auto group) {
        constexpr int fixed=decltype(group)::value;
        if(share_norm) {
            const auto launch=[&](auto threads) {
              const auto format=[&](auto injection_kind,auto static_format) {
                const auto packed=[&](auto vector_layout) {
                  gated_residual_projection_kernel<fixed,std::decay_t<decltype(threads)>::value,std::decay_t<decltype(injection_kind)>::value,std::decay_t<decltype(static_format)>::value,decltype(vector_layout)::value><<<dim3(unsigned((down.out+tile-1)/tile*ds),unsigned(rows)),unsigned(threads),unsigned(shared),mfq_current_cuda_stream()>>>(
                    projection_input,d,i,projection_normalized,parts.data_ptr<float>(),ds,is,tile,down_capacity,
                    dense_warp?(promoted_normalized.defined()?promoted_normalized.data_ptr<float>():normalized.data_ptr<float>()):nullptr,asynchronous);
                };
                if constexpr(decltype(injection_kind)::value!=2) {
                    if(vector_injection)packed(std::true_type{});
                    else packed(std::false_type{});
                }else packed(std::false_type{});
              };
              if constexpr(fixed>0) {
                if(fixed_formats) {
                    if(dense_injection.defined())format(std::integral_constant<int,1>{},std::true_type{});
                    else if(inject->aligned_q8 && !inject->q8_zero)format(std::integral_constant<int,2>{},std::true_type{});
                    else format(std::integral_constant<int,0>{},std::true_type{});
                }else format(std::integral_constant<int,0>{},std::false_type{});
              }else format(std::integral_constant<int,0>{},std::false_type{});
            };
            if(projection_threads==128)launch(std::integral_constant<int,128>{});
            else if(projection_threads==512)launch(std::integral_constant<int,512>{});
            else launch(std::integral_constant<int,256>{});
        }else gated_residual_down_kernel<fixed><<<dim3(unsigned((down.out+tile-1)/tile*ds),unsigned(rows)),256,unsigned(shared),mfq_current_cuda_stream()>>>(
            input,d,i,updated.data_ptr(),normalized.data_ptr(),promoted_normalized.defined()?promoted_normalized.data_ptr<float>():nullptr,parts.data_ptr<float>(),ds,is,float(eps),tile,down_capacity);
    };
    if(fixed_groups && down.aligned_q8 && !down.q8_zero && down.gs==32)launch_down(std::integral_constant<int,32>{});
    else if(fixed_groups && down.aligned_q8 && !down.q8_zero && down.gs==48)launch_down(std::integral_constant<int,48>{});
    else if(fixed_groups && down.aligned_q8 && !down.q8_zero && down.gs==64)launch_down(std::integral_constant<int,64>{});
    else launch_down(std::integral_constant<int,0>{});
    if(marker)marker("down");
    tb::Tensor dense_projection;
    if(dense_injection.defined() && !dense_warp) {
        const auto right=vector_injection?
            prepared_injection_right.transpose(1,2).reshape({down.neuron_len,streams}):
            prepared_injection_right;
        dense_projection=tb::matmul(promoted_normalized.defined()?promoted_normalized:normalized,
            right.defined()?right:dense_injection.transpose(-1,-2).to(tb::kFloat32));
    }
    if(marker && dense_projection.defined())marker("dense_projection");
    const auto launch_up=[&](auto group) {
        constexpr int fixed=decltype(group)::value;
        const auto launch=[&](auto width) {
            gated_residual_up_kernel<fixed,fixed?4:0,decltype(width)::value><<<dim3(unsigned((hidden+up_threads/32-1)/(up_threads/32)),unsigned(rows)),unsigned(up_threads),unsigned(down.out*sizeof(float)),mfq_current_cuda_stream()>>>(
                d,u,i,normalized.data_ptr(),parts.data_ptr<float>(),mixed.data_ptr(),next_injection.data_ptr(),hidden,int(streams),half,ds,is,dense_projection.defined()?dense_projection.data_ptr<float>():nullptr);
        };
        if constexpr(fixed>0) {
            if(up.neuron_len==320)launch(std::integral_constant<int,320>{});
            else launch(std::integral_constant<int,0>{});
        } else launch(std::integral_constant<int,0>{});
    };
    if(fixed_groups && streams==4 && up.aligned_q8 && !up.q8_zero && up.gs==32)launch_up(std::integral_constant<int,32>{});
    else if(fixed_groups && streams==4 && up.aligned_q8 && !up.q8_zero && up.gs==48)launch_up(std::integral_constant<int,48>{});
    else if(fixed_groups && streams==4 && up.aligned_q8 && !up.q8_zero && up.gs==64)launch_up(std::integral_constant<int,64>{});
    else launch_up(std::integral_constant<int,0>{});
    if(marker)marker("up_mix");
    MFQ_CUDA_CHECK(cudaGetLastError());
    if(marker)marker("end");
    return {mixed,updated,next_injection};
}
