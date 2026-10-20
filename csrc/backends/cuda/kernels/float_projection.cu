#include "float_projection.h"
#include "gr_prefill_blas.h"
#include "gr_prefill_decode.cuh"

#include <mma.h>
#include "native/env.h"
#include "kernels/cache/async_copy.cuh"
#include "native/nint_weight.h"
#include "qwen4_exp.h"

#include <limits>
#include <algorithm>
#include <map>
#include <mutex>
#include <type_traits>
#include <cstdlib>
#include <array>
#include <cstdio>
// Compressed GR tiles with three TF32 products: Ah*Bh + Al*Bh + Ah*Bl.
// The residual products retain near-FP32 precision without dense weights.
namespace {
template<class Input,bool AlignedQ8,bool Zero>
__global__ void gr_prefill_tf32_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,float* output,
        int rows,int width,int outputs,int groups,int gs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
    using namespace nvcuda;
    constexpr int BM=32,BN=32,BK=32,Pad=40;
    __shared__ __align__(32) float ah[BM][Pad],al[BM][Pad],bh[BN][Pad],bl[BN][Pad],result[BM][BN];
    const int tid=threadIdx.x,warp=tid/32,wm=(warp/2)*16,wn=(warp%2)*16;
    const int row0=int(blockIdx.y)*BM,out0=int(blockIdx.x)*BN;
    wmma::fragment<wmma::accumulator,16,16,8,float> total;
    wmma::fill_fragment(total,0.0f);
    for(int k0=0;k0<width;k0+=BK) {
        wmma::fragment<wmma::accumulator,16,16,8,float> high,low;
        wmma::fill_fragment(high,0.0f);wmma::fill_fragment(low,0.0f);
        for(int i=tid;i<BM*BK;i+=128) {
            const int r=i/BK,c=i%BK;
            const float value=row0+r<rows && k0+c<width ? float(input[int64_t(row0+r)*width+k0+c]) : 0;
            const float top=wmma::__float_to_tf32(value);
            ah[r][c]=top;al[r][c]=wmma::__float_to_tf32(__fsub_rn(value,top));
        }
        for(int i=tid;i<BN*BK;i+=128) {
            const int r=i/BK,c=i%BK,neuron=out0+r,column=k0+c;
            float value=0;
            if(neuron<outputs && column<width) {
                const int64_t meta=int64_t(neuron)*groups+column/gs;
                float q,scale,minimum;
                if constexpr(Zero) {
                    q=float(static_cast<int8_t>(packed[meta*gs+column%gs]));
                    scale=__half2float(zero_scales[meta]);minimum=0;
                } else {
                    const int bits=AlignedQ8?8:row_bits[neuron];
                    const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
                    unsigned word=packed[bit>>3];const int shift=int(bit&7);
                    if(shift+bits>8)word|=unsigned(packed[(bit>>3)+1])<<8;
                    q=float((word>>shift)&((1u<<bits)-1));
                    scale=row_scales[neuron]*float(scales[meta]);
                    minimum=row_minima[neuron]*float(minima[meta]);
                }
                value=__fsub_rn(__fmul_rn(scale,q),minimum);
            }
            const float top=wmma::__float_to_tf32(value);
            bh[r][c]=top;bl[r][c]=wmma::__float_to_tf32(__fsub_rn(value,top));
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k+=8) {
            wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a_hi,a_lo;
            wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> b_hi,b_lo;
            wmma::load_matrix_sync(a_hi,&ah[wm][k],Pad);
            wmma::load_matrix_sync(b_hi,&bh[wn][k],Pad);
            wmma::load_matrix_sync(b_lo,&bl[wn][k],Pad);
            wmma::mma_sync(low,a_hi,b_lo,low);
            if constexpr(!std::is_same<Input,__half>::value) {
                wmma::load_matrix_sync(a_lo,&al[wm][k],Pad);
                wmma::mma_sync(low,a_lo,b_hi,low);
            }
            wmma::mma_sync(high,a_hi,b_hi,high);
        }
        #pragma unroll
        for(int i=0;i<total.num_elements;++i)total.x[i]+=high.x[i]+low.x[i];
        __syncthreads();
    }
    wmma::store_matrix_sync(&result[wm][wn],total,BN,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<BM*BN;i+=128) {
        const int m=i/BN,n=i%BN;
        if(row0+m<rows && out0+n<outputs)output[int64_t(row0+m)*outputs+out0+n]=result[m][n];
    }
#endif
}
} // namespace
// Transient half high/residual tiles. Residuals are scaled by 4096 before
// half conversion; two/three Tensor Core products retain FP32 accuracy.
namespace {
template<class Input,bool AlignedQ8,bool Zero>
__global__ void gr_prefill_half2_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,float* output,
        int rows,int width,int outputs,int groups,int gs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=700
    using namespace nvcuda;
    constexpr int BM=64,BN=32,BK=32,Pad=40;
    __shared__ __align__(32) __half ah[BM][Pad],al[BM][Pad],bh[BN][Pad],bl[BN][Pad];
    __shared__ __align__(32) float result[BM][BN];
    const int tid=threadIdx.x,warp=tid/32,wm=(warp/2)*16,wn=(warp%2)*16;
    const int row0=int(blockIdx.y)*BM,out0=int(blockIdx.x)*BN;
    wmma::fragment<wmma::accumulator,16,16,16,float> total;
    wmma::fill_fragment(total,0.0f);
    for(int k0=0;k0<width;k0+=BK) {
        for(int i=tid;i<BM*BK;i+=256) {
            const int r=i/BK,c=i%BK;
            const float value=row0+r<rows && k0+c<width?
                float(input[int64_t(row0+r)*width+k0+c]):0;
            const auto top=__float2half_rn(value);
            ah[r][c]=top;
            if constexpr(!std::is_same<Input,__half>::value)
                al[r][c]=__float2half_rn(__fsub_rn(value,__half2float(top))*4096.0f);
        }
        for(int i=tid;i<BN*BK;i+=256) {
            const int r=i/BK,c=i%BK,neuron=out0+r,column=k0+c;
            float value=0;
            if(neuron<outputs && column<width) {
                const int64_t meta=int64_t(neuron)*groups+column/gs;
                float q,scale,minimum;
                if constexpr(Zero) {
                    q=float(static_cast<int8_t>(packed[meta*gs+column%gs]));
                    scale=__half2float(zero_scales[meta]);minimum=0;
                } else {
                    const int bits=AlignedQ8?8:row_bits[neuron];
                    const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
                    unsigned word=packed[bit>>3];const int shift=int(bit&7);
                    if(shift+bits>8)word|=unsigned(packed[(bit>>3)+1])<<8;
                    q=float((word>>shift)&((1u<<bits)-1));
                    scale=row_scales[neuron]*float(scales[meta]);
                    minimum=row_minima[neuron]*float(minima[meta]);
                }
                value=__fsub_rn(__fmul_rn(scale,q),minimum);
            }
            const auto top=__float2half_rn(value);
            bh[r][c]=top;bl[r][c]=__float2half_rn(__fsub_rn(value,__half2float(top))*4096.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> high,low;
        wmma::fill_fragment(high,0.0f);wmma::fill_fragment(low,0.0f);
        #pragma unroll
        for(int k=0;k<BK;k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_hi,a_lo;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b_hi,b_lo;
            wmma::load_matrix_sync(a_hi,&ah[wm][k],Pad);
            wmma::load_matrix_sync(b_hi,&bh[wn][k],Pad);
            wmma::load_matrix_sync(b_lo,&bl[wn][k],Pad);
            wmma::mma_sync(high,a_hi,b_hi,high);
            wmma::mma_sync(low,a_hi,b_lo,low);
            if constexpr(!std::is_same<Input,__half>::value) {
                wmma::load_matrix_sync(a_lo,&al[wm][k],Pad);
                wmma::mma_sync(low,a_lo,b_hi,low);
            }
        }
        #pragma unroll
        for(int i=0;i<total.num_elements;++i)
            total.x[i]+=high.x[i]+low.x[i]*(1.0f/4096.0f);
        __syncthreads();
    }
    wmma::store_matrix_sync(&result[wm][wn],total,BN,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<BM*BN;i+=256) {
        const int m=i/BN,n=i%BN;
        if(row0+m<rows && out0+n<outputs)
            output[int64_t(row0+m)*outputs+out0+n]=result[m][n];
    }
#endif
}
} // namespace
namespace {
// Four adjacent columns share subgroup and neuron metadata. Decode directly
// into a transient shared tile while retaining the original affine rounding.

}
namespace {
bool gr_prefill_pipeline_supported(const mfq_tensor_backend::Tensor& input) {
    thread_local std::map<int,std::pair<int,int>> capabilities;
    auto found=capabilities.find(input.get_device());
    if(found==capabilities.end()) {
        int major=0,shared=0;
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
        MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,input.get_device()));
        found=capabilities.emplace(input.get_device(),std::pair<int,int>{major,shared}).first;
    }
    const int bytes=input.scalar_type()==mfq_tensor_backend::kFloat16?50176:82944;
    return found->second.first>=8 && found->second.second>=bytes;
}
template<class Input,int BM,int BN,int BK,bool Packed=false> union GrPrefillPipelineShared {
    struct {Input a[2][BM][BK+(std::is_same<Input,float>::value?4:8)];float w[2][BN][BK+(Packed?4:1)];} tile;
    float products[BM][BN];
};
template<class Input,class Value,bool AlignedQ8,bool Zero,bool Mix,int RN,int RM,int BK=32,int BM=256,int TileN=32,bool Packed=false>
__global__ __launch_bounds__(BM*TileN/(RN*RM),1) void gr_prefill_pipeline_kernel(
        const Input* input,const uint8_t* packed,const uint8_t* row_bits,
        const int64_t* row_offsets,const uint8_t* scales,const uint8_t* minima,
        const float* row_scales,const float* row_minima,const __half* zero_scales,
        const Value* normalized,std::conditional_t<!Mix || std::is_same<Input,float>::value || std::is_same_v<Value,float>,float,__half>* output,int rows,int width,int outputs,int groups,int gs) {
    using Output=std::conditional_t<!Mix || std::is_same<Input,float>::value || std::is_same_v<Value,float>,float,__half>;
    constexpr int BN=TileN,Threads=BM*BN/(RN*RM),Columns=BN/RN,HiddenN=BN/4;
    extern __shared__ __align__(16) unsigned char bytes[];
    auto& shared=*reinterpret_cast<GrPrefillPipelineShared<Input,BM,BN,BK,Packed>*>(bytes);
    const auto weight_row=[](int r) {return Packed?r^(r>>3):r;};
    const int tid=threadIdx.x,m=tid/Columns,n=(tid%Columns)*RN;
    const int row0=blockIdx.y*BM,col0=blockIdx.x*(Mix?HiddenN:BN),hidden=outputs/4;
    float acc[RM][RN][4]{};
    const auto load=[&](int buffer,int k0) {
        constexpr int Vector=16/sizeof(Input);
        for(int i=tid;i<BM*(BK/Vector);i+=Threads) {
            const int r=i/(BK/Vector),c=(i%(BK/Vector))*Vector;
            auto* destination=&shared.tile.a[buffer][r][c];
            if(row0+r<rows && k0+c+Vector-1<width && width%Vector==0) {
                mfq::cuda_detail::copy_16_async(reinterpret_cast<__half*>(destination),
                    reinterpret_cast<const __half*>(input+int64_t(row0+r)*width+k0+c));
            } else {
                #pragma unroll
                for(int j=0;j<Vector;++j)destination[j]=row0+r<rows && k0+c+j<width?
                    input[int64_t(row0+r)*width+k0+c+j]:Input(0.f);
            }
        }
        mfq::cuda_detail::copy_async_commit();
        for(int i=tid;i<BN*(BK/4);i+=Threads) {
            const int r=i/(BK/4),c=(i%(BK/4))*4;
            const int hcol=col0+r%HiddenN;
            const int neuron=Mix?(r/HiddenN)*hidden+hcol:col0+r;
            float values[4]{};
            if((Mix?hcol<hidden:neuron<outputs) && k0+c<width)
                gr_prefill_decode_four<AlignedQ8,Zero>(values,packed,row_bits,row_offsets,
                    scales,minima,row_scales,row_minima,zero_scales,neuron,k0+c,width,groups,gs);
            #pragma unroll
            for(int j=0;j<4;++j)shared.tile.w[buffer][weight_row(r)][c+j]=values[j];
        }
    };
    load(0,0);mfq::cuda_detail::copy_async_wait();__syncthreads();
    for(int k0=0;k0<width;k0+=BK) {
        const int buffer=(k0/BK)&1;
        if(k0+BK<width)load(buffer^1,k0+BK);
        const auto accumulate=[&](int k,auto slot) {
            #pragma unroll
            for(int r=0;r<RM;++r) {
                const float x=float(shared.tile.a[buffer][m+r*(BM/RM)][k]);
                #pragma unroll
                for(int j=0;j<RN;++j)
                    acc[r][j][decltype(slot)::value]=fmaf(x,shared.tile.w[buffer][n+j][k],acc[r][j][decltype(slot)::value]);
            }
        };
        #pragma unroll
        for(int k=0;k<BK;k+=4) {
            if constexpr(Packed) {
                // One aligned shared load supplies all four canonical chains.
                // Permute weight rows so the vector loads use separate banks.
                #pragma unroll
                for(int j=0;j<RN;++j) {
                    const float4 w=*reinterpret_cast<const float4*>(
                        &shared.tile.w[buffer][weight_row(n+j)][k]);
                    #pragma unroll
                    for(int r=0;r<RM;++r) {
                        const float x0=float(shared.tile.a[buffer][m+r*(BM/RM)][k]);
                        const float x1=float(shared.tile.a[buffer][m+r*(BM/RM)][k+1]);
                        const float x2=float(shared.tile.a[buffer][m+r*(BM/RM)][k+2]);
                        const float x3=float(shared.tile.a[buffer][m+r*(BM/RM)][k+3]);
                        acc[r][j][0]=fmaf(x0,w.x,acc[r][j][0]);
                        acc[r][j][1]=fmaf(x1,w.y,acc[r][j][1]);
                        acc[r][j][2]=fmaf(x2,w.z,acc[r][j][2]);
                        acc[r][j][3]=fmaf(x3,w.w,acc[r][j][3]);
                    }
                }
            } else {
                accumulate(k,std::integral_constant<int,0>{});
                accumulate(k+1,std::integral_constant<int,1>{});
                accumulate(k+2,std::integral_constant<int,2>{});
                accumulate(k+3,std::integral_constant<int,3>{});
            }
        }
        mfq::cuda_detail::copy_async_wait();__syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RM;++r) {
        const int row=row0+m+r*(BM/RM);
        #pragma unroll
        for(int j=0;j<RN;++j) {
            const float projection=(acc[r][j][0]+acc[r][j][1])+(acc[r][j][2]+acc[r][j][3]);
            if constexpr(Mix) {
                const int hcol=col0+(n+j)%HiddenN,neuron=((n+j)/HiddenN)*hidden+hcol;
                float product=0;
                if(row<rows && hcol<hidden) {
                    const Input projected=static_cast<Input>(projection);
                    const Input gate=static_cast<Input>(1.0f/(1.0f+expf(-float(projected))));
                    product=float(static_cast<Output>(__fmul_rn(float(gate),float(normalized[int64_t(row)*hidden*4+neuron]))));
                }
                shared.products[m+r*(BM/RM)][n+j]=product;
            } else if(row<rows && col0+n+j<outputs)
                output[int64_t(row)*outputs+col0+n+j]=projection;
        }
    }
    if constexpr(Mix) {
        __syncthreads();
        for(int i=tid;i<BM*HiddenN;i+=Threads) {
            const int r=i/HiddenN,c=i%HiddenN;
            if(row0+r<rows && col0+c<hidden) {
                float sum=0;
                #pragma unroll
                for(int stream=0;stream<4;++stream)sum=__fadd_rn(sum,shared.products[r][stream*HiddenN+c]);
                output[int64_t(row0+r)*hidden+col0+c]=static_cast<Output>(__fdiv_rn(sum,4.0f));
            }
        }
    }
}
template<bool Mix,class Input,class Value,bool Aligned,bool Zero>
void launch_gr_prefill_pipeline(const NintWeight& weight,const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& normalized,mfq_tensor_backend::Tensor& output) {
    namespace tb=mfq_tensor_backend;
    using Output=std::conditional_t<!Mix || std::is_same<Input,float>::value || std::is_same_v<Value,float>,float,__half>;
    const auto rows=input.numel()/weight.neuron_len;
    const auto launch=[&](auto rn_tag,auto rm_tag,auto bm_tag,auto bn_tag,auto bk_tag,auto packed_tag) {
        constexpr int rn=decltype(rn_tag)::value,rm=decltype(rm_tag)::value;
        constexpr int bm=decltype(bm_tag)::value,bn=decltype(bn_tag)::value,bk=decltype(bk_tag)::value;
        constexpr bool packed=decltype(packed_tag)::value;
        constexpr int shared_bytes=sizeof(GrPrefillPipelineShared<Input,bm,bn,bk,packed>);
        const auto kernel=gr_prefill_pipeline_kernel<Input,Value,Aligned,Zero,Mix,rn,rm,bk,bm,bn,packed>;
        thread_local int configured_device=-1;
        if(configured_device!=input.get_device()) {
            MFQ_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
            configured_device=input.get_device();
        }
        const int64_t outputs=Mix?weight.out/4:weight.out;
        constexpr int columns=Mix?bn/4:bn;
        kernel<<<dim3(unsigned((outputs+columns-1)/columns),unsigned((rows+bm-1)/bm)),bm*bn/(rn*rm),shared_bytes,mfq_current_cuda_stream()>>>(
            input.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            Zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),Zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            Zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),Zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            Zero?nullptr:weight.neuron_scale.data_ptr<float>(),Zero?nullptr:weight.neuron_min.data_ptr<float>(),
            Zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,Mix?normalized.data_ptr<Value>():nullptr,
            output.data_ptr<Output>(),int(rows),int(weight.neuron_len),int(weight.out),int(weight.ng),int(weight.gs));
    };
    const auto* option=mfq::cuda::env_lookup(Mix?"MFQ_GR_PREFILL_FUSED_MIX":"MFQ_GR_PREFILL_MATMUL");
    int mode=option?std::atoi(option):(Mix?(std::is_same<Input,__half>::value?12:10):31);
    if constexpr(!Mix)if(mode==32)mode=31;
    if constexpr(!Mix) {
        if((mode==29 || mode==31) && weight.out<=4) {
            launch(std::integral_constant<int,1>{},std::integral_constant<int,2>{},
                std::integral_constant<int,64>{},std::integral_constant<int,4>{},
                std::integral_constant<int,32>{},std::false_type{});return;
        }
    }
        const auto wide=[&](auto rn,auto rm) {
            const auto invoke=[&](auto packed) {launch(rn,rm,
            std::integral_constant<int,256>{},std::integral_constant<int,32>{},
            std::integral_constant<int,32>{},packed);};
            if(mode==(Mix?12:31))invoke(std::true_type{});else invoke(std::false_type{});
        };
        if constexpr(Mix)wide(std::integral_constant<int,4>{},std::integral_constant<int,4>{});
        else if(weight.out>=weight.neuron_len)wide(std::integral_constant<int,4>{},std::integral_constant<int,4>{});
        else wide(std::integral_constant<int,8>{},std::integral_constant<int,2>{});
}
}
// FP32 batched GR projection from compressed NINT weights. The only decoded
// weights are a small shared-memory tile reused by sixty-four token rows.
namespace {
bool gr_prefill_selected(const mfq_tensor_backend::Tensor& input) {
    const auto* option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_MATMUL");
    return (!option || std::atoi(option)!=0) && input.dim()>0 && input.size(-1)>0 &&
        input.numel()/input.size(-1)>=32;
}

template<class Input,bool AlignedQ8,bool Zero,int BM=64,int RegisterN=4,int BK=32,int RegisterM=2,bool CompactK=false,bool VectorDecode=false,bool TransposeWeight=false,int TileN=32>
__global__ __launch_bounds__(BM*TileN/(RegisterN*RegisterM),1) void gr_prefill_matmul_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,float* output,
        int rows,int width,int outputs,int groups,int gs) {
    constexpr int BN=TileN,Pad=BK+1,Threads=BM*BN/(RegisterN*RegisterM),Columns=BN/RegisterN;
    static_assert(BK%4==0,"GR K tile must preserve the four FP32 accumulator streams");
    __shared__ float a[BM][Pad];
    union WeightTile {float row[BN][Pad];float col[BK][BN];};
    __shared__ __align__(16) WeightTile w;
    const int tid=threadIdx.x,m=tid/Columns,n=(tid%Columns)*RegisterN;
    const int row0=int(blockIdx.y)*BM,out0=int(blockIdx.x)*BN;
    float acc[RegisterM][RegisterN][4]{};
    for(int k0=0;k0<width;k0+=BK) {
        for(int i=tid;i<BM*BK;i+=Threads) {
            const int r=i/BK,c=i%BK;
            a[r][c]=row0+r<rows && k0+c<width ? float(input[(int64_t(row0+r)*width)+k0+c]) : 0;
        }
        if constexpr(VectorDecode) {
            for(int i=tid;i<BN*(BK/4);i+=Threads) {
                const int r=i/(BK/4),c=(i%(BK/4))*4,neuron=out0+r;
                float values[4]{};
                if(neuron<outputs && k0+c<width)gr_prefill_decode_four<AlignedQ8,Zero>(
                    values,packed,row_bits,row_offsets,scales,minima,row_scales,row_minima,zero_scales,
                    neuron,k0+c,width,groups,gs);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    if constexpr(TransposeWeight)w.col[c+j][r]=values[j];
                    else w.row[TileN>32?r^(r>>5):r][c+j]=values[j];
                }
            }
        } else for(int i=tid;i<BN*BK;i+=Threads) {
            const int r=i/BK,c=i%BK,neuron=out0+r,column=k0+c;
            float value=0;
            if(neuron<outputs && column<width) {
                const int64_t meta=int64_t(neuron)*groups+column/gs;
                float q,scale,minimum;
                if constexpr(Zero) {
                    q=float(static_cast<int8_t>(packed[meta*gs+column%gs]));
                    scale=__half2float(zero_scales[meta]);minimum=0;
                } else {
                    const int bits=AlignedQ8?8:row_bits[neuron];
                    const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
                    unsigned word=packed[bit>>3];const int shift=int(bit&7);
                    if(shift+bits>8)word|=unsigned(packed[(bit>>3)+1])<<8;
                    q=float((word>>shift)&((1u<<bits)-1));
                    scale=row_scales[neuron]*float(scales[meta]);
                    minimum=row_minima[neuron]*float(minima[meta]);
                }
                value=__fsub_rn(__fmul_rn(scale,q),minimum);
            }
            w.row[TileN>32?r^(r>>5):r][c]=value;
        }
        __syncthreads();
        if constexpr(TransposeWeight) {
            #pragma unroll
            for(int k=0;k<BK;++k) {
                float weight[RegisterN];
                #pragma unroll
                for(int j=0;j<RegisterN;j+=4) {
                    const float4 value=*reinterpret_cast<const float4*>(&w.col[k][n+j]);
                    weight[j]=value.x;weight[j+1]=value.y;weight[j+2]=value.z;weight[j+3]=value.w;
                }
                #pragma unroll
                for(int r=0;r<RegisterM;++r) {
                    const float x=a[m+r*(BM/RegisterM)][k];
                    #pragma unroll
                    for(int j=0;j<RegisterN;++j)acc[r][j][k&3]=fmaf(x,weight[j],acc[r][j][k&3]);
                }
            }
        } else if constexpr(CompactK) {
        // Explicit constant accumulator slots keep the arrays in registers;
        // only four K positions are live rather than the entire shared tile.
        #pragma unroll 1
        for(int kb=0;kb<BK;kb+=4) {
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                #pragma unroll
                for(int r=0;r<RegisterM;++r) {
                    const float x=a[m+r*(BM/RegisterM)][kb+slot];
                    #pragma unroll
                    for(int j=0;j<RegisterN;++j)
                        acc[r][j][slot]=fmaf(x,w.row[TileN>32?(n+j)^((n+j)>>5):n+j][kb+slot],acc[r][j][slot]);
                }
            }
        }
        } else {
        #pragma unroll
        for(int k=0;k<BK;++k) {
            #pragma unroll
            for(int r=0;r<RegisterM;++r) {
                const float x=a[m+r*(BM/RegisterM)][k];
                #pragma unroll
                for(int j=0;j<RegisterN;++j)acc[r][j][k&3]=fmaf(x,w.row[TileN>32?(n+j)^((n+j)>>5):n+j][k],acc[r][j][k&3]);
            }
        }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RegisterM;++r)if(row0+m+r*(BM/RegisterM)<rows) {
        #pragma unroll
        for(int j=0;j<RegisterN;++j)if(out0+n+j<outputs)
            output[int64_t(row0+m+r*(BM/RegisterM))*outputs+out0+n+j]=
                (acc[r][j][0]+acc[r][j][1])+(acc[r][j][2]+acc[r][j][3]);
    }
}

mfq_tensor_backend::Tensor gr_prefill_matmul(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input) {
    namespace tb=mfq_tensor_backend;
    auto x=input.scalar_type()==tb::kFloat16 ? input.contiguous() : input.to(tb::kFloat32).contiguous();
    MFQ_RUNTIME_CHECK(input.is_cuda() && input.size(-1)==weight.neuron_len && weight.q_packed.is_cuda() &&
        input.device()==weight.q_packed.device() && weight.gs>0 && weight.neuron_len>0 &&
        weight.neuron_len<=std::numeric_limits<int>::max() && weight.out>0 && weight.out<=std::numeric_limits<int>::max() &&
        weight.ng==(weight.neuron_len+weight.gs-1)/weight.gs,
        "GR prefill projection geometry mismatch");
    const auto rows=x.numel()/weight.neuron_len;
    if(gr_prefill_blas_selected(weight,input))return gr_prefill_blas_fp32(weight,x);
    const auto* trace_geometry=mfq::cuda::env_lookup("MFQ_GR_PREFILL_GEOMETRY_TRACE");
    if(trace_geometry && trace_geometry[0]=='1') {
        thread_local std::map<std::array<int64_t,5>,bool> seen;
        const std::array<int64_t,5> shape={weight.out,weight.neuron_len,weight.gs,int64_t(x.scalar_type()==tb::kFloat16),int64_t(weight.q8_zero)};
        if(seen.emplace(shape,true).second)std::fprintf(stderr,
            "gr_prefill_geometry rows=%lld outputs=%lld width=%lld gs=%lld half=%d zero=%d\n",
            (long long)rows,(long long)weight.out,(long long)weight.neuron_len,(long long)weight.gs,
            int(x.scalar_type()==tb::kFloat16),int(weight.q8_zero));
    }
    MFQ_RUNTIME_CHECK(rows<=65535*16,"GR prefill projection exceeds CUDA grid geometry");
    MfqCudaGuard guard(input.device());
    const auto* option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_MATMUL");
    int major=0;
    const bool requested_tf32=option && std::atoi(option)==2;
    const bool requested_half2=option && std::atoi(option)==3;
    int mode=option?std::atoi(option):31;
    if(mode==32)mode=31;
    const bool wide_contraction=mode==24 && weight.out<weight.neuron_len && rows>=128;
    const bool transpose_weight=mode==22 || mode==23;
    const bool fewer_registers=mode==23;
    const bool pipeline_mode=mode==27 || mode==29 || mode==31;
    const bool vector_decode=mode==21 || mode==24 || pipeline_mode || transpose_weight;
    if(vector_decode)mode=17;
    const bool compact_k=mode==19 || mode==20;
    if(compact_k)mode=mode==19?6:17;
    if(mode==16 || mode==17 || mode==18) {
        const bool expansion=weight.out>=weight.neuron_len;
        mode=expansion?(mode>=17?(x.scalar_type()==tb::kFloat32?14:(mode==18?15:9)):(x.scalar_type()==tb::kFloat32?5:9))
            :(rows>=2048?9:rows>=128?(mode==17?11:13):1);
    }
    if(mode==12)mode=weight.out>=weight.neuron_len?(x.scalar_type()==tb::kFloat32?5:9):(rows>=2048?9:rows>=128?11:1);
    if(mode==6)mode=weight.out>=weight.neuron_len?(x.scalar_type()==tb::kFloat32?5:9):(rows>=2048?9:1);
    const int exact_tile=mode==4?128:mode==5?256:64;
    if(requested_tf32 || requested_half2)MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
    const bool tf32=requested_tf32 && major>=8;
    const bool half2=requested_half2 && major>=7;
    auto shape=input.sizes().vec();shape.back()=weight.out;
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    const auto typed=[&](auto tag,auto aligned_tag,auto zero_tag) {
        using Input=decltype(tag);constexpr bool aligned=decltype(aligned_tag)::value,zero=decltype(zero_tag)::value;
        if(pipeline_mode && rows>=128 && gr_prefill_pipeline_supported(x)) {
            launch_gr_prefill_pipeline<false,Input,float,aligned,zero>(weight,x,{},output);return;
        }
        const auto launch_kernel=[&](auto kernel,int tile_m,int threads,int tile_n=32) {
        kernel<<<dim3(unsigned((weight.out+tile_n-1)/tile_n),unsigned((rows+tile_m-1)/tile_m)),threads,0,mfq_current_cuda_stream()>>>(
            x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            zero?nullptr:weight.neuron_scale.data_ptr<float>(),zero?nullptr:weight.neuron_min.data_ptr<float>(),
            zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,output.data_ptr<float>(),int(rows),
            int(weight.neuron_len),int(weight.out),int(weight.ng),int(weight.gs));
        };
        if(transpose_weight) {
            if(fewer_registers)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,32,2,false,true,true>,128,512);
            else if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4,false,true,true>,256,512);
            else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,2,false,true,true>,128,512);
            else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,2,false,true,true>,256,512);
            else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,32,2,false,true,true>,64,256);
            return;
        }
        if(vector_decode) {
            if(wide_contraction)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,8,32,2,false,true,false,64>,128,512,64);
            else if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4,false,true>,256,512);
            else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,2,false,true>,128,512);
            else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,2,false,true>,256,512);
            else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,32,2,false,true>,64,256);
            return;
        }
        if(compact_k) {
            if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4,true>,256,512);
            else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,2,true>,128,512);
            else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,2,true>,256,512);
            else if(mode==5)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,2,true>,256,1024);
            else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,32,2,true>,64,256);
            return;
        }
        if(half2)launch_kernel(gr_prefill_half2_kernel<Input,aligned,zero>,64,256);
        else if(tf32)launch_kernel(gr_prefill_tf32_kernel<Input,aligned,zero>,32,128);
        else if(mode==10)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,4,64>,64,256);
        else if(mode==11)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64>,128,512);
        else if(mode==13)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,4,64,4>,128,256);
        else if(mode==14)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,4,32,4>,256,512);
        else if(mode==15)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8,32,4>,256,256);
        else if(mode==9)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256,8>,256,512);
        else if(mode==8)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128,8>,128,256);
        else if(mode==7)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,64,8>,64,128);
        else if(exact_tile==256)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,256>,256,1024);
        else if(exact_tile==128)launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero,128>,128,512);
        else launch_kernel(gr_prefill_matmul_kernel<Input,aligned,zero>,64,256);
    };
    const auto launch=[&](auto tag) {
        if(weight.q8_zero)typed(tag,std::false_type{},std::true_type{});
        else if(weight.aligned_q8)typed(tag,std::true_type{},std::false_type{});
        else typed(tag,std::false_type{},std::false_type{});
    };
    if(x.scalar_type()==tb::kFloat16)launch(__half{});else launch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}
} // namespace
// Four stream projections share the activation tile. Products overwrite the
// decoded tile after the last FMA, then reduce in canonical stream order.
namespace {
template<int BM,int BN,int BK> union GrPrefillMixShared {
    struct {float a[BM][BK+1];union {float w[BN][BK+1];float wt[BK][BN];};} tile;
    float products[BM][BN];
};
template<class Input,class Value,bool AlignedQ8,bool Zero,bool Swizzle=false,int BM=128,int HiddenN=16,int BK=32,bool CompactK=false,bool VectorDecode=false,bool TransposeWeight=false,int RegisterM=4>
__global__ __launch_bounds__(512,1) void gr_prefill_mix_kernel(const Input* input,
        const uint8_t* packed,const uint8_t* row_bits,const int64_t* row_offsets,
        const uint8_t* scales,const uint8_t* minima,const float* row_scales,
        const float* row_minima,const __half* zero_scales,const Value* normalized,
        std::conditional_t<std::is_same<Input,float>::value || std::is_same_v<Value,float>,float,__half>* output,
        int rows,int width,int hidden,int groups,int gs) {
    using Output=std::conditional_t<std::is_same<Input,float>::value || std::is_same_v<Value,float>,float,__half>;
    constexpr int BN=HiddenN*4,RN=4,RM=RegisterM,Threads=512,Columns=BN/RN;
    static_assert(BM*BN/(RN*RM)==Threads,"GR fused tile and register geometry mismatch");
    using Shared=GrPrefillMixShared<BM,BN,BK>;
    extern __shared__ __align__(16) unsigned char gr_mix_dynamic[];
    Shared* storage;
    if constexpr(BK>32)storage=reinterpret_cast<Shared*>(gr_mix_dynamic);
    else {__shared__ __align__(16) Shared fixed;storage=&fixed;}
    auto& shared=*storage;
    const int tid=threadIdx.x,m=tid/Columns,n=(tid%Columns)*RN;
    const int row0=int(blockIdx.y)*BM,col0=int(blockIdx.x)*HiddenN;
    float acc[RM][RN][4]{};
    for(int k0=0;k0<width;k0+=BK) {
        for(int i=tid;i<BM*BK;i+=Threads) {
            const int r=i/BK,c=i%BK;
            shared.tile.a[r][c]=row0+r<rows && k0+c<width?float(input[int64_t(row0+r)*width+k0+c]):0;
        }
        if constexpr(VectorDecode) {
            for(int i=tid;i<BN*(BK/4);i+=Threads) {
                const int r=i/(BK/4),c=(i%(BK/4))*4,hcol=col0+r%HiddenN;
                const int neuron=(r/HiddenN)*hidden+hcol;
                float values[4]{};
                if(hcol<hidden && k0+c<width)gr_prefill_decode_four<AlignedQ8,Zero>(
                    values,packed,row_bits,row_offsets,scales,minima,row_scales,row_minima,zero_scales,
                    neuron,k0+c,width,groups,gs);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    if constexpr(TransposeWeight)shared.tile.wt[c+j][r]=values[j];
                    else shared.tile.w[r][c+j]=values[j];
                }
            }
        } else for(int i=tid;i<BN*BK;i+=Threads) {
            const int r=i/BK,c=i%BK,column=k0+c,hcol=col0+r%HiddenN;
            const int neuron=(r/HiddenN)*hidden+hcol;
            float value=0;
            if(hcol<hidden && column<width) {
                const int64_t meta=int64_t(neuron)*groups+column/gs;
                float q,scale,minimum;
                if constexpr(Zero) {
                    q=float(static_cast<int8_t>(packed[meta*gs+column%gs]));
                    scale=__half2float(zero_scales[meta]);minimum=0;
                } else {
                    const int bits=AlignedQ8?8:row_bits[neuron];
                    const uint64_t bit=uint64_t(row_offsets[neuron])+uint64_t(column)*bits;
                    unsigned word=packed[bit>>3];const int shift=int(bit&7);
                    if(shift+bits>8)word|=unsigned(packed[(bit>>3)+1])<<8;
                    q=float((word>>shift)&((1u<<bits)-1));
                    scale=row_scales[neuron]*float(scales[meta]);
                    minimum=row_minima[neuron]*float(minima[meta]);
                }
                value=__fsub_rn(__fmul_rn(scale,q),minimum);
            }
            const int stored_row=Swizzle?r^(r>>5):r;
            shared.tile.w[stored_row][c]=value;
        }
        __syncthreads();
        const auto accumulate=[&](int k,auto slot) {
            #pragma unroll
            for(int r=0;r<RM;++r) {
                const float x=shared.tile.a[m+r*(BM/RM)][k];
                #pragma unroll
                for(int j=0;j<RN;++j) {
                    const int stored_row=Swizzle?(n+j)^((n+j)>>5):n+j;
                    acc[r][j][decltype(slot)::value]=fmaf(x,shared.tile.w[stored_row][k],acc[r][j][decltype(slot)::value]);
                }
            }
        };
        if constexpr(TransposeWeight) {
            #pragma unroll
            for(int k=0;k<BK;++k) {
                const float4 vector=*reinterpret_cast<const float4*>(&shared.tile.wt[k][n]);
                const float weight[4]={vector.x,vector.y,vector.z,vector.w};
                #pragma unroll
                for(int r=0;r<RM;++r) {
                    const float x=shared.tile.a[m+r*(BM/RM)][k];
                    #pragma unroll
                    for(int j=0;j<RN;++j)acc[r][j][k&3]=fmaf(x,weight[j],acc[r][j][k&3]);
                }
            }
        } else if constexpr(CompactK) {
            #pragma unroll 1
            for(int k=0;k<BK;k+=4) {
                accumulate(k,std::integral_constant<int,0>{});
                accumulate(k+1,std::integral_constant<int,1>{});
                accumulate(k+2,std::integral_constant<int,2>{});
                accumulate(k+3,std::integral_constant<int,3>{});
            }
        } else {
            #pragma unroll
            for(int k=0;k<BK;++k) {
                #pragma unroll
                for(int r=0;r<RM;++r) {
                    const float x=shared.tile.a[m+r*(BM/RM)][k];
                    #pragma unroll
                    for(int j=0;j<RN;++j) {
                        const int stored_row=Swizzle?(n+j)^((n+j)>>5):n+j;
                        acc[r][j][k&3]=fmaf(x,shared.tile.w[stored_row][k],acc[r][j][k&3]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<RM;++r) {
        const int row=row0+m+r*(BM/RM);
        #pragma unroll
        for(int j=0;j<RN;++j) {
            const int hcol=col0+(n+j)%HiddenN,neuron=((n+j)/HiddenN)*hidden+hcol;
            float product=0;
            if(row<rows && hcol<hidden) {
                const Input projected=static_cast<Input>((acc[r][j][0]+acc[r][j][1])+(acc[r][j][2]+acc[r][j][3]));
                const Input gate=static_cast<Input>(1.0f/(1.0f+expf(-static_cast<float>(projected))));
                product=float(static_cast<Output>(__fmul_rn(float(gate),float(normalized[int64_t(row)*hidden*4+neuron]))));
            }
            shared.products[m+r*(BM/RM)][n+j]=product;
        }
    }
    __syncthreads();
    for(int i=tid;i<BM*HiddenN;i+=Threads) {
        const int r=i/HiddenN,c=i%HiddenN;
        if(row0+r<rows && col0+c<hidden) {
            float sum=0;
            #pragma unroll
            for(int s=0;s<4;++s)sum=__fadd_rn(sum,shared.products[r][s*HiddenN+c]);
            output[int64_t(row0+r)*hidden+col0+c]=static_cast<Output>(__fdiv_rn(sum,4.0f));
        }
    }
}

bool gr_prefill_mix_selected(const NintWeight& weight,const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& normalized,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    const auto* option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_FUSED_MIX");
    const auto* projection=mfq::cuda::env_lookup("MFQ_GR_PREFILL_MATMUL");
    if(projection && (std::atoi(projection)==2 || std::atoi(projection)==3))return false;
    if(!option && (weight.out<weight.neuron_len || input.numel()/input.size(-1)<128))return false;
    const auto supported=[](const tb::Tensor& t){return t.scalar_type()==tb::kFloat16 || t.scalar_type()==tb::kFloat32;};
    return (!option || std::atoi(option)!=0) && streams==4 && input.is_contiguous() && normalized.is_contiguous()
        && supported(input) && supported(normalized);
}

mfq_tensor_backend::Tensor gr_prefill_fused_mix(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized) {
    namespace tb=mfq_tensor_backend;
    MfqCudaGuard guard(input.device());
    const auto rows=input.numel()/weight.neuron_len,hidden=weight.out/4;
    MFQ_RUNTIME_CHECK(rows<=65535*int64_t(128) && hidden<=std::numeric_limits<int>::max() &&
        weight.neuron_len<=std::numeric_limits<int>::max() && weight.ng<=std::numeric_limits<int>::max(),
        "fused GR prefill exceeds CUDA launch geometry");
    auto shape=input.sizes().vec();shape.back()=hidden;
    const auto dtype=input.scalar_type()==tb::kFloat32 || normalized.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
    auto output=tb::empty(shape,input.options().dtype(dtype));
    const auto typed=[&](auto input_tag,auto value_tag,auto aligned_tag,auto zero_tag,auto swizzle_tag) {
        using Input=decltype(input_tag);using Value=decltype(value_tag);
        using Output=std::conditional_t<std::is_same<Input,float>::value || std::is_same_v<Value,float>,float,__half>;
        constexpr bool aligned=decltype(aligned_tag)::value,zero=decltype(zero_tag)::value,swizzle=decltype(swizzle_tag)::value;
        {
            const auto* pipeline_option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_FUSED_MIX");
            if((!pipeline_option || std::atoi(pipeline_option)==10 || std::atoi(pipeline_option)==12) && gr_prefill_pipeline_supported(input)) {
                launch_gr_prefill_pipeline<true,Input,Value,aligned,zero>(weight,input,normalized,output);return;
            }
        }
        const auto invoke=[&](auto tile_tag,auto k_tag,auto compact_tag,auto vector_tag,auto transpose_tag,auto register_tag) {
        constexpr int tile=decltype(tile_tag)::value,register_m=decltype(register_tag)::value;
        constexpr int columns=register_m==2 || tile==256?8:16,k=decltype(k_tag)::value;
        constexpr bool compact=decltype(compact_tag)::value;
        constexpr bool vector_decode=decltype(vector_tag)::value;
        constexpr bool transpose_weight=decltype(transpose_tag)::value;
        constexpr size_t shared_bytes=k>32?sizeof(GrPrefillMixShared<tile,columns*4,k>):0;
        if constexpr(shared_bytes>48*1024) {
            thread_local int configured_device=-1;
            if(configured_device!=input.device().index) {
                MFQ_CUDA_CHECK(cudaFuncSetAttribute(gr_prefill_mix_kernel<Input,Value,aligned,zero,swizzle,tile,columns,k,compact,vector_decode,transpose_weight,register_m>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared_bytes)));
                configured_device=input.device().index;
            }
        }
        gr_prefill_mix_kernel<Input,Value,aligned,zero,swizzle,tile,columns,k,compact,vector_decode,transpose_weight,register_m><<<dim3(unsigned((hidden+columns-1)/columns),unsigned((rows+tile-1)/tile)),512,shared_bytes,mfq_current_cuda_stream()>>>(
            input.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            zero?nullptr:weight.neuron_scale.data_ptr<float>(),zero?nullptr:weight.neuron_min.data_ptr<float>(),
            zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,normalized.data_ptr<Value>(),output.data_ptr<Output>(),
            int(rows),int(weight.neuron_len),int(hidden),int(weight.ng),int(weight.gs));
        };
        if constexpr(swizzle)invoke(std::integral_constant<int,128>{},std::integral_constant<int,32>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
        else {
            const auto* option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_FUSED_MIX");
            const int requested_mode=option?std::atoi(option):(std::is_same<Input,__half>::value?12:10);
            const int mode=requested_mode==10 || requested_mode==12?6:requested_mode;
            if(mode==8)invoke(std::integral_constant<int,128>{},std::integral_constant<int,32>{},std::false_type{},std::true_type{},std::true_type{},std::integral_constant<int,2>{});
            else if(mode==7)invoke(std::integral_constant<int,256>{},std::integral_constant<int,32>{},std::false_type{},std::true_type{},std::true_type{},std::integral_constant<int,4>{});
            else if(mode==6)invoke(std::integral_constant<int,256>{},std::integral_constant<int,32>{},std::false_type{},std::true_type{},std::false_type{},std::integral_constant<int,4>{});
            else if(mode==4)invoke(std::integral_constant<int,256>{},std::integral_constant<int,64>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
            else if(mode==5)invoke(std::integral_constant<int,256>{},std::integral_constant<int,64>{},std::true_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
            else if(mode==3)invoke(std::integral_constant<int,256>{},std::integral_constant<int,32>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
            else invoke(std::integral_constant<int,128>{},std::integral_constant<int,32>{},std::false_type{},std::false_type{},std::false_type{},std::integral_constant<int,4>{});
        }
    };
    const auto launch=[&](auto input_tag,auto value_tag) {
        const auto dispatch=[&](auto swizzle_tag) {
            if(weight.q8_zero)typed(input_tag,value_tag,std::false_type{},std::true_type{},swizzle_tag);
            else if(weight.aligned_q8)typed(input_tag,value_tag,std::true_type{},std::false_type{},swizzle_tag);
            else typed(input_tag,value_tag,std::false_type{},std::false_type{},swizzle_tag);
        };
        const auto* option=mfq::cuda::env_lookup("MFQ_GR_PREFILL_FUSED_MIX");
        if(option && std::atoi(option)==2)dispatch(std::true_type{});else dispatch(std::false_type{});
    };
    if(input.scalar_type()==tb::kFloat16) {
        if(normalized.scalar_type()==tb::kFloat16)launch(__half{},__half{});else launch(__half{},float{});
    } else {
        if(normalized.scalar_type()==tb::kFloat16)launch(float{},__half{});else launch(float{},float{});
    }
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}
}

#include "gr_prefill_blas.h"
#include "kernel_dispatch_trace.h"

namespace {
template<class Input>
__device__ __forceinline__ float4 gr_dense_input4(const Input* input) {
    if constexpr(std::is_same<Input,float>::value)return *reinterpret_cast<const float4*>(input);
    else {
        const uint2 x=*reinterpret_cast<const uint2*>(input);
        return make_float4(__half2float(__ushort_as_half(x.x)),__half2float(__ushort_as_half(x.x>>16)),
            __half2float(__ushort_as_half(x.y)),__half2float(__ushort_as_half(x.y>>16)));
    }
}
__device__ __forceinline__ float4 gr_dense_weight4(const uint16_t* weight) {
    const uint2 w=*reinterpret_cast<const uint2*>(weight);
    return make_float4(__uint_as_float(w.x<<16),__uint_as_float(w.x&0xffff0000u),
        __uint_as_float(w.y<<16),__uint_as_float(w.y&0xffff0000u));
}
__device__ __forceinline__ float gr_dense_dot4(float sum,float4 w,float4 x) {
    sum=fmaf(w.x,x.x,sum);sum=fmaf(w.y,x.y,sum);
    sum=fmaf(w.z,x.z,sum);return fmaf(w.w,x.w,sum);
}
template<int Rows,class Input>
__global__ void __launch_bounds__(128) gr_bf16_down_kernel(
        const Input* input,const uint16_t* weight,float* output) {
    constexpr int Width=10240,Rank=320;
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    float sums[Rows]{};
    for(int column=tid*4;column<Width;column+=128*4) {
        const float4 w=gr_dense_weight4(weight+size_t(row)*Width+column);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=gr_dense_dot4(sums[m],w,
            gr_dense_input4(input+size_t(m)*Width+column));
    }
    __shared__ float partial[Rows][4];
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],offset);
        if(lane==0)partial[m][warp]=sums[m];
    }
    __syncthreads();
    if(tid<Rows) {
        const float sum=(partial[tid][0]+partial[tid][2])+(partial[tid][1]+partial[tid][3]);
        const float low=__fdiv_rn(sum,4.f);
        const float gate=1.f/(1.f+expf(-low));
        output[size_t(tid)*Rank+row]=__fmul_rn(low,gate);
    }
}
template<int Rows,class Normalized>
__global__ void __launch_bounds__(128) gr_bf16_up_mix_kernel(
        const float* input,const Normalized* normalized,const uint16_t* weight,float* output) {
    constexpr int Rank=320,Hidden=2560,Streams=4;
    const int column=blockIdx.x,tid=threadIdx.x,lane=tid&31,stream=tid>>5;
    const int row=stream*Hidden+column;
    float sums[Rows]{};
    for(int begin=lane*4;begin<Rank;begin+=128) {
        const float4 w=gr_dense_weight4(weight+size_t(row)*Rank+begin);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=gr_dense_dot4(sums[m],w,
            gr_dense_input4(input+size_t(m)*Rank+begin));
    }
    __shared__ float products[Rows][Streams];
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],offset);
        if(lane==0) {
            const float gate=1.f/(1.f+expf(-sums[m]));
            products[m][stream]=__fmul_rn(gate,static_cast<float>(normalized[size_t(m)*Hidden*Streams+row]));
        }
    }
    __syncthreads();
    if(tid<Rows) {
        float sum=0.f;
        #pragma unroll
        for(int s=0;s<Streams;++s)sum=__fadd_rn(sum,products[tid][s]);
        output[size_t(tid)*Hidden+column]=sum/float(Streams);
    }
}

template<bool AlignedQ8=false,int FixedGroup=0,class Input>
__device__ __forceinline__ float affine_sum(
    const Input* input, const unsigned char* values,
    const unsigned char* row_bits, const int64_t* row_offsets,
    const unsigned char* scales, const unsigned char* minima,
    const float* row_scales, const float* row_minima, const __half* zero_scales,
    int64_t width, int64_t groups, int group_size, bool zero,
    int64_t neuron,int64_t sample,int64_t first,int64_t last,int lane) {
    const int bits = AlignedQ8 || zero ? 8 : row_bits[neuron];
    float sum = 0;
    for (int64_t group = first; group < last; ++group) {
        const int64_t meta = neuron * groups + group;
        const float scale = FixedGroup==0 && zero ? __half2float(zero_scales[meta])
            : row_scales[neuron] * float(scales[meta]);
        const float minimum = FixedGroup==0 && zero ? 0 : row_minima[neuron] * float(minima[meta]);
        if constexpr(FixedGroup>0) {
            const auto* packed=values+(row_offsets[neuron]>>3)+group*FixedGroup;
            const auto* activation=input+sample*width+group*FixedGroup;
            if((group+1)*FixedGroup<=width) {
#pragma unroll
                for(int j=lane;j<FixedGroup;j+=32) {
                    const float weight=__fsub_rn(__fmul_rn(scale,float(packed[j])),minimum);
                    sum=fmaf(weight,static_cast<float>(activation[j]),sum);
                }
            } else {
#pragma unroll
                for(int j=lane;j<FixedGroup;j+=32) {
                    if(group*FixedGroup+j>=width)continue;
                    const float weight=__fsub_rn(__fmul_rn(scale,float(packed[j])),minimum);
                    sum=fmaf(weight,static_cast<float>(activation[j]),sum);
                }
            }
        } else {
            for (int j = lane; j < group_size; j += 32) {
                const int64_t column = group * group_size + j;
                if (column >= width) continue;
                float q;
                if (zero) {
                    q = float(static_cast<signed char>(values[meta * group_size + j]));
                } else if constexpr(AlignedQ8) {
                    q=float(values[(row_offsets[neuron]>>3)+column]);
                } else {
                    const uint64_t bit = uint64_t(row_offsets[neuron]) + uint64_t(column) * bits;
                    const int shift = int(bit & 7);
                    unsigned packed = values[bit >> 3];
                    if (shift + bits > 8) packed |= unsigned(values[(bit >> 3) + 1]) << 8;
                    q = float((packed >> shift) & ((1u << bits) - 1));
                }
                // Separate affine rounding matches the canonical FP32 decoder.
                const float weight = __fsub_rn(__fmul_rn(scale, q), minimum);
                sum = fmaf(weight, static_cast<float>(input[sample * width + column]), sum);
            }
        }
    }
    for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, step);
    return sum;
}

template<class Output,int Activation>
__device__ __forceinline__ Output projection_activation(float sum,float streams) {
    const Output projected=static_cast<Output>(sum);
    const Output low=static_cast<Output>(__fdiv_rn(static_cast<float>(projected),streams));
    const Output gate=static_cast<Output>(1.0f/(1.0f+expf(-static_cast<float>(low))));
    const float multiplier=Activation==2?2.0f:static_cast<float>(low);
    return static_cast<Output>(__fmul_rn(multiplier,static_cast<float>(gate)));
}

template<class Input,bool AlignedQ8=false,int FixedGroup=0,class Output=float,int Activation=0>
__global__ void affine_projection(
    const Input* input,const unsigned char* values,
    const unsigned char* row_bits,const int64_t* row_offsets,
    const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const __half* zero_scales,
    Output* output,int64_t width,int64_t outputs,int64_t groups,int group_size,bool zero,
    int splits,float streams) {
    const int lane=threadIdx.x&31;
    const int64_t neuron=int64_t(blockIdx.x)*4+(threadIdx.x>>5);
    if(neuron>=outputs)return;
    const int64_t sample=blockIdx.y;
    const float sum=affine_sum<AlignedQ8,FixedGroup>(input,values,row_bits,row_offsets,scales,minima,row_scales,
        row_minima,zero_scales,width,groups,group_size,zero,neuron,sample,
        groups*blockIdx.z/splits,groups*(blockIdx.z+1)/splits,lane);
    if (!lane) {
        if constexpr(Activation==0)output[(sample * outputs + neuron)*splits+blockIdx.z]=sum;
        else output[sample*outputs+neuron]=projection_activation<Output,Activation>(sum,streams);
    }
}

template<class Gate,class Value,bool AlignedQ8,class Input,int FixedGroup=0>
__global__ void affine_projection_mix(const Input* input,const unsigned char* values,
    const unsigned char* row_bits,const int64_t* row_offsets,
    const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const __half* zero_scales,
    const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>* output,
    int64_t width,int64_t hidden,int64_t groups,int group_size,bool zero,int streams) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int stream=warp;
    const int64_t hidden_column=blockIdx.x;
    const int64_t neuron=int64_t(stream)*hidden+hidden_column,sample=blockIdx.y;
    __shared__ float products[32];
    if(hidden_column<hidden) {
        const float sum=affine_sum<AlignedQ8,FixedGroup>(input,values,row_bits,row_offsets,scales,minima,row_scales,
            row_minima,zero_scales,width,groups,group_size,zero,neuron,sample,0,groups,lane);
        if(!lane) {
            const Gate projection=static_cast<Gate>(sum);
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection))));
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),
                static_cast<float>(normalized[sample*hidden*streams+neuron])));
            products[warp]=static_cast<float>(product);
        }
    }
    __syncthreads();
    if(!lane && !stream && hidden_column<hidden) {
        float mixed=0;
        for(int s=0;s<streams;++s)mixed=__fadd_rn(mixed,products[s]);
        output[sample*hidden+hidden_column]=static_cast<Output>(__fdiv_rn(mixed,float(streams)));
    }
}
template<class Gate,class Value,class Input,int FixedGroup,int Streams>
__global__ void affine_projection_warp_mix(const Input* input,const unsigned char* values,
    const int64_t* row_offsets,const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>* output,
    int64_t width,int64_t hidden,int64_t groups) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
    const int lane=threadIdx.x&31;
    const int64_t column=int64_t(blockIdx.x)*Streams+(threadIdx.x>>5),sample=blockIdx.y;
    if(column>=hidden)return;
    float sums[Streams]{};
    const unsigned char* packed[Streams];
    float row_scale[Streams],row_minimum[Streams];
#pragma unroll
    for(int s=0;s<Streams;++s) {
        const auto neuron=int64_t(s)*hidden+column;
        packed[s]=values+(row_offsets[neuron]>>3);
        row_scale[s]=row_scales[neuron];
        row_minimum[s]=row_minima[neuron];
    }
    for(int64_t group=0;group<groups;++group) {
        float group_scale[Streams],group_minimum[Streams];
#pragma unroll
        for(int s=0;s<Streams;++s) {
            const auto meta=(int64_t(s)*hidden+column)*groups+group;
            group_scale[s]=__fmul_rn(row_scale[s],float(scales[meta]));
            group_minimum[s]=__fmul_rn(row_minimum[s],float(minima[meta]));
        }
#pragma unroll
        for(int j=lane;j<FixedGroup;j+=32) {
            if(group*FixedGroup+j>=width)continue;
            const float activation=static_cast<float>(input[sample*width+group*FixedGroup+j]);
#pragma unroll
            for(int s=0;s<Streams;++s) {
                const float q=float(packed[s][group*FixedGroup+j]);
                const float weight=__fsub_rn(__fmul_rn(group_scale[s],q),group_minimum[s]);
                sums[s]=fmaf(weight,activation,sums[s]);
            }
        }
    }
    for(int step=16;step;step>>=1) {
#pragma unroll
        for(int s=0;s<Streams;++s)sums[s]+=__shfl_down_sync(0xffffffffu,sums[s],step);
    }
    if(!lane) {
        float mixed=0;
#pragma unroll
        for(int s=0;s<Streams;++s) {
            const Gate projection=static_cast<Gate>(sums[s]);
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection))));
            const auto neuron=int64_t(s)*hidden+column;
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),
                static_cast<float>(normalized[sample*hidden*Streams+neuron])));
            mixed=__fadd_rn(mixed,static_cast<float>(product));
        }
        output[sample*hidden+column]=static_cast<Output>(__fdiv_rn(mixed,float(Streams)));
    }
}
__global__ void reduce_projection_splits(const float* partial,float* output,int64_t count,int splits) {
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=int64_t(gridDim.x)*blockDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[row*splits+split];
        output[row]=sum;
    }
}
template<class Output,int Activation>
__global__ void reduce_projection_activation(const float* partial,Output* output,int64_t count,int splits,float streams) {
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=int64_t(gridDim.x)*blockDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[row*splits+split];
        output[row]=projection_activation<Output,Activation>(sum,streams);
    }
}
int projection_resident_blocks(int device) {
    static std::mutex mutex;
    static std::map<int,int> devices;
    std::lock_guard lock(mutex);
    if(const auto found=devices.find(device);found!=devices.end())return found->second;
    cudaDeviceProp properties{};
    MFQ_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
    int active=0;
    MFQ_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,affine_projection<float,false>,128,0));
    const int blocks=active*properties.multiProcessorCount;
    devices.emplace(device,blocks);return blocks;
}
} // namespace

namespace {
mfq_tensor_backend::Tensor float_projection_impl(
        const NintWeight& weight, const mfq_tensor_backend::Tensor& input,bool parallel_groups,bool native_input,
        bool fixed_groups,int activation,int64_t streams) {
    namespace tb = mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.is_cuda() && weight.q_packed.is_cuda() &&
        input.device() == weight.q_packed.device() && input.dim() >= 1 &&
        input.size(-1) == weight.neuron_len && weight.neuron_len > 0 && weight.out > 0 &&
        weight.gs > 0 && weight.gs <= 64 &&
        weight.ng == (weight.neuron_len + weight.gs - 1) / weight.gs,
        "packed float projection dimensions/device disagree");
    MfqCudaGuard guard(input.device());
    auto x = native_input && input.scalar_type()==tb::kFloat16
        ? input.contiguous() : input.to(tb::kFloat32).contiguous();
    auto shape = x.sizes().vec();
    const auto rows = x.numel() / weight.neuron_len;
    MFQ_RUNTIME_CHECK(rows <= 65535 && (weight.out + 3) / 4 <= std::numeric_limits<int>::max(),
        "packed float projection exceeds CUDA launch geometry");
    shape.back() = weight.out;
    const auto output_dtype=activation && input.scalar_type()==tb::kFloat16?tb::kFloat16:tb::kFloat32;
    auto result = tb::empty(shape, x.options().dtype(output_dtype));
    if (!rows) return result;
    const int64_t blocks=((weight.out+3)/4)*rows;
    const int target=parallel_groups?projection_resident_blocks(x.get_device()):1;
    const int splits=int(std::min<int64_t>(weight.ng,std::max<int64_t>(1,(target+blocks-1)/blocks)));
    auto partial=splits>1?tb::empty({rows,weight.out,splits},result.options().dtype(tb::kFloat32)):result;
    const auto launch=[&](auto input_tag,auto aligned_tag,auto group_tag) {
        using Input=decltype(input_tag);constexpr bool aligned=decltype(aligned_tag)::value;
        constexpr int group=decltype(group_tag)::value;
        const auto invoke=[&](auto output_tag,auto activation_tag,const tb::Tensor& target) {
        using Output=decltype(output_tag);constexpr int action=decltype(activation_tag)::value;
        affine_projection<Input,aligned,group,Output,action><<<dim3(unsigned((weight.out + 3) / 4), unsigned(rows),unsigned(splits)), 128, 0,
        mfq_current_cuda_stream()>>>(x.data_ptr<Input>(), weight.q_packed.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bits.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bit_offsets.data_ptr<int64_t>(),
        weight.q8_zero ? nullptr : weight.sub_scale.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.sub_min.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.neuron_scale.data_ptr<float>(),
        weight.q8_zero ? nullptr : weight.neuron_min.data_ptr<float>(),
        weight.q8_zero ? weight.q8_zero_scale.data_ptr<__half>() : nullptr,
        target.data_ptr<Output>(), weight.neuron_len, weight.out, weight.ng, int(weight.gs),
        weight.q8_zero,splits,float(streams));
        };
        if(!activation || splits>1)invoke(float{},std::integral_constant<int,0>{},partial);
        else {
            const auto activated=[&](auto output_tag) {
                if(activation==2)invoke(output_tag,std::integral_constant<int,2>{},result);
                else invoke(output_tag,std::integral_constant<int,1>{},result);
            };
            if(output_dtype==tb::kFloat16)activated(__half{});else activated(float{});
        }
    };
    const auto dispatch=[&](auto input_tag) {
        if(native_input && weight.aligned_q8) {
            if(fixed_groups && !weight.q8_zero) {
                switch(weight.gs) {
                    case 32:launch(input_tag,std::true_type{},std::integral_constant<int,32>{});return;
                    case 48:launch(input_tag,std::true_type{},std::integral_constant<int,48>{});return;
                    case 64:launch(input_tag,std::true_type{},std::integral_constant<int,64>{});return;
                }
            }
            launch(input_tag,std::true_type{},std::integral_constant<int,0>{});
        } else launch(input_tag,std::false_type{},std::integral_constant<int,0>{});
    };
    if(x.scalar_type()==tb::kFloat16)dispatch(__half{});else dispatch(float{});
    MFQ_CUDA_CHECK(cudaGetLastError());
    if(splits>1) {
        const int64_t count=rows*weight.out;
        if(!activation)reduce_projection_splits<<<unsigned((count+255)/256),256,0,mfq_current_cuda_stream()>>>(
                partial.data_ptr<float>(),result.data_ptr<float>(),count,splits);
        else {
            const auto invoke=[&](auto output_tag,auto activation_tag) {
                using Output=decltype(output_tag);constexpr int action=decltype(activation_tag)::value;
                reduce_projection_activation<Output,action><<<unsigned((count+255)/256),256,0,mfq_current_cuda_stream()>>>(
                    partial.data_ptr<float>(),result.data_ptr<Output>(),count,splits,float(streams));
            };
            const auto activated=[&](auto output_tag) {
                if(activation==2)invoke(output_tag,std::integral_constant<int,2>{});
                else invoke(output_tag,std::integral_constant<int,1>{});
            };
            if(output_dtype==tb::kFloat16)activated(__half{});else activated(float{});
        }
        MFQ_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}
} // namespace

mfq_tensor_backend::Tensor try_gr_bf16_down_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    if(streams!=4 || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)!=10240 || input.numel()/10240<1 || input.numel()/10240>6 ||
            (input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kFloat32) ||
            !weight.is_cuda() || !weight.is_contiguous() || weight.device()!=input.device() ||
            weight.scalar_type()!=tb::kBFloat16 || weight.sizes().vec()!=std::vector<int64_t>({320,10240}))return {};
    const MfqCudaGuard guard(input.device());
    const int rows=int(input.numel()/10240);
    auto shape=input.sizes().vec();shape.back()=320;
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    const auto launch=[&](auto capacity,auto type) {
        using Input=decltype(type);
        gr_bf16_down_kernel<decltype(capacity)::value,Input><<<320,128,0,mfq_current_cuda_stream()>>>(
            input.data_ptr<Input>(),static_cast<const uint16_t*>(weight.data_ptr()),output.data_ptr<float>());
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            "gr_bf16_down_silu",reinterpret_cast<const void*>(gr_bf16_down_kernel<decltype(capacity)::value,Input>),
            "M="+std::to_string(rows)+" N=320 K=10240");
    };
    const auto types=[&](auto capacity) {
        if(input.scalar_type()==tb::kFloat16)launch(capacity,__half{});else launch(capacity,float{});
    };
    switch(rows) {
        case 1:types(std::integral_constant<int,1>{});break;
        case 2:types(std::integral_constant<int,2>{});break;
        case 3:types(std::integral_constant<int,3>{});break;
        case 4:types(std::integral_constant<int,4>{});break;
        case 5:types(std::integral_constant<int,5>{});break;
        case 6:types(std::integral_constant<int,6>{});break;
    }
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}

mfq_tensor_backend::Tensor try_gr_bf16_up_mix_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& normalized,const mfq_tensor_backend::Tensor& weight,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    if(streams!=4 || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)!=320 || input.numel()/320<1 || input.numel()/320>6 ||
            input.scalar_type()!=tb::kFloat32 || !normalized.is_cuda() || !normalized.is_contiguous() ||
            normalized.device()!=input.device() ||
            (normalized.scalar_type()!=tb::kFloat16 && normalized.scalar_type()!=tb::kFloat32) ||
            !weight.is_cuda() || !weight.is_contiguous() || weight.device()!=input.device() ||
            weight.scalar_type()!=tb::kBFloat16 || weight.sizes().vec()!=std::vector<int64_t>({10240,320}))return {};
    auto expected=input.sizes().vec();expected.back()=10240;
    if(normalized.sizes().vec()!=expected)return {};
    const MfqCudaGuard guard(input.device());
    const int rows=int(input.numel()/320);
    auto shape=input.sizes().vec();shape.back()=2560;
    auto output=tb::empty(shape,input.options());
    const auto launch=[&](auto capacity,auto type) {
        using Normalized=decltype(type);
        gr_bf16_up_mix_kernel<decltype(capacity)::value,Normalized><<<2560,128,0,mfq_current_cuda_stream()>>>(
            input.data_ptr<float>(),normalized.data_ptr<Normalized>(),static_cast<const uint16_t*>(weight.data_ptr()),output.data_ptr<float>());
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            "gr_bf16_up_mix",reinterpret_cast<const void*>(gr_bf16_up_mix_kernel<decltype(capacity)::value,Normalized>),
            "M="+std::to_string(rows)+" N=10240 K=320");
    };
    const auto types=[&](auto capacity) {
        if(normalized.scalar_type()==tb::kFloat16)launch(capacity,__half{});else launch(capacity,float{});
    };
    switch(rows) {
        case 1:types(std::integral_constant<int,1>{});break;
        case 2:types(std::integral_constant<int,2>{});break;
        case 3:types(std::integral_constant<int,3>{});break;
        case 4:types(std::integral_constant<int,4>{});break;
        case 5:types(std::integral_constant<int,5>{});break;
        case 6:types(std::integral_constant<int,6>{});break;
    }
    MFQ_CUDA_CHECK(cudaGetLastError());return output;
}

mfq_tensor_backend::Tensor nint_float_projection_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,bool parallel_groups,bool native_input,bool fixed_groups) {
    if(gr_prefill_selected(input))return gr_prefill_matmul(weight,input);
    return float_projection_impl(weight,input,parallel_groups,native_input,fixed_groups,0,1);
}

int nint_float_projection_resident_blocks(int device) {
    return projection_resident_blocks(device);
}

mfq_tensor_backend::Tensor nint_float_projection_activation_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,int64_t streams,bool injection,
        bool parallel_groups,bool native_input,bool fixed_groups) {
    MFQ_RUNTIME_CHECK(streams>0,"packed residual activation requires positive stream count");
    if(gr_prefill_selected(input)) {
        auto projected=gr_prefill_matmul(weight,input);
        if(input.scalar_type()==mfq_tensor_backend::kFloat16)projected=projected.to(mfq_tensor_backend::kFloat16);
        return injection ? mfq_qwen4_exp::gated_residual_injection(projected,streams)
                         : mfq_qwen4_exp::gated_residual_bottleneck(projected,streams);
    }
    return float_projection_impl(weight,input,parallel_groups,native_input,fixed_groups,injection?2:1,streams);
}

mfq_tensor_backend::Tensor nint_float_projection_mix_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized,
        int64_t streams,bool native_input,bool fixed_groups,bool compact) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(input.dim()>0 && streams>0 && weight.out>0 && weight.out%streams==0 &&
        normalized.dim()==input.dim(),"packed residual mixing dimensions disagree");
    auto expected=input.sizes().vec();expected.back()=weight.out;
    MFQ_RUNTIME_CHECK(normalized.sizes().vec()==expected && input.is_cuda() && normalized.is_cuda() &&
        input.device()==normalized.device() && input.device()==weight.q_packed.device() &&
        input.size(-1)==weight.neuron_len && weight.ng>0 && weight.gs>0 && weight.gs<=64 &&
        weight.ng==(weight.neuron_len+weight.gs-1)/weight.gs,"packed residual mixing geometry/device disagree");
    const auto rows=input.numel()/weight.neuron_len,hidden=weight.out/streams;
    if(gr_prefill_selected(input)) {
        if(gr_prefill_blas_selected(weight,input) && normalized.is_contiguous() &&
                (normalized.scalar_type()==tb::kFloat16 || normalized.scalar_type()==tb::kFloat32))
            return gr_prefill_blas_mix(weight,input,normalized,streams);
        if(gr_prefill_mix_selected(weight,input,normalized,streams))return gr_prefill_fused_mix(weight,input,normalized);
        auto projected=gr_prefill_matmul(weight,input);
        if(input.scalar_type()==tb::kFloat16)projected=projected.to(tb::kFloat16);
        return mfq_qwen4_exp::gated_residual_mix(projected,normalized,streams);
    }
    const auto ordinary_blocks=((weight.out+3)/4)*rows;
    MfqCudaGuard guard(input.device());
    // Retain the original group partition whenever the ordinary projection splits.
    const bool unsplit=weight.ng==1 || !rows || ordinary_blocks>=projection_resident_blocks(input.get_device());
    const auto half_or_float=[](const tb::Tensor& value) {
        return value.scalar_type()==tb::kFloat16 || value.scalar_type()==tb::kFloat32;
    };
    if(!unsplit || streams>32 || !input.is_contiguous() || !normalized.is_contiguous() ||
        !half_or_float(input) || !half_or_float(normalized)) {
        auto projection=nint_float_projection_cuda(weight,input,true,native_input,fixed_groups);
        if(input.scalar_type()==tb::kFloat16)projection=projection.to(tb::kFloat16);
        return mfq_qwen4_exp::gated_residual_mix(projection,normalized,streams);
    }
    MFQ_RUNTIME_CHECK(rows<=65535 && hidden<=std::numeric_limits<int>::max(),
        "packed residual mixing exceeds CUDA launch geometry");
    auto x=native_input && input.scalar_type()==tb::kFloat16?input:input.to(tb::kFloat32);
    auto shape=input.sizes().vec();shape.back()=hidden;
    const auto dtype=input.scalar_type()==tb::kFloat32 || normalized.scalar_type()==tb::kFloat32
        ?tb::kFloat32:tb::kFloat16;
    auto output=tb::empty(shape,input.options().dtype(dtype));
    if(!rows)return output;
    const auto launch=[&](auto gate_tag,auto value_tag,auto aligned_tag,auto input_tag,auto group_tag) {
        using Gate=decltype(gate_tag);using Value=decltype(value_tag);
        using Input=decltype(input_tag);
        using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
        constexpr bool aligned=decltype(aligned_tag)::value;
        constexpr int group=decltype(group_tag)::value;
        if constexpr(aligned && group>0) {
            if(compact && (streams==3 || streams==4)) {
                const auto warp_launch=[&](auto stream_tag) {
                    constexpr int count=decltype(stream_tag)::value;
                    affine_projection_warp_mix<Gate,Value,Input,group,count><<<
                        dim3(unsigned((hidden+count-1)/count),unsigned(rows)),count*32,0,mfq_current_cuda_stream()>>>(
                        x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),weight.row_q_bit_offsets.data_ptr<int64_t>(),
                        weight.sub_scale.data_ptr<uint8_t>(),weight.sub_min.data_ptr<uint8_t>(),
                        weight.neuron_scale.data_ptr<float>(),weight.neuron_min.data_ptr<float>(),
                        normalized.data_ptr<Value>(),output.data_ptr<Output>(),weight.neuron_len,hidden,weight.ng);
                };
                if(streams==3)warp_launch(std::integral_constant<int,3>{});
                else warp_launch(std::integral_constant<int,4>{});
                return;
            }
        }
        affine_projection_mix<Gate,Value,aligned,Input,group><<<dim3(unsigned(hidden),unsigned(rows)),unsigned(streams*32),0,
            mfq_current_cuda_stream()>>>(x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            weight.q8_zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.neuron_scale.data_ptr<float>(),
            weight.q8_zero?nullptr:weight.neuron_min.data_ptr<float>(),
            weight.q8_zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,
            normalized.data_ptr<Value>(),output.data_ptr<Output>(),weight.neuron_len,hidden,weight.ng,
            int(weight.gs),weight.q8_zero,int(streams));
    };
    const auto dispatch=[&](auto gate_tag,auto value_tag) {
        const auto typed=[&](auto input_tag) {
            if(weight.aligned_q8) {
                if(fixed_groups && !weight.q8_zero) {
                    switch(weight.gs) {
                        case 32:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,32>{});return;
                        case 48:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,48>{});return;
                        case 64:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,64>{});return;
                    }
                }
                launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,0>{});
            } else launch(gate_tag,value_tag,std::false_type{},input_tag,std::integral_constant<int,0>{});
        };
        if(x.scalar_type()==tb::kFloat16)typed(__half{});else typed(float{});
    };
    if(input.scalar_type()==tb::kFloat16 && normalized.scalar_type()==tb::kFloat16)dispatch(__half{},__half{});
    else if(input.scalar_type()==tb::kFloat16)dispatch(__half{},float{});
    else if(normalized.scalar_type()==tb::kFloat16)dispatch(float{},__half{});
    else dispatch(float{},float{});
    MFQ_CUDA_CHECK(cudaGetLastError());
    return output;
}
