#pragma once
#include "async_copy.cuh"

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
    struct {Input a[2][BM][BK+(std::is_same_v<Input,float>?4:8)];float w[2][BN][BK+(Packed?4:1)];} tile;
    float products[BM][BN];
};
template<class Input,class Value,bool AlignedQ8,bool Zero,bool Mix,int RN,int RM,int BK=32,int BM=256,int TileN=32,bool Packed=false>
__global__ __launch_bounds__(BM*TileN/(RN*RM),1) void gr_prefill_pipeline_kernel(
        const Input* input,const uint8_t* packed,const uint8_t* row_bits,
        const int64_t* row_offsets,const uint8_t* scales,const uint8_t* minima,
        const float* row_scales,const float* row_minima,const __half* zero_scales,
        const Value* normalized,std::conditional_t<!Mix || std::is_same_v<Input,float> || std::is_same_v<Value,float>,float,__half>* output,int rows,int width,int outputs,int groups,int gs) {
    using Output=std::conditional_t<!Mix || std::is_same_v<Input,float> || std::is_same_v<Value,float>,float,__half>;
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
    using Output=std::conditional_t<!Mix || std::is_same_v<Input,float> || std::is_same_v<Value,float>,float,__half>;
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
    const auto* option=std::getenv(Mix?"MFQ_GR_PREFILL_FUSED_MIX":"MFQ_GR_PREFILL_MATMUL");
    const int mode=option?std::atoi(option):(Mix?(std::is_same_v<Input,__half>?12:10):31);
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
