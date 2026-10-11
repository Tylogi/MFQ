#include "mfq_cuda_nvq1_decode.h"
#include "packed_nvq.cuh"
#include <cstdlib>
#include <type_traits>

namespace {
using namespace mfq::cuda::packed_nvq;
struct DecodeArgs {
    Nvq1DecodeView weight;
    const int8_t* input;
    const float* scales;
    const int32_t* ids;
    const int32_t* local;
    __half* output;
    int experts,rows,routes,pairs;
    bool down;
};
// Same rounding and padded-tail convention as the production GS24 quantizer.
__global__ void quantize_input(const __half* x,int8_t* q,float* scales,int width,int groups) {
    const int token=blockIdx.x,group=blockIdx.y*blockDim.y+threadIdx.y,lane=threadIdx.x,column=group*24+lane;
    if(group>=groups)return;
    const bool valid=lane<24 && column<width;
    const float value=valid?__half2float(x[int64_t(token)*width+column]):0.f;
    float maximum=fabsf(value);
#pragma unroll
    for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
    const float scale=maximum>0.f?maximum/127.f:1.f;
    if(!lane)scales[int64_t(token)*groups+group]=scale;
    if(lane<24)q[(int64_t(token)*groups+group)*24+lane]=
        valid?int8_t(max(-127,min(127,int(roundf(value/scale))))):0;
}
template<bool Small>
__device__ __forceinline__ uint64_t record(const Nvq1DecodeView& w,int row,int group) {
    const auto* records=reinterpret_cast<const uint8_t*>(w.pointers[5]);
    const int64_t linear=int64_t(row)*w.groups+group;
    if constexpr(Small)return reinterpret_cast<const uint32_t*>(records)[linear];
    else {
        const int64_t byte=linear*5,aligned=byte&~int64_t(3);
        const int shift=int(byte&3)*8;
        if(aligned+8<=w.sizes[3]) {
            const auto* words=reinterpret_cast<const uint32_t*>(records+aligned);
            return uint64_t(__funnelshift_r(words[0],words[1],shift))|
                (uint64_t(words[1]>>shift)<<32);
        }
        uint64_t value=0;
#pragma unroll
        for(int i=0;i<5;++i)value|=uint64_t(records[byte+i])<<(8*i);
        return value;
    }
}
template<bool Small,bool Records,bool IntegerDelta,int Rows,int Lanes>
__global__ void __launch_bounds__(128) decode(DecodeArgs args) {
    constexpr int bits=Small?9:11,bank_bytes=Small?4096:16384,lanes=Lanes;
    constexpr int rows_per_block=4*(32/Lanes)*Rows;
    const int pair=blockIdx.y,lane=threadIdx.x&(Lanes-1);
    const int first=blockIdx.x*rows_per_block+(threadIdx.y*(32/Lanes)+threadIdx.x/Lanes)*Rows;
    if(first>=args.rows)return;
    const int expert=args.ids[pair];if(unsigned(expert)>=unsigned(args.experts))return;
    const int local=args.local[expert];if(unsigned(local)>=unsigned(args.weight.local_experts))return;
    const auto w=args.weight;
    const auto* indices=reinterpret_cast<const uint8_t*>(w.pointers[0]);
    const auto* auxiliary=reinterpret_cast<const uint8_t*>(w.pointers[1]);
    const auto* states=reinterpret_cast<const uint8_t*>(w.pointers[2]);
    const auto* anchors=reinterpret_cast<const float*>(w.pointers[3]);
    const auto* books=reinterpret_cast<const int8_t*>(w.pointers[IntegerDelta?6:4]);
    const int source=args.down?pair:pair/args.routes;
    const auto* input=args.input+int64_t(source)*w.groups*24;
    const auto* scales=args.scales+int64_t(source)*w.groups;
    float partial[Rows]{};
    for(int group=lane;group<w.groups;group+=lanes) {
        const int2 x[3]={*reinterpret_cast<const int2*>(input+group*24),
            *reinterpret_cast<const int2*>(input+group*24+8),*reinterpret_cast<const int2*>(input+group*24+16)};
        int input_sum=0;
        if constexpr(!IntegerDelta) {
#pragma unroll
            for(int segment=0;segment<3;++segment)
                input_sum=__dp4a(0x01010101,x[segment].y,__dp4a(0x01010101,x[segment].x,input_sum));
        }
        const float input_scale=scales[group];
#pragma unroll
        for(int r=0;r<Rows;++r)if(first+r<args.rows) {
            const int row=local*args.rows+first+r;
            const int64_t linear=int64_t(row)*w.groups+group;
            uint64_t packed;uint32_t state;int negative;
            if constexpr(Records) {
                packed=record<Small>(w,row,group);
                state=uint32_t(packed>>(3*bits))&(Small?15u:7u);
                negative=int((packed>>(Small?31:36))&1u);
            }else {
                packed=load_packed_group_window<3*bits>(indices,(int64_t(row)*w.vectors+group*3)*bits,w.sizes[0]);
                state=Small?load_packed_4(states,linear):load_packed_bits(states,linear*3,3,w.sizes[2]);
                negative=int(load_packed_bits(auxiliary,linear,1,w.sizes[1]));
            }
            const int8_t* bank=books+((IntegerDelta || Small)?negative*bank_bytes:0);
            int dot=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                const uint32_t index=uint32_t(packed>>(segment*bits))&((1u<<bits)-1u);
                const int2 value=reinterpret_cast<const int2*>(bank)[index];
                dot=__dp4a(value.y,x[segment].y,__dp4a(value.x,x[segment].x,dot));
            }
            float value=float(dot);
            if constexpr(IntegerDelta)value*=Small?0.03125f:0.125f;
            else value+=(Small?0.15625f:0.125f)*float((negative?-1:1)*input_sum);
            partial[r]=fmaf((anchors[row]*float(state))*input_scale,value,partial[r]);
        }
    }
#pragma unroll
    for(int r=0;r<Rows;++r) {
#pragma unroll
        for(int offset=Lanes/2;offset;offset>>=1)partial[r]+=__shfl_xor_sync(0xffffffffu,partial[r],offset,Lanes);
        if(!lane && first+r<args.rows)args.output[int64_t(pair)*args.rows+first+r]=__float2half_rn(partial[r]);
    }
}
template<bool Small,bool Records,bool IntegerDelta>
void launch(const DecodeArgs& args,cudaStream_t stream) {
    const auto* lane_setting=std::getenv("MFQ_NVQ1_LANES");
    const auto* row_setting=std::getenv("MFQ_NVQ1_ROWS");
    const int lanes=lane_setting?std::atoi(lane_setting):16;
    const int rows=row_setting?std::atoi(row_setting):(args.weight.groups<=32?2:1);
    const auto shape=[&](auto l,auto r) {
        constexpr int per_block=4*(32/decltype(l)::value)*decltype(r)::value;
        decode<Small,Records,IntegerDelta,decltype(r)::value,decltype(l)::value>
            <<<dim3((args.rows+per_block-1)/per_block,args.pairs),dim3(32,4),0,stream>>>(args);
    };
    const auto columns=[&](auto l) {
        if(rows==4)shape(l,std::integral_constant<int,4>{});
        else if(rows==2)shape(l,std::integral_constant<int,2>{});
        else shape(l,std::integral_constant<int,1>{});
    };
    if(lanes==32)columns(std::integral_constant<int,32>{});
    else columns(std::integral_constant<int,16>{});
}
template<bool Small>
void select(const DecodeArgs& args,cudaStream_t stream) {
    if(args.weight.pointers[5]) {
        if(args.weight.pointers[6])launch<Small,true,true>(args,stream);
        else launch<Small,true,false>(args,stream);
    }else if(args.weight.pointers[6])launch<Small,false,true>(args,stream);
    else launch<Small,false,false>(args,stream);
}
}
bool nvq1_try_decode_cuda(Nvq1DecodeView w,mfq_tensor_backend::Tensor x,mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor local,mfq_tensor_backend::Tensor qx,mfq_tensor_backend::Tensor scales,
        mfq_tensor_backend::Tensor output,int experts,int rows,bool quantized) {
    const auto* group_dot=std::getenv("MFQ_NVQ_MOE_GROUP_DOT");
    const auto* native=std::getenv("MFQ_NVQ1_NATIVE_DECODE");
    if((native && native[0]=='0') || (group_dot && group_dot[0]=='0') || ids.size(0)>8 ||
       (w.format!=1 && w.format!=8) || w.groups<=0)return false;
    const int routes=int(ids.size(1)),pairs=int(ids.numel()),input_rows=int(x.numel()/x.size(-1));
    auto stream=mfq_current_cuda_stream();
    const auto* quant_setting=std::getenv("MFQ_NVQ1_QUANT_WARPS");
    int automatic_warps=1;
    if(!quant_setting && !quantized && w.groups<=32 && input_rows>=10) {
        thread_local int cached_device=-1,major=0,minor=0;
        const int device=output.get_device();
        if(device!=cached_device) {
            MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
            MFQ_CUDA_CHECK(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
            cached_device=device;
        }
        if(major==8 && minor==6)automatic_warps=4;
    }
    const int requested=quant_setting?std::atoi(quant_setting):automatic_warps;
    const int quant_warps=requested==2 || requested==4 || requested==8?requested:1;
    if(!quantized)quantize_input<<<dim3(input_rows,(w.groups+quant_warps-1)/quant_warps),dim3(32,quant_warps),0,stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),qx.data_ptr<int8_t>(),scales.data_ptr<float>(),int(x.size(-1)),w.groups);
    const DecodeArgs args{w,qx.data_ptr<int8_t>(),scales.data_ptr<float>(),ids.data_ptr<int32_t>(),local.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(output.data_ptr()),experts,rows,routes,pairs,x.dim()==3};
    if(w.format==8)select<true>(args,stream);else select<false>(args,stream);
    return true;
}
