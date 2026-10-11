#include "e8_dense_group_probe.h"
#include "mfq/kernels/cuda/nvq_e8_dense_group.cuh"
#include "mfq/kernels/cuda/nvq_jsc_bank_map.cuh"
#include "mfq/kernels/cuda/mfe_e8_narrow.cuh"
#include "mfq/kernels/cuda/nvq_byte_signs.cuh"
#include "mfq/kernels/cuda/nvq_dense_window32.cuh"
#include "e8_cache_policy.cuh"
#include <stdexcept>

namespace pn=mfq::cuda::packed_nvq;

// 0: original MFE arithmetic/reader; 1: packed bank map; 2: current narrow
// MFE reader; 3: narrow two-word index reader and the proven byte-sign method
// for E8-256/1024. Layout comparisons use the same decoder variant.
// 4: current narrow reader with one prepared (scale-half, bank-id) word.
// 5: compact groups only, one group of lookahead metadata. It preserves each
// lane's FMA order and does not retain a second set of activation/book vectors.
// 6: compact groups with proven expert-local 32-bit bit offsets.
// 7: streaming compact weights; 8: retained codebook; 9: both cache hints.
// 10: explicitly global compact loads with the default cache policy, isolating
// the streaming hint from the change from generic LD to global LDG.
template<int Format,bool Dense,int Lanes,int Decoder=0>
__global__ void __launch_bounds__(128) e8_dense_dot(const E8DenseProbeWeight* weights,
        const int8_t* input,const float* scales,float* output) {
    const auto w=weights[blockIdx.y];
    const int row=blockIdx.x*(128/Lanes)+threadIdx.x/Lanes,lane=threadIdx.x&(Lanes-1);
    // Match the MFE pair/quad group order and floating point accumulation.
    float sum=0.f;
    constexpr int bits=pn::format_index_bits(Format);
    for(int group=lane;group<w.groups;group+=Lanes) {
        const auto* x=reinterpret_cast<const int2*>(input+(blockIdx.y*w.groups+group)*24);
        const int2 activation[3]={x[0],x[1],x[2]};
        const float input_scale=scales[blockIdx.y*w.groups+group];
        if(row<w.rows) {
            uint64_t indices=0;uint32_t signs=0,state=0;
            pn::DenseE8Group<bits> packed{};
            if constexpr(Dense) {
                if constexpr(Decoder>=7) {
                    packed=e8_cache_policy::group<bits,Decoder==8?0:Decoder==10?2:1>(w.dense,w.dense_bytes,row,group,w.vectors,w.groups);
                }else if constexpr(Decoder==6) {
                    const int bit=row*(w.vectors*(bits+7)+w.groups*4)+group*(4+3*(bits+7));
                    const auto words=pn::load_dense_window32<4+3*(bits+7)>(w.dense,int(w.dense_bytes),bit);
                    packed.bits=uint64_t(words.x)|(uint64_t(words.y)<<32);
                }else packed=pn::load_dense_e8_group<bits>(w.dense,w.dense_bytes,row,group,w.vectors,w.groups);
                state=packed.state();
            }else if constexpr(Decoder>=2) {
                state=mfq::cuda::load_e8_state32(w.states,row*w.groups+group);
                signs=mfq::cuda::load_mfe_e8_signs_words32(w.signs,(row*w.vectors+group*3)*7,int(w.sign_bytes));
                indices=mfq::cuda::load_e8_indices32<bits,Decoder==3>(w.indices,row*w.vectors+group*3,int(w.index_bytes));
            }else {
                state=pn::load_packed_4(w.states,int64_t(row)*w.groups+group);
                signs=pn::load_packed_sign_group3_words(w.signs,(int64_t(row)*w.vectors+group*3)*7,w.sign_bytes);
                indices=pn::load_packed_group_window<bits*3>(w.indices,(int64_t(row)*w.vectors+group*3)*bits,w.index_bytes);
            }
            const int8_t* bank;float weight_scale;
            if constexpr(Decoder==4) {
                const uint32_t metadata=reinterpret_cast<const uint32_t*>(w.codebook-64)[state];
                bank=w.codebook+64+(metadata>>16)*(8<<bits);
                weight_scale=w.anchors[row]*__half2float(__ushort_as_half(uint16_t(metadata)));
            }else {
                bank=Decoder==1?pn::packed_jsc_codebook<Format>(w.codebook,w.bank_map,state):
                    pn::active_codebook<Format>(w.codebook,state);
                weight_scale=pn::format_scale<Format>(w.anchors[row],state,w.codebook);
            }
            int dot=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                int2 value{};
                if constexpr(Dense && Decoder>=7) {
                    if(segment==0)value=e8_cache_policy::decode<0,bits,Decoder==8 || Decoder==9>(packed,bank,w.sign_mode);
                    if(segment==1)value=e8_cache_policy::decode<1,bits,Decoder==8 || Decoder==9>(packed,bank,w.sign_mode);
                    if(segment==2)value=e8_cache_policy::decode<2,bits,Decoder==8 || Decoder==9>(packed,bank,w.sign_mode);
                }else if constexpr(Dense && !(Decoder==3 && bits!=12)) {
                    if(segment==0)value=pn::decode_dense_e8_vector<0>(packed,bank,w.sign_mode);
                    if(segment==1)value=pn::decode_dense_e8_vector<1>(packed,bank,w.sign_mode);
                    if(segment==2)value=pn::decode_dense_e8_vector<2>(packed,bank,w.sign_mode);
                }else {
                    const uint32_t index=uint32_t((Dense?packed.bits:indices) >>
                        (Dense?4+segment*(bits+7):segment*bits))&((1u<<bits)-1);
                    const uint32_t sign=Dense?uint32_t(packed.bits>>(4+segment*(bits+7)+bits))&127u:
                        (signs>>(segment*7))&127u;
                    const uint32_t parity=(__popc(sign)&1)^((w.sign_mode&1)?((index>>7)&1u):0u);
                    const auto digits=reinterpret_cast<const int2*>(bank)[index];
                    if constexpr(Decoder==3 && bits!=12)
                        value=mfq::cuda::nvq_apply_byte_signs8(digits,sign|(parity<<7));
                    else value=pn::apply_sign8(digits,sign|(parity<<7));
                }
                dot=__dp4a(value.y,activation[segment].y,__dp4a(value.x,activation[segment].x,dot));
            }
            sum=fmaf(weight_scale*input_scale,float(dot),sum);
        }
    }
#pragma unroll
    for(int offset=Lanes/2;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,offset,Lanes);
    if(lane==0 && row<w.rows)output[int64_t(blockIdx.y)*w.rows+row]=sum;
}

template<int Format>
__global__ void e8_dense_check(const E8DenseProbeWeight* weights,uint32_t* errors) {
    constexpr int bits=pn::format_index_bits(Format);
    const auto w=weights[blockIdx.y];
    for(int linear=blockIdx.x*blockDim.x+threadIdx.x;linear<w.rows*w.groups;linear+=gridDim.x*blockDim.x) {
        const int row=linear/w.groups,group=linear%w.groups;
        const auto packed=pn::load_dense_e8_group<bits>(w.dense,w.dense_bytes,row,group,w.vectors,w.groups);
        const auto state=pn::load_packed_4(w.states,linear);
        bool bad=packed.state()!=state;
        const auto* bank=pn::active_codebook<Format>(w.codebook,state);
        const auto* packed_bank=pn::packed_jsc_codebook<Format>(w.codebook,w.bank_map,state);
        bad|=bank!=packed_bank;
#pragma unroll
        for(int segment=0;segment<3;++segment)if(group*3+segment<w.vectors) {
            const int64_t vector=int64_t(row)*w.vectors+group*3+segment;
            const auto index=pn::load_packed_bits(w.indices,vector*bits,bits,w.index_bytes);
            const auto sign=pn::load_packed_bits(w.signs,vector*7,7,w.sign_bytes);
            uint32_t di=0,ds=0;int2 value{};
            if(segment==0){di=packed.template index<0>();ds=packed.template signs<0>();value=pn::decode_dense_e8_vector<0>(packed,bank,w.sign_mode);}
            if(segment==1){di=packed.template index<1>();ds=packed.template signs<1>();value=pn::decode_dense_e8_vector<1>(packed,bank,w.sign_mode);}
            if(segment==2){di=packed.template index<2>();ds=packed.template signs<2>();value=pn::decode_dense_e8_vector<2>(packed,bank,w.sign_mode);}
            const auto expected=pn::load_nvq_vec8<Format>(w.indices,w.index_bytes,w.signs,w.sign_bytes,
                w.codebook,row,group*3+segment,group,w.groups,w.vectors,w.vectors,w.sign_mode,state);
            bad|=di!=index || ds!=sign || value.x!=expected.values.x || value.y!=expected.values.y;
        }
        if(bad)atomicAdd(errors,1u);
    }
}

template<int Format>
struct E8PipelineGroup {
    pn::DenseE8Group<pn::format_index_bits(Format)> packed;
    const int8_t* bank;
    float weight_scale;
};

template<int Format>
__device__ __forceinline__ E8PipelineGroup<Format> prepare_e8_group(
        const E8DenseProbeWeight& w,int row,int group) {
    constexpr int bits=pn::format_index_bits(Format);
    const auto packed=pn::load_dense_e8_group<bits>(w.dense,w.dense_bytes,row,group,w.vectors,w.groups);
    const auto state=packed.state();
    return {packed,pn::active_codebook<Format>(w.codebook,state),
        pn::format_scale<Format>(w.anchors[row],state,w.codebook)};
}

template<int Format,int Lanes>
__global__ void __launch_bounds__(128) e8_pipeline_dense_dot(const E8DenseProbeWeight* weights,
        const int8_t* input,const float* scales,float* output) {
    const auto w=weights[blockIdx.y];
    const int row=blockIdx.x*(128/Lanes)+threadIdx.x/Lanes,lane=threadIdx.x&(Lanes-1);
    float sum=0.f;
    if(row<w.rows && lane<w.groups) {
        auto current=prepare_e8_group<Format>(w,row,lane);
        for(int group=lane;group<w.groups;group+=Lanes) {
            // Only metadata and the packed bits are live across an iteration.
            // The final iteration performs no speculative/out-of-range read.
            E8PipelineGroup<Format> next{};
            if(group+Lanes<w.groups)next=prepare_e8_group<Format>(w,row,group+Lanes);
            const auto* x=reinterpret_cast<const int2*>(input+(blockIdx.y*w.groups+group)*24);
            const int2 activation[3]={x[0],x[1],x[2]};
            const float input_scale=scales[blockIdx.y*w.groups+group];
            int dot=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                int2 value{};
                if(segment==0)value=pn::decode_dense_e8_vector<0>(current.packed,current.bank,w.sign_mode);
                if(segment==1)value=pn::decode_dense_e8_vector<1>(current.packed,current.bank,w.sign_mode);
                if(segment==2)value=pn::decode_dense_e8_vector<2>(current.packed,current.bank,w.sign_mode);
                dot=__dp4a(value.y,activation[segment].y,__dp4a(value.x,activation[segment].x,dot));
            }
            sum=fmaf(current.weight_scale*input_scale,float(dot),sum);
            current=next;
        }
    }
#pragma unroll
    for(int offset=Lanes/2;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,offset,Lanes);
    if(lane==0 && row<w.rows)output[int64_t(blockIdx.y)*w.rows+row]=sum;
}

template<int Format,int Decoder> void launch(bool dense,int lanes,const E8DenseProbeWeight* weights,
        int experts,int rows,const int8_t* input,const float* scales,float* output,cudaStream_t stream) {
    if constexpr(Decoder==5) {
        if(!dense) {launch<Format,2>(false,lanes,weights,experts,rows,input,scales,output,stream);return;}
        if(lanes==16)e8_pipeline_dense_dot<Format,16><<<dim3((rows+7)/8,experts),128,0,stream>>>(weights,input,scales,output);
        else e8_pipeline_dense_dot<Format,8><<<dim3((rows+15)/16,experts),128,0,stream>>>(weights,input,scales,output);
        return;
    }
#define RUN(D,L) e8_dense_dot<Format,D,L,Decoder><<<dim3((rows+128/L-1)/(128/L),experts),128,0,stream>>>(weights,input,scales,output)
    if(lanes==16){if(dense){RUN(true,16);}else{RUN(false,16);}}
    else {if(dense){RUN(true,8);}else{RUN(false,8);}}
#undef RUN
}
void e8_dense_probe_launch(int format,bool dense,int lanes,const E8DenseProbeWeight* weights,
        int experts,int rows,const int8_t* input,const float* scales,float* output,cudaStream_t stream,int decoder) {
    if(lanes!=8 && lanes!=16)throw std::invalid_argument("E8 lane count");
    if(decoder<0 || decoder>10)throw std::invalid_argument("E8 decoder variant");
    if(format==10 || format==11 || format==12 || format==15) {
        if(decoder!=0 && decoder!=6)throw std::invalid_argument("D4 probe supports decoders 0 and 6");
        d4_dense_probe_launch(format,dense,lanes,weights,experts,rows,input,scales,output,stream,decoder==6);return;
    }
    switch(format) {
#define L(F,D) launch<F,D>(dense,lanes,weights,experts,rows,input,scales,output,stream)
#define F(X) case X: switch(decoder) {case 0:L(X,0);break;case 1:L(X,1);break; \
        case 2:L(X,2);break;case 3:L(X,3);break;case 4:L(X,4);break;case 5:L(X,5);break;case 6:L(X,6);break; \
        case 7:L(X,7);break;case 8:L(X,8);break;case 9:L(X,9);break;case 10:L(X,10);break;}break
        F(5);F(13);F(14);
#undef F
#undef L
        default:throw std::invalid_argument("E8 format");
    }
}
void e8_dense_probe_validate(int format,const E8DenseProbeWeight* weights,int experts,int rows,
        uint32_t* errors,cudaStream_t stream) {
    if(format==10 || format==11 || format==12 || format==15) {
        d4_dense_probe_validate(format,weights,experts,rows,errors,stream);return;
    }
    switch(format) {
#define F(X) case X:e8_dense_check<X><<<dim3((rows+127)/128,experts),128,0,stream>>>(weights,errors);break
        F(5);F(13);F(14);
#undef F
        default:throw std::invalid_argument("E8 format");
    }
}
template<int Format,int Decoder> cudaFuncAttributes attributes(bool dense,int lanes) {
    cudaFuncAttributes value{};cudaError_t status;
    if constexpr(Decoder==5) {
        if(!dense)return attributes<Format,2>(false,lanes);
        if(lanes==16)status=cudaFuncGetAttributes(&value,e8_pipeline_dense_dot<Format,16>);
        else status=cudaFuncGetAttributes(&value,e8_pipeline_dense_dot<Format,8>);
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
        return value;
    }
#define A(D,L) cudaFuncGetAttributes(&value,e8_dense_dot<Format,D,L,Decoder>)
    if(lanes==16)status=dense?A(true,16):A(false,16);
    else status=dense?A(true,8):A(false,8);
#undef A
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    return value;
}
cudaFuncAttributes e8_dense_probe_attributes(int format,bool dense,int lanes,int decoder) {
    if(format==10 || format==11 || format==12 || format==15)
        return d4_dense_probe_attributes(format,dense,lanes,decoder==6);
#define A(F) decoder==10?attributes<F,10>(dense,lanes):decoder==9?attributes<F,9>(dense,lanes):decoder==8?attributes<F,8>(dense,lanes):decoder==7?attributes<F,7>(dense,lanes):decoder==6?attributes<F,6>(dense,lanes):decoder==5?attributes<F,5>(dense,lanes):decoder==4?attributes<F,4>(dense,lanes):decoder==3?attributes<F,3>(dense,lanes):decoder==2?attributes<F,2>(dense,lanes): \
        decoder==1?attributes<F,1>(dense,lanes):attributes<F,0>(dense,lanes)
    switch(format){case 5:return A(5);case 13:return A(13);
        case 14:return A(14);default:throw std::invalid_argument("E8 format");}
#undef A
}
