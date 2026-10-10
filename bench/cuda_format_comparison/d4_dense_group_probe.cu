#include "e8_dense_group_probe.h"
#include "mfq/kernels/cuda/nvq_d4_dense_group.cuh"
#include "mfq/kernels/cuda/nvq_dense_window32.cuh"
#include <stdexcept>

namespace pn=mfq::cuda::packed_nvq;

template<int Format,bool Dense,int Lanes,bool Narrow=false>
__global__ void __launch_bounds__(128) d4_dense_dot(const E8DenseProbeWeight* weights,
        const int8_t* input,const float* scales,float* output) {
    const auto w=weights[blockIdx.y];
    const int row=blockIdx.x*(128/Lanes)+threadIdx.x/Lanes,lane=threadIdx.x&(Lanes-1);
    constexpr int bits=pn::format_index_bits(Format);
    float sum=0.f;
    for(int group=lane;group<w.groups;group+=Lanes) {
        const auto* x=reinterpret_cast<const int2*>(input+(blockIdx.y*w.groups+group)*24);
        const int2 activation[3]={x[0],x[1],x[2]};
        const float input_scale=scales[blockIdx.y*w.groups+group];
        if(row<w.rows) {
            uint64_t indices=0;uint32_t signs=0,state=0;
            pn::DenseD4Group<bits> packed{};
            if constexpr(Dense) {
                if constexpr(Narrow) {
                    const int bit=row*(w.vectors*(2*bits+7)+w.groups*4)+group*(4+3*(2*bits+7));
                    const auto words=pn::load_dense_window32<4+3*(2*bits+7)>(w.dense,int(w.dense_bytes),bit);
                    packed={words.x,words.y,words.z};
                }else packed=pn::load_dense_d4_group<bits>(w.dense,w.dense_bytes,row,group,w.vectors,w.groups);
                state=packed.state();
            }else {
                state=pn::load_packed_4(w.states,int64_t(row)*w.groups+group);
                signs=pn::load_packed_sign_group3_words(w.signs,(int64_t(row)*w.vectors+group*3)*7,w.sign_bytes);
                indices=pn::load_packed_group_window<bits*6>(w.indices,(int64_t(row)*w.vectors+group*3)*2*bits,w.index_bytes);
            }
            const auto* bank=pn::active_codebook<Format>(w.codebook,state);
            const float weight_scale=pn::format_scale<Format>(w.anchors[row],state,w.codebook);
            int dot=0;
#pragma unroll
            for(int segment=0;segment<3;++segment) {
                int2 value{};
                if constexpr(Dense) {
                    if(segment==0)value=pn::decode_dense_d4_vector<0>(packed,bank);
                    if(segment==1)value=pn::decode_dense_d4_vector<1>(packed,bank);
                    if(segment==2)value=pn::decode_dense_d4_vector<2>(packed,bank);
                }else {
                    const auto* book=reinterpret_cast<const int*>(bank);
                    const auto a=uint32_t(indices>>(segment*2*bits))&((1u<<bits)-1);
                    const auto b=uint32_t(indices>>((segment*2+1)*bits))&((1u<<bits)-1);
                    const auto sign=(signs>>(segment*7))&127u;
                    value=pn::apply_sign8(make_int2(book[a],book[b]),sign|((__popc(sign)&1u)<<7));
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
__global__ void d4_dense_check(const E8DenseProbeWeight* weights,uint32_t* errors) {
    constexpr int bits=pn::format_index_bits(Format);
    const auto w=weights[blockIdx.y];
    for(int linear=blockIdx.x*blockDim.x+threadIdx.x;linear<w.rows*w.groups;linear+=gridDim.x*blockDim.x) {
        const int row=linear/w.groups,group=linear%w.groups;
        const auto packed=pn::load_dense_d4_group<bits>(w.dense,w.dense_bytes,row,group,w.vectors,w.groups);
        const auto state=pn::load_packed_4(w.states,linear);
        bool bad=packed.state()!=state;
        const auto* bank=pn::active_codebook<Format>(w.codebook,state);
#pragma unroll
        for(int segment=0;segment<3;++segment)if(group*3+segment<w.vectors) {
            uint32_t a=0,b=0,sign=0;int2 value{};
#define DECODE(S) if(segment==S) {a=packed.template index<S,0>();b=packed.template index<S,1>(); \
            sign=packed.template signs<S>();value=pn::decode_dense_d4_vector<S>(packed,bank);}
            DECODE(0);DECODE(1);DECODE(2);
#undef DECODE
            const int64_t pair=int64_t(row)*w.vectors+group*3+segment;
            bad|=a!=pn::load_packed_bits(w.indices,pair*2*bits,bits,w.index_bytes) ||
                 b!=pn::load_packed_bits(w.indices,(pair*2+1)*bits,bits,w.index_bytes) ||
                 sign!=pn::load_packed_bits(w.signs,pair*7,7,w.sign_bytes);
            const auto expected0=pn::decode_chunk4<Format>(w.indices,w.index_bytes,w.signs,w.sign_bytes,
                w.codebook,row,group,segment*2,w.vectors*2,w.vectors,w.groups,w.sign_mode,state);
            const auto expected1=pn::decode_chunk4<Format>(w.indices,w.index_bytes,w.signs,w.sign_bytes,
                w.codebook,row,group,segment*2+1,w.vectors*2,w.vectors,w.groups,w.sign_mode,state);
            bad|=value.x!=expected0 || value.y!=expected1;
        }
        if(bad)atomicAdd(errors,1u);
    }
}

template<int Format,bool Narrow> void launch(bool dense,int lanes,const E8DenseProbeWeight* weights,
        int experts,int rows,const int8_t* input,const float* scales,float* output,cudaStream_t stream) {
#define RUN(D,L) d4_dense_dot<Format,D,L,Narrow><<<dim3((rows+128/L-1)/(128/L),experts),128,0,stream>>>(weights,input,scales,output)
    if(lanes==16){if(dense){RUN(true,16);}else{RUN(false,16);}}
    else {if(dense){RUN(true,8);}else{RUN(false,8);}}
#undef RUN
}
void d4_dense_probe_launch(int format,bool dense,int lanes,const E8DenseProbeWeight* weights,
        int experts,int rows,const int8_t* input,const float* scales,float* output,cudaStream_t stream,bool narrow) {
    switch(format) {
#define F(X) case X:if(narrow)launch<X,true>(dense,lanes,weights,experts,rows,input,scales,output,stream); \
        else launch<X,false>(dense,lanes,weights,experts,rows,input,scales,output,stream);break
        F(10);F(11);F(12);F(15);
#undef F
        default:throw std::invalid_argument("D4 format");
    }
}
void d4_dense_probe_validate(int format,const E8DenseProbeWeight* weights,int experts,int rows,
        uint32_t* errors,cudaStream_t stream) {
    switch(format) {
#define F(X) case X:d4_dense_check<X><<<dim3((rows+127)/128,experts),128,0,stream>>>(weights,errors);break
        F(10);F(11);F(12);F(15);
#undef F
        default:throw std::invalid_argument("D4 format");
    }
}
template<int Format,bool Narrow> cudaFuncAttributes attributes(bool dense,int lanes) {
    cudaFuncAttributes value{};cudaError_t status;
#define A(D,L) cudaFuncGetAttributes(&value,d4_dense_dot<Format,D,L,Narrow>)
    if(lanes==16)status=dense?A(true,16):A(false,16);
    else status=dense?A(true,8):A(false,8);
#undef A
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    return value;
}
cudaFuncAttributes d4_dense_probe_attributes(int format,bool dense,int lanes,bool narrow) {
#define A(F) narrow?attributes<F,true>(dense,lanes):attributes<F,false>(dense,lanes)
    switch(format){case 10:return A(10);case 11:return A(11);
        case 12:return A(12);case 15:return A(15);
        default:throw std::invalid_argument("D4 format");}
#undef A
}
