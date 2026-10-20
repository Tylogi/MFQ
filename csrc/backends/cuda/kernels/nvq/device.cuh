#pragma once
#include "kernels/quant/packed_nvq.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace mfq::cuda {
__device__ __forceinline__ int nvq_apply_byte_signs4(int values,uint32_t signs,int base) {
    const uint32_t nibble=(signs>>base)&15u;
    const uint32_t ones=(nibble*0x00204081u)&0x01010101u;
    const uint32_t inverted=uint32_t(values)^(ones*255u);
    // Each low byte is at most 127 + 1, so the addition cannot carry
    // between bytes. XOR restores the high bit after the per-byte +1.
    return int(((inverted&0x7f7f7f7fu)+ones)^(inverted&0x80808080u));
}
__device__ __forceinline__ int2 nvq_apply_byte_signs8(int2 values,uint32_t signs) {
    return make_int2(nvq_apply_byte_signs4(values.x,signs,0),nvq_apply_byte_signs4(values.y,signs,4));
}
}

namespace mfq::cuda {
__device__ __forceinline__ uint32_t nvq_checked_word(const uint8_t* data,int64_t byte,int64_t bytes) {
    if(byte+4<=bytes)return *reinterpret_cast<const uint32_t*>(data+byte);
    uint32_t value=0;
#pragma unroll
    for(int i=0;i<4;++i)if(byte+i<bytes)value|=uint32_t(data[byte+i])<<(i*8);
    return value;
}
template<int Bits,int Lanes>
__device__ __forceinline__ uint64_t nvq_cooperative_window(
        const uint8_t* data,int64_t bytes,int64_t base_bit,int lane) {
    static_assert(Bits>0 && Bits<64 && (Lanes==16 || Lanes==32));
    const int64_t byte=base_bit>>3;
    const int64_t aligned=byte-int64_t((reinterpret_cast<uintptr_t>(data)+byte)&3u);
    if(aligned<0)return packed_nvq::load_packed_group_window<Bits>(data,base_bit+int64_t(lane)*Bits,bytes);
    const uint32_t first=nvq_checked_word(data,aligned+int64_t(lane)*4,bytes);
    [[maybe_unused]] uint32_t second=0;
    if constexpr(Bits>=32)second=nvq_checked_word(data,aligned+int64_t(lane+Lanes)*4,bytes);
    const unsigned active=__activemask();
    const auto word=[&](int index) {
        const uint32_t a=__shfl_sync(active,first,index&(Lanes-1),Lanes);
        if constexpr(Bits>=32) {
            const uint32_t b=__shfl_sync(active,second,index&(Lanes-1),Lanes);
            return index<Lanes?a:b;
        }
        return a;
    };
    const int bit=int((byte-aligned)*8+(base_bit&7))+lane*Bits;
    const int index=bit>>5,shift=bit&31;
    const uint32_t a=word(index),b=word(index+1);
    uint64_t value=__funnelshift_r(a,b,shift);
    if constexpr(Bits>32)value|=uint64_t(__funnelshift_r(b,word(index+2),shift))<<32;
    return value&((uint64_t(1)<<Bits)-1);
}
}

namespace mfq::cuda {

// Prove every row/group bit offset, including the padded last group, before
// selecting signed 32-bit metadata arithmetic. Device pointers remain 64-bit.
inline bool nvq_e8_narrow_eligible(const packed_nvq::NvqDeviceWeight& w,
        int rows,int experts,int format) {
    if((format!=5 && format!=13 && format!=14) || rows<=0 || experts<=0 ||
            w.N<=0 || w.ng<=0 || w.nvec<=0 || w.nvec!=w.nsign || w.sub_bits!=4 ||
            int64_t(w.ng)!=(int64_t(w.nvec)+2)/3)return false;
    const int64_t total_rows=int64_t(rows)*experts;
    if(total_rows>w.N)return false;
    const int bits=packed_nvq::format_index_bits(format);
    const int64_t first=(total_rows-1)*w.nvec+int64_t(w.ng-1)*3;
    // Reserve space for the complete three-vector window and byte guards.
    if(first>(INT_MAX-64)/bits || first>(INT_MAX-64)/7)return false;
    const int64_t states=total_rows*w.ng;
    if(states>INT_MAX)return false;
    if(w.indices_nbytes<=0 || w.indices_nbytes>INT_MAX ||
            w.aux_nbytes<=0 || w.aux_nbytes>INT_MAX ||
            w.sub_scale_nbytes<=0 || w.sub_scale_nbytes>INT_MAX)return false;
    return w.indices_nbytes>=(total_rows*w.nvec*bits+7)/8 &&
           w.aux_nbytes>=(total_rows*w.nsign*7+7)/8 &&
           w.sub_scale_nbytes>=(states+1)/2;
}

__device__ __forceinline__ uint32_t load_e8_state32(const uint8_t* data,int linear) {
    return (data[linear>>1]>>((linear&1)*4))&15u;
}

__device__ __forceinline__ uint32_t load_e8_signs32(const uint8_t* data,int bit,int nbytes) {
    const int byte=bit>>3;
    uint32_t word=0;
#pragma unroll
    for(int i=0;i<4;++i)if(byte+i<nbytes)word|=uint32_t(data[byte+i])<<(i*8);
    return (word>>(bit&7))&0x1fffffu;
}

template<int IndexBits,bool WordWindow>
__device__ __forceinline__ uint64_t load_e8_indices32(const uint8_t* data,int first_vector,int nbytes) {
    static_assert(IndexBits==8 || IndexBits==10 || IndexBits==12);
    constexpr int Bits=3*IndexBits;
    const int bit=first_vector*IndexBits,byte=bit>>3;
    const int aligned=byte-int((uint32_t(reinterpret_cast<uintptr_t>(data))+uint32_t(byte))&3u);
    if(aligned>=0 && aligned+(WordWindow?8:12)<=nbytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(data+aligned);
        const int shift=(byte-aligned)*8+(bit&7);
        const uint32_t a=words[0],b=words[1];
        if constexpr(WordWindow) {
            if constexpr(Bits<=32)return __funnelshift_r(a,b,shift)&((1u<<Bits)-1u);
            else return ((uint64_t(a)|(uint64_t(b)<<32))>>shift)&((uint64_t(1)<<Bits)-1);
        }else {
            const uint32_t c=words[2];
            return (uint64_t(__funnelshift_r(a,b,shift))|
                (uint64_t(__funnelshift_r(b,c,shift))<<32))&((uint64_t(1)<<Bits)-1);
        }
    }
    uint64_t value=0;
#pragma unroll
    for(int i=0;i<(Bits+14)/8;++i)if(byte+i<nbytes)value|=uint64_t(data[byte+i])<<(i*8);
    return (value>>(bit&7))&((uint64_t(1)<<Bits)-1);
}

}

namespace mfq::cuda::packed_nvq {
// Lossless E8 group stream: state[4], then (index[B], sign[7]) for
// each valid vector. The final group has no padding vectors. Rows need
// not start on a byte boundary; the backing allocation has no read padding.
template<int IndexBits>
struct DenseE8Group {
    static_assert(IndexBits==8 || IndexBits==10 || IndexBits==12);
    uint64_t bits;
    __device__ uint32_t state() const { return uint32_t(bits)&15u; }
    template<int Segment> __device__ uint32_t index() const {
        return uint32_t(bits>>(4+Segment*(IndexBits+7)))&((1u<<IndexBits)-1);
    }
    template<int Segment> __device__ uint32_t signs() const {
        return uint32_t(bits>>(4+Segment*(IndexBits+7)+IndexBits))&127u;
    }
};

template<int IndexBits>
__device__ __forceinline__ DenseE8Group<IndexBits> load_dense_e8_group(
        const uint8_t* stream,int64_t bytes,int row,int group,int vectors,int groups) {
    constexpr int group_bits=4+3*(IndexBits+7);
    const int row_bits=vectors*(IndexBits+7)+groups*4;
    const int64_t bit=int64_t(row)*row_bits+group*group_bits;
    // Unused last-group fields may overlap the next row. Callers mask these
    // vectors or use the existing zero-padded GS24 activation contract.
    return {load_packed_group_window<group_bits>(stream,bit,bytes)};
}

template<int Segment,int IndexBits>
__device__ __forceinline__ int2 decode_dense_e8_vector(
        DenseE8Group<IndexBits> group,const int8_t* bank,int sign_mode) {
    const auto index=group.template index<Segment>();
    const auto signs=group.template signs<Segment>();
    const auto parity=(__popc(signs)&1)^((sign_mode&1)?((index>>7)&1u):0u);
    return apply_sign8(reinterpret_cast<const int2*>(bank)[index],signs|(parity<<7));
}
}

namespace mfq::cuda::packed_nvq {
// Byte-preserving D4 records: state[4], then three pairs of indices and
// sign[7]. Full groups occupy 73/79/85 bits. The final group retains only
// valid pairs, without rounding each group or each row to a byte boundary.
// This variant requires an even number of D4 vectors per row.
template<int IndexBits>
struct DenseD4Group {
    static_assert(IndexBits==8 || IndexBits==9 || IndexBits==10);
    uint32_t lo,mid,hi;
    template<int Bit,int Bits> __device__ __forceinline__ uint32_t field() const {
        static_assert(Bit>=0 && Bit+Bits<=96 && Bits>0 && Bits<32);
        constexpr int shift=Bit&31,word=Bit/32;
        const uint32_t a=word==0?lo:word==1?mid:hi;
        if constexpr(shift+Bits<=32)return (a>>shift)&((1u<<Bits)-1);
        else {
            const uint32_t b=word==0?mid:hi;
            return __funnelshift_r(a,b,shift)&((1u<<Bits)-1);
        }
    }
    __device__ uint32_t state() const {return lo&15u;}
    template<int Segment,int Part> __device__ uint32_t index() const {
        return field<4+Segment*(2*IndexBits+7)+Part*IndexBits,IndexBits>();
    }
    template<int Segment> __device__ uint32_t signs() const {
        return field<4+Segment*(2*IndexBits+7)+2*IndexBits,7>();
    }
};

template<int IndexBits>
__device__ __forceinline__ DenseD4Group<IndexBits> load_dense_d4_group(
        const uint8_t* stream,int64_t bytes,int row,int group,int pairs,int groups) {
    constexpr int group_bits=4+3*(2*IndexBits+7);
    const int row_bits=pairs*(2*IndexBits+7)+groups*4;
    const int64_t bit=int64_t(row)*row_bits+group*group_bits,byte=bit>>3;
    const int64_t aligned=byte-int((reinterpret_cast<uintptr_t>(stream)+byte)&3u);
    if(aligned>=0 && aligned+16<=bytes) {
        const auto* words=reinterpret_cast<const uint32_t*>(stream+aligned);
        const int shift=int((byte-aligned)*8+(bit&7));
        return {__funnelshift_r(words[0],words[1],shift),
                __funnelshift_r(words[1],words[2],shift),
                __funnelshift_r(words[2],words[3],shift)};
    }
    uint32_t words[3]{};
#pragma unroll
    for(int i=0;i<(group_bits+14)/8;++i)if(byte+i<bytes)
        words[i/4]|=uint32_t(stream[byte+i])<<((i&3)*8);
    const int shift=int(bit&7);
    return {__funnelshift_r(words[0],words[1],shift),
            __funnelshift_r(words[1],words[2],shift),words[2]>>shift};
}

template<int Segment,int IndexBits>
__device__ __forceinline__ int2 decode_dense_d4_vector(DenseD4Group<IndexBits> group,const int8_t* bank) {
    const auto* codes=reinterpret_cast<const int*>(bank);
    const int2 values=make_int2(codes[group.template index<Segment,0>()],
                               codes[group.template index<Segment,1>()]);
    const uint32_t signs=group.template signs<Segment>();
    return apply_sign8(values,signs|((__popc(signs)&1u)<<7));
}
}

namespace mfq::cuda {
struct NvqRoutedReuseParams {
    packed_nvq::NvqDeviceWeight weight{};
    const int64_t* pointers=nullptr;
    const int64_t* sizes=nullptr;
    const int32_t* params=nullptr;
    const int32_t* expert_pool=nullptr;
    const int32_t* expert_local=nullptr;
    const int32_t* ids=nullptr;
    const int32_t* active=nullptr;
    const int8_t* input=nullptr;
    const float* scales=nullptr;
    __half* output=nullptr;
    int format=0,experts=0,local_experts=0,pools=0,rows=0,routes=0,pairs=0;
    bool down=false;
};
bool nvq_launch_routed_reuse(const NvqRoutedReuseParams&,cudaStream_t);
}
