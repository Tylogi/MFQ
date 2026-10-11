#pragma once
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nvq_dense_probe {
static_assert(std::endian::native==std::endian::little);

template<class T> inline T read_word(const uint8_t* data,size_t bytes,uint64_t bit) {
    const size_t byte=size_t(bit>>3);
    T word=0;
    if(bytes-byte>=sizeof(T))std::memcpy(&word,data+byte,sizeof(T));
    else std::memcpy(&word,data+byte,bytes-byte);
    return word>>int(bit&7);
}

// Groups are written in stream order. A wide store can zero future bits, but
// must preserve the already written prefix in its first byte. Both stores are
// bounded by the actual allocation; no padding is required at its tail.
inline void store_group(std::vector<uint8_t>& output,uint64_t bit,uint64_t lo,uint32_t hi) {
    const size_t byte=size_t(bit>>3);const int shift=int(bit&7);
    const uint32_t high=(hi<<shift)|(shift?uint32_t(lo>>(64-shift)):0u);
    const uint64_t low=(lo<<shift)|(output[byte]&((1u<<shift)-1));
    const size_t remaining=output.size()-byte;
    if(remaining>=12) {
        std::memcpy(output.data()+byte,&low,8);
        std::memcpy(output.data()+byte+8,&high,4);
    }else {
        std::memcpy(output.data()+byte,&low,std::min(size_t(8),remaining));
        if(remaining>8)std::memcpy(output.data()+byte+8,&high,std::min(size_t(4),remaining-8));
    }
}

template<int Bits,int Parts>
std::vector<uint8_t> pack(const uint8_t* indices,size_t index_bytes,
        const uint8_t* signs,size_t sign_bytes,const uint8_t* states,size_t state_bytes,
        int rows,int pairs,int groups) {
    static_assert((Parts==1 && (Bits==8 || Bits==10 || Bits==12)) ||
                  (Parts==2 && (Bits==8 || Bits==9 || Bits==10)));
    if(rows<=0 || pairs<=0 || groups<=0 || int64_t(groups)!=(int64_t(pairs)+2)/3)
        throw std::invalid_argument("dense group geometry");
    constexpr int chunk_bits=Parts*Bits+7,group_bits=4+3*chunk_bits;
    const uint64_t row_bits=uint64_t(pairs)*chunk_bits+uint64_t(groups)*4;
    if(row_bits>(std::numeric_limits<size_t>::max()-7)/uint64_t(rows))
        throw std::overflow_error("dense group payload size");
    const uint64_t count=uint64_t(rows)*pairs;
    if(!indices || !signs || !states || index_bytes<(count*Parts*Bits+7)/8 ||
            sign_bytes<(count*7+7)/8 || state_bytes<(uint64_t(rows)*groups*4+7)/8)
        throw std::invalid_argument("truncated dense group source");
    std::vector<uint8_t> output(size_t((uint64_t(rows)*row_bits+7)/8));
    for(int row=0;row<rows;++row)for(int group=0;group<groups;++group) {
        const uint64_t pair=uint64_t(row)*pairs+group*3;
        const uint64_t packed_indices=read_word<uint64_t>(indices,index_bytes,pair*Parts*Bits);
        const uint32_t packed_signs=read_word<uint32_t>(signs,sign_bytes,pair*7);
        const uint64_t state=uint64_t(row)*groups+group;
        uint64_t lo=(states[state>>1]>>((state&1)*4))&15u;uint32_t hi=0;
        const int valid=std::min(3,pairs-group*3);
        for(int segment=0;segment<3;++segment)if(segment<valid) {
            const uint32_t index=uint32_t(packed_indices>>(segment*Parts*Bits))&((1u<<(Parts*Bits))-1);
            const uint32_t sign=(packed_signs>>(segment*7))&127u;
            const uint32_t chunk=index|(sign<<(Parts*Bits));
            const int position=4+segment*chunk_bits;
            lo|=uint64_t(chunk)<<position;
            if(position+chunk_bits>64)hi|=chunk>>(64-position);
        }
        store_group(output,uint64_t(row)*row_bits+uint64_t(group)*group_bits,lo,hi);
    }
    return output;
}
}
