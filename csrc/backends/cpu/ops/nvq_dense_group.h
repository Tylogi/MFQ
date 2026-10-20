#pragma once
#include "nvq_group.h"
#include <algorithm>
#include <cstring>

namespace mfq::cpu::detail {
// Borrow the compact stream in NvqDecodeView::indices. The loader validates
// its exact length; no padding or expanded code vectors are retained.
struct DenseNvqRecord {
    std::uint64_t lo=0;
    std::uint32_t hi=0;
    std::uint32_t field(int bit,int width) const {
        const auto mask=(1u<<width)-1u;
        if(bit>=64)return (hi>>(bit-64))&mask;
        auto value=lo>>bit;
        if(bit+width>64)value|=std::uint64_t(hi)<<(64-bit);
        return std::uint32_t(value)&mask;
    }
};

template<int Format> constexpr int dense_nvq_parts=
    Format==10 || Format==11 || Format==12 || Format==15 ? 2 : 1;
template<int Format> constexpr int dense_nvq_bits=
    Format==12 ? 9 : Format==13 || Format==15 ? 10 : Format==14 ? 12 : 8;

template<int Format>
inline std::int64_t dense_nvq_group_bit(const NvqDecodeView& w,std::int64_t row,int group) {
    constexpr int chunk=dense_nvq_parts<Format>*dense_nvq_bits<Format>+7;
    return row*(std::int64_t(w.nsign)*chunk+std::int64_t(w.groups)*4)+std::int64_t(group)*(4+3*chunk);
}

template<int Format>
inline std::uint32_t dense_nvq_state(const NvqDecodeView& w,std::int64_t row,int group) {
    const auto bit=dense_nvq_group_bit<Format>(w,row,group),byte=bit>>3;
    std::uint32_t word=w.indices[byte];
    if((bit&7)>4 && byte+1<w.index_bytes)word|=std::uint32_t(w.indices[byte+1])<<8;
    return (word>>(bit&7))&15u;
}

template<int Format>
inline DenseNvqRecord load_dense_nvq_group(const NvqDecodeView& w,std::int64_t row,int group) {
    const auto bit=dense_nvq_group_bit<Format>(w,row,group),byte=bit>>3;
    const auto available=w.index_bytes-byte;
    DenseNvqRecord result;
    if(available>=12) {
        std::memcpy(&result.lo,w.indices+byte,8);
        std::memcpy(&result.hi,w.indices+byte+8,4);
    }else {
        std::memcpy(&result.lo,w.indices+byte,std::size_t(std::min<std::int64_t>(8,available)));
        if(available>8)std::memcpy(&result.hi,w.indices+byte+8,std::size_t(available-8));
    }
    const int shift=int(bit&7);
    if(shift) {
        result.lo=(result.lo>>shift)|(std::uint64_t(result.hi)<<(64-shift));
        result.hi>>=shift;
    }
    return result;
}
} // namespace mfq::cpu::detail
