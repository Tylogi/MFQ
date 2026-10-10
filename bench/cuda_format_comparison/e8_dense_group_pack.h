#pragma once
#include "vq.h"
#include <vector>
#include <cstdint>
#include <stdexcept>

static uint32_t get_bits(const uint8_t* data,uint64_t bit,int bits) {
    uint32_t value=0;
    for(int i=0;i<bits;++i)value|=uint32_t((data[(bit+i)/8]>>((bit+i)&7))&1u)<<i;
    return value;
}
static void put_bits(std::vector<uint8_t>& data,uint64_t bit,int bits,uint32_t value) {
    for(int i=0;i<bits;++i)data.at((bit+i)/8)|=uint8_t(((value>>i)&1u)<<((bit+i)&7));
}
static bool dense_d4_format(int format) {return format==10 || format==11 || format==12 || format==15;}
static int index_bits(int format) {
    switch(format) {
        case 5:case 10:case 11:return 8;
        case 12:return 9;
        case 13:case 15:return 10;
        case 14:return 12;
        default:throw std::runtime_error("dense NVQ format unsupported");
    }
}
static std::vector<uint8_t> pack_dense(const NvqWeight& w) {
    const int parts=dense_d4_format(int(w.kernel_format))?2:1;
    const int bits=index_bits(int(w.kernel_format)),pairs=int((w.neuron_len+7)/8),groups=int(w.ng);
    const int vectors=int((w.neuron_len+(parts==2?3:7))/(parts==2?4:8));
    if(w.gs!=24 || w.sub_bits!=4 || w.out<=0 || w.neuron_len<=0 || groups!=(pairs+2)/3 || vectors!=pairs*parts)
        throw std::runtime_error("dense NVQ layout requires complete index pairs and GS24/s4");
    const uint64_t row_bits=uint64_t(vectors)*bits+uint64_t(pairs)*7+groups*4;
    if(uint64_t(w.indices_packed.numel())<(uint64_t(w.out)*vectors*bits+7)/8 ||
       uint64_t(w.aux_packed.numel())<(uint64_t(w.out)*pairs*7+7)/8 ||
       uint64_t(w.sub_scale_packed.numel())<(uint64_t(w.out)*groups*4+7)/8)
        throw std::runtime_error("dense NVQ source is truncated");
    std::vector<uint8_t> packed((uint64_t(w.out)*row_bits+7)/8);
    const auto* indices=w.indices_packed.data_ptr<uint8_t>();
    const auto* signs=w.aux_packed.data_ptr<uint8_t>();
    const auto* states=w.sub_scale_packed.data_ptr<uint8_t>();
    for(int row=0;row<w.out;++row)for(int group=0;group<groups;++group) {
        uint64_t target=row*row_bits+group*(4+3*(parts*bits+7));
        put_bits(packed,target,4,get_bits(states,(uint64_t(row)*groups+group)*4,4));target+=4;
        for(int segment=0;segment<3 && group*3+segment<pairs;++segment) {
            const uint64_t pair=uint64_t(row)*pairs+group*3+segment;
            for(int part=0;part<parts;++part) {
                put_bits(packed,target,bits,get_bits(indices,(pair*parts+part)*bits,bits));target+=bits;
            }
            put_bits(packed,target,7,get_bits(signs,pair*7,7));target+=7;
        }
    }
    // Independent bit-by-bit round trip covers every valid source field,
    // including partial groups and half-byte row starts.
    for(int row=0;row<w.out;++row)for(int group=0;group<groups;++group) {
        uint64_t target=row*row_bits+group*(4+3*(parts*bits+7));
        if(get_bits(packed.data(),target,4)!=get_bits(states,(uint64_t(row)*groups+group)*4,4))
            throw std::runtime_error("dense NVQ state round trip");
        target+=4;
        for(int segment=0;segment<3 && group*3+segment<pairs;++segment) {
            const uint64_t pair=uint64_t(row)*pairs+group*3+segment;
            for(int part=0;part<parts;++part) {
                if(get_bits(packed.data(),target,bits)!=get_bits(indices,(pair*parts+part)*bits,bits))
                    throw std::runtime_error("dense NVQ index round trip");
                target+=bits;
            }
            if(get_bits(packed.data(),target,7)!=get_bits(signs,pair*7,7))
                throw std::runtime_error("dense NVQ signs round trip");
            target+=7;
        }
    }
    if(packed.size()>std::size_t(w.indices_packed.numel()+w.aux_packed.numel()+w.sub_scale_packed.numel()))
        throw std::runtime_error("dense NVQ expanded payload");
    return packed;
}
