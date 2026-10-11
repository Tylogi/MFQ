#pragma once
#include "mfe_ffn.h"
#include <climits>

namespace mfq::cuda {
inline bool mfe_e8_narrow_view_eligible(const MfePackedProjection& d,bool allow_empty=true) {
    if(d.family!=2 || (d.format!=5 && d.format!=13 && d.format!=14))return true;
    if(d.local_experts<0 || d.output_rows<=0 || d.input_width<=0 || d.groups<=0 || d.nvec<=0 ||
            int64_t(d.nvec)!=(int64_t(d.input_width)+7)/8 ||
            d.nvec!=d.nsign || d.sub_bits!=4 || d.group_size!=24 ||
            int64_t(d.groups)!=(int64_t(d.nvec)+2)/3)return false;
    if(d.local_experts==0 && !allow_empty)return false;
    // An empty resident pool still describes transfer geometry. Its concrete
    // matrix-local buffer lengths are checked again by expert_view().
    // The device rebases each expert with 64-bit pointer arithmetic before
    // entering its group loop. Only matrix-local offsets must fit int32.
    const int64_t rows=d.output_rows;
    const int bits=d.format==5?8:d.format==13?10:12;
    const int64_t first=(rows-1)*d.nvec+int64_t(d.groups-1)*3;
    if(first>(INT_MAX-64)/bits || first>(INT_MAX-64)/7 || rows*d.groups>INT_MAX)return false;
    const int64_t index_bits=rows*d.nvec*bits,sign_bits=rows*d.nsign*7,state_bits=rows*d.groups*4;
    // Byte rebasing cannot preserve an expert boundary in the middle of a
    // packed byte. Such geometries retain the ordinary 64-bit reader.
    if((index_bits&7) || (sign_bits&7) || (state_bits&7))return false;
    for(int i=0;i<3;++i)if(d.sizes[i]<0)return false;
    if(d.e8_dense_groups)return false;
    if(d.local_experts==0)return true;
    return d.sizes[0]>=(index_bits/8)*d.local_experts && d.sizes[1]>=(sign_bits/8)*d.local_experts &&
           d.sizes[2]>=(state_bits/8)*d.local_experts;
}
}
