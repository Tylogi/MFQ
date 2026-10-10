#pragma once
#include "mfe_ffn.h"
#include <climits>
#include <cstdlib>

namespace mfq::cuda {
inline bool mfe_nvq_dense_narrow_eligible(const MfePackedProjection& d,bool allow_empty=true) {
    if(d.family!=2 || d.e8_dense_groups==d.d4_dense_groups || d.local_experts<0 ||
            (!allow_empty && d.local_experts==0) || d.output_rows<=0 || d.input_width<=0 ||
            d.group_size!=24 || d.sub_bits!=4 || d.groups<=0 || d.nvec<=0 || d.nsign<=0)
        return false;
    const bool d4=d.d4_dense_groups;
    int bits=0;
    if(!d4)bits=d.format==5?8:d.format==13?10:d.format==14?12:0;
    else bits=(d.format==10 || d.format==11)?8:d.format==12?9:d.format==15?10:0;
    if(!bits || int64_t(d.nsign)!=(int64_t(d.input_width)+7)/8 ||
            int64_t(d.nvec)!=(int64_t(d.input_width)+(d4?3:7))/(d4?4:8) ||
            int64_t(d.nvec)!=int64_t(d.nsign)*(d4?2:1) ||
            int64_t(d.groups)!=(int64_t(d.nsign)+2)/3)return false;
    // The existing projection arithmetic still addresses row anchors with an
    // int neuron index. Large byte arenas are legal; overflowing row indices
    // must not be admitted by the new local-window proof.
    if(d.local_experts>INT_MAX/d.output_rows)return false;
    const int64_t row_bits=int64_t(d.nvec)*bits+int64_t(d.nsign)*7+int64_t(d.groups)*4;
    if(row_bits>(INT_MAX-64)/int64_t(d.output_rows))return false;
    const int64_t expert_bits=row_bits*d.output_rows;
    if((expert_bits&7) || d.sizes[0]<0 || d.sizes[1]!=0 || d.sizes[2]!=0)return false;
    return d.sizes[0]>=(expert_bits/8)*d.local_experts;
}

inline bool mfe_nvq_dense_narrow_selected(const MfePackedProjection& d) {
    const auto* value=std::getenv("MFQ_MFE_NVQ_DENSE_NARROW");
    if(!value || value[0]!='1' || value[1])return false;
    // Candidate policy follows the paired projection results. The remaining
    // legal shapes retain the existing wide reader until separately measured.
    const bool shape=(d.input_width==640 && d.output_rows==2560 &&
            (d.e8_dense_groups || d.format==11)) ||
        (d.input_width==2560 && d.output_rows==640 &&
            (d.format==5 || d.format==14 || d.format==11));
    return shape && mfe_nvq_dense_narrow_eligible(d);
}
}
