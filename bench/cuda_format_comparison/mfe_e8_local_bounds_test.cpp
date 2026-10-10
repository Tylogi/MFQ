#include "mfq/kernels/cuda/mfe_e8_narrow_bounds.h"
#include "mfq/kernels/cuda/mfe_nvq_dense_bounds.h"
#include <iostream>
#include <stdexcept>
#include <string>

using mfq::cuda::MfePackedProjection;
using mfq::cuda::mfe_e8_narrow_view_eligible;
int main()try {
    int checks=0,large=0;
    const auto check=[&](bool actual,bool expected) {
        ++checks;if(actual!=expected)throw std::runtime_error("expert-local E8 bounds check "+std::to_string(checks));
    };
    for(int format:{5,13,14})for(int width:{640,2560})for(int experts:{0,1,512,1509,2653,100000,INT_MAX}) {
        MfePackedProjection d;d.family=2;d.format=format;d.local_experts=experts;
        d.output_rows=width==640?2560:640;d.input_width=width;d.nvec=d.nsign=(width+7)/8;
        d.groups=(width+23)/24;d.group_size=24;d.sub_bits=4;
        const int bits=format==5?8:format==13?10:12;
        d.sizes[0]=int64_t(d.output_rows)*d.nvec*bits/8*experts;
        d.sizes[1]=int64_t(d.output_rows)*d.nsign*7/8*experts;
        d.sizes[2]=int64_t(d.output_rows)*d.groups/2*experts;
        check(mfe_e8_narrow_view_eligible(d),true);
        check(mfe_e8_narrow_view_eligible(d,false),experts!=0);
        if(d.sizes[0]>INT_MAX)++large;
        for(int field=0;field<3;++field) {
            auto bad=d;bad.sizes[field]=-1;check(mfe_e8_narrow_view_eligible(bad),false);
            if(experts){bad=d;--bad.sizes[field];check(mfe_e8_narrow_view_eligible(bad),false);}
        }
        auto bad=d;bad.sub_bits=3;check(mfe_e8_narrow_view_eligible(bad),false);
        bad=d;++bad.nvec;check(mfe_e8_narrow_view_eligible(bad),false);
        bad=d;bad.e8_dense_groups=true;check(mfe_e8_narrow_view_eligible(bad),false);
        bad=d;bad.output_rows=INT_MAX;check(mfe_e8_narrow_view_eligible(bad),false);
    }
    for(int format:{5,13,14})for(int rows=1;rows<=16;++rows) {
        MfePackedProjection d;d.family=2;d.format=format;d.local_experts=2;
        d.output_rows=rows;d.input_width=8;d.nvec=d.nsign=1;d.groups=1;d.group_size=24;d.sub_bits=4;
        d.sizes[0]=d.sizes[1]=d.sizes[2]=INT64_MAX;
        check(mfe_e8_narrow_view_eligible(d),rows%8==0);
    }
    MfePackedProjection other;other.family=1;check(mfe_e8_narrow_view_eligible(other),true);
    std::cout<<"E8 expert-local bounds PASS checks="<<checks<<" arenas_over_2GiB="<<large<<'\n';
    checks=0;large=0;
    const auto valid=mfq::cuda::mfe_nvq_dense_narrow_eligible;
    for(int format:{5,13,14,10,11,12,15})for(int width:{640,2560})
        for(int experts:{0,1,512,100000,INT_MAX}) {
        MfePackedProjection d;d.family=2;d.format=format;d.local_experts=experts;
        d.output_rows=width==640?2560:640;d.input_width=width;
        const bool d4=format==10 || format==11 || format==12 || format==15;
        const int bits=format==13?10:format==14?12:format==12?9:format==15?10:8;
        d.e8_dense_groups=!d4;d.d4_dense_groups=d4;d.nsign=(width+7)/8;d.nvec=d.nsign*(d4?2:1);
        d.groups=(width+23)/24;d.group_size=24;d.sub_bits=4;
        const int64_t row_bits=int64_t(d.nvec)*bits+d.nsign*7+d.groups*4;
        d.sizes[0]=row_bits*d.output_rows/8*experts;
        const bool row_index_fits=int64_t(d.output_rows)*experts<=INT_MAX;
        check(valid(d,true),row_index_fits);check(valid(d,false),row_index_fits && experts!=0);
        if(row_index_fits && d.sizes[0]>INT_MAX)++large;
        auto bad=d;bad.sizes[0]=-1;check(valid(bad,true),false);
        if(experts){bad=d;--bad.sizes[0];check(valid(bad,true),false);}
        bad=d;bad.sizes[1]=1;check(valid(bad,true),false);
        bad=d;bad.sizes[2]=-1;check(valid(bad,true),false);
        bad=d;bad.e8_dense_groups=bad.d4_dense_groups=true;check(valid(bad,true),false);
        bad=d;bad.group_size=32;check(valid(bad,true),false);
        bad=d;++bad.nvec;check(valid(bad,true),false);
        bad=d;bad.output_rows=INT_MAX;check(valid(bad,true),false);
        bad=d;bad.input_width=INT_MAX;check(valid(bad,true),false);
        bad=d;bad.local_experts=-1;check(valid(bad,true),false);
    }
    for(int format:{5,13,14,10,11,12,15})for(int rows=1;rows<=16;++rows) {
        MfePackedProjection d;d.family=2;d.format=format;d.local_experts=1;
        const bool d4=format==10 || format==11 || format==12 || format==15;
        const int bits=format==13?10:format==14?12:format==12?9:format==15?10:8;
        d.output_rows=rows;d.input_width=8;d.nvec=d4?2:1;d.nsign=1;d.groups=1;d.group_size=24;d.sub_bits=4;
        d.e8_dense_groups=!d4;d.d4_dense_groups=d4;d.sizes[0]=INT64_MAX;
        check(valid(d,true),(int64_t(rows)*(d.nvec*bits+7+4)&7)==0);
    }
    check(valid(other,true),false);
    static_assert(sizeof(MfePackedProjection)==184,"dense flag must reuse descriptor padding");
    std::cout<<"NVQ compact local bounds PASS checks="<<checks<<" arenas_over_2GiB="<<large<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
