#pragma once

// Up to 96 original groups. One warp computes one row, keeping the original
// three 32-group reductions independent and combining them in the exact
// lane-0 order of the original four-warp XOR tree: (p0+p2)+(p1+0).
// Four rows share a CTA; there is no block barrier or partial-sum storage.
template<int GS,int Bits,int Rows,bool CachedSum=false,bool Interleaved=false,typename Packed=NintWholeGroup<GS,Bits>,bool ExactRows=false>
__device__ __forceinline__ void nint_whole_warp_row(NintWholeArgs a,int row) {
    const int lane=threadIdx.x;
    const auto row_bit=uint64_t(a.offsets[row]);
    const float outer_scale=a.outer_scale[row],outer_minimum=a.outer_minimum[row];
    float total[Rows]{};
    #pragma unroll 1
    for(int part=0;part<3;++part) {
        const int slice=part==0?0:part==1?2:1;
        const int group=slice*32+lane;
        float acc[Rows]{};
        if(group<a.groups) {
            const Packed packed(a.weights,row_bit+uint64_t(group*GS)*Bits,a.bytes,a.group_load);
            int dots[Rows]{},sums[CachedSum?1:Rows]{};
            const auto consume=[&](auto chunk_constant) {
                constexpr int chunk=decltype(chunk_constant)::value;
                const int qw=packed.template codes<chunk>();
                if constexpr(Interleaved) {
                    int qx[Rows];nint_interleaved_codes(a,group,chunk,qx);
                    #pragma unroll
                    for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                        if constexpr(!CachedSum)sums[m]=__dp4a(0x01010101,qx[m],sums[m]);
                        dots[m]=__dp4a(Bits==8?qw^int(0x80808080u):qw,qx[m],dots[m]);
                    }
                }else {
                const int column=group*GS+chunk*4;
                #pragma unroll
                for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                    const int qx=*reinterpret_cast<const int*>(a.activation+size_t(m)*a.groups*GS+column);
                    if constexpr(!CachedSum)sums[m]=__dp4a(0x01010101,qx,sums[m]);
                    dots[m]=__dp4a(Bits==8?qw^int(0x80808080u):qw,qx,dots[m]);
                }
                }
            };
            consume(std::integral_constant<int,0>{});consume(std::integral_constant<int,1>{});
            consume(std::integral_constant<int,2>{});consume(std::integral_constant<int,3>{});
            consume(std::integral_constant<int,4>{});consume(std::integral_constant<int,5>{});
            if constexpr(GS==28)consume(std::integral_constant<int,6>{});
            const float scale=outer_scale*float(a.scale[size_t(row)*a.groups+group]);
            const float minimum=outer_minimum*float(a.minimum[size_t(row)*a.groups+group]);
            #pragma unroll
            for(int m=0;m<Rows;++m)if(ExactRows || (CachedSum && Rows!=4 && Rows!=8) || m<a.activation_rows) {
                const int sum=CachedSum?a.activation_sum[size_t(m)*a.groups+group]:sums[m];
                const int dot=dots[m]+(Bits==8?128*sum:0);
                acc[m]+=a.activation_scale[size_t(m)*a.groups+group]*(scale*float(dot)-minimum*float(sum));
            }
        }
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            #pragma unroll
            for(int offset=16;offset>0;offset>>=1)acc[m]+=__shfl_xor_sync(0xffffffffu,acc[m],offset);
            // Original second tree adds empty lanes at offsets 16, 8 and 4.
            #pragma unroll
            for(int pad=0;pad<3;++pad)acc[m]=__fadd_rn(acc[m],0.f);
            if(part==0)total[m]=acc[m];
            else if(part==1)total[m]=__fadd_rn(total[m],acc[m]);
            else total[m]=__fadd_rn(total[m],__fadd_rn(acc[m],0.f));
        }
    }
    if(lane==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m)if(ExactRows || m<a.activation_rows)
            a.output[size_t(m)*a.rows+row]=__float2half_rn(total[m]);
    }
}

template<int GS,int Rows,bool CachedSum=false>
__global__ void __launch_bounds__(128) nint_whole_warp_rows_kernel(NintWholeArgs a) {
    const int row=int(blockIdx.x)*4+threadIdx.y;
    if(row>=a.rows)return;
    switch(a.bits[row]) {
    case 1:nint_whole_warp_row<GS,1,Rows,CachedSum>(a,row);break;
    case 2:nint_whole_warp_row<GS,2,Rows,CachedSum>(a,row);break;
    case 3:nint_whole_warp_row<GS,3,Rows,CachedSum>(a,row);break;
    case 4:nint_whole_warp_row<GS,4,Rows,CachedSum>(a,row);break;
    case 5:nint_whole_warp_row<GS,5,Rows,CachedSum>(a,row);break;
    case 6:nint_whole_warp_row<GS,6,Rows,CachedSum>(a,row);break;
    case 7:nint_whole_warp_row<GS,7,Rows,CachedSum>(a,row);break;
    case 8:nint_whole_warp_row<GS,8,Rows,CachedSum>(a,row);break;
    }
}
